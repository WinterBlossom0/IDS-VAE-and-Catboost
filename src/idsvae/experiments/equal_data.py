"""Re-run methods given the further capture days of the unseen-attack detector, and judged with the detector's cut-off.

Each method trains on its Stage 1 features of fit_train plus the detector's rows of the further days (detector_runs.
multi_day_training_rows): every attack flow of those days, labelled by family, and the same Benign negatives, drawn with the same
generator in the same order. A flow is flagged when 1 - P(Benign) exceeds the score at the detector's false-alarm target on the
cut-off halves of the Benign flows of those days and of 02-23, each day with equal weight (detector_runs.multi_day_cutoff).
- day: trained without the evaluated day and its families; detection per family and Benign false alarms on that day.
- fold: trained without the left-out class on every day; detection of the class and hold-out Benign false alarms, as baselines.run_fold.
- test: trained on every day; the test-day report of baselines.run.
isolation_forest is the open-set comparison: it trains on the Benign flows of the same rows only, and its anomaly score replaces
1 - P(Benign).
"""
import argparse
import importlib
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import IsolationForest

from idsvae import config, files, metrics
from idsvae.data import extra_days, splits
from idsvae.experiments import baselines, detector_runs, final_run
from idsvae.framework import detector, features

OUTPUT_DIR = config.PROJECT_ROOT / "data" / "final" / "equal_data"
MODES = ("day", "fold", "test")
FLAGGED = "attack"
OPEN_SET_METHODS = ("isolation_forest",)


def day_flows(day, columns):
    """Flows of one detector day with the given feature columns and the split metadata, in the order of its detector features."""
    if day == "02-28":
        return splits.load_split("vae_val", columns=columns)
    path = extra_days.OUTPUT_DIR / f"{day}.parquet"
    available = set(pq.read_schema(path).names)
    return pd.read_parquet(path, columns=[*columns, *[column for column in ("label", "day", "time") if column in available]])


def further_rows(seed, days, columns, excluded_labels=()):
    """(training rows, cut-off rows per day) of the further days: the rows of detector_runs.multi_day_training_rows and
    detector_runs.multi_day_cutoff, as flows with features and labels."""
    rng = np.random.default_rng(seed)
    training, cutoff = [], []
    for day in days:
        frame = day_flows(day, columns)
        labels = frame["label"].to_numpy()
        is_positive = (labels != config.BENIGN) & ~np.isin(labels, list(excluded_labels))
        training.append(frame[is_positive])
        training_half, cutoff_half = detector_runs.benign_halves(frame, detector_runs.day_salt(day))
        training.append(frame.iloc[rng.choice(training_half, min(len(training_half), detector_runs.NEGATIVE_ROWS_PER_DAY), replace=False)])
        cutoff.append(frame.iloc[cutoff_half][columns].to_numpy("float32"))
        del frame
    calib = splits.load_split("calib", columns=columns)
    negative_rows, cutoff_rows = detector_runs.calib_halves(calib)
    training.append(calib.iloc[rng.choice(negative_rows, min(len(negative_rows), detector_runs.NEGATIVE_ROWS_PER_DAY), replace=False)])
    # multi_day_cutoff puts 02-23 first; the order does not change the weighted quantile.
    cutoff.insert(0, calib.iloc[cutoff_rows][columns].to_numpy("float32"))
    return pd.concat(training, ignore_index=True), cutoff


def attack_scores(classes, probabilities):
    return 1.0 - probabilities[:, list(classes).index(config.BENIGN)]


def cutoff_of(classes, cutoff_probabilities):
    scores = [attack_scores(classes, probabilities) for probabilities in cutoff_probabilities]
    weights = [np.full(len(day_scores), 1.0 / len(day_scores)) for day_scores in scores]
    return detector.weighted_quantile(np.concatenate(scores), np.concatenate(weights), 1 - detector.FALSE_ALARM_TARGET)


def flagged_labels(classes, probabilities, threshold):
    return np.where(attack_scores(classes, probabilities) > threshold, FLAGGED, config.BENIGN)


def train_and_predict(method, seed, threads, base_train, columns, days, excluded_labels, evaluation):
    """(classes, probabilities of each evaluation part, cut-off) of one model trained on base_train and the further days."""
    extra, cutoff_values = further_rows(seed, days, columns, excluded_labels)
    keep = [column for column in base_train.columns if column in extra.columns]
    train = pd.concat([base_train[keep], extra[keep]], ignore_index=True)
    del extra
    parts = evaluation + [(values, None) for values in cutoff_values]
    if method in OPEN_SET_METHODS:
        classes, probabilities = open_set_scores(seed, threads, train, columns, parts)
    else:
        classes, probabilities, _, _ = baselines.predict_parts(method, seed, threads, train, columns, parts)
    return classes, probabilities[:len(evaluation)], cutoff_of(classes, probabilities[len(evaluation):])


def open_set_scores(seed, threads, train, columns, parts):
    """(classes, two-column scores per part) of an Isolation Forest on the Benign training rows; the second column is the
    anomaly score, so that attack_scores reads it as 1 - P(Benign)."""
    benign = train.loc[(train["label"] == config.BENIGN).to_numpy(), columns].to_numpy("float64")
    standardizer = baselines.Standardizer(benign)
    model = IsolationForest(n_jobs=threads, random_state=seed).fit(standardizer.transform(benign))
    scores = [-model.score_samples(standardizer.transform(values.astype("float64"))) for values, _ in parts]
    return [config.BENIGN, FLAGGED], [np.column_stack([1.0 - score, score]) for score in scores]


def run_day(method, seed, threads, held_out):
    columns = baselines.selected_features()
    days = detector_runs.qualified_days()
    if held_out not in days:
        raise ValueError(f"{held_out} is not a detector day ({days})")
    frame = day_flows(held_out, columns)
    labels = frame["label"].to_numpy()
    families = [label for label in pd.unique(labels) if label != config.BENIGN]
    others = [day for day in days if day != held_out]
    base_train = splits.load_split("fit_train", columns=columns)
    classes, (probabilities,), threshold = train_and_predict(method, seed, threads, base_train, columns, others, families, [(frame[columns].to_numpy("float32"), frame)])
    predicted = flagged_labels(classes, probabilities, threshold)
    report = metrics.other_day_report(labels, predicted, np.zeros(len(labels), dtype=bool))
    return {"method": method, "seed": seed, "mode": "day", "held_out": held_out, "training_days": others, "cutoff": threshold, "report": report}


def run_fold(method, seed, threads, left_out_class):
    columns = baselines.selected_features(left_out_class)
    train_all = splits.load_split("fit_train", columns=columns)
    holdout = splits.load_split("fit_holdout", columns=columns)
    is_left_out = features.left_out_mask(train_all, left_out_class)
    base_train, left_out_rows = train_all[~is_left_out].reset_index(drop=True), train_all[is_left_out].reset_index(drop=True)
    # float64 as baselines.run_fold: Standardizer.transform keeps the input dtype, and training rows go through float64.
    evaluation = [(part[columns].to_numpy("float64"), part) for part in (left_out_rows, holdout)]
    classes, (left_out_probabilities, holdout_probabilities), threshold = train_and_predict(
        method, seed, threads, base_train, columns, detector_runs.qualified_days(), (), evaluation)
    left_out_attack = flagged_labels(classes, left_out_probabilities, threshold) != config.BENIGN
    holdout_labels = holdout["label"].to_numpy()
    holdout_attack = flagged_labels(classes, holdout_probabilities, threshold) != config.BENIGN
    detected = np.concatenate([left_out_attack, holdout_attack[holdout_labels == left_out_class]])
    return {"method": method, "seed": seed, "mode": "fold", "left_out_class": left_out_class, "cutoff": threshold,
            "left_out_detection": float(detected.mean()), "holdout_benign_false_alarms": float(holdout_attack[holdout_labels == config.BENIGN].mean())}


def run_test(method, seed, threads):
    columns = baselines.selected_features()
    base_train = splits.load_split("fit_train", columns=columns)
    test = splits.load_split("test", columns=[*columns, "in_benign_burst"])
    classes, (probabilities,), threshold = train_and_predict(
        method, seed, threads, base_train, columns, detector_runs.qualified_days(), (), [(test[columns].to_numpy("float64"), test)])
    predicted = flagged_labels(classes, probabilities, threshold)
    report = metrics.other_day_report(test["label"].to_numpy(), predicted, test["in_benign_burst"].to_numpy(dtype=bool))
    return {"method": method, "seed": seed, "mode": "test", "cutoff": threshold, "report": report}


def output_path(method, seed, mode, target=""):
    suffix = f"_{features.class_slug(target)}" if target else ""
    return OUTPUT_DIR / f"{method}_s{seed}_{mode}{suffix}.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=(*baselines.METHODS, *OPEN_SET_METHODS), required=True)
    parser.add_argument("--seed", type=int, choices=baselines.SEEDS, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--target", default="", help="held-out day (mode day) or left-out class (mode fold)")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.mode == "test" and args.target:
        raise ValueError("mode test takes no --target")
    output = output_path(args.method, args.seed, args.mode, args.target)
    if output.exists():
        return
    if args.mode == "day":
        result = run_day(args.method, args.seed, args.threads, args.target)
    elif args.mode == "fold":
        if args.target not in final_run.LEFT_OUT_CLASSES:
            raise ValueError(f"--target must be one of {final_run.LEFT_OUT_CLASSES}")
        result = run_fold(args.method, args.seed, args.threads, args.target)
    else:
        result = run_test(args.method, args.seed, args.threads)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files.write_json_atomic(output, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    importlib.import_module("idsvae.experiments.equal_data").main()
