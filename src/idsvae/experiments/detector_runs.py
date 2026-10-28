"""Training, configuration choice and evaluation of the unseen-attack detector (framework/detector.py).

Positives are flows of attack families that the VAE and Stage 3 never saw: the training-day flows of the left-out class of
each leave-one-attack-out fold (with that fold's VAE), and the attack flows of capture days outside the protocol splits
(data/extra_days.py, with the main VAE). Negatives are the training half of the Benign flows of those days and of 02-23. The
cut-off gives the target false-alarm rate on the other half of the Benign flows, each day with equal weight. The
configuration (novelty reference, feature set) is chosen by leaving one day (with its attack families) or one fold out at a
time. No hold-out or test-day label is used.
"""
import argparse
import json
import multiprocessing
import os

import numpy as np
import pandas as pd
import torch

from idsvae import config, files, metrics
from idsvae.data import extra_days, splits
from idsvae.experiments import final_run, inner_selection
from idsvae.framework import correction, detector, onnx_graph, vae

ROOT = final_run.FINAL_DIR / "unseen"
# "calib": novelty against the calibration-day Benign flows; "day": novelty against the running background of the same day.
NOVELTY_REFERENCES = ("calib", "day")
NOVELTY_REFERENCE = os.environ.get("IDSVAE_NOVELTY_REFERENCE", "day")
if NOVELTY_REFERENCE not in NOVELTY_REFERENCES:
    raise ValueError(f"IDSVAE_NOVELTY_REFERENCE must be one of {NOVELTY_REFERENCES}, not {NOVELTY_REFERENCE!r}")
FEATURE_DIR = ROOT / f"features_{NOVELTY_REFERENCE}"
MODEL_DIR = ROOT / "models"
# A flow is outside a burst when its MIDAS-R score is at or below this quantile of the training Benign scores.
BACKGROUND_BURST_QUANTILE = 0.9
# The held-out reports run in four parallel processes of four threads; apply and leave-one-out use detector.THREADS.
SELECTION_THREADS = 4
SELECTION_PROCESSES = 4
NEGATIVE_ROWS_PER_DAY = 200_000


def feature_file(arm, seed, variant, part):
    return FEATURE_DIR / f"{arm}_s{seed}{variant.suffix()}_{part}.parquet"


class FeatureBuilder:
    """Detector features of any day for one VAE.

    References: NLL and KL against the calibration-day Benign flows; the stream values against the training-day Benign flows
    of the fit-day stream (training + hold-out flows, no hold-out label), without the left-out class of a fold. KL enters as
    a two-sided tail score, because an unseen class can have a lower KL than Benign (GoldenEye does) as well as a higher one.
    """

    def __init__(self, arm, seed, variant):
        self.encoded_dir, _, _ = final_run.rule_inputs(arm, seed, variant)
        self.latent_dim = correction.latent_dim_of(self.encoded_dir)
        self.settings = inner_selection.chosen_rule_settings()
        self.train = pd.read_parquet(self.encoded_dir / "fit_train.parquet")
        self.holdout = pd.read_parquet(self.encoded_dir / "fit_holdout.parquet")
        is_seen_train = (self.train["label"] != variant.left_out_class).to_numpy() if variant.left_out_class else np.ones(len(self.train), bool)
        self.scale = self.train.loc[is_seen_train, vae.latent_columns(self.latent_dim)].to_numpy(np.float64).std(axis=0)
        self.calib = pd.read_parquet(self.encoded_dir / "calib.parquet")
        calib_benign = (self.calib["label"] == config.BENIGN).to_numpy()
        self.kl_reference = self.calib.loc[calib_benign, "kl"].to_numpy()
        if NOVELTY_REFERENCE == "day":
            seen_train = self.train[is_seen_train]
            train_burst = correction.stream_scores(seen_train, self.latent_dim, self.scale, self.settings.cell_width)
            self.burst_cutoff = float(np.quantile(train_burst[(seen_train["label"] == config.BENIGN).to_numpy()], BACKGROUND_BURST_QUANTILE))
            self.fallback = float(np.median(self.calib.loc[calib_benign, "recon_nll"]))
        self.calib_novelty = self.novelty_values(self.calib)
        self.nll_reference = self.calib_novelty[calib_benign]
        self.novelty = float(np.quantile(self.nll_reference, self.settings.novelty_quantile))
        self.fit = pd.concat([self.train, self.holdout], ignore_index=True)
        self.fit_novelty = self.novelty_values(self.fit)
        self.fit_stream = detector.stream_values(self.fit, self.latent_dim, self.scale, self.fit_novelty > self.novelty, self.settings)
        is_reference = np.r_[is_seen_train & (self.train["label"] == config.BENIGN).to_numpy(), np.zeros(len(self.holdout), bool)]
        self.references = [values[is_reference] for values in self.fit_stream]

    def novelty_values(self, frame):
        if NOVELTY_REFERENCE == "day":
            return detector.relative_nll(frame, self.latent_dim, self.scale, self.settings, self.burst_cutoff, self.fallback)
        return frame["recon_nll"].to_numpy(np.float64)

    def features(self, frame, novelty_values=None, stream=None):
        novelty_values = self.novelty_values(frame) if novelty_values is None else novelty_values
        stream = detector.stream_values(frame, self.latent_dim, self.scale, novelty_values > self.novelty, self.settings) if stream is None else stream
        kl_quantile = detector.quantile_of(self.kl_reference, frame["kl"].to_numpy())
        columns = {"nll_q": detector.quantile_of(self.nll_reference, novelty_values), "kl_tail_q": 1 - 2 * np.minimum(kl_quantile, 1 - kl_quantile)}
        columns.update({name: detector.quantile_of(reference, value) for name, reference, value in zip(detector.STREAM_FEATURES, self.references, stream)})
        return pd.DataFrame({**columns, "label": frame["label"].to_numpy(), "time": frame["time"].to_numpy()})

    def online(self, columns):
        """An OnlineDetector with the references of this VAE; a new one per stream."""
        background = detector.DayBackground(self.burst_cutoff, self.fallback, np.random.default_rng(0)) if NOVELTY_REFERENCE == "day" else None
        return detector.OnlineDetector(columns, self.scale, self.settings, self.novelty, self.nll_reference, self.kl_reference, self.references, background)

    def encode_extra_day(self, day):
        """Encodings of the flows of an extra capture day with this VAE (as vae.run_vae encodes the protocol splits)."""
        model, standardizer, columns = vae.load_model(self.encoded_dir)
        settings = vae.VaeSettings(**json.loads((self.encoded_dir / "summary.json").read_text())["settings"])
        frame = extra_day_vae_inputs(day, columns)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        with files.gpu_lock():
            encoded, _ = vae.encode_frame(model.to(device), frame, columns, standardizer, settings, device)
        return encoded


def extra_day_vae_inputs(day, columns):
    """Flows of an extra capture day with only the columns that the encoding by a VAE with these input columns reads (the
    inputs and the sources of their undefined masks) and the label and time; day 02-20 has 5 million flows."""
    needed = sorted({*columns, *vae.mask_source_columns(columns), "label", "time"})
    return pd.read_parquet(extra_days.OUTPUT_DIR / f"{day}.parquet", columns=needed)


def write_features(task):
    """Feature files of one VAE: the fit-day stream, the calibration day, 02-28 and the test day."""
    arm, seed, variant = task
    if feature_file(arm, seed, variant, "test").exists():
        return task
    builder = FeatureBuilder(arm, seed, variant)
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    fit_features = builder.features(builder.fit, builder.fit_novelty, builder.fit_stream)
    files.write_parquet_atomic(fit_features.assign(holdout=np.arange(len(builder.fit)) >= len(builder.train)), feature_file(arm, seed, variant, "fit"))
    # Of a fold variant only the fit part is read; the other parts stay, because the test file marks a finished task.
    files.write_parquet_atomic(builder.features(builder.calib, builder.calib_novelty), feature_file(arm, seed, variant, "calib"))
    for part in ("vae_val", "test"):
        files.write_parquet_atomic(builder.features(pd.read_parquet(builder.encoded_dir / f"{part}.parquet")), feature_file(arm, seed, variant, part))
    return task


def write_extra_features(task):
    """Feature files of the extra capture days (extra_days.py) for one VAE of the main variant."""
    arm, seed, days = task
    pending = [day for day in days if not feature_file(arm, seed, final_run.MAIN, f"extra_{day}").exists()]
    if not pending:
        return task
    builder = FeatureBuilder(arm, seed, final_run.MAIN)
    for day in pending:
        files.write_parquet_atomic(builder.features(builder.encode_extra_day(day)), feature_file(arm, seed, final_run.MAIN, f"extra_{day}"))
    return task


def detection_report(labels, predicted, classes):
    """Seen-class macro-F1 (an unknown-attack label counts as a wrong class), attack detection per class and the Benign false-alarm rate."""
    per_class = metrics.per_class_label_metrics(labels, predicted, classes)
    is_attack = predicted != config.BENIGN
    return {"macro_f1_min_support": metrics.macro_f1_with_min_support(per_class)["value"],
            "detected": {label: float(is_attack[labels == label].mean()) for label in pd.unique(labels) if label != config.BENIGN},
            "benign_false_alarms": float(is_attack[labels == config.BENIGN].mean()),
            "unknown_predictions": int((predicted == detector.UNKNOWN).sum()),
            "benign_set_to_unknown": int(((predicted == detector.UNKNOWN) & (labels == config.BENIGN)).sum())}


def same_rows(first, second):
    """True when two frames hold the same flows in the same order (equal labels and times)."""
    return all(np.array_equal(first[column].to_numpy(), second[column].to_numpy()) for column in ("label", "time"))


def apply(arm, seed, feature_set):
    """The detector trained on every available day and fold of a seed (multi_day_training_rows), applied after Stage 3 and the
    correction of an arm with its own VAE (main variant)."""
    days = available_days(seed)
    model = detector.fit_detector(*multi_day_training_rows(seed, days, final_run.LEFT_OUT_CLASSES, feature_set, arm=arm), seed)
    threshold = multi_day_cutoff(model, seed, days, feature_set, arm=arm)
    fit, test = (pd.read_parquet(feature_file(arm, seed, final_run.MAIN, part)) for part in ("fit", "test"))
    classes = json.loads((final_run.stage3_root(arm) / final_run.classifier_settings(arm, seed).run_name() / "result.json").read_text())["classes"]
    rule_file = final_run.rule_output(arm, seed, final_run.MAIN)
    holdout = pd.read_parquet(rule_file.with_suffix(".holdout.parquet"))
    holdout_features = fit[fit["holdout"].to_numpy()]
    if not same_rows(holdout, holdout_features):
        raise ValueError("hold-out rows of the rule output and of the detector features differ")
    holdout_flags = detector.flags(model, holdout_features, feature_set, threshold)
    labels = holdout["label"].to_numpy()
    test_rows = pd.read_parquet(rule_file.with_suffix(".test.parquet"))
    if not same_rows(test_rows, test):
        raise ValueError("test rows of the rule output and of the detector features differ")
    test_flags = detector.flags(model, test, feature_set, threshold)
    burst = test_rows["in_benign_burst"].to_numpy(dtype=bool)
    output = {
        "arm": arm, "seed": seed, "feature_set": feature_set, "cutoff": threshold, "false_alarm_target": detector.FALSE_ALARM_TARGET, "training_days": days,
        # The hold-out classes are training classes of this detector, so these numbers show its effect on seen classes and its
        # false alarms; unseen detection comes from held_out_report (selection_multi_*.json) and from the test day.
        "holdout": {"stage3_rule": detection_report(labels, holdout["rule"].to_numpy(), classes),
                    "with_detector": detection_report(labels, detector.combine(holdout["rule"], holdout_flags), classes)},
        "test": {"stage3_rule": metrics.other_day_report(test_rows["label"].to_numpy(), test_rows["rule"].to_numpy(), burst),
                 "with_detector": metrics.other_day_report(test_rows["label"].to_numpy(), detector.combine(test_rows["rule"], test_flags), burst)},
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_file(arm, seed)))
    files.write_json_atomic(ROOT / f"{arm}_s{seed}_{NOVELTY_REFERENCE}.json", output)
    return output


def model_file(arm, seed):
    return MODEL_DIR / f"{arm}_s{seed}_{NOVELTY_REFERENCE}.cbm"


def applied_output(arm, seed, feature_set):
    """Output of apply for an arm and seed; raises when that detector was trained with another feature set."""
    output = json.loads((ROOT / f"{arm}_s{seed}_{NOVELTY_REFERENCE}.json").read_text())
    if output["feature_set"] != feature_set:
        raise ValueError(f"{arm} seed {seed}: the applied detector uses the feature set {output['feature_set']!r}, not {feature_set!r}")
    return output


def deployed_detector(arm, seed, threads):
    """(factory of a fresh OnlineDetector, rows -> attack scores, cut-off) of an arm, for python_parts.py; the arm's VAE gives the
    references. The model runs in ONNX Runtime: a CatBoost call per flow costs more than the whole detector feature step."""
    output = applied_output(arm, seed, chosen_feature_set_multi())
    exported = onnx_graph.catboost_onnx(model_file(arm, seed), model_file(arm, seed).with_suffix(".onnx"))
    run = onnx_graph.graph_function(onnx_graph.session(exported, threads), [onnx_graph.PROBABILITIES_NAME])
    builder = FeatureBuilder("base" if arm == "base" else "topology", seed, final_run.MAIN)
    columns = list(detector.FEATURE_SETS[output["feature_set"]])
    return (lambda: builder.online(columns)), (lambda rows: run(rows)[0][:, 1]), output["cutoff"]


# ---------------------------------------------------------------- detector training on several days

# Days with unseen attack families and Benign traffic: the extra capture days and 02-28 (the VAE validation day, used without
# its labels by the VAE). 02-23 supplies Benign flows only, because the NLL and KL references come from its Benign flows.
VALIDATION_DAYS = ("02-28", *extra_days.EXTRA_DAY_FILES)
# A configuration qualifies when its mean Benign false-alarm rate on held-out days is at most this multiple of the target.
FALSE_ALARM_TOLERANCE = 2.0


def day_part(day):
    return "vae_val" if day == "02-28" else f"extra_{day}"


def day_frame(day):
    if day == "02-28":
        return splits.load_split("vae_val", columns=["label", "time"])
    return pd.read_parquet(extra_days.OUTPUT_DIR / f"{day}.parquet", columns=["label", "time"])


def qualified_days():
    """Validation days whose Benign flows cover the capture (extra_days.benign_covers_day)."""
    return [day for day in VALIDATION_DAYS if benign_covers_day_cached(day)]


_COVERAGE = {}


def benign_covers_day_cached(day):
    if day not in _COVERAGE:
        _COVERAGE[day] = extra_days.benign_covers_day(day_frame(day))
    return _COVERAGE[day]


def available_days(seed):
    return [day for day in qualified_days() if feature_file("topology", seed, final_run.MAIN, day_part(day)).exists()]


def benign_halves(frame, salt):
    """(training half, cut-off half) of the Benign flows of one day, fixed for every model."""
    benign = np.flatnonzero((frame["label"] == config.BENIGN).to_numpy())
    shuffled = np.random.default_rng(salt).permutation(benign)
    return np.sort(shuffled[: len(shuffled) // 2]), np.sort(shuffled[len(shuffled) // 2:])


def calib_halves(calib):
    """(negative rows, cut-off rows) of the calibration-day Benign flows, the same for every model."""
    return benign_halves(calib, salt=0)


def day_salt(day):
    return int(day.replace("-", ""))


def feature_part_loader(arm, seed):
    """Part name, variant -> detector feature rows of that part; the fold parts come from the design-VAE fold runs."""
    def load(part, variant=final_run.MAIN):
        return pd.read_parquet(feature_file("topology" if variant.left_out_class else arm, seed, variant, part))
    return load


def multi_day_training_rows(seed, days, folds, feature_set=None, arm="topology", excluded_labels=(), load_part=None, columns=None):
    """Positives: the attack flows of the given days (unseen by the main models) and the training-day flows of the left-out
    class of the given folds; negatives: the training half of the Benign flows of the given days and of 02-23. Within each
    target, each family of each day, the class of each fold and each Benign day weighs the same in total, so a family on two
    days (Infiltration on 02-28 and 03-01) weighs twice. load_part and columns replace the detector features of feature_set by
    other inputs of the same rows."""
    columns = list(columns or detector.FEATURE_SETS[feature_set])
    load = load_part or feature_part_loader(arm, seed)
    parts = []
    rng = np.random.default_rng(seed)

    def add(frame, target):
        if len(frame):
            parts.append(frame[columns].assign(target=target, weight=1.0 / len(frame)))
    for day in days:
        frame = load(day_part(day))
        for label in pd.unique(frame["label"]):
            if label != config.BENIGN and label not in excluded_labels:
                add(frame[(frame["label"] == label).to_numpy()], 1)
        training_half, _ = benign_halves(frame, day_salt(day))
        add(frame.iloc[rng.choice(training_half, min(len(training_half), NEGATIVE_ROWS_PER_DAY), replace=False)], 0)
    calib = load("calib")
    negative_rows, _ = calib_halves(calib)
    add(calib.iloc[rng.choice(negative_rows, min(len(negative_rows), NEGATIVE_ROWS_PER_DAY), replace=False)], 0)
    for left_out in folds:
        fit = load("fit", final_run.Variant(0, left_out))
        add(fit[(fit["label"] == left_out).to_numpy() & ~fit["holdout"].to_numpy()], 1)
    rows = pd.concat(parts, ignore_index=True)
    positive = rows["target"].to_numpy() == 1
    # Positives and negatives each weigh one half in total.
    rows.loc[positive, "weight"] *= 0.5 / rows.loc[positive, "weight"].sum()
    rows.loc[~positive, "weight"] *= 0.5 / rows.loc[~positive, "weight"].sum()
    return rows[columns], rows["target"].to_numpy(), rows["weight"].to_numpy()


def multi_day_cutoff(model, seed, days, feature_set=None, arm="topology", target=detector.FALSE_ALARM_TARGET, load_part=None, columns=None):
    """Score quantile on the cut-off halves of the Benign flows of the given days and of 02-23, each day with equal weight."""
    columns = list(columns or detector.FEATURE_SETS[feature_set])
    load = load_part or feature_part_loader(arm, seed)
    scores, weights = [], []
    calib = load("calib")
    frames = [(calib, calib_halves(calib)[1])]
    for day in days:
        frame = load(day_part(day))
        frames.append((frame, benign_halves(frame, day_salt(day))[1]))
    for frame, rows in frames:
        day_scores = model.predict_proba(frame.iloc[rows][columns])[:, 1]
        scores.append(day_scores)
        weights.append(np.full(len(day_scores), 1.0 / len(day_scores)))
    return detector.weighted_quantile(np.concatenate(scores), np.concatenate(weights), 1 - target)


def held_out_day_flags(seed, feature_set, held_out, days, depth=detector.DEPTH):
    """(detector features of the held-out day, detector flags) from a detector trained without that day and its families."""
    frame, scores, threshold = held_out_day_scores(seed, feature_set, held_out, days, depth)
    return frame, scores > threshold


def held_out_day_scores(seed, feature_set, held_out, days, depth=detector.DEPTH):
    """(detector features of the held-out day, attack scores, cut-off) of a detector trained without that day and its families."""
    others = [day for day in days if day != held_out]
    frame = pd.read_parquet(feature_file("topology", seed, final_run.MAIN, day_part(held_out)))
    # A family on the held-out day leaves detector training on every day (Infiltration is on 02-28 and 03-01).
    families = [label for label in pd.unique(frame["label"]) if label != config.BENIGN]
    model = detector.fit_detector(*multi_day_training_rows(seed, others, final_run.LEFT_OUT_CLASSES, feature_set, excluded_labels=families), seed,
                       SELECTION_THREADS, depth=depth)
    threshold = multi_day_cutoff(model, seed, others, feature_set)
    return frame, model.predict_proba(frame[list(detector.FEATURE_SETS[feature_set])])[:, 1], threshold


def held_out_report(seed, feature_set, depth=detector.DEPTH):
    """Leave one day out (detection per family and Benign false alarms on that day) and leave one fold out (detection of
    the fold's class), each with a detector that never saw the held-out day or class."""
    days = available_days(seed)
    columns = list(detector.FEATURE_SETS[feature_set])
    report = {"days": {}, "folds": {}}
    for held_out in days:
        frame, flagged = held_out_day_flags(seed, feature_set, held_out, days, depth)
        labels = frame["label"].to_numpy()
        report["days"][held_out] = {"benign_false_alarms": float(flagged[labels == config.BENIGN].mean()),
                                    "detected": {label: float(flagged[labels == label].mean()) for label in pd.unique(labels) if label != config.BENIGN}}
    for left_out in final_run.LEFT_OUT_CLASSES:
        others = [label for label in final_run.LEFT_OUT_CLASSES if label != left_out]
        model = detector.fit_detector(*multi_day_training_rows(seed, days, others, feature_set), seed, SELECTION_THREADS, depth=depth)
        threshold = multi_day_cutoff(model, seed, days, feature_set)
        fit = pd.read_parquet(feature_file("topology", seed, final_run.Variant(0, left_out), "fit"))
        flagged = model.predict_proba(fit[columns])[:, 1] > threshold
        is_left_out, is_holdout = (fit["label"] == left_out).to_numpy(), fit["holdout"].to_numpy()
        report["folds"][left_out] = {"detection_training_days": float(flagged[is_left_out & ~is_holdout].mean()),
                                     "detection_holdout": float(flagged[is_left_out & is_holdout].mean()),
                                     "holdout_benign_flags": float(flagged[is_holdout & (fit["label"] == config.BENIGN).to_numpy()].mean())}
    return report


def leave_one_out_report(seed, feature_set, arm="topology"):
    """Per leave-one-out fold: detection of the left-out class and hold-out Benign false alarms after Stage 3, after the
    correction, and after the detector (trained on every day and on the two other folds, so the class is unseen by every stage)."""
    days = available_days(seed)
    columns = list(detector.FEATURE_SETS[feature_set])
    report = {}
    for left_out in final_run.LEFT_OUT_CLASSES:
        variant = final_run.Variant(0, left_out)
        others = [label for label in final_run.LEFT_OUT_CLASSES if label != left_out]
        model = detector.fit_detector(*multi_day_training_rows(seed, days, others, feature_set), seed)
        threshold = multi_day_cutoff(model, seed, days, feature_set)
        fit = pd.read_parquet(feature_file("topology", seed, variant, "fit"))
        rows = pd.read_parquet(final_run.rule_output(arm, seed, variant).with_suffix(".stream.parquet"))
        if not same_rows(rows, fit):
            raise ValueError(f"{left_out}: stream rows of the rule output and of the detector features differ")
        detector_flags = model.predict_proba(fit[columns])[:, 1] > threshold
        labels, is_holdout = rows["label"].to_numpy(), rows["holdout"].to_numpy()
        is_left_out, holdout_benign = labels == left_out, is_holdout & (labels == config.BENIGN)
        report[left_out] = {}
        for name, predicted in (("stage3", rows["base"].to_numpy()), ("correction", rows["rule"].to_numpy()),
                                ("detector", detector.combine(rows["rule"], detector_flags))):
            is_attack = predicted != config.BENIGN
            report[left_out][name] = {"left_out_detection": float(is_attack[is_left_out].mean()), "holdout_benign_false_alarms": float(is_attack[holdout_benign].mean())}
    files.write_json_atomic(ROOT / f"loo_{arm}_s{seed}_{NOVELTY_REFERENCE}.json", report)
    return report


# LOIC-HTTP flows of 02-20 have two templates (forward, backward packets): requests (3, 4) and attempts without a reply (2, 0).
LOIC_HTTP = "DDoS attacks-LOIC-HTTP"
REQUEST_PACKETS = (3, 4)


def template_report(feature_set, seeds=final_run.SEEDS):
    """Held-out 02-20: share of the LOIC-HTTP request flows and of the LOIC-HTTP flows without a reply that the detector flags."""
    raw = pd.read_parquet(extra_days.OUTPUT_DIR / "02-20.parquet", columns=["label", "time", *vae.COUNT_COLUMNS])
    forward, backward = (raw[column].to_numpy() for column in vae.COUNT_COLUMNS)
    is_attack = raw["label"].to_numpy() == LOIC_HTTP
    is_request = is_attack & (forward == REQUEST_PACKETS[0]) & (backward == REQUEST_PACKETS[1])
    is_no_reply = is_attack & (backward == 0)
    if not is_attack.any():
        raise ValueError("day 02-20 has no LOIC-HTTP flow")
    report = {"request_share": float(is_request.sum() / is_attack.sum()), "no_reply_share": float(is_no_reply.sum() / is_attack.sum()), "seeds": {}}
    for seed in seeds:
        days = available_days(seed)
        if "02-20" not in days:
            raise ValueError(f"seed {seed}: day 02-20 has no detector features or fails the Benign coverage rule")
        frame, flagged = held_out_day_flags(seed, feature_set, "02-20", days)
        if not same_rows(frame, raw):
            raise ValueError("rows of the 02-20 detector features and of the processed day differ")
        report["seeds"][str(seed)] = {"request": float(flagged[is_request].mean()), "no_reply": float(flagged[is_no_reply].mean())}
    files.write_json_atomic(ROOT / f"loic_http_templates_{NOVELTY_REFERENCE}.json", report)
    return report


def summary_scores(reports):
    """(mean detection over held-out families and folds, mean Benign false alarms over held-out days) over seeds."""
    detections, false_alarms = [], []
    for report in reports.values():
        for day in report["days"].values():
            detections.extend(day["detected"].values())
            false_alarms.append(day["benign_false_alarms"])
        detections.extend(fold["detection_training_days"] for fold in report["folds"].values())
    return float(np.mean(detections)), float(np.mean(false_alarms))


def chosen_configuration(scores):
    """(chosen, qualified) by the pre-set rule: among the configurations whose mean held-out-day Benign false-alarm rate is at
    most FALSE_ALARM_TOLERANCE times the target, the one with the best mean held-out detection; when no configuration
    qualifies, the one with the best mean held-out detection of all. scores maps a configuration to its mean_detection and
    mean_benign_false_alarms; the first configuration wins a tie."""
    qualified = [key for key, score in scores.items() if score["mean_benign_false_alarms"] <= FALSE_ALARM_TOLERANCE * detector.FALSE_ALARM_TARGET]
    chosen = max(qualified or scores, key=lambda key: scores[key]["mean_detection"])
    return chosen, qualified


def select_multi(seeds=final_run.SEEDS):
    """The pre-set rule (chosen_configuration) over the feature sets of this novelty reference; no hold-out or test-day data."""
    tasks = [(seed, name) for name in detector.FEATURE_SETS for seed in seeds]
    with multiprocessing.get_context("fork").Pool(SELECTION_PROCESSES) as pool:
        results = pool.starmap(held_out_report, tasks)
    reports = {name: {} for name in detector.FEATURE_SETS}
    for (seed, name), report in zip(tasks, results):
        reports[name][seed] = report
    scores = {}
    for name, by_seed in reports.items():
        detection, false_alarms = summary_scores(by_seed)
        scores[name] = {"mean_detection": detection, "mean_benign_false_alarms": false_alarms}
    chosen, qualified = chosen_configuration(scores)
    output = {"chosen": chosen, "qualified": qualified, "days": available_days(seeds[0]), "scores": scores,
              "reports": {name: {str(seed): report for seed, report in by_seed.items()} for name, by_seed in reports.items()}}
    files.write_json_atomic(ROOT / f"selection_multi_{NOVELTY_REFERENCE}.json", output)
    print(json.dumps({"chosen": chosen, "qualified": qualified, "scores": output["scores"]}, indent=1), flush=True)


def choose_multi():
    """The pre-set rule (chosen_configuration) over both novelty references and all feature sets together."""
    candidates = {}
    for reference in NOVELTY_REFERENCES:
        selection = json.loads((ROOT / f"selection_multi_{reference}.json").read_text())
        for name, score in selection["scores"].items():
            candidates[(reference, name)] = score
    (reference, name), qualified = chosen_configuration(candidates)
    output = {"novelty_reference": reference, "feature_set": name, "qualified": [list(key) for key in qualified],
              "scores": {f"{key[0]}/{key[1]}": score for key, score in candidates.items()}}
    files.write_json_atomic(ROOT / "choice.json", output)
    print(json.dumps(output, indent=1), flush=True)


def chosen_feature_set_multi():
    """Feature set of choice.json; the novelty reference of this process must be the chosen one (IDSVAE_NOVELTY_REFERENCE)."""
    choice = json.loads((ROOT / "choice.json").read_text())
    if choice["novelty_reference"] != NOVELTY_REFERENCE:
        raise ValueError(f"choice.json chose the novelty reference {choice['novelty_reference']!r}; run with IDSVAE_NOVELTY_REFERENCE={choice['novelty_reference']}")
    return choice["feature_set"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("features", "extra-features", "select-multi", "choose-multi", "apply", "leave-one-out", "templates"))
    parser.add_argument("--days", nargs="*", default=list(extra_days.EXTRA_DAY_FILES))
    parser.add_argument("--processes", type=int, default=3)
    args = parser.parse_args()
    if args.step == "features":
        tasks = [("topology", seed, final_run.Variant(0, left_out)) for seed in final_run.SEEDS for left_out in final_run.LEFT_OUT_CLASSES]
        # Detector features use only a VAE and the flow times: the raw arm takes the design VAE (as its correction does), the base arm its own VAE.
        tasks += [(arm, seed, final_run.MAIN) for arm in ("topology", "base") for seed in final_run.SEEDS]
        with multiprocessing.get_context("fork").Pool(args.processes) as pool:
            for task in pool.imap_unordered(write_features, tasks):
                print("features done:", task[0], task[1], task[2].suffix(), flush=True)
    elif args.step == "extra-features":
        days = [day for day in args.days if (extra_days.OUTPUT_DIR / f"{day}.parquet").exists()]
        for arm in ("topology", "base"):
            for seed in final_run.SEEDS:
                write_extra_features((arm, seed, days))
                print("extra features done:", arm, seed, days, flush=True)
    elif args.step == "select-multi":
        select_multi()
    elif args.step == "choose-multi":
        choose_multi()
    elif args.step == "leave-one-out":
        feature_set = chosen_feature_set_multi()
        for arm in ("topology", "raw"):
            for seed in final_run.SEEDS:
                print(arm, seed, json.dumps(leave_one_out_report(seed, feature_set, arm)), flush=True)
    elif args.step == "templates":
        print(json.dumps(template_report(chosen_feature_set_multi())), flush=True)
    else:
        feature_set = chosen_feature_set_multi()
        for arm in ("topology", "base"):
            for seed in final_run.SEEDS:
                output = apply(arm, seed, feature_set)
                print(arm, seed, json.dumps({part: output[part] for part in ("holdout", "test")}), flush=True)


if __name__ == "__main__":
    main()
