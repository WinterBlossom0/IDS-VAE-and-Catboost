"""The full framework on the held-out days of the detector (02-28 and the further days), as the re-run methods are evaluated
there: the design VAE, the Stage 3 classifier and the correction of each seed (main variant), then the detector trained
without the day and its families (detector_runs.held_out_day_scores). The report gives, per day, the share of each unseen
family predicted as any attack and the Benign false-positive rate after each step, and the ROC-AUC and PR-AUC of the detector
score of each family against the Benign flows of the day.

Step "encode" writes the VAE encodings of the further days (one GPU lock for all seeds); step "report" runs on the CPU;
step "test" gives the ROC-AUC of the final detector score on the test day.
"""
import argparse
import gc
import json

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from idsvae import config, files, metrics
from idsvae.data import splits
from idsvae.experiments import detector_runs, final_run
from idsvae.framework import classify, correction, detector, vae

ENCODED_DIR = detector_runs.ROOT / "encoded_extra"
ARM = "topology"


def encoded_file(seed, day):
    return ENCODED_DIR / f"{ARM}_s{seed}_{day}.parquet"


def further_days():
    return [day for day in detector_runs.qualified_days() if day != "02-28"]


def encode():
    """VAE encodings of the further days for each seed, under one GPU lock: the queue of other GPU jobs waits only once."""
    ENCODED_DIR.mkdir(parents=True, exist_ok=True)
    pending = [(seed, day) for seed in final_run.SEEDS for day in further_days() if not encoded_file(seed, day).exists()]
    if not pending:
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with files.gpu_lock():
        for seed, day in pending:
            encoded_dir, _, _ = final_run.rule_inputs(ARM, seed)
            model, standardizer, columns = vae.load_model(encoded_dir)
            settings = vae.VaeSettings(**json.loads((encoded_dir / "summary.json").read_text())["settings"])
            frame = detector_runs.extra_day_vae_inputs(day, columns)
            encoded, _ = vae.encode_frame(model.to(device), frame, columns, standardizer, settings, device)
            files.write_parquet_atomic(encoded, encoded_file(seed, day))
            print("encoded", seed, day, len(encoded), flush=True)
            # Each day is freed before the next, because 02-20 has 5 million flows.
            del frame, encoded, model
            gc.collect()
            torch.cuda.empty_cache()


def day_encodings(seed, day, encoded_dir):
    if day == "02-28":
        return pd.read_parquet(encoded_dir / "vae_val.parquet")
    return pd.read_parquet(encoded_file(seed, day))


def score_quality(labels, scores):
    """ROC-AUC and PR-AUC of the detector score of each unseen family against the Benign flows of the day."""
    benign = labels == config.BENIGN
    quality = {}
    for label in pd.unique(labels[~benign]):
        rows = benign | (labels == label)
        is_attack = labels[rows] == label
        quality[label] = {"roc_auc": float(roc_auc_score(is_attack, scores[rows])), "pr_auc": float(average_precision_score(is_attack, scores[rows]))}
    return quality


def report(seed, feature_set, threads):
    encoded_dir, run_dir, result = final_run.rule_inputs(ARM, seed)
    *_, rule_thresholds = final_run.main_rule_thresholds(encoded_dir, run_dir, result)
    settings = final_run.classifier_settings(ARM, seed)
    predict = classify.load_probability_function(settings.model, run_dir / "model.bin", threads)
    days = detector_runs.available_days(seed)
    output = {}
    for day in days:
        encoded = day_encodings(seed, day, encoded_dir)
        predicted = classify.predicted_labels(predict(encoded[result["columns"]].to_numpy(dtype="float32")), result["classes"]).astype(object)
        corrected, _ = correction.apply_rule(encoded, predicted, rule_thresholds)
        features, scores, threshold = detector_runs.held_out_day_scores(seed, feature_set, day, days)
        if not detector_runs.same_rows(features, encoded):
            raise ValueError(f"seed {seed}, {day}: rows of the detector features and of the encodings differ")
        labels = encoded["label"].to_numpy()
        no_burst = np.zeros(len(labels), dtype=bool)
        output[day] = {"classifier": metrics.other_day_report(labels, predicted, no_burst),
                       "correction": metrics.other_day_report(labels, corrected, no_burst),
                       "detector": metrics.other_day_report(labels, detector.combine(corrected, scores > threshold), no_burst),
                       "detector_score": score_quality(labels, scores)}
        print(seed, day, json.dumps(output[day]["detector"]), flush=True)
    return output


def test_day_scores(feature_set):
    """ROC-AUC of the final detector score (detector_runs.apply models) of each test-day class against all Benign flows and
    against the Benign flows outside the burst; PR-AUC is not given, because HOIC is 98.9 % of the rows of the second set."""
    test = splits.load_split("test", columns=["label"])
    labels, burst = test["label"].to_numpy(), test["in_benign_burst"].to_numpy(dtype=bool)
    benign_sets = {"all_benign": labels == config.BENIGN, "benign_outside_burst": (labels == config.BENIGN) & ~burst}
    output = {}
    for seed in final_run.SEEDS:
        detector_runs.applied_output(ARM, seed, feature_set)
        model = CatBoostClassifier()
        model.load_model(str(detector_runs.model_file(ARM, seed)))
        features = pd.read_parquet(detector_runs.feature_file(ARM, seed, final_run.MAIN, "test"))
        if not detector_runs.same_rows(features, test):
            raise ValueError(f"seed {seed}: rows of the test-day features and of the test split differ")
        scores = model.predict_proba(features[list(detector.FEATURE_SETS[feature_set])])[:, 1]
        output[str(seed)] = {label: {name: float(roc_auc_score(labels[benign | (labels == label)] == label, scores[benign | (labels == label)]))
                                     for name, benign in benign_sets.items()}
                             for label in pd.unique(labels[labels != config.BENIGN])}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("encode", "report", "test"))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.step == "encode":
        encode()
        return
    feature_set = detector_runs.chosen_feature_set_multi()
    if args.step == "test":
        files.write_json_atomic(detector_runs.ROOT / f"test_detector_scores_{detector_runs.NOVELTY_REFERENCE}.json", test_day_scores(feature_set))
        return
    output = {str(seed): report(seed, feature_set, args.threads) for seed in final_run.SEEDS}
    files.write_json_atomic(detector_runs.ROOT / f"held_out_framework_{detector_runs.NOVELTY_REFERENCE}.json", output)


if __name__ == "__main__":
    main()
