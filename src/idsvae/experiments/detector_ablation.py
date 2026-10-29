"""The unseen-attack detector on the features of each VAE arm, by leaving one day out (detector_runs.held_out_report).

Only the main split has VAEs of every arm, so these detectors train on the further days without the fold positives, for every
arm alike; the detection of the chosen configuration in Table 7 keeps the fold positives.
"""
import importlib
import json

import numpy as np
import pandas as pd

from idsvae import config, files
from idsvae.experiments import detector_runs, final_run
from idsvae.framework import detector

ARMS = ("topology", "base", "gaussian")
OUTPUT = detector_runs.ROOT / f"vae_arms_{detector_runs.NOVELTY_REFERENCE}.json"


def arm_days(arm, seed):
    return [day for day in detector_runs.qualified_days()
            if detector_runs.feature_file(arm, seed, final_run.MAIN, detector_runs.day_part(day)).exists()]


def held_out_days(arm, seed, feature_set):
    """Per held-out day: Benign false alarms and detection per family, with a detector trained on the other days of this arm."""
    days = arm_days(arm, seed)
    columns = list(detector.FEATURE_SETS[feature_set])
    report = {}
    for held_out in days:
        others = [day for day in days if day != held_out]
        frame = pd.read_parquet(detector_runs.feature_file(arm, seed, final_run.MAIN, detector_runs.day_part(held_out)))
        families = [label for label in pd.unique(frame["label"]) if label != config.BENIGN]
        rows = detector_runs.multi_day_training_rows(seed, others, (), feature_set, arm=arm, excluded_labels=families)
        model = detector.fit_detector(*rows, seed, detector_runs.SELECTION_THREADS)
        threshold = detector_runs.multi_day_cutoff(model, seed, others, feature_set, arm=arm)
        flagged = model.predict_proba(frame[columns])[:, 1] > threshold
        labels = frame["label"].to_numpy()
        report[held_out] = {"benign_false_alarms": float(flagged[labels == config.BENIGN].mean()),
                            "detected": {label: float(flagged[labels == label].mean()) for label in families}}
    return report


def youden(report):
    """Mean over day and family of the detection rate minus the Benign false-alarm rate of that day."""
    return float(np.mean([rate - day["benign_false_alarms"] for day in report.values() for rate in day["detected"].values()]))


def main():
    if OUTPUT.exists():
        return
    feature_set = detector_runs.chosen_feature_set_multi()
    output = {}
    for arm in ARMS:
        for seed in final_run.SEEDS:
            if len(arm_days(arm, seed)) != len(detector_runs.qualified_days()):
                print(f"skipped {arm} seed {seed}: detector features missing for some days", flush=True)
                continue
            report = held_out_days(arm, seed, feature_set)
            output.setdefault(arm, {})[str(seed)] = {"days": report, "youden": youden(report)}
            print(arm, seed, output[arm][str(seed)]["youden"], flush=True)
    files.write_json_atomic(OUTPUT, output)
    print(json.dumps({arm: {seed: value["youden"] for seed, value in by_seed.items()} for arm, by_seed in output.items()}), flush=True)


if __name__ == "__main__":
    importlib.import_module("idsvae.experiments.detector_ablation").main()
