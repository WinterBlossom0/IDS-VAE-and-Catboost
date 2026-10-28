"""Ablation of the detector inputs on the held-out days: a CatBoost detector on the 44 raw Stage 1 features instead of the
five VAE-based inputs, trained on the same rows with the same weights and the same 1 % cut-off
(detector_runs.multi_day_training_rows and multi_day_cutoff), and evaluated on each held-out day as the leave-one-day-out
part of detector_runs.held_out_report does; the leave-one-fold-out part is not repeated. It separates the effect of the
VAE signals from the effect of the labelled attack days that only the detector sees. Raw features carry no known direction,
so the trees have no monotone constraint. Two tree depths run: the CatBoost default depth and the depth of the adopted
detector, so that the change of the inputs is separated from the change of the tree capacity.
Output raw_detector.json: {depth: {seed: {held-out day: report}}}."""
import json

import pandas as pd
from catboost import CatBoostClassifier

from idsvae import config, files
from idsvae.data import extra_days, splits
from idsvae.experiments import baselines, detector_runs, final_run
from idsvae.framework import detector

# 6 is the CatBoost default depth; the second depth is the depth of the adopted detector.
RAW_DEPTHS = (6, detector.DEPTH)


class RawParts:
    """Raw feature rows of the detector parts, each checked once against the rows of its detector feature file."""

    def __init__(self, columns):
        self.columns = columns
        self.fit = None
        self.checked = set()

    def rows(self, part):
        if part == "fit":
            # Every fold of every seed and depth reads the same fit rows, so they are read once.
            if self.fit is None:
                train = splits.load_split("fit_train", columns=self.columns)
                holdout = splits.load_split("fit_holdout", columns=self.columns)
                self.fit = pd.concat([train.assign(holdout=False), holdout.assign(holdout=True)], ignore_index=True)
            return self.fit
        if part.startswith("extra_"):
            return pd.read_parquet(extra_days.OUTPUT_DIR / f"{part.removeprefix('extra_')}.parquet", columns=[*self.columns, "label", "time"])
        return splits.load_split(part, columns=self.columns)

    def loader(self, seed):
        """Part name, variant -> raw feature rows of that part, as detector_runs.feature_part_loader gives the detector features."""
        def load(part, variant=final_run.MAIN):
            frame = self.rows(part)
            if (seed, part, variant) not in self.checked:
                detector_rows = pd.read_parquet(detector_runs.feature_file("topology", seed, variant, part), columns=["label", "time"])
                if not detector_runs.same_rows(frame, detector_rows):
                    raise ValueError(f"seed {seed}, {part}{variant.suffix()}: raw rows and detector feature rows differ")
                self.checked.add((seed, part, variant))
            return frame
        return load


def held_out_days_report(seed, depth, parts):
    """Per held-out day: Benign false alarms and detection of each family by a raw detector trained without that day and its families."""
    days = detector_runs.available_days(seed)
    load = parts.loader(seed)
    report = {}
    for held_out in days:
        others = [day for day in days if day != held_out]
        frame = load(detector_runs.day_part(held_out))
        families = [label for label in pd.unique(frame["label"]) if label != config.BENIGN]
        features, targets, weights = detector_runs.multi_day_training_rows(seed, others, final_run.LEFT_OUT_CLASSES, excluded_labels=families,
                                                                            load_part=load, columns=parts.columns)
        model = CatBoostClassifier(iterations=detector.ITERATIONS, depth=depth, learning_rate=detector.LEARNING_RATE, random_seed=seed,
                                   verbose=0, thread_count=detector_runs.SELECTION_THREADS, allow_writing_files=False)
        model.fit(features, targets, sample_weight=weights)
        threshold = detector_runs.multi_day_cutoff(model, seed, others, load_part=load, columns=parts.columns)
        flagged = model.predict_proba(frame[parts.columns])[:, 1] > threshold
        labels = frame["label"].to_numpy()
        report[held_out] = {"benign_false_alarms": float(flagged[labels == config.BENIGN].mean()),
                            "detected": {label: float(flagged[labels == label].mean()) for label in families}}
        print(depth, seed, held_out, json.dumps(report[held_out]), flush=True)
    return report


def main():
    parts = RawParts(baselines.selected_features())
    output = {str(depth): {str(seed): held_out_days_report(seed, depth, parts) for seed in final_run.SEEDS} for depth in RAW_DEPTHS}
    files.write_json_atomic(detector_runs.ROOT / "raw_detector.json", output)


if __name__ == "__main__":
    main()
