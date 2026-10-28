"""Final run of the camera-ready design on new seeds: VAE with the topological loss, tuned Stage 3 CatBoost, novel-burst correction.

Arms per seed: "topology" (the design), "base" (the VAE without the topological loss), "gaussian" (the VAE with a Gaussian
in place of the Student-t likelihood), "latent_only" (CatBoost on the design VAE's latent means without recon_nll and the
KL term), "lightgbm" and "xgboost" (LightGBM or XGBoost in place of CatBoost on the design VAE), "concat" (CatBoost on the
selected raw features and the design VAE's outputs) and "raw" (CatBoost on the selected raw features). Steps: `vae` trains one VAE under the shared GPU lock, `classifier` tunes and fits Stage 3 for one
arm, `rule` applies the novel-burst correction to an arm (arms without a VAE of their own take the design VAE's signals; for the
raw arm this is the raw + rule ablation) and writes the seen-class and other-day results. run_timing.sh measures the CPU
inference time.
"""
import argparse
import dataclasses
import json

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from idsvae import config, files, metrics
from idsvae.data import splits
from idsvae.experiments import inner_selection
from idsvae.framework import classify, correction, features, vae

FINAL_DIR = config.PROJECT_ROOT / "data" / "final"
# Seeds of the final run; the development runs used seeds 0-2.
SEEDS = (3, 4, 5)
# VAE per arm: (with the topological loss, likelihood). "base" drops the topological loss, "gaussian" the Student-t likelihood.
# The topology weight and the rule settings come from inner_selection.py, which chooses them without the hold-out.
VAE_ARMS = {"topology": (True, "student_t"), "base": (False, "student_t"), "gaussian": (True, "gaussian")}
# Stage 3 ablations on the design VAE: (model, view); "latent_only" leaves out recon_nll and the KL term, "concat" adds the
# Stage 1 raw features to the seven VAE outputs.
CLASSIFIER_ARMS = {"latent_only": ("catboost", "latent"), "lightgbm": ("lightgbm", "latent_aug"), "xgboost": ("xgboost", "latent_aug"),
                   "concat": ("catboost", "concat")}
ARMS = (*VAE_ARMS, *CLASSIFIER_ARMS, "raw")
# Arms without a VAE of their own (raw, the Stage 3 ablations) take the novelty and burst signals of the design VAE.
DESIGN_ARM = "topology"


def vae_arm(arm):
    """The arm whose VAE an arm uses (None for raw)."""
    if arm in VAE_ARMS:
        return arm
    return DESIGN_ARM if arm in CLASSIFIER_ARMS else None


def stage2_root(arm):
    """Folder of the VAE runs of an arm; the Stage 3 ablations read the design VAE."""
    arm = vae_arm(arm) or arm
    return FINAL_DIR / f"stage2_{arm}"


def stage3_root(arm):
    return FINAL_DIR / f"stage3_{arm}"


@dataclasses.dataclass(frozen=True)
class Variant:
    """A run of the final protocol: feature_count 0 is the Stage 1 cut; a left-out class removes it from every fitted stage."""
    feature_count: int = 0
    left_out_class: str = ""

    def suffix(self):
        parts = [f"_k{self.feature_count}"] if self.feature_count else []
        if self.left_out_class:
            parts.append(f"_loo-{features.class_slug(self.left_out_class)}")
        return "".join(parts)


MAIN = Variant()
# The feature-count ablation tests how the result depends on the Stage 1 cut (64 = all candidate features after the dropped columns).
FEATURE_COUNTS = (10, 15, 23, 30, 40, 64)
# FTP-BruteForce has 30 training rows after the label-conflict removal, too few for a leave-one-out fold.
LEFT_OUT_CLASSES = ("SSH-Bruteforce", "DoS attacks-GoldenEye", "DoS attacks-Slowloris")
# Three steps of 5 threads share the 16 logical CPUs of the run machine.
DEFAULT_THREADS = 5


def ranking_dir(variant):
    return features.ranking_dir_for(features.SelectionSettings(left_out_class=variant.left_out_class))


def feature_count(variant):
    """The 95 % cut of the variant's Stage 1 ranking unless the variant fixes the count."""
    return variant.feature_count or features.load_ranking(ranking_dir(variant))["selected_count"]


def vae_settings(arm, seed, variant=MAIN):
    with_topology, likelihood = VAE_ARMS[vae_arm(arm)]
    weight = inner_selection.chosen_topology_weight() if with_topology else 0.0
    selection = features.SelectionSettings(left_out_class=variant.left_out_class)
    return vae.VaeSettings(likelihood=likelihood, topology_weight=weight, feature_count=feature_count(variant), left_out_class=variant.left_out_class,
                           seed=seed, feature_set=features.feature_set_name(selection))


def classifier_settings(arm, seed, variant=MAIN):
    if arm == "raw":
        return classify.ClassifySettings(view="raw", feature_count=feature_count(variant), left_out_class=variant.left_out_class, seed=seed)
    model, view = CLASSIFIER_ARMS.get(arm, ("catboost", "latent_aug"))
    return classify.ClassifySettings(model=model, view=view, vae_run=vae_settings(arm, seed, variant).run_name(), feature_count=feature_count(variant),
                                     left_out_class=variant.left_out_class, seed=seed)


def run_vae(arm, seed, variant=MAIN):
    settings = vae_settings(arm, seed, variant)
    if (stage2_root(arm) / settings.run_name() / "summary.json").exists():
        return
    with files.gpu_lock():
        vae.run_vae(settings, ranking_dir=ranking_dir(variant), output_root=stage2_root(arm))


def run_classifier(arm, seed, threads, variant=MAIN):
    settings = classifier_settings(arm, seed, variant)
    if (stage3_root(arm) / settings.run_name() / "result.json").exists():
        return
    classify.run_classifier(settings, config.PROCESSED_DIR, ranking_dir(variant), stage2_root(arm), stage3_root(arm), threads=threads)


def label_report(true_labels, predicted, classes):
    """The seen-class metrics of metrics.seen_multiclass_report that need only predicted labels (the rule changes labels, not probabilities)."""
    per_class = metrics.per_class_label_metrics(true_labels, predicted, classes)
    return {
        "macro_f1_min_support": metrics.macro_f1_with_min_support(per_class)["value"],
        "balanced_accuracy": float(balanced_accuracy_score(true_labels, predicted)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(true_labels, predicted, labels=classes).tolist(),
    }


def rule_inputs(arm, seed, variant=MAIN):
    """(encoded_dir, run_dir, result) of the rule step of an arm: its own VAE, or for the raw arm the design VAE."""
    run_dir = stage3_root(arm) / classifier_settings(arm, seed, variant).run_name()
    signal_arm = vae_arm(arm) or DESIGN_ARM
    encoded_dir = stage2_root(signal_arm) / vae_settings(signal_arm, seed, variant).run_name()
    return encoded_dir, run_dir, json.loads((run_dir / "result.json").read_text())


def rule_output(arm, seed, variant):
    return FINAL_DIR / "rule" / f"{arm}_s{seed}{variant.suffix()}.json"


def main_rule_thresholds(encoded_dir, run_dir, result):
    """(training rows, hold-out rows and predictions, calibration rows and predictions, rule thresholds) of a main-variant run."""
    attack_classes = [label for label in result["classes"] if label != config.BENIGN]
    train, _ = correction.load_split(encoded_dir, run_dir, result, "fit_train")
    holdout, holdout_predicted = correction.load_split(encoded_dir, run_dir, result, "fit_holdout")
    calib, calib_predicted = correction.load_split(encoded_dir, run_dir, result, "calib")
    rule_thresholds = correction.thresholds(train, calib, calib_predicted, attack_classes, correction.latent_dim_of(encoded_dir),
                                              inner_selection.chosen_rule_settings())
    return train, holdout, holdout_predicted, calib, calib_predicted, rule_thresholds


def run_rule(arm, seed, variant=MAIN, threads=DEFAULT_THREADS):
    if variant.left_out_class:
        run_leave_one_out_rule(arm, seed, variant, threads)
        return
    encoded_dir, run_dir, result = rule_inputs(arm, seed, variant)
    classes = result["classes"]
    train, holdout, holdout_predicted, calib, calib_predicted, rule_thresholds = main_rule_thresholds(encoded_dir, run_dir, result)

    stream, context = correction.fit_stream(train, holdout, holdout_predicted)
    fit_changed, fit_override = correction.apply_rule(stream, context, rule_thresholds)
    holdout_changed, holdout_override = fit_changed[len(train):], fit_override[len(train):]
    holdout_labels = holdout["label"].to_numpy()
    output = {
        "arm": arm, "seed": seed, "rule_settings": dataclasses.asdict(rule_thresholds.settings),
        "novelty_threshold": rule_thresholds.novelty, "burst_threshold": rule_thresholds.burst,
        "attribution_min_support": rule_thresholds.min_support,
        "holdout": {"base": label_report(holdout_labels, holdout_predicted, classes), "rule": label_report(holdout_labels, holdout_changed, classes)},
        "holdout_overrides": {"rows": int(holdout_override.sum()), "changed": int((holdout_changed != holdout_predicted).sum()),
                              "changed_by_true_class": pd.Series(holdout_labels[holdout_changed != holdout_predicted]).value_counts().to_dict()},
    }
    # Per-row labels for the paired bootstrap of the paper tables.
    rule_output(arm, seed, variant).parent.mkdir(parents=True, exist_ok=True)
    files.write_parquet_atomic(pd.DataFrame({"label": holdout_labels, "time": holdout["time"].to_numpy(), "base": holdout_predicted, "rule": holdout_changed}),
                               rule_output(arm, seed, variant).with_suffix(".holdout.parquet"), index=False)
    test, test_predicted = correction.load_split(encoded_dir, run_dir, result, "test")
    # The novelty threshold and the minimum support are quantiles of the calibration-day Benign flows, so the calib results are in-sample.
    for part, frame, predicted in (("calib_in_sample", calib, calib_predicted), ("test", test, test_predicted)):
        changed, _ = correction.apply_rule(frame, predicted, rule_thresholds)
        labels = frame["label"].to_numpy()
        burst_rows = frame["in_benign_burst"].to_numpy(dtype=bool)
        output[part] = {"base": metrics.other_day_report(labels, predicted, burst_rows), "rule": metrics.other_day_report(labels, changed, burst_rows), "changed": int((changed != predicted).sum())}
        if part == "test":
            # Per-row labels for the detector step (detector_runs.py).
            files.write_parquet_atomic(pd.DataFrame({"label": labels, "time": frame["time"].to_numpy(), "in_benign_burst": burst_rows, "base": predicted, "rule": changed}),
                                       rule_output(arm, seed, variant).with_suffix(".test.parquet"), index=False)
    write_rule_output(arm, seed, variant, output)
    print(json.dumps({"arm": arm, "seed": seed, "variant": variant.suffix(), "base": output["holdout"]["base"]["macro_f1_min_support"], "rule": output["holdout"]["rule"]["macro_f1_min_support"]}), flush=True)


def write_rule_output(arm, seed, variant, output):
    rule_output(arm, seed, variant).parent.mkdir(parents=True, exist_ok=True)
    files.write_json_atomic(rule_output(arm, seed, variant), output)


def run_leave_one_out_rule(arm, seed, variant, threads=DEFAULT_THREADS):
    """Detection of the left-out class (its fit_train and fit_holdout rows) and Benign false alarms, without and with the rule.

    No fitted stage saw the left-out class, so its fit_train rows are test rows: they keep their Stage 3 predictions in the
    attribution context, while the other training rows count as Benign (see correction.fit_stream). Leave-one-out
    classifiers write no calibration-day predictions, so this step predicts them for the minimum support.
    """
    encoded_dir, run_dir, result = rule_inputs(arm, seed, variant)
    classes = result["classes"]
    attack_classes = [label for label in classes if label != config.BENIGN]
    latent_dim = correction.latent_dim_of(encoded_dir)
    train, _ = correction.load_split(encoded_dir, run_dir, result, "fit_train")
    holdout, holdout_predicted = correction.load_split(encoded_dir, run_dir, result, "fit_holdout")
    calib = pd.read_parquet(encoded_dir / "calib.parquet")
    settings = classifier_settings(arm, seed, variant)
    predict = classify.load_probability_function(settings.model, run_dir / "model.bin", threads)
    view_columns = result["columns"]
    calib_source = calib if arm != "raw" else splits.load_split("calib", columns=view_columns)
    calib_predicted = classify.predicted_labels(predict(calib_source[view_columns].to_numpy(dtype="float32")), classes).astype(object)
    left_out_predictions = pd.read_parquet(run_dir / result["prediction_files"]["fit_train_left_out"])
    stream, context, threshold_rows = correction.leave_one_out_stream(train, holdout, holdout_predicted, left_out_predictions, variant.left_out_class)
    rule_thresholds = correction.thresholds(threshold_rows, calib, calib_predicted, attack_classes, latent_dim, inner_selection.chosen_rule_settings())
    changed, _ = correction.apply_rule(stream, context, rule_thresholds)
    labels = stream["label"].to_numpy()
    is_left_out = labels == variant.left_out_class
    is_holdout = np.arange(len(stream)) >= len(train)
    # Per-row labels for the detector step (detector_runs.py), in the stream order: training rows, then hold-out rows.
    rule_output(arm, seed, variant).parent.mkdir(parents=True, exist_ok=True)
    files.write_parquet_atomic(pd.DataFrame({"label": labels, "time": stream["time"].to_numpy(), "holdout": is_holdout, "base": context, "rule": changed}),
                               rule_output(arm, seed, variant).with_suffix(".stream.parquet"), index=False)
    holdout_benign = is_holdout & (labels == config.BENIGN)

    def detection(predicted):
        return {"left_out_detection": float((predicted[is_left_out] != config.BENIGN).mean()),
                "left_out_rows": int(is_left_out.sum()),
                "left_out_mapped_to": pd.Series(predicted[is_left_out]).value_counts().to_dict(),
                "holdout_benign_fpr": float((predicted[holdout_benign] != config.BENIGN).mean())}
    seen = is_holdout & ~is_left_out
    output = {
        "arm": arm, "seed": seed, "left_out_class": variant.left_out_class, "attribution_min_support": rule_thresholds.min_support,
        "base": detection(context), "rule": detection(changed),
        "holdout_seen_macro_f1": {"base": label_report(labels[seen], context[seen], classes)["macro_f1_min_support"],
                                  "rule": label_report(labels[seen], changed[seen], classes)["macro_f1_min_support"]},
    }
    write_rule_output(arm, seed, variant, output)
    print(json.dumps({"arm": arm, "seed": seed, "left_out": variant.left_out_class, "base": output["base"]["left_out_detection"], "rule": output["rule"]["left_out_detection"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("vae", "classifier", "rule"))
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--feature-count", type=int, choices=FEATURE_COUNTS, default=0)
    parser.add_argument("--left-out-class", choices=LEFT_OUT_CLASSES, default="")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    args = parser.parse_args()
    if args.step == "vae" and args.arm not in VAE_ARMS:
        parser.error(f"the {args.arm} arm has no VAE of its own")
    variant = Variant(args.feature_count, args.left_out_class)
    if args.step == "vae":
        run_vae(args.arm, args.seed, variant)
    elif args.step == "classifier":
        run_classifier(args.arm, args.seed, args.threads, variant)
    else:
        run_rule(args.arm, args.seed, variant, args.threads)


if __name__ == "__main__":
    main()
