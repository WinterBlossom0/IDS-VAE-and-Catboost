"""Choice of the topology weight and of the rule settings without the hold-out, on an inner split of fit_train.

Inner split: the latest 20 % per day and class of fit_train is the inner validation part (the rule of fit_holdout); the
rest is the inner training part. All models here train on the inner training part only.

1. Topology weight, from the range of Moor et al. (ICML 2020, weights in [0.1, 3]): for each candidate, the VAE and
   Stage 3 train on the inner training part; the score is the macro-F1 (classes with at least 30 rows) after the rule on
   the inner validation part. A tie keeps the weight in use.
2. Rule settings, by leaving one attack class out: for SSH-Bruteforce, DoS-GoldenEye and DoS-Slowloris (FTP-BruteForce
   has 30 training rows), the VAE (with the chosen weight) and Stage 3 train without that class. The score is the F1 of
   attack versus Benign on the inner validation part, where any attack label counts as attack, averaged over the
   three folds. The settings are chosen one at a time in the order of RULE_CANDIDATES, each over its candidates with
   the others fixed; a tie keeps the value in use. Raw-feature CatBoost trains on the same folds for comparison.
The Stage 1 rankings come from all of fit_train (data/stage1 and data/stage1_loo, written by python -m idsvae.framework.features), so the inner
validation rows took part in the feature ranking.
"""
import argparse
import dataclasses
import fcntl
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from idsvae import config, files, metrics
from idsvae.data import splits
from idsvae.framework import classify, correction, features, vae

ROOT = config.PROJECT_ROOT / "data" / "final" / "inner_selection"
PROCESSED_DIR = ROOT / "processed"
SEED = 3
FOLD_CLASSES = ("SSH-Bruteforce", "DoS attacks-GoldenEye", "DoS attacks-Slowloris")
TOPOLOGY_CANDIDATES = (0.1, 0.3, 1.0, 3.0)
CURRENT_TOPOLOGY_WEIGHT = 3.0
RULE_CANDIDATES = {
    "cell_width": (0.1, 0.2, 0.5),
    "burst_quantile": (0.975, 0.99, 0.995),
    "window_seconds": (30, 60, 120),
    "support_quantile": (0.975, 0.99, 0.995),
    "novelty_quantile": (0.90, 0.95, 0.975, 0.99),
}
# vae.run_vae encodes every split of vae.ENCODED_SPLITS, so the 02-21 test split is linked too; no inner score reads it.
LINKED_SPLITS = ("vae_val", "calib", "test")
# Each rule evaluation is one single-thread pass over about a million flows, so the evaluations run in parallel processes.
MAX_WORKERS = 6


def write_inner_splits():
    """fit_train -> inner training rows (as fit_train) and inner validation rows (as fit_holdout); other splits are linked.

    The pipeline starts several inner steps at once, so the first one writes the split under a lock and the others wait.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_DIR / ".write.lock", "a") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if not (PROCESSED_DIR / "fit_holdout.parquet").exists():
            write_inner_split_files()


def write_inner_split_files():
    for split in LINKED_SPLITS:
        link = PROCESSED_DIR / f"{split}.parquet"
        link.unlink(missing_ok=True)
        # A relative link stays valid when the project folder moves.
        link.symlink_to(os.path.relpath(config.PROCESSED_DIR / f"{split}.parquet", PROCESSED_DIR))
    train = splits.load_split("fit_train", config.PROCESSED_DIR)
    inner_validation = splits.assign_seen_holdout(train)
    files.write_parquet_atomic(train[~inner_validation].reset_index(drop=True), PROCESSED_DIR / "fit_train.parquet")
    # The validation file is written last: its presence marks a complete inner split.
    files.write_parquet_atomic(train[inner_validation].reset_index(drop=True), PROCESSED_DIR / "fit_holdout.parquet")


def ranking_dir(left_out_class):
    return features.ranking_dir_for(features.SelectionSettings(left_out_class=left_out_class))


def vae_settings(topology_weight, left_out_class=""):
    selection = features.SelectionSettings(left_out_class=left_out_class)
    count = features.load_ranking(ranking_dir(left_out_class))["selected_count"]
    return vae.VaeSettings(topology_weight=topology_weight, feature_count=count, left_out_class=left_out_class, seed=SEED,
                           feature_set=features.feature_set_name(selection))


def classifier_settings(topology_weight, left_out_class=""):
    settings = vae_settings(topology_weight, left_out_class)
    return classify.ClassifySettings(view="latent_aug", vae_run=settings.run_name(), feature_count=settings.feature_count,
                                     left_out_class=left_out_class, seed=SEED)


def raw_settings(left_out_class):
    count = features.load_ranking(ranking_dir(left_out_class))["selected_count"]
    return classify.ClassifySettings(view="raw", feature_count=count, left_out_class=left_out_class, seed=SEED)


def run_vae(topology_weight, left_out_class=""):
    settings = vae_settings(topology_weight, left_out_class)
    if (ROOT / "stage2" / settings.run_name() / "summary.json").exists():
        return
    write_inner_splits()
    with files.gpu_lock():
        vae.run_vae(settings, processed_dir=PROCESSED_DIR, ranking_dir=ranking_dir(left_out_class), output_root=ROOT / "stage2")


def run_classifier(settings, threads):
    if (ROOT / "stage3" / settings.run_name() / "result.json").exists():
        return
    write_inner_splits()
    classify.run_classifier(settings, processed_dir=PROCESSED_DIR, ranking_dir=ranking_dir(settings.left_out_class),
                            stage2_root=ROOT / "stage2", output_root=ROOT / "stage3", threads=threads)


def stage3_predictions(settings, encoded, run_dir, result, threads):
    """Stage 3 labels for encoded rows; leave-one-out runs write no calibration-day predictions, so this predicts them."""
    predict = classify.load_probability_function(settings.model, run_dir / "model.bin", threads)
    return classify.predicted_labels(predict(encoded[result["columns"]].to_numpy(dtype="float32")), result["classes"]).astype(object)


class RuleRun:
    """Inputs of the rule for one inner-split run: encodings, Stage 3 predictions, and the fit stream."""

    def __init__(self, topology_weight, left_out_class, threads):
        settings = classifier_settings(topology_weight, left_out_class)
        run_dir = ROOT / "stage3" / settings.run_name()
        self.encoded_dir = ROOT / "stage2" / settings.vae_run
        self.result = json.loads((run_dir / "result.json").read_text())
        self.classes = self.result["classes"]
        self.left_out_class = left_out_class
        self.latent_dim = correction.latent_dim_of(self.encoded_dir)
        self.train, _ = correction.load_split(self.encoded_dir, run_dir, self.result, "fit_train")
        self.validation, self.validation_predicted = correction.load_split(self.encoded_dir, run_dir, self.result, "fit_holdout")
        self.calib = pd.read_parquet(self.encoded_dir / "calib.parquet")
        self.calib_predicted = stage3_predictions(settings, self.calib, run_dir, self.result, threads)
        if left_out_class:
            left_out_predictions = pd.read_parquet(run_dir / self.result["prediction_files"]["fit_train_left_out"])
            self.stream, self.context, self.threshold_rows = correction.leave_one_out_stream(
                self.train, self.validation, self.validation_predicted, left_out_predictions, left_out_class)
        else:
            self.stream, self.context = correction.fit_stream(self.train, self.validation, self.validation_predicted)
            self.threshold_rows = self.train
        self.labels = self.validation["label"].to_numpy()

    def predictions(self, rule_settings):
        attack_classes = [label for label in self.classes if label != config.BENIGN]
        rule_thresholds = correction.thresholds(self.threshold_rows, self.calib, self.calib_predicted, attack_classes, self.latent_dim, rule_settings)
        changed, _ = correction.apply_rule(self.stream, self.context, rule_thresholds)
        return changed[len(self.train):]

    def report(self, predicted):
        is_attack = self.labels != config.BENIGN
        is_seen = self.labels != self.left_out_class
        known = metrics.per_class_label_metrics(self.labels[is_seen], predicted[is_seen], self.classes)
        report = {
            "attack_f1": float(f1_score(is_attack, predicted != config.BENIGN)),
            "known_class_macro_f1": metrics.macro_f1_with_min_support(known)["value"],
            "benign_false_alarms": int(((predicted != config.BENIGN) & ~is_attack).sum()),
        }
        if self.left_out_class:
            report["left_out_detection"] = float((predicted[~is_seen] != config.BENIGN).mean())
        return report


def raw_report(rule_run, left_out_class):
    run_dir = ROOT / "stage3" / raw_settings(left_out_class).run_name()
    result = json.loads((run_dir / "result.json").read_text())
    _, predicted = correction.load_split(rule_run.encoded_dir, run_dir, result, "fit_holdout")
    return rule_run.report(predicted)


def best(candidates, scores, current):
    """Candidate with the best score; a tie keeps the current value."""
    top = max(scores.values())
    tied = [candidate for candidate in candidates if scores[candidate] == top]
    return current if current in tied else tied[0]


# The loaded rule runs of the parent process; forked workers inherit them, so no worker copies the encoded frames.
_RULE_RUNS = {}


def rule_report(task):
    """Report of one (rule run key, rule settings or None for the Stage 3 labels) evaluation, in a worker process."""
    key, rule_settings = task
    rule_run = _RULE_RUNS[key]
    if rule_settings is None:
        return rule_run.report(rule_run.validation_predicted)
    return rule_run.report(rule_run.predictions(rule_settings))


def rule_reports(rule_runs, tasks):
    """Reports of (key, rule settings) tasks, in task order."""
    _RULE_RUNS.clear()
    _RULE_RUNS.update(rule_runs)
    # A worker that the out-of-memory killer stops raises BrokenProcessPool here; multiprocessing.Pool would wait forever.
    # Known risk: fork after the parent started threads can hang a worker, which BrokenProcessPool does not catch; fork shares the frames' memory pages with the workers.
    with ProcessPoolExecutor(min(MAX_WORKERS, len(tasks)), mp_context=multiprocessing.get_context("fork")) as pool:
        return list(pool.map(rule_report, tasks))


def choose_topology_weight(threads):
    defaults = correction.RuleSettings()
    rule_runs = {weight: RuleRun(weight, "", threads) for weight in TOPOLOGY_CANDIDATES}
    tasks = [(weight, settings) for weight in TOPOLOGY_CANDIDATES for settings in (None, defaults)]
    results = iter(rule_reports(rule_runs, tasks))
    reports = {weight: {"no_rule": next(results), "rule": next(results)} for weight in TOPOLOGY_CANDIDATES}
    scores = {weight: report["rule"]["known_class_macro_f1"] for weight, report in reports.items()}
    chosen = best(TOPOLOGY_CANDIDATES, scores, CURRENT_TOPOLOGY_WEIGHT)
    output = {"chosen": chosen, "reports": {str(weight): report for weight, report in reports.items()}}
    files.write_json_atomic(ROOT / "topology_choice.json", output)
    print(json.dumps(output, indent=1), flush=True)


def chosen_topology_weight():
    return json.loads((ROOT / "topology_choice.json").read_text())["chosen"]


def chosen_rule_settings():
    return correction.RuleSettings(**json.loads((ROOT / "rule_choice.json").read_text())["chosen"])


def choose_rule_settings(threads):
    weight = chosen_topology_weight()
    folds = {left_out: RuleRun(weight, left_out, threads) for left_out in FOLD_CLASSES}
    settings = correction.RuleSettings()
    steps = []
    for name, candidates in RULE_CANDIDATES.items():
        tasks = [(left_out, dataclasses.replace(settings, **{name: candidate})) for candidate in candidates for left_out in FOLD_CLASSES]
        results = iter(rule_reports(folds, tasks))
        fold_reports = {candidate: {left_out: next(results) for left_out in FOLD_CLASSES} for candidate in candidates}
        scores = {candidate: float(np.mean([report["attack_f1"] for report in reports.values()])) for candidate, reports in fold_reports.items()}
        settings = dataclasses.replace(settings, **{name: best(candidates, scores, getattr(settings, name))})
        steps.append({"setting": name, "mean_attack_f1": {str(candidate): score for candidate, score in scores.items()},
                      "chosen": getattr(settings, name), "fold_reports": {str(candidate): reports for candidate, reports in fold_reports.items()}})
    final = iter(rule_reports(folds, [(left_out, task_settings) for left_out in FOLD_CLASSES for task_settings in (None, settings)]))
    comparison = {left_out: {"raw": raw_report(fold, left_out), "no_rule": next(final), "rule": next(final)} for left_out, fold in folds.items()}
    output = {"topology_weight": weight, "chosen": dataclasses.asdict(settings), "steps": steps, "comparison": comparison}
    files.write_json_atomic(ROOT / "rule_choice.json", output)
    print(json.dumps({"topology_weight": weight, "chosen": output["chosen"]}, indent=1), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("vae", "classifier", "raw", "choose-topology", "choose-rule"))
    parser.add_argument("--left-out-class", choices=FOLD_CLASSES, default="")
    parser.add_argument("--topology-weight", type=float, choices=TOPOLOGY_CANDIDATES)
    parser.add_argument("--threads", type=int, default=5)
    args = parser.parse_args()
    if args.step in ("vae", "classifier") and args.topology_weight is None:
        parser.error("vae and classifier need --topology-weight")
    if args.step == "raw" and not args.left_out_class:
        parser.error("raw needs --left-out-class")
    if args.step == "vae":
        run_vae(args.topology_weight, args.left_out_class)
    elif args.step == "classifier":
        run_classifier(classifier_settings(args.topology_weight, args.left_out_class), args.threads)
    elif args.step == "raw":
        run_classifier(raw_settings(args.left_out_class), args.threads)
    elif args.step == "choose-topology":
        choose_topology_weight(args.threads)
    else:
        choose_rule_settings(args.threads)


if __name__ == "__main__":
    main()
