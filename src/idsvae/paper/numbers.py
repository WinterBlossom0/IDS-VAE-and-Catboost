"""Tables and inline numbers of the paper from the final-run outputs (data/final); writes paper/generated/*.tex.

Run `uv run --frozen python -m idsvae.paper.numbers` after the final run; a missing run shows as "pending" in its cell.
"""
import json
import math
import statistics

import pandas as pd

from idsvae import config, metrics
from idsvae.data import extra_days, splits
from idsvae.experiments import baselines, equal_data, final_run, inner_selection
from idsvae.framework import features, vae
from idsvae.timing import python_parts

OUTPUT_DIR = config.PROJECT_ROOT / "paper" / "generated"
SEEDS = final_run.SEEDS
PENDING = r"\textit{pending}"
CLASS_NAMES = {"Benign": "Benign", "DoS attacks-GoldenEye": "DoS GoldenEye", "DoS attacks-Slowloris": "DoS Slowloris",
               "FTP-BruteForce": "FTP-BruteForce", "SSH-Bruteforce": "SSH-BruteForce"}
# Row label of the framework with the detector in Tables 4 and 8.
WITH_DETECTOR_LABEL = r"\textbf{Ours + detector}"
# (row label, arm, "rule" or "base"): the main table, design first, then the ablations.
MAIN_ROWS = (
    (r"\textbf{Ours (with the correction)}", "topology", "rule"),
    ("Without the correction", "topology", "base"),
    ("Without the topological loss", "base", "rule"),
    ("No topological loss or corr.", "base", "base"),
    ("Gaussian likelihood", "gaussian", "rule"),
    ("Latent mean only (5 inputs)", "latent_only", "rule"),
    ("LightGBM in Stage 3", "lightgbm", "rule"),
    ("XGBoost in Stage 3", "xgboost", "rule"),
    ("Raw + VAE inputs (51) + corr.", "concat", "rule"),
    ("Raw + VAE inputs (51)", "concat", "base"),
    ("Raw CatBoost + correction", "raw", "rule"),
    ("Raw CatBoost (44 features)", "raw", "base"),
)
# Ablations that remove one robust-VAE part: the topological loss, the Student-t likelihood, the NLL and KL inputs.
VAE_PART_ARMS = ("base", "gaussian", "latent_only")
BASELINE_ROWS = (
    ("Random forest~\\cite{sakr2026}", "random_forest"), ("MLP~\\cite{ferreira2020}", "mlp"), ("XGBoost~\\cite{chen2016}", "xgboost"),
    ("LightGBM~\\cite{ke2017}", "lightgbm"), ("IDAE-XG~\\cite{kang2022}", "kang_idae_xg"), ("CNN-1D~\\cite{donoso2026}", "cnn1d"),
    ("FlowTransformer~\\cite{manocchio2024}", "flow_transformer"), ("TabM~\\cite{gorishniy2025}", "tabm"),
    ("FT-Transformer~\\cite{gorishniy2021}", "ft_transformer"), ("TabNet~\\cite{zegarra2023}", "tabnet"),
)


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


UNSEEN_DIR = final_run.FINAL_DIR / "unseen"
# Re-run methods given the further days and the detector's cut-off (experiments/equal_data.py), and an open-set detector.
EQUAL_ROWS = (("Random forest$^+$", "random_forest"), ("MLP$^+$", "mlp"), ("XGBoost$^+$", "xgboost"), ("LightGBM$^+$", "lightgbm"),
              ("IDAE-XG$^+$", "kang_idae_xg"), ("Isolation Forest$^+$~\\cite{liu2008}", "isolation_forest"))
FAMILY_NAMES = {"Infilteration": "Infiltration", "DDoS attacks-LOIC-HTTP": "LOIC-HTTP", "Brute Force -Web": "Web brute force",
                "Brute Force -XSS": "XSS", "SQL Injection": "SQL injection", "Bot": "Bot"}


def detector_choice():
    return read_json(UNSEEN_DIR / "choice.json")


def unseen_result(arm, seed):
    """Detector output of an arm (detector_runs.apply) for the chosen novelty reference."""
    choice = detector_choice()
    result = None if choice is None else read_json(UNSEEN_DIR / f"{arm}_s{seed}_{choice['novelty_reference']}.json")
    # detector_runs reuses existing outputs, so an output of an earlier configuration must not reach the paper.
    if result is not None and result["feature_set"] != choice["feature_set"]:
        raise ValueError(f"{arm} seed {seed}: Detector output has feature set {result['feature_set']}, choice.json {choice['feature_set']}")
    return result


def baseline_result(method, seed, left_out_class=""):
    return read_json(baselines.output_path(method, seed, left_out_class))


def rule_result(arm, seed, variant=final_run.MAIN):
    return read_json(final_run.rule_output(arm, seed, variant))


def stage3_result(arm, seed, variant=final_run.MAIN):
    return read_json(final_run.stage3_root(arm) / final_run.classifier_settings(arm, seed, variant).run_name() / "result.json")


def number(value, digits=4):
    return PENDING if value is None else f"{value:.{digits}f}"


def exact_text(value, digits):
    """The value with the given decimals; an exact zero prints as 0, and a nonzero value that would print as zero keeps one
    significant digit (0.002), so that a table never shows a nonzero rate as 0.00."""
    if value == 0:
        return "0"
    text = f"{value:.{digits}f}"
    if float(text) != 0:
        return text
    return f"{value:.{-math.floor(math.log10(abs(value)))}f}"


def time_text(value):
    """CPU time with two significant digits (4.7, 0.098, 0.00078): the medians of the two timing sessions differ by a few per cent."""
    if value is None:
        return PENDING
    if value <= 0:
        raise ValueError(f"a CPU time must be positive: {value}")
    # Rounding first fixes the decimals of a value that rounds up to the next power of ten (0.0996 -> 0.10).
    rounded = float(f"{value:.2g}")
    return f"{rounded:.{max(0, 1 - math.floor(math.log10(rounded)))}f}"


def mean_sd(values, digits=4):
    if not values or any(value is None for value in values):
        return PENDING
    if len(values) == 1:
        return exact_text(values[0], digits)
    return rf"\pmsd{{{exact_text(statistics.mean(values), digits)}}}{{{exact_text(statistics.stdev(values), digits)}}}"


def holdout_score(arm, seed, key, field="macro_f1_min_support"):
    result = rule_result(arm, seed)
    return None if result is None else result["holdout"][key][field]


def write(name, text):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / f"{name}.tex").write_text(text)


# Timing entry (compiled timing program, src/cpp/timing.cpp) of each main-table row that the benchmark times.
TIMED_ROWS = {("topology", "rule"): "stage3_catboost", ("raw", "rule"): "raw_correction", ("lightgbm", "rule"): "stage3_lightgbm", ("xgboost", "rule"): "stage3_xgboost",
              ("raw", "base"): "raw", ("topology", "detector"): "framework"}


def time_cell(name):
    """CPU ms per flow of a timed pipeline at python_parts.BATCH_SIZE, '--' for an untimed row."""
    if name is None:
        return "--"
    return time_text(timing_median(f"{name}_batch{python_parts.BATCH_SIZE}"))


def main_table():
    lines = []
    for label, arm, key in MAIN_ROWS:
        scores = [holdout_score(arm, seed, key) for seed in SEEDS]
        balanced = [holdout_score(arm, seed, key, "balanced_accuracy") for seed in SEEDS]
        relabelled = PENDING
        results = [rule_result(arm, seed) for seed in SEEDS]
        if key == "base":
            relabelled = "--"
        elif all(results):
            relabelled = str(sum(result["holdout_overrides"]["changed_by_true_class"].get(config.BENIGN, 0) for result in results))
        cells = [number(score) for score in scores] + [mean_sd(scores), mean_sd(balanced), relabelled, time_cell(TIMED_ROWS.get((arm, key)))]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    for label, arm in ((WITH_DETECTOR_LABEL, "topology"),):
        results = [unseen_result(arm, seed) for seed in SEEDS]
        scores = [None if result is None else result["holdout"]["with_detector"]["macro_f1_min_support"] for result in results]
        rules = [rule_result(arm, seed) for seed in SEEDS]
        # Benign hold-out flows that the correction or the detector relabels; Stage 3 false alarms are not relabels.
        unknown = PENDING if not (all(results) and all(rules)) else str(sum(result["holdout"]["with_detector"]["benign_set_to_unknown"] + rule["holdout_overrides"]["changed_by_true_class"].get(config.BENIGN, 0)
                                                                         for result, rule in zip(results, rules)))
        lines.append(f"{label} & " + " & ".join(number(score) for score in scores) + f" & {mean_sd(scores)} & -- & {unknown} & {time_cell(TIMED_ROWS[(arm, 'detector')])}" + r" \\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{8}{l}{\textit{Re-run methods (Table~\ref{tab:baselines})}} \\")
    for label, method in BASELINE_ROWS:
        values = [baseline_result(method, seed) for seed in SEEDS]
        scores = [None if value is None else value["macro_f1_min_support"] for value in values]
        balanced = [None if value is None else value["balanced_accuracy"] for value in values]
        lines.append(f"{label} & " + " & ".join(number(score) for score in scores) + f" & {mean_sd(scores)} & {mean_sd(balanced)} & -- & {time_cell(method)}" + r" \\")
    write("tab_main_rows", "\n".join(lines) + "\n")


def per_class_table():
    lines = []
    ours = [rule_result("topology", seed) for seed in SEEDS]
    raw = [rule_result("raw", seed) for seed in SEEDS]
    if not all(ours) or not all(raw):
        write("tab_perclass_rows", PENDING + r" \\" + "\n")
        return
    for label, name in CLASS_NAMES.items():
        cells = [f"{ours[0]['holdout']['rule']['per_class'][label]['support']:,}".replace(",", "{,}")]
        for results, key in ((ours, "rule"), (raw, "base")):
            for field in ("precision", "recall", "f1"):
                cells.append(number(statistics.mean(result["holdout"][key]["per_class"][label][field] for result in results)))
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")

    def summary(label, values_ours, values_raw):
        lines.append(rf"\multicolumn{{2}}{{l}}{{{label}}} & \multicolumn{{3}}{{c}}{{{values_ours}}} & \multicolumn{{3}}{{c}}{{{values_raw}}} \\")
    summary("Balanced accuracy", mean_sd([r["holdout"]["rule"]["balanced_accuracy"] for r in ours]), mean_sd([r["holdout"]["base"]["balanced_accuracy"] for r in raw]))
    # ROC-AUC and PR-AUC rank the classifier probabilities; the correction changes labels, not probabilities.
    classifier = [[stage3_result(arm, seed) for seed in SEEDS] for arm in ("topology", "raw")]
    for label, field in (("Macro ROC-AUC (one vs rest, classifier)", "macro_roc_auc_ovr"), ("Macro PR-AUC (one vs rest, classifier)", "macro_pr_auc_ovr")):
        summary(label, *(mean_sd([None if r is None else r["seen_multiclass"][field] for r in results]) for results in classifier))
    write("tab_perclass_rows", "\n".join(lines) + "\n")


def confusion_table(seed=SEEDS[0]):
    result = rule_result("topology", seed)
    if result is None:
        write("tab_confusion_rows", PENDING + r" \\" + "\n")
        return
    if list(result["holdout"]["rule"]["per_class"]) != list(CLASS_NAMES):
        raise ValueError("the confusion matrix rows do not follow the class order of CLASS_NAMES")
    matrix = result["holdout"]["rule"]["confusion_matrix"]
    lines = [f"{CLASS_NAMES[label]} & " + " & ".join(f"{value:,}".replace(",", "{,}") for value in row) + r" \\" for label, row in zip(CLASS_NAMES, matrix)]
    write("tab_confusion_rows", "\n".join(lines) + "\n")


def paired_bootstrap(seed):
    """Paired bootstrap over hold-out flows of (ours with the correction) - (raw CatBoost), mean F1 of the classes with >= 30 rows."""
    path = final_run.rule_output("topology", seed, final_run.MAIN).with_suffix(".holdout.parquet")
    raw = stage3_result("raw", seed)
    if not path.exists() or raw is None:
        return None
    ours = pd.read_parquet(path)
    raw_run = final_run.stage3_root("raw") / final_run.classifier_settings("raw", seed).run_name()
    raw_rows = pd.read_parquet(raw_run / raw["prediction_files"]["fit_holdout"])
    if not (raw_rows["label"].to_numpy() == ours["label"].to_numpy()).all():
        raise ValueError("raw and design hold-out rows differ")
    classes = raw["classes"]
    support = ours["label"].value_counts()
    kept = [index for index, label in enumerate(classes) if metrics.has_min_support(int(support.get(label, 0)))]
    tables = metrics.paired_resampled_tables(ours["label"].to_numpy(), ours["rule"].to_numpy(), raw_rows["predicted"].to_numpy(), classes)
    return metrics.paired_bootstrap_test(tables, metrics.macro_f1_of_classes(kept))


INLINE_MACROS = ("OursMean", "RawMean", "OursSd", "RawSd", "GainMin", "GainMax", "BootstrapIntervals",
                 "BestBaselineMean", "DetectorMean", "DetectorSd",
                 "TestBenignOutside", "TestBenignOutsideLoic", "LoicRequestDetection", "LoicNoReplyDetection",
                 "NoCorrMean", "RawCorrMean", "ConcatMean", "ConcatNoCorrMean", "ConcatFtpScore", "OursFtpScore", "AllClassOurs", "AllClassRaw", "SlowRecallBase", "SlowRecallCorr", "PrAucOurs", "PrAucRaw",
                 "TestUdpMean", "TestBurstFprMean", "TestBurstFprRange", "VaePartGap",
                 "DepthTwoDetection", "DepthThreeDetection", "DepthFourDetection",
                 "FrameworkDayDetection", "FrameworkDayFpr", "DetectorRocAuc", "DetectorPrAuc",
                 "DetectorAloneDetection", "DetectorAloneFpr", "RawDetectorDetection", "RawDetectorFpr",
                 "TestRocUdp", "TestRocHoic", "TestRocHoicAll",
                 "JOursDays", "JOursFolds", "JOursUdp", "JBestDaysName", "JBestDays", "JBestFoldsName", "JBestFolds",
                 "JBestUdpName", "JBestUdp", "JClassifierDays", "JClassifierFolds",
                 "JUdpAboveCount", "JHoicAboveCount", "JHoicPositiveCount",
                 "RawDetectorTwoDetection", "RawDetectorTwoFpr", "TestFprOutside", "TestHoicDetection",
                 "LightgbmMean", "XgboostMean", "LightgbmBalanced", "XgboostBalanced", "OursBalanced",
                 "NoTopoNoCorrMean", "JOursDaysRange", "JBestDaysRange", "JEqualBestDays", "JEqualBestDaysName",
                 "JEqualBestFolds", "JEqualBestFoldsName", "JIsoDays", "JArmTopology", "JArmBase", "JOursDaysEqual", "JOursFoldsEqual")


def test_score_macros(test_scores):
    """Ranges over seeds of the test-day ROC-AUC of the detector score: LOIC-UDP and HOIC against the Benign flows outside the
    burst, and the largest HOIC value against all Benign flows."""
    def values(label, benign):
        return [test_scores[str(seed)][label][benign] for seed in SEEDS]
    def value_range(label, benign):
        return f"{min(values(label, benign)):.2f}--{max(values(label, benign)):.2f}"
    if max(values("DDOS attack-HOIC", "benign_outside_burst")) > 0.55:
        raise ValueError("the text says the detector ranks HOIC only near or below chance; revise it")
    return {"TestRocUdp": value_range("DDOS attack-LOIC-UDP", "benign_outside_burst"), "TestRocHoic": value_range("DDOS attack-HOIC", "benign_outside_burst"),
            "TestRocHoicAll": f"{max(values('DDOS attack-HOIC', 'all_benign')):.2f}"}


def day_detection_macros(prefix, reports):
    """Mean share of flagged flows over seeds and held-out-day families, and mean Benign FPR over seeds and days, of
    held_out_report-style reports ({seed: {"days": {day: {"detected": ..., "benign_false_alarms": ...}}}})."""
    days = [report for seed in SEEDS for report in reports[str(seed)]["days"].values()]
    detection = [value for report in days for value in report["detected"].values()]
    return {f"{prefix}Detection": f"{100 * statistics.mean(detection):.2f}",
            f"{prefix}Fpr": f"{100 * statistics.mean(report['benign_false_alarms'] for report in days):.2f}"}


def held_out_framework_macros(framework):
    """The full framework on the held-out days, averaged as the re-run methods there: mean detection over seeds and unseen
    families, mean Benign FPR over seeds and days, and the mean ROC-AUC and PR-AUC of the detector score over the families."""
    seed_reports = list(framework.values())
    detection = [value for by_day in seed_reports for report in by_day.values() for key, value in report["detector"].items() if key.startswith("detected::")]
    false_alarms = [report["detector"]["benign_fpr_outside_burst"] for by_day in seed_reports for report in by_day.values()]
    quality = [family for by_day in seed_reports for report in by_day.values() for family in report["detector_score"].values()]
    infiltration = [value for by_day in seed_reports for report in by_day.values() for key, value in report["detector"].items() if key == "detected::Infilteration"]
    if not infiltration or max(infiltration) >= 0.5:
        raise ValueError("the text says Infiltration stays mostly undetected; revise it")
    return {"FrameworkDayDetection": f"{100 * statistics.mean(detection):.2f}", "FrameworkDayFpr": f"{100 * statistics.mean(false_alarms):.2f}",
            "DetectorRocAuc": f"{statistics.mean(item['roc_auc'] for item in quality):.2f}", "DetectorPrAuc": f"{statistics.mean(item['pr_auc'] for item in quality):.2f}"}


def all_class_f1(result, key):
    return statistics.mean(values["f1"] for values in result["holdout"][key]["per_class"].values())


def numbers():
    macros = dict.fromkeys(INLINE_MACROS, PENDING)
    ours = [holdout_score("topology", seed, "rule") for seed in SEEDS]
    raw = [holdout_score("raw", seed, "base") for seed in SEEDS]
    without_correction = [holdout_score("topology", seed, "base") for seed in SEEDS]
    no_topology_no_correction = [holdout_score("base", seed, "base") for seed in SEEDS]
    if None not in without_correction:
        macros["NoCorrMean"] = f"{statistics.mean(without_correction):.3f}"
    if None not in without_correction and None not in no_topology_no_correction:
        macros["NoTopoNoCorrMean"] = f"{statistics.mean(no_topology_no_correction):.3f}"
        if statistics.mean(without_correction) >= statistics.mean(no_topology_no_correction):
            raise ValueError("the text says the topological loss lowers the F1 without the correction; revise it")
    raw_corrected = [holdout_score("raw", seed, "rule") for seed in SEEDS]
    if None not in raw_corrected:
        macros["RawCorrMean"] = f"{statistics.mean(raw_corrected):.3f}"
    # Stage 3 on the 44 raw features plus the seven VAE outputs (ablation "concat").
    concat_results = [rule_result("concat", seed) for seed in SEEDS]
    if all(concat_results):
        for macro, key in (("ConcatMean", "rule"), ("ConcatNoCorrMean", "base")):
            macros[macro] = f"{statistics.mean(result['holdout'][key]['macro_f1_min_support'] for result in concat_results):.3f}"
        macros["ConcatFtpScore"] = f"{statistics.mean(result['holdout']['rule']['per_class']['FTP-BruteForce']['f1'] for result in concat_results):.2f}"
    ours_results, raw_results = [rule_result("topology", seed) for seed in SEEDS], [rule_result("raw", seed) for seed in SEEDS]
    if all(ours_results) and all(raw_results):
        # Mean F1 over every class, FTP (8 hold-out rows) included.
        macros["AllClassOurs"] = f"{statistics.mean(all_class_f1(result, 'rule') for result in ours_results):.3f}"
        macros["AllClassRaw"] = f"{statistics.mean(all_class_f1(result, 'base') for result in raw_results):.3f}"
        macros["OursFtpScore"] = f"{statistics.mean(result['holdout']['rule']['per_class']['FTP-BruteForce']['f1'] for result in ours_results):.2f}"
        if float(macros["AllClassRaw"]) <= float(macros["AllClassOurs"]):
            raise ValueError("the text says raw CatBoost scores higher with FTP included; revise it")
        for macro, key in (("SlowRecallBase", "base"), ("SlowRecallCorr", "rule")):
            macros[macro] = f"{statistics.mean(result['holdout'][key]['per_class']['DoS attacks-Slowloris']['recall'] for result in ours_results):.2f}"
        if float(macros["SlowRecallCorr"]) <= float(macros["SlowRecallBase"]):
            raise ValueError("the text says the correction raises the Slowloris recall; revise it")
    # Largest change of the mean F1 (with the correction) when one robust-VAE part is removed.
    part_means = [[holdout_score(arm, seed, "rule") for seed in SEEDS] for arm in ("topology", *VAE_PART_ARMS)]
    if all(None not in means for means in part_means):
        macros["VaePartGap"] = f"{max(abs(statistics.mean(means) - statistics.mean(part_means[0])) for means in part_means[1:]):.3f}"
    classifiers = [[stage3_result(arm, seed) for seed in SEEDS] for arm in ("topology", "raw")]
    if all(all(results) for results in classifiers):
        macros["PrAucOurs"], macros["PrAucRaw"] = (f"{statistics.mean(result['seen_multiclass']['macro_pr_auc_ovr'] for result in results):.3f}" for results in classifiers)
        if float(macros["PrAucOurs"]) >= float(macros["PrAucRaw"]):
            raise ValueError("the text says the uncorrected framework has the lower macro PR-AUC; revise it")
    # Stage 3 with LightGBM or XGBoost: the text says the mean F1 stays within the seed spread of ours and the balanced
    # accuracy is lower.
    ours_balanced = [holdout_score("topology", seed, "rule", "balanced_accuracy") for seed in SEEDS]
    boosting = {arm: ([holdout_score(arm, seed, "rule") for seed in SEEDS], [holdout_score(arm, seed, "rule", "balanced_accuracy") for seed in SEEDS])
                for arm in ("lightgbm", "xgboost")}
    if None not in ours and None not in ours_balanced and all(None not in scores + balanced for scores, balanced in boosting.values()):
        macros["OursBalanced"] = f"{statistics.mean(ours_balanced):.3f}"
        for arm, (scores, balanced) in boosting.items():
            if abs(statistics.mean(scores) - statistics.mean(ours)) > statistics.stdev(ours) or statistics.mean(balanced) >= statistics.mean(ours_balanced):
                raise ValueError(f"the text on the {arm} Stage 3 ablation no longer holds; revise it")
            prefix = {"lightgbm": "Lightgbm", "xgboost": "Xgboost"}[arm]
            macros[f"{prefix}Mean"], macros[f"{prefix}Balanced"] = f"{statistics.mean(scores):.3f}", f"{statistics.mean(balanced):.3f}"
    # The text calls this gap "within the seed spread".
    if macros["VaePartGap"] != PENDING and None not in ours and float(macros["VaePartGap"]) > statistics.stdev(ours):
        raise ValueError("a robust-VAE part changes the mean F1 by more than the seed standard deviation of ours; revise the text")
    if None not in ours and None not in raw:
        macros["OursMean"] = f"{statistics.mean(ours):.3f}"
        macros["RawMean"] = f"{statistics.mean(raw):.3f}"
        macros["OursSd"] = f"{statistics.stdev(ours):.3f}"
        macros["RawSd"] = f"{statistics.stdev(raw):.3f}"
        macros["GainMin"] = f"{min(o - r for o, r in zip(ours, raw)):.3f}"
        if min(o - r for o, r in zip(ours, raw)) <= 0:
            raise ValueError("the text reports a gain over raw CatBoost for every seed; revise it")
        if None not in without_correction and statistics.mean(without_correction) >= statistics.mean(raw):
            raise ValueError("the text says the uncorrected framework scores below raw CatBoost; revise it")
        macros["GainMax"] = f"{max(o - r for o, r in zip(ours, raw)):.3f}"
    # The best re-run method is defined only when every method has every seed.
    baseline_values = [[baseline_result(method, seed) for seed in SEEDS] for _, method in BASELINE_ROWS]
    if all(all(values) for values in baseline_values):
        macros["BestBaselineMean"] = f"{max(statistics.mean(value['macro_f1_min_support'] for value in values) for values in baseline_values):.3f}"
    macros.update(detector_macros())
    macros.update(youden_macros())
    macros.update(equal_macros())
    intervals = [paired_bootstrap(seed) for seed in SEEDS]
    if all(intervals):
        macros["BootstrapIntervals"] = ", ".join(f"[{test['ci95_low']:.3f}, {test['ci95_high']:.3f}]" for test in intervals)
    macros.update(chosen_setting_macros())
    macros.update(data_macros())
    text = "".join(rf"\newcommand{{\{name}}}{{{value}}}" + "\n" for name, value in macros.items())
    write("numbers", text)


def detector_macros():
    """Inline numbers of the unseen-attack detector: hold-out F1, held-out-day scores of the chosen configuration,
    and the test day 02-21 (per seed), plus the count of 02-21 Benign flows outside the burst."""
    macros = {}
    results = [unseen_result("topology", seed) for seed in SEEDS]
    if all(results):
        scores = [result["holdout"]["with_detector"]["macro_f1_min_support"] for result in results]
        macros["DetectorMean"], macros["DetectorSd"] = f"{statistics.mean(scores):.3f}", f"{statistics.stdev(scores):.3f}"
        tests = [result["test"]["with_detector"] for result in results]
        udp = [100 * test["detected::DDOS attack-LOIC-UDP"] for test in tests]
        macros["TestUdpMean"] = f"{statistics.mean(udp):.2f}"
        macros["TestFprOutside"] = f"{100 * statistics.mean(test['benign_fpr_outside_burst'] for test in tests):.2f}"
        macros["TestHoicDetection"] = f"{100 * statistics.mean(test['detected::DDOS attack-HOIC'] for test in tests):.2f}"
        burst_fpr = [100 * test["benign_fpr_burst"] for test in tests]
        macros["TestBurstFprMean"] = f"{statistics.mean(burst_fpr):.2f}"
        macros["TestBurstFprRange"] = f"{min(burst_fpr):.2f}--{max(burst_fpr):.2f}"
        if min(burst_fpr) <= 50:
            raise ValueError("the text says the detector flags most of the test-day Benign burst; revise it")
    choice = detector_choice()
    if choice is not None:
        framework = read_json(UNSEEN_DIR / f"held_out_framework_{choice['novelty_reference']}.json")
        if framework is not None:
            macros.update(held_out_framework_macros(framework))
        selection = read_json(UNSEEN_DIR / f"selection_multi_{choice['novelty_reference']}.json")
        if selection is not None:
            macros.update(day_detection_macros("DetectorAlone", selection["reports"][choice["feature_set"]]))
        test_scores = read_json(UNSEEN_DIR / f"test_detector_scores_{choice['novelty_reference']}.json")
        if test_scores is not None:
            macros.update(test_score_macros(test_scores))
        raw_detector = read_json(UNSEEN_DIR / "raw_detector.json")
        if raw_detector is not None:
            # Depth 6 is CatBoost's default depth, depth 2 the depth of our detector.
            for prefix, depth in (("RawDetector", "6"), ("RawDetectorTwo", "2")):
                macros.update(day_detection_macros(prefix, {seed: {"days": days} for seed, days in raw_detector[depth].items()}))
            # The text says that the gain over the raw-feature detector comes from the VAE inputs.
            if selection is not None and max(float(macros["RawDetectorDetection"]), float(macros["RawDetectorTwoDetection"])) >= float(macros["DetectorAloneDetection"]):
                raise ValueError("a raw-feature detector flags at least as many unseen flows as ours; revise the text")
        depths = read_json(UNSEEN_DIR / f"depths_{choice['novelty_reference']}.json")
        if depths is not None:
            for macro, depth in (("DepthTwoDetection", "2"), ("DepthThreeDetection", "3"), ("DepthFourDetection", "4")):
                macros[macro] = f"{100 * depths[depth]['mean_detection']:.2f}"
        templates = read_json(UNSEEN_DIR / f"loic_http_templates_{choice['novelty_reference']}.json")
        if templates is not None:
            by_seed = [templates["seeds"][str(seed)] for seed in SEEDS]
            macros["LoicRequestDetection"] = f"{100 * statistics.mean(seed['request'] for seed in by_seed):.2f}"
            macros["LoicNoReplyDetection"] = f"{100 * statistics.mean(seed['no_reply'] for seed in by_seed):.2f}"
    test = splits.load_split("test", columns=["in_benign_burst"])
    outside = test[(test["label"] == config.BENIGN).to_numpy() & ~test["in_benign_burst"].to_numpy()]
    loic = test.loc[test["label"] == "DDOS attack-LOIC-UDP", "time"]
    inside_loic = outside["time"].between(loic.min(), loic.max()).sum()
    macros["TestBenignOutside"] = f"{len(outside):,}".replace(",", "{,}")
    macros["TestBenignOutsideLoic"] = f"{int(inside_loic):,}".replace(",", "{,}")
    return macros


# Paper macro of each setting that inner_selection.py chooses without the hold-out.
RULE_SETTING_MACROS = {"cellwidth": "cell_width", "burstq": "burst_quantile", "windowsec": "window_seconds", "supportq": "support_quantile", "noveltyq": "novelty_quantile"}


def chosen_setting_macros():
    macros = dict.fromkeys(("topoweight", *RULE_SETTING_MACROS), PENDING)
    if (inner_selection.ROOT / "topology_choice.json").exists():
        macros["topoweight"] = f"{inner_selection.chosen_topology_weight():g}"
    if (inner_selection.ROOT / "rule_choice.json").exists():
        settings = inner_selection.chosen_rule_settings()
        for macro, field in RULE_SETTING_MACROS.items():
            macros[macro] = f"{getattr(settings, field):g}"
    return macros


def unseen_section():
    """Unseen attack families for our classifier, the framework and the re-run methods, mean over seeds: Youden's J on the
    held-out days and in the folds, and on the test day the detection rate of each class, the Benign FPR outside and inside
    the burst, and J against all Benign flows."""
    def mean_percent(values, digits=2):
        return PENDING if any(value is None for value in values) else exact_text(100 * statistics.mean(values), digits)

    def test_reports(label):
        if label == "Ours, classifier":
            return [None if rule is None else rule["test"]["base"] for rule in (rule_result("topology", seed) for seed in SEEDS)]
        if label == WITH_DETECTOR_LABEL:
            return [None if result is None else result["test"]["with_detector"] for result in (unseen_result("topology", seed) for seed in SEEDS)]
        method = dict(BASELINE_ROWS)[label]
        return [None if result is None else result["test"] for result in (baseline_result(method, seed) for seed in SEEDS)]
    lines = []
    for label, days, folds, udp, hoic in youden_rows():
        # The text gives the classifier-only J; the table keeps the framework with the detector.
        if label == "Ours, classifier":
            continue
        if label == BASELINE_ROWS[0][0]:
            lines.append(r"\midrule")
        reports = test_reports(label)

        def field(name, reports=reports):
            return [None if report is None else report.get(name) for report in reports]
        cells = [PENDING if value is None else signed(value) for value in (days, folds)]
        cells += [mean_percent(field("detected::DDOS attack-LOIC-UDP")), mean_percent(field("detected::DDOS attack-HOIC")),
                  mean_percent(field("benign_fpr_outside_burst")), mean_percent(field("benign_fpr_burst"))]
        cells += [PENDING if value is None else signed(value) for value in (udp, hoic)]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    seeds = equal_seeds()
    if seeds:
        lines.append(r"\midrule")
        for label, days, folds, udp, hoic, reports in equal_rows(seeds):
            def field(name, reports=reports):
                return [report.get(name) for report in reports]
            cells = [signed(days), signed(folds), mean_percent(field("detected::DDOS attack-LOIC-UDP")), mean_percent(field("detected::DDOS attack-HOIC")),
                     mean_percent(field("benign_fpr_outside_burst")), mean_percent(field("benign_fpr_burst")), signed(udp), signed(hoic)]
            lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    text = r"""\begin{table}[t]
\caption{Unseen attack families, in \%, mean over seeds 3--5: Youden's $J$ is the detection rate (DR) minus the Benign false-positive rate (FPR); a re-run method flags a flow whose argmax class is not Benign. $^+$: trained also on the further days and flagged at the detector's cut-off. Left-out days: mean over day and family; folds: mean over the three folds. Test day 02-21: DR, Benign FPR outside and inside the burst, and $J$ against all Benign flows.}\label{tab:unseen}
\centering\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{L{2.6cm}cccccccc}
\toprule
 & \multicolumn{2}{c}{$J$} & \multicolumn{6}{c}{Test day} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-9}
 & Left-out days & Folds & DR UDP & DR HOIC & FPR out & FPR burst & $J$ UDP & $J$ HOIC \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("unseen", text)


def raw_gap_text(raw_gaps):
    """Caption clause on the mean DR after the detector of raw CatBoost minus ours, over the folds."""
    if not raw_gaps:
        return PENDING
    clause = "with raw CatBoost as classifier (same correction and detector), the mean final DR"
    low, high = min(raw_gaps), max(raw_gaps)
    if low >= 0:
        return f"{clause} is {low:.2f} to {high:.2f} points higher per fold."
    if high <= 0:
        return f"{clause} is {-high:.2f} to {-low:.2f} points lower per fold."
    return f"{clause} differs by {low:+.2f} to {high:+.2f} points per fold."


def detector_days_section():
    """Held-out days (each left out of detector training with all of its families) and leave-one-attack-out folds, side by side."""
    choice = detector_choice()
    selection = None if choice is None else read_json(UNSEEN_DIR / f"selection_multi_{choice['novelty_reference']}.json")
    day_lines = []
    if selection is not None:
        reports = selection["reports"][choice["feature_set"]]
        for day in selection["days"]:
            by_seed = [reports[str(seed)]["days"][day] for seed in SEEDS]
            for index, family in enumerate(by_seed[0]["detected"]):
                detection = mean_sd([100.0 * report["detected"][family] for report in by_seed], 2)
                false_alarms = mean_sd([100.0 * report["benign_false_alarms"] for report in by_seed], 2) if index == 0 else ""
                day_lines.append(f"{day if index == 0 else ''} & {FAMILY_NAMES.get(family, family)} & {detection} & {false_alarms}" + r" \\")
    fold_lines, raw_gaps = [], []
    for left_out in final_run.LEFT_OUT_CLASSES:
        reports = [None if choice is None else read_json(UNSEEN_DIR / f"loo_topology_s{seed}_{choice['novelty_reference']}.json") for seed in SEEDS]
        cells = [mean_sd([None if report is None else 100.0 * report[left_out][stage]["left_out_detection"] for report in reports], 2)
                 for stage in ("stage3", "correction", "detector")]
        cells.append(mean_sd([None if report is None else 100.0 * report[left_out]["detector"]["holdout_benign_false_alarms"] for report in reports], 2))
        fold_lines.append(f"{CLASS_NAMES[left_out]} & " + " & ".join(cells) + r" \\")
        raw_reports = [None if choice is None else read_json(UNSEEN_DIR / f"loo_raw_s{seed}_{choice['novelty_reference']}.json") for seed in SEEDS]
        if all(reports) and all(raw_reports):
            raw_gaps.append(statistics.mean(100.0 * report[left_out]["detector"]["left_out_detection"] for report in raw_reports)
                            - statistics.mean(100.0 * report[left_out]["detector"]["left_out_detection"] for report in reports))
    text = r"""\begin{table}[t]
\caption{Attack families absent from all training labels, in \%, mean $\pm$ SD over seeds 3--5: (a) left-out days, DR and Benign FPR of the detector alone; (b) leave-one-attack-out folds, DR on days 02-14 and 02-15 after each step, and hold-out Benign FPR; """ + raw_gap_text(raw_gaps) + r"""}\label{tab:detectordays}\label{tab:loo}
\centering\scriptsize
\setlength{\tabcolsep}{1.5pt}
\renewcommand{\pmsd}[2]{#1$\pm$#2}
\begin{minipage}[t]{0.4\textwidth}\centering
(a) Left-out days\\[2pt]
\begin{tabular}{llcc}
\toprule
Day & Family & DR & FPR \\
\midrule
""" + ("\n".join(day_lines) if day_lines else PENDING + r" \\") + r"""
\bottomrule
\end{tabular}
\end{minipage}\hfill
\begin{minipage}[t]{0.595\textwidth}\centering
(b) Leave-one-attack-out folds\\[2pt]
\begin{tabular}{lcccc}
\toprule
Removed class & Classifier & + corr. & + det. & FPR \\
\midrule
""" + "\n".join(fold_lines) + r"""
\bottomrule
\end{tabular}
\end{minipage}
\end{table}
"""
    write("detector_days", text)


# The compiled timing program (src/cpp/timing.cpp, run_timing.sh): the reported run and the first of the two full runs.
TIMING_FILE = final_run.FINAL_DIR / "timing" / "compiled_threads1.json"
RUNTIME_PREFIX = "C++ ONNX Runtime "
REPEAT_FILE = TIMING_FILE.with_name("compiled_threads1_run1.json")


def timing_report():
    report = read_json(TIMING_FILE)
    if report is not None and not report.get("runtime", "").startswith(RUNTIME_PREFIX):
        raise ValueError(f"{TIMING_FILE} does not come from the compiled timing program")
    return report


def timed_main_pipelines():
    """Timing entries of the main table: our timed rows, then the re-run methods."""
    return [*TIMED_ROWS.values(), *(method for _, method in BASELINE_ROWS)]


def repeat_comparison(report, earlier):
    """(pipelines of the main table, % difference of the framework time of the other run against the reported run); raises
    when the other run orders the main-table pipelines differently."""
    names = [f"{name}_batch{python_parts.BATCH_SIZE}" for name in timed_main_pipelines()]
    if sorted(names, key=lambda name: report["timings"][name]["median"]) != sorted(names, key=lambda name: earlier["timings"][name]["median"]):
        raise ValueError("the other timing run orders the main-table pipelines differently; revise the timing text")
    framework = f"framework_batch{python_parts.BATCH_SIZE}"
    return len(names), 100 * (earlier["timings"][framework]["median"] - report["timings"][framework]["median"]) / report["timings"][framework]["median"]


def timing_median(name):
    """Median ms per flow over both timing sessions of one configuration (src/cpp/timing.cpp), or None."""
    report = timing_report()
    entry = None if report is None else report["timings"].get(name)
    return None if entry is None else entry["median"]


def feature_count_entries(count, selected):
    """Timing entries (ours, raw) of a feature count; the selected count is the main run, timed as stage3_catboost and raw."""
    return ("stage3_catboost", "raw") if count == selected else (f"ours_k{count}", f"raw_k{count}")


def feature_count_section(seed=SEEDS[0]):
    lines, ours_scores = [], {}
    selected = features.load_ranking()["selected_count"]
    for count in sorted((*final_run.FEATURE_COUNTS, selected)):
        variant = final_run.MAIN if count == selected else final_run.Variant(count)
        ours, raw = rule_result("topology", seed, variant), rule_result("raw", seed, variant)
        if ours is not None:
            ours_scores[count] = ours["holdout"]["rule"]["macro_f1_min_support"]
        vae_summary = read_json(final_run.stage2_root("topology") / final_run.vae_settings("topology", seed, variant).run_name() / "summary.json")
        cells = [number(None if ours is None else ours["holdout"]["rule"]["macro_f1_min_support"]),
                 number(None if raw is None else raw["holdout"]["base"]["macro_f1_min_support"]),
                 number(None if raw is None else raw["holdout"]["rule"]["macro_f1_min_support"]),
                 number(None if vae_summary is None else vae_summary["training"]["seconds"], 0),
                 *(time_text(timing_median(f"{name}_batch{python_parts.BATCH_SIZE}")) for name in feature_count_entries(count, selected))]
        label = rf"\textbf{{{count}}}" if count == selected else str(count)
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    smallest = min(ours_scores, default=None)
    larger = [score for count, score in ours_scores.items() if count != smallest]
    spread = (f"; for $k>{smallest}$, our F1 stays within {min(larger):.4f}--{max(larger):.4f}"
              if larger else "")
    better = sorted(count for count, score in ours_scores.items() if selected in ours_scores and score > ours_scores[selected])
    if better:
        spread += rf", and $k\in\{{{', '.join(str(count) for count in better)}\}}$ score above the $k={selected}$ of the 95\,\% rule"
    text = r"""Table~\ref{tab:featurecount} measures F1 and cost with the $k$ most important features of Stage~1 (seed~3; $k=64$: all 79 columns minus the 15 dropped)""" + spread + r""".

\begin{table}[t]
\caption{Feature-count ablation (seed 3): hold-out mean per-class F1, VAE training time in s (one VAE at a time on the GPU) and inference time in ms per flow (batch 1, one thread, no detector, as Table~\ref{tab:main})}\label{tab:featurecount}
\centering\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{rcccrcc}
\toprule
 & \multicolumn{3}{c}{F1} & & \multicolumn{2}{c}{ms per flow} \\
\cmidrule(lr){2-4}\cmidrule(lr){6-7}
$k$ & Ours & Raw & Raw + corr. & VAE (s) & Ours & Raw \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write("featurecount", text)


def thousands(value):
    return f"{value:,}".replace(",", "{,}")


def timing_window():
    """(Benign burst flows, other Benign flows, attack flows) of the timing sample: python_parts.SAMPLE_COUNT consecutive test-day flows."""
    window = python_parts.sample_window(splits.load_split("test", columns=["in_benign_burst"]))
    benign = (window["label"] == config.BENIGN).to_numpy()
    burst = window["in_benign_burst"].to_numpy(dtype=bool)
    return int((benign & burst).sum()), int((benign & ~burst).sum()), int((~benign).sum())


def window_composition(burst, other, attacks):
    """'9{,}928 Benign burst and 72 attack flows'; a group without flows is left out."""
    groups = [f"{thousands(count)} {name}" for count, name in ((burst, "Benign burst"), (other, "other Benign"), (attacks, "attack")) if count > 0]
    if not groups:
        raise ValueError("the timing window holds no flows")
    return (", ".join(groups[:-1]) + " and " + groups[-1] if len(groups) > 1 else groups[0]) + " flows"


def timing_section():
    """The CPU timing of the deployed framework, its parts and its ratio to raw CatBoost, with the protocol."""
    burst, other, attacks = timing_window()
    report, earlier = timing_report(), read_json(REPEAT_FILE)
    if report is None or earlier is None:
        write("timing", PENDING + "\n")
        return
    timings = report["timings"]

    def median_of(name, part=None):
        entry = timings[f"{name}_batch{python_parts.BATCH_SIZE}"]
        return entry["median"] if part is None else entry["parts"][part]
    framework = median_of("framework")
    passes = {session["passes"] for entry in timings.values() for session in entry["sessions"]}
    if len(passes) != 1:
        raise ValueError("the sessions hold different pass counts; revise the timing text")
    per_session = passes.pop()
    sessions = len(timings[f"framework_batch{python_parts.BATCH_SIZE}"]["sessions"])
    pipelines, difference = repeat_comparison(report, earlier)
    hardware = report["cpu"].replace("(R)", "").replace("(TM)", "")
    # "C++ ONNX Runtime 1.31.0" -> "1.31"
    runtime_version = ".".join(report["runtime"].removeprefix(RUNTIME_PREFIX).split(".")[:2])
    text = (rf"On a single CPU thread, end-to-end processing with the detector takes {time_text(framework)}\,ms per flow, compared to {time_text(median_of('raw'))}\,ms for raw "
            rf"CatBoost: the fused VAE and CatBoost graph accounts for {time_text(median_of('framework', 'model'))}\,ms "
            rf"({time_text(median_of('framework_parts', 'vae') + median_of('framework_parts', 'classifier'))}\,ms as two separate graphs), the correction takes "
            rf"{time_text(median_of('framework', 'rule'))}\,ms, and the detector takes {time_text(median_of('framework', 'detector'))}\,ms. We timed each pipeline with a time in "
            rf"Table~\ref{{tab:main}} as ONNX Runtime {runtime_version} graphs and C++ code on one thread of an {hardware}, one flow per call from the extracted "
            rf"features to the final label, over the {thousands(python_parts.SAMPLE_COUNT)} consecutive flows in the middle of day 02-21 "
            rf"({window_composition(burst, other, attacks)}). Each time is the median of {sessions * per_session} passes; a full repeat gave the same order of the "
            rf"{pipelines} pipelines and a framework time {abs(difference):.1f}\,\% {'higher' if difference >= 0 else 'lower'}.") + "\n"
    write("timing", text)


def test_benign_counts():
    """(Benign flows of the test day outside the burst, inside the burst), from the manifest of the processed splits."""
    test = splits.load_manifest()["splits"]["test"]
    burst = test["benign_burst_rows"]
    return test["class_counts"][config.BENIGN] - burst, burst


def test_youden(report):
    """(J of LOIC-UDP, J of HOIC) on the test day against all Benign flows, burst included."""
    outside, burst = test_benign_counts()
    fpr = (outside * report["benign_fpr_outside_burst"] + burst * report["benign_fpr_burst"]) / (outside + burst)
    return report["detected::DDOS attack-LOIC-UDP"] - fpr, report["detected::DDOS attack-HOIC"] - fpr


def youden_of_day_reports(day_reports):
    """J per day and unseen family of other_day_report-style reports of the held-out days."""
    return [report[key] - report["benign_fpr_outside_burst"] for report in day_reports for key in report if key.startswith("detected::")]


def day_youden_per_seed(method):
    """Mean J over the held-out days and families of each seed, in %: the framework (method None) or a re-run method."""
    if method is None:
        framework = read_json(UNSEEN_DIR / f"held_out_framework_{detector_choice()['novelty_reference']}.json")
        return [100 * statistics.mean(youden_of_day_reports([by_day["detector"] for by_day in framework[str(seed)].values()])) for seed in SEEDS]
    return [100 * statistics.mean(youden_of_day_reports(list(baseline_result(method, seed)["held_out_days"].values()))) for seed in SEEDS]


def youden_rows():
    """(label, J held-out days, J folds, J test LOIC-UDP, J test HOIC) in %, mean over seeds, or None where results are missing."""
    choice = detector_choice()
    if choice is None:
        return []
    reference = choice["novelty_reference"]
    framework = read_json(UNSEEN_DIR / f"held_out_framework_{reference}.json")
    loo = [read_json(UNSEEN_DIR / f"loo_topology_s{seed}_{reference}.json") for seed in SEEDS]
    unseen = [unseen_result("topology", seed) for seed in SEEDS]
    rules = [rule_result("topology", seed) for seed in SEEDS]

    def percent(values):
        return None if values is None else 100 * statistics.mean(values)
    rows = []
    for label, stage, fold_stage, test_part in (("Ours, classifier", "classifier", "stage3", "base"), (WITH_DETECTOR_LABEL, "detector", "detector", "detector")):
        days = None if framework is None else youden_of_day_reports([by_day[stage] for seed_reports in framework.values() for by_day in seed_reports.values()])
        folds = None if not all(loo) else [report[left_out][fold_stage]["left_out_detection"] - report[left_out][fold_stage]["holdout_benign_false_alarms"]
                                           for report in loo for left_out in final_run.LEFT_OUT_CLASSES]
        if test_part == "base":
            tests = None if not all(rules) else [test_youden(rule["test"]["base"]) for rule in rules]
        else:
            tests = None if not all(unseen) else [test_youden(result["test"]["with_detector"]) for result in unseen]
        rows.append((label, percent(days), percent(folds), percent(None if tests is None else [t[0] for t in tests]), percent(None if tests is None else [t[1] for t in tests])))
    for label, method in BASELINE_ROWS:
        results = [baseline_result(method, seed) for seed in SEEDS]
        days = youden_of_day_reports([report for result in results for report in result["held_out_days"].values()]) if all(r is not None and "held_out_days" in r for r in results) else None
        fold_results = [baseline_result(method, seed, left_out) for seed in SEEDS for left_out in final_run.LEFT_OUT_CLASSES]
        folds = [r["left_out_detection"] - r["holdout_benign_false_alarms"] for r in fold_results] if all(fold_results) else None
        tests = [test_youden(r["test"]) for r in results] if all(results) else None
        rows.append((label, percent(days), percent(folds), percent(None if tests is None else [t[0] for t in tests]), percent(None if tests is None else [t[1] for t in tests])))
    return rows


def equal_result(method, seed, mode, target=""):
    return read_json(equal_data.output_path(method, seed, mode, target))


def equal_targets():
    from idsvae.experiments import detector_runs
    return [("day", day) for day in detector_runs.qualified_days()] + [("fold", left_out) for left_out in final_run.LEFT_OUT_CLASSES] + [("test", "")]


def equal_seeds():
    """Seeds with every equal-data result of every row."""
    targets = equal_targets()
    return [seed for seed in SEEDS if all(equal_result(method, seed, mode, target) for _, method in EQUAL_ROWS for mode, target in targets)]


def equal_rows(seeds):
    """(label, J left-out days, J folds, J test LOIC-UDP, J test HOIC, test reports) in %, mean over the given seeds."""
    rows = []
    for label, method in EQUAL_ROWS:
        days = youden_of_day_reports([equal_result(method, seed, "day", day)["report"] for seed in seeds for mode, day in equal_targets() if mode == "day"])
        folds = [result["left_out_detection"] - result["holdout_benign_false_alarms"]
                 for result in (equal_result(method, seed, "fold", left_out) for seed in seeds for left_out in final_run.LEFT_OUT_CLASSES)]
        reports = [equal_result(method, seed, "test")["report"] for seed in seeds]
        tests = [test_youden(report) for report in reports]
        rows.append((label, 100 * statistics.mean(days), 100 * statistics.mean(folds), 100 * statistics.mean(t[0] for t in tests),
                     100 * statistics.mean(t[1] for t in tests), reports))
    return rows


def equal_macros():
    """Best J of the equal-data rows on the left-out days and in the folds, the open-set J, and the detector J per VAE arm."""
    macros = {}
    seeds = equal_seeds()
    if seeds:
        rows = dict(zip((method for _, method in EQUAL_ROWS), equal_rows(seeds)))
        # The text calls the best row a re-run method, so the open-set detector is left out of the maximum.
        rerun_rows = [row for method, row in rows.items() if method not in equal_data.OPEN_SET_METHODS]
        for column, name in ((1, "Days"), (2, "Folds")):
            row = max(rerun_rows, key=lambda item: item[column])
            macros[f"JEqualBest{name}Name"], macros[f"JEqualBest{name}"] = row[0].split("~")[0].replace("$^+$", ""), signed(row[column])
        macros["JIsoDays"] = signed(rows["isolation_forest"][1])
        # The framework with the detector on the same seeds as the equal-data rows.
        days = dict(zip(SEEDS, day_youden_per_seed(None)))
        reference = detector_choice()["novelty_reference"]
        folds = [100 * (report[left_out]["detector"]["left_out_detection"] - report[left_out]["detector"]["holdout_benign_false_alarms"])
                 for report in (read_json(UNSEEN_DIR / f"loo_topology_s{seed}_{reference}.json") for seed in seeds) for left_out in final_run.LEFT_OUT_CLASSES]
        macros["JOursDaysEqual"] = signed(statistics.mean(days[seed] for seed in seeds))
        macros["JOursFoldsEqual"] = signed(statistics.mean(folds))
    choice = detector_choice()
    arms = None if choice is None else read_json(UNSEEN_DIR / f"vae_arms_{choice['novelty_reference']}.json")
    if arms is not None:
        for arm, name in (("topology", "JArmTopology"), ("base", "JArmBase")):
            if len(arms.get(arm, {})) == len(SEEDS):
                macros[name] = signed(100 * statistics.mean(value["youden"] for value in arms[arm].values()))
    return macros


def signed(value):
    """exact_text with two decimals and a typeset minus sign."""
    text = exact_text(value, 2)
    return f"$-${text[1:]}" if text.startswith("-") else text


def youden_macros():
    """Inline values of the J table: ours, the best other method per column, and counts of methods for the text; checks the
    claims of the text that depend on these values."""
    rows = youden_rows()
    # Best-of and count statements need every method with every seed and fold.
    if len(rows) != 2 + len(BASELINE_ROWS) or any(None in row[1:] for row in rows):
        return {}
    classifier, ours, *others = rows

    def best(column):
        row = max(others, key=lambda item: item[column])
        return row[0].split("~")[0], signed(row[column])
    macros = {"JOursDays": signed(ours[1]), "JOursFolds": signed(ours[2]), "JOursUdp": signed(ours[3]), "JClassifierDays": signed(classifier[1]), "JClassifierFolds": signed(classifier[2])}
    for column, name in ((1, "Days"), (2, "Folds"), (3, "Udp")):
        macros[f"JBest{name}Name"], macros[f"JBest{name}"] = best(column)
    # Counts of re-run methods, as words for the text: a higher J than ours on LOIC-UDP and on HOIC, and a positive J on HOIC.
    udp_above, hoic_above = sum(row[3] > ours[3] for row in others), sum(row[4] > ours[4] for row in others)
    hoic_positive = sum(row[4] > 0 for row in rows)
    if not (ours[1] > max(row[1] for row in others) and ours[2] > max(row[2] for row in others)):
        raise ValueError("the text says the framework leads on the held-out days and in the folds; revise it")
    if udp_above < 2 or hoic_above < 2 or hoic_positive > 1:
        raise ValueError("the text uses a plural verb for the LOIC-UDP and HOIC counts and a singular one for positive HOIC; revise it")
    best_days_method = next(method for (label, method), row in zip(BASELINE_ROWS, others) if row[1] == max(item[1] for item in others))
    for name, per_seed in (("JOursDaysRange", day_youden_per_seed(None)), ("JBestDaysRange", day_youden_per_seed(best_days_method))):
        macros[name] = f"{signed(min(per_seed))}--{signed(max(per_seed))}"
    macros["JUdpAboveCount"] = count_words(udp_above, len(others))
    macros["JHoicAboveCount"] = count_words(hoic_above, len(others))
    # Over every row of the table, ours included.
    macros["JHoicPositiveCount"] = count_words(hoic_positive, len(rows))
    return macros


def count_words(count, total):
    """A count for a sentence: "no", "all ten", or the number word."""
    words = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve")
    if count == total and count > 0:
        return f"all {words[count]}"
    return words[count]


# Short class names of the data table.
DATA_CLASS_NAMES = {"SSH-Bruteforce": "SSH", "DoS attacks-GoldenEye": "GoldenEye", "DoS attacks-Slowloris": "Slowloris", "FTP-BruteForce": "FTP",
                    "Infilteration": "Infiltration", "DDOS attack-HOIC": "HOIC", "DDOS attack-LOIC-UDP": "LOIC-UDP",
                    "DDoS attacks-LOIC-HTTP": "LOIC-HTTP", "Bot": "Bot", "Brute Force -Web": "web brute force", "Brute Force -XSS": "XSS",
                    "SQL Injection": "SQL injection"}
# (row label, days, split) of the data table; the calibration day shows its three web attack classes as one count.
DATA_ROWS = (("Training (Stages 1--3)", "02-14, 02-15", "fit_train"), ("Hold-out (seen classes)", "02-14, 02-15", "fit_holdout"),
             ("Validation (VAE stop, detector)", "02-28", "vae_val"), ("Calibration", "02-23", "calib"), ("Test (unseen classes)", "02-21", "test"))
# Counts that the architecture figure (paper/figures/architecture.tex) draws; data_macros checks them against the data.
FIGURE_CANDIDATE_COUNT, FIGURE_SELECTED_COUNT = 64, 44


def class_counts_cell(counts, merged_attack_name=None):
    """Total rows, then Benign and the attack classes by falling count."""
    attacks = {DATA_CLASS_NAMES[label]: count for label, count in counts.items() if label != config.BENIGN}
    if merged_attack_name is not None:
        attacks = {merged_attack_name: sum(attacks.values())}
    parts = [f"Benign {thousands(counts.get(config.BENIGN, 0))}"] + [f"{name} {thousands(count)}" for name, count in sorted(attacks.items(), key=lambda item: -item[1])]
    return f"{thousands(sum(counts.values()))} ({'; '.join(parts)})"


def further_days():
    """The capture days outside the protocol splits that serve the detector, and their manifest records."""
    # Imported here: the detector modules load CatBoost and the VAE code.
    from idsvae.experiments import held_out_days
    days = held_out_days.further_days()
    records = json.loads((extra_days.OUTPUT_DIR / "manifest.json").read_text())
    return days, {day: records[day] for day in days}


def data_table():
    split_records = splits.load_manifest()["splits"]
    lines = [f"{label} & {days} & {class_counts_cell(split_records[split]['class_counts'], 'web attacks' if split == 'calib' else None)}" + r" \\"
             for label, days, split in DATA_ROWS]
    days, records = further_days()
    counts = {}
    for record in records.values():
        for label, count in record["class_counts"].items():
            counts[label] = counts.get(label, 0) + count
    lines.append(f"Further days (detector) & {', '.join(days)} & {class_counts_cell(counts)}" + r" \\")
    write("tab_data_rows", "\n".join(lines) + "\n")


def undefined_shares(columns):
    """Share of undefined entries per selected feature in the training rows, and the count of features that every flow defines."""
    train = splits.load_split("fit_train", columns=vae.mask_source_columns(columns))
    shares = dict(zip(columns, vae.undefined_entries(train, columns).mean(axis=0)))
    return shares, int(vae.error_feature_mask(columns).sum())


def data_macros():
    """Counts of the data section and of Stage 1, from the manifests and the Stage 1 ranking."""
    manifest = splits.load_manifest()
    ranking = features.load_ranking()
    if len(manifest["feature_columns"]) != FIGURE_CANDIDATE_COUNT or ranking["selected_count"] != FIGURE_SELECTED_COUNT:
        raise ValueError("the candidate or selected feature count differs from the architecture figure")
    fit, test = manifest["splits"]["fit"], manifest["splits"]["test"]
    # The text says that the label-conflict removal takes no row of the test day.
    if test["label_conflict_rows_removed_by_class"]:
        raise ValueError("label conflicts removed rows of the test day; revise the text")
    ftp_vectors = fit["label_conflict_distinct_vectors_by_class"]["FTP-BruteForce"]
    _, records = further_days()
    stability = ranking["stability"]["subset_ranking"]["aggregated_halves"]
    columns = features.top_features(ranking, ranking["selected_count"])
    shares, always_defined = undefined_shares(columns)
    return {"ConstantCount": str(len(manifest["constant_columns_dropped"])),
            "FtpConflictRows": thousands(fit["label_conflict_rows_removed_by_class"]["FTP-BruteForce"]),
            "FtpConflictVectors": str(ftp_vectors["distinct_vectors"]), "FtpConflictBenignVectors": str(ftp_vectors["distinct_vectors_equal_to_benign"]),
            "BotConflictRows": thousands(records["03-02"]["label_conflict_rows_removed_by_class"]["Bot"]),
            "MislabelledUdpRows": thousands(records["02-20"]["mislabelled_udp_rows_removed"]),
            "BurstRows": thousands(test_benign_counts()[1]),
            "StabilitySpearman": f"{stability['spearman']:.3f}", "StabilityJaccard": f"{stability['top_k_jaccard']:.3f}",
            "UndefinedIdleMax": f"{100 * shares['Idle Max']:.2f}", "UndefinedFlowIatStd": f"{100 * shares['Flow IAT Std']:.2f}",
            "AlwaysDefinedCount": str(always_defined)}


def main():
    data_table()
    unseen_section()
    detector_days_section()
    feature_count_section()
    timing_section()
    main_table()
    per_class_table()
    confusion_table()
    numbers()
    print(f"wrote {sorted(path.name for path in OUTPUT_DIR.glob('*.tex'))}")


if __name__ == "__main__":
    main()
