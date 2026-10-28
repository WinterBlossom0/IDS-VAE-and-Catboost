"""Metrics of the two evaluations: seen-class multiclass and unseen-class detection."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from idsvae import config

# Below this hold-out support, one row changes the recall of the class by more than 1/30.
MIN_SUPPORT = 30
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 0
INTERVAL_CONFIDENCE = 0.95
# Koehn, EMNLP 2004, Sect. 6, resamples the test set 1000 times.
PAIRED_BOOTSTRAP_RESAMPLES = 1000


def binary_ranking_scores(is_positive, scores):
    """ROC-AUC and PR-AUC of the scores for the positive rows; None when the rows hold one class only."""
    if is_positive.all() or not is_positive.any():
        return {"roc_auc": None, "pr_auc": None}
    return {"roc_auc": float(roc_auc_score(is_positive, scores)), "pr_auc": float(average_precision_score(is_positive, scores))}


def one_vs_rest_scores(true_labels, probabilities, classes):
    return {label: binary_ranking_scores(true_labels == label, probabilities[:, index]) for index, label in enumerate(classes)}


def mean_of_defined(values):
    defined = [value for value in values if value is not None]
    return float(np.mean(defined)) if defined else None


def has_min_support(support, min_support=MIN_SUPPORT):
    return support >= min_support


def macro_f1_with_min_support(per_class, min_support=MIN_SUPPORT):
    included = [label for label, values in per_class.items() if has_min_support(values["support"], min_support)]
    return {
        "value": mean_of_defined([per_class[label]["f1"] for label in included]),
        "excluded_classes": [label for label in per_class if label not in included],
        "min_support": min_support,
    }


def per_class_f1(matrices):
    """F1 of each class from confusion matrices with shape (..., classes, classes); rows are true classes."""
    true_positive = np.diagonal(matrices, axis1=-2, axis2=-1).astype("float64")
    # F1 = 2 TP / (2 TP + FP + FN) = 2 TP / (support + predicted); 0 when both are 0, as zero_division=0 in scikit-learn.
    f1_denominator = matrices.sum(axis=-1) + matrices.sum(axis=-2)
    return np.divide(2.0 * true_positive, f1_denominator, out=np.zeros_like(true_positive), where=f1_denominator > 0)


def balanced_accuracy_of_matrices(matrices):
    true_positive = np.diagonal(matrices, axis1=-2, axis2=-1).astype("float64")
    support = matrices.sum(axis=-1)
    # Balanced accuracy averages recall over the classes that have true rows only, as scikit-learn does.
    recall = np.divide(true_positive, support, out=np.full_like(true_positive, np.nan), where=support > 0)
    return np.nanmean(recall, axis=-1)


def confusion_scores(matrices):
    """Macro-F1 and balanced accuracy of confusion matrices with shape (..., classes, classes); rows are true classes."""
    return per_class_f1(matrices).mean(axis=-1), balanced_accuracy_of_matrices(matrices)


def macro_f1_of_classes(class_indices):
    """Metric on confusion matrices: the mean F1 of the classes at class_indices only."""
    class_indices = list(class_indices)

    def macro_f1(matrices):
        return per_class_f1(matrices)[..., class_indices].mean(axis=-1)

    return macro_f1


def bootstrap_ci(matrix, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    """Stratified bootstrap 95 % percentile intervals: each true class keeps its hold-out size in each resample."""
    matrix = np.asarray(matrix, dtype="int64")
    rng = np.random.default_rng(seed)
    resampled = np.zeros((resamples, *matrix.shape), dtype="int64")
    for true_index, class_rows in enumerate(matrix.sum(axis=1)):
        if class_rows == 0:
            continue
        # A multinomial draw of one confusion row has the distribution of a resample of the rows of that class, without a pass over the rows.
        resampled[:, true_index, :] = rng.multinomial(class_rows, matrix[true_index] / class_rows, size=resamples)
    macro_f1, balanced_accuracy = confusion_scores(resampled)
    return {
        "macro_f1": np.percentile(macro_f1, [2.5, 97.5]).tolist(),
        "balanced_accuracy": np.percentile(balanced_accuracy, [2.5, 97.5]).tolist(),
        "resamples": resamples,
    }


def label_codes(values, labels):
    codes = pd.Index(labels).get_indexer(np.asarray(values))
    if (codes < 0).any():
        raise ValueError(f"values outside the labels {list(labels)}: {sorted(set(np.asarray(values)[codes < 0]))}")
    return codes


@dataclass(frozen=True)
class PairedTables:
    """Confusion matrices of two arms on the same rows: observed with shape (labels, labels), resampled with shape (resamples, labels, labels)."""

    observed_first: np.ndarray
    observed_second: np.ndarray
    resampled_first: np.ndarray
    resampled_second: np.ndarray


def paired_resampled_tables(true_labels, first_predicted, second_predicted, labels, resamples=PAIRED_BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    """Confusion matrices of both arms on each bootstrap resample of the rows; both arms get the same drawn rows in each resample."""
    true_codes, first_codes, second_codes = (label_codes(values, labels) for values in (true_labels, first_predicted, second_predicted))
    row_count = len(true_codes)
    if len(first_codes) != row_count or len(second_codes) != row_count:
        raise ValueError("the true labels and both arms must cover the same rows")
    if row_count == 0:
        raise ValueError("no rows to resample")
    size = len(labels)
    # One code per (true, first, second) cell lets one bincount per resample count both arms on the same drawn rows.
    joint_codes = (true_codes * size + first_codes) * size + second_codes
    cell_count = size**3
    rng = np.random.default_rng(seed)
    resampled = np.empty((resamples, cell_count), dtype="int64")
    for resample in range(resamples):
        # Koehn, EMNLP 2004, Sect. 6, and Berg-Kirkpatrick et al., EMNLP 2012, Fig. 1: draw n rows with replacement.
        drawn_rows = rng.integers(0, row_count, size=row_count)
        resampled[resample] = np.bincount(joint_codes[drawn_rows], minlength=cell_count)
    observed = np.bincount(joint_codes, minlength=cell_count).reshape(size, size, size)
    resampled = resampled.reshape(resamples, size, size, size)
    # Axis -3 is the true class, axis -2 the class of the first arm, axis -1 the class of the second arm.
    return PairedTables(observed.sum(axis=-1), observed.sum(axis=-2), resampled.sum(axis=-1), resampled.sum(axis=-2))


def bootstrap_p_value(delta, resampled_deltas):
    """One-sided p-value that the arm with the better observed value is not better (Berg-Kirkpatrick et al., EMNLP 2012, Fig. 1)."""
    # The resampled deltas centre on delta, not on 0, so the test counts resamples beyond 2 delta (their Sect. 2.2).
    if delta > 0:
        return float(np.mean(resampled_deltas > 2.0 * delta))
    if delta < 0:
        return float(np.mean(resampled_deltas < 2.0 * delta))
    # With delta = 0 no arm is better on the observed rows, so no evidence against the null hypothesis exists.
    return 1.0


def paired_bootstrap_test(tables, metric, confidence=INTERVAL_CONFIDENCE):
    """Observed delta = metric(first) - metric(second), percentile interval of the resampled deltas, one-sided p-value and the resample count."""
    first = float(metric(tables.observed_first))
    second = float(metric(tables.observed_second))
    delta = first - second
    resampled_deltas = metric(tables.resampled_first) - metric(tables.resampled_second)
    # A resample without rows of a class leaves its share undefined; such resamples do not count.
    resampled_deltas = resampled_deltas[np.isfinite(resampled_deltas)]
    result = {"first": first, "second": second, "delta": delta, "ci95_low": np.nan, "ci95_high": np.nan, "p_value": np.nan, "resamples": len(resampled_deltas)}
    if len(resampled_deltas) == 0:
        return result
    tail = 100.0 * (1.0 - confidence) / 2.0
    low, high = np.percentile(resampled_deltas, [tail, 100.0 - tail])
    result.update({"ci95_low": float(low), "ci95_high": float(high), "p_value": bootstrap_p_value(delta, resampled_deltas)})
    return result


def per_class_label_metrics(true_labels, predicted, classes):
    """Precision, recall, F1 and support per class from predicted labels."""
    precision, recall, f1, support = precision_recall_fscore_support(true_labels, predicted, labels=classes, zero_division=0)
    return {label: {"precision": float(precision[index]), "recall": float(recall[index]), "f1": float(f1[index]), "support": int(support[index])} for index, label in enumerate(classes)}


def seen_multiclass_report(true_labels, probabilities, classes):
    true_labels = np.asarray(true_labels)
    predicted = np.asarray(classes)[probabilities.argmax(axis=1)]
    ranking_scores = one_vs_rest_scores(true_labels, probabilities, classes)
    per_class = {label: {**values, **ranking_scores[label]} for label, values in per_class_label_metrics(true_labels, predicted, classes).items()}
    matrix = confusion_matrix(true_labels, predicted, labels=classes)
    return {
        "rows": len(true_labels),
        "accuracy": float(accuracy_score(true_labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(true_labels, predicted)),
        "macro_f1": float(f1_score(true_labels, predicted, labels=classes, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true_labels, predicted, labels=classes, average="weighted", zero_division=0)),
        # The one-vs-rest macro value is the unweighted mean of the per-class values, as roc_auc_score(multi_class="ovr") computes it.
        "macro_roc_auc_ovr": mean_of_defined([scores["roc_auc"] for scores in ranking_scores.values()]),
        "macro_pr_auc_ovr": mean_of_defined([scores["pr_auc"] for scores in ranking_scores.values()]),
        "macro_f1_min_support": macro_f1_with_min_support(per_class),
        "bootstrap_ci": bootstrap_ci(matrix),
        "per_class": per_class,
        "confusion_matrix": {"labels": list(classes), "matrix": matrix.tolist()},
    }


def detection_per_class(true_labels, attack_score, attack_predicted):
    is_benign = true_labels == config.BENIGN
    per_class = {}
    for label in pd.unique(true_labels):
        in_class = true_labels == label
        flagged = proportion_report(attack_predicted[in_class].sum(), in_class.sum())
        per_class[label] = {"rows": flagged["rows"], "flagged_as_attack": flagged["rate"], "flagged": flagged}
        if label == config.BENIGN:
            continue
        # Each attack class is ranked against the Benign rows only, so that its AUC does not depend on the other attack classes of the view.
        versus_benign = in_class | is_benign
        scores = binary_ranking_scores(in_class[versus_benign], attack_score[versus_benign])
        per_class[label].update({"roc_auc_vs_benign": scores["roc_auc"], "pr_auc_vs_benign": scores["pr_auc"]})
    return per_class


def detection_report(true_labels, attack_score, attack_predicted):
    true_labels = np.asarray(true_labels)
    attack_score = np.asarray(attack_score)
    attack_predicted = np.asarray(attack_predicted, dtype=bool)
    is_attack = true_labels != config.BENIGN
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(is_attack, attack_predicted, labels=[False, True]).ravel()
    return {
        "rows": len(true_labels),
        "attack_share": float(is_attack.mean()),
        "accuracy": float(accuracy_score(is_attack, attack_predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(is_attack, attack_predicted)),
        "precision": float(true_positive / max(true_positive + false_positive, 1)),
        "recall": float(true_positive / max(true_positive + false_negative, 1)),
        "f1": float(f1_score(is_attack, attack_predicted, zero_division=0)),
        "false_positive_rate": float(false_positive / max(false_positive + true_negative, 1)),
        "roc_auc": float(roc_auc_score(is_attack, attack_score)),
        "pr_auc": float(average_precision_score(is_attack, attack_score)),
        "confusion_matrix": {"tn": int(true_negative), "fp": int(false_positive), "fn": int(false_negative), "tp": int(true_positive)},
        "benign_false_positive_rate": proportion_report(false_positive, false_positive + true_negative),
        "per_class": detection_per_class(true_labels, attack_score, attack_predicted),
    }


def wilson_interval(successes, trials, confidence=INTERVAL_CONFIDENCE):
    """Wilson score interval of a binomial proportion (E. B. Wilson, J. Am. Stat. Assoc. 22(158):209-212, 1927)."""
    # The Wald interval fails for small counts and for shares near 0 or 1 (Brown, Cai and DasGupta, Stat. Sci. 16(2), 2001); a left-out class can have 38 rows.
    if trials < 1:
        raise ValueError(f"trials must be at least 1, got {trials}")
    quantile = norm.ppf((1.0 + confidence) / 2.0)
    share = successes / trials
    shrinkage = 1.0 + quantile**2 / trials
    centre = (share + quantile**2 / (2.0 * trials)) / shrinkage
    half_width = quantile * np.sqrt(share * (1.0 - share) / trials + quantile**2 / (4.0 * trials**2)) / shrinkage
    # Rounding can put a bound a little outside [0, 1] when the share is 0 or 1.
    return max(0.0, float(centre - half_width)), min(1.0, float(centre + half_width))


def proportion_report(count, rows):
    low, high = wilson_interval(count, rows)
    return {"count": int(count), "rows": int(rows), "rate": float(count / rows), "wilson95_low": low, "wilson95_high": high}


def other_day_report(labels, predicted, burst):
    """Detection on a day of unseen attack classes: Benign false alarms outside and inside the Benign burst windows, and the
    share of each attack class predicted as any attack (a model cannot output an unseen class)."""
    labels, predicted, burst = np.asarray(labels), np.asarray(predicted), np.asarray(burst, dtype=bool)
    benign = labels == config.BENIGN
    is_attack = predicted != config.BENIGN
    report = {"benign_fpr_outside_burst": float(is_attack[benign & ~burst].mean())}
    if burst.any():
        report["benign_fpr_burst"] = float(is_attack[benign & burst].mean())
    for label in pd.unique(labels[~benign]):
        report[f"detected::{label}"] = float(is_attack[labels == label].mean())
    return report
