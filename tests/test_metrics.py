import numpy as np
import pytest
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from idsvae import metrics

CLASSES = ["AttackA", "AttackB", "Benign", "Rare"]


def noisy_predictions(seed=0):
    rng = np.random.default_rng(seed)
    true_labels = np.repeat(CLASSES, [200, 150, 600, 12])
    probabilities = rng.dirichlet(np.ones(len(CLASSES)), size=len(true_labels))
    true_index = np.searchsorted(CLASSES, true_labels)
    probabilities[np.arange(len(true_labels)), true_index] += rng.uniform(0.0, 1.5, size=len(true_labels))
    return true_labels, probabilities / probabilities.sum(axis=1, keepdims=True)


def test_confusion_scores_match_scikit_learn():
    true_labels, probabilities = noisy_predictions()
    predicted = np.asarray(CLASSES)[probabilities.argmax(axis=1)]
    macro_f1, balanced_accuracy = metrics.confusion_scores(confusion_matrix(true_labels, predicted, labels=CLASSES))
    assert np.isclose(macro_f1, f1_score(true_labels, predicted, labels=CLASSES, average="macro", zero_division=0))
    assert np.isclose(balanced_accuracy, balanced_accuracy_score(true_labels, predicted))


def test_bootstrap_ci_contains_point_estimate_and_keeps_class_sizes():
    true_labels, probabilities = noisy_predictions()
    report = metrics.seen_multiclass_report(true_labels, probabilities, CLASSES)
    interval = report["bootstrap_ci"]
    assert interval["resamples"] == metrics.BOOTSTRAP_RESAMPLES
    for key in ("macro_f1", "balanced_accuracy"):
        low, high = interval[key]
        assert low < report[key] < high
    assert metrics.bootstrap_ci(report["confusion_matrix"]["matrix"]) == interval


def test_macro_f1_with_min_support_excludes_small_classes():
    true_labels, probabilities = noisy_predictions()
    report = metrics.seen_multiclass_report(true_labels, probabilities, CLASSES)
    restricted = report["macro_f1_min_support"]
    assert restricted["min_support"] == 30
    assert restricted["excluded_classes"] == ["Rare"]
    predicted = np.asarray(CLASSES)[probabilities.argmax(axis=1)]
    expected = f1_score(true_labels, predicted, labels=["AttackA", "AttackB", "Benign"], average="macro", zero_division=0)
    assert np.isclose(restricted["value"], expected)


def test_seen_report_ranking_scores_are_one_vs_rest():
    true_labels, probabilities = noisy_predictions()
    report = metrics.seen_multiclass_report(true_labels, probabilities, CLASSES)
    assert np.isclose(report["macro_roc_auc_ovr"], roc_auc_score(true_labels, probabilities, labels=CLASSES, multi_class="ovr", average="macro"))
    per_class_average_precision = [average_precision_score(true_labels == label, probabilities[:, index]) for index, label in enumerate(CLASSES)]
    assert np.isclose(report["macro_pr_auc_ovr"], np.mean(per_class_average_precision))
    assert set(report["per_class"]["Rare"]) == {"precision", "recall", "f1", "support", "roc_auc", "pr_auc"}
    assert np.isclose(report["per_class"]["Rare"]["pr_auc"], per_class_average_precision[3])


def test_wilson_interval_matches_hand_calculation():
    # 8 of 10, z = 1.959964: centre (0.8 + z^2 / 20) / (1 + z^2 / 10) = 0.716740, half width 0.226578.
    low, high = metrics.wilson_interval(8, 10)
    assert low == pytest.approx(0.490162, abs=1e-6)
    assert high == pytest.approx(0.943318, abs=1e-6)
    # With no successes the interval is [0, z^2 / (n + z^2)].
    assert metrics.wilson_interval(0, 38) == pytest.approx((0.0, 1.959964**2 / (38 + 1.959964**2)), abs=1e-6)
    with pytest.raises(ValueError):
        metrics.wilson_interval(0, 0)


def test_proportion_report_holds_counts_rate_and_interval():
    proportion = metrics.proportion_report(30, 38)
    assert (proportion["count"], proportion["rows"]) == (30, 38)
    assert proportion["rate"] == pytest.approx(30 / 38)
    assert (proportion["wilson95_low"], proportion["wilson95_high"]) == metrics.wilson_interval(30, 38)


def test_detection_report_ranks_each_attack_class_against_benign_with_wilson_intervals():
    rng = np.random.default_rng(1)
    true_labels = np.repeat(["Benign", "AttackA", "AttackB"], [300, 100, 40])
    attack_score = rng.uniform(size=len(true_labels)) + np.select([true_labels == "AttackA", true_labels == "AttackB"], [0.6, 0.1], 0.0)
    flagged = attack_score > 0.8
    report = metrics.detection_report(true_labels, attack_score, flagged)

    is_benign = true_labels == "Benign"
    false_positives = int(flagged[is_benign].sum())
    assert report["benign_false_positive_rate"] == metrics.proportion_report(false_positives, 300)
    assert report["benign_false_positive_rate"]["rate"] == report["false_positive_rate"]
    for label in ("AttackA", "AttackB"):
        in_class = true_labels == label
        versus_benign = in_class | is_benign
        values = report["per_class"][label]
        assert values["roc_auc_vs_benign"] == pytest.approx(roc_auc_score(in_class[versus_benign], attack_score[versus_benign]))
        assert values["pr_auc_vs_benign"] == pytest.approx(average_precision_score(in_class[versus_benign], attack_score[versus_benign]))
        assert values["flagged"] == metrics.proportion_report(int(flagged[in_class].sum()), int(in_class.sum()))
        assert (values["rows"], values["flagged_as_attack"]) == (int(in_class.sum()), pytest.approx(flagged[in_class].mean()))
    assert set(report["per_class"]["Benign"]) == {"rows", "flagged_as_attack", "flagged"}
    # The per-class AUC of the weaker class must not equal the AUC over all attack rows.
    assert report["per_class"]["AttackB"]["roc_auc_vs_benign"] < report["roc_auc"] < report["per_class"]["AttackA"]["roc_auc_vs_benign"]


def arm_with_accuracy(true_labels, accuracy, seed):
    """Predicted labels: the true class with probability accuracy, else a random class."""
    rng = np.random.default_rng(seed)
    keep = rng.uniform(size=len(true_labels)) < accuracy
    return np.where(keep, true_labels, rng.choice(CLASSES, size=len(true_labels)))


def paired_bootstrap(true_labels, first_predicted, second_predicted, metric, labels, resamples=metrics.PAIRED_BOOTSTRAP_RESAMPLES, seed=metrics.BOOTSTRAP_SEED):
    return metrics.paired_bootstrap_test(metrics.paired_resampled_tables(true_labels, first_predicted, second_predicted, labels, resamples, seed), metric)


def test_paired_bootstrap_of_identical_arms_gives_zero_delta():
    true_labels, _ = noisy_predictions()
    predicted = arm_with_accuracy(true_labels, 0.6, seed=1)
    result = paired_bootstrap(true_labels, predicted, predicted, metrics.balanced_accuracy_of_matrices, CLASSES)
    assert result["delta"] == 0.0
    # Rows drawn separately for each arm would spread the resampled deltas around 0; the same rows give exactly 0.
    assert result["ci95_low"] == result["ci95_high"] == 0.0
    assert result["p_value"] == 1.0
    assert result["resamples"] == metrics.PAIRED_BOOTSTRAP_RESAMPLES


def test_paired_bootstrap_of_a_better_arm_gives_an_interval_above_zero():
    true_labels, _ = noisy_predictions()
    better = arm_with_accuracy(true_labels, 0.95, seed=1)
    worse = arm_with_accuracy(true_labels, 0.6, seed=2)
    result = paired_bootstrap(true_labels, better, worse, metrics.macro_f1_of_classes(range(len(CLASSES))), CLASSES)
    assert result["delta"] > 0.0
    assert 0.0 < result["ci95_low"] < result["delta"] < result["ci95_high"]
    assert result["p_value"] < 0.05
    mirrored = paired_bootstrap(true_labels, worse, better, metrics.macro_f1_of_classes(range(len(CLASSES))), CLASSES)
    assert mirrored["delta"] == pytest.approx(-result["delta"])
    assert mirrored["ci95_high"] < 0.0
    assert mirrored["p_value"] == result["p_value"]


def reference_bootstrap(true_labels, first, second, resamples, seed, same_rows):
    """Resampled macro-F1 deltas from explicit row indices and scikit-learn, with the draw order of metrics.paired_resampled_tables."""
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(resamples):
        first_rows = rng.integers(0, len(true_labels), size=len(true_labels))
        second_rows = first_rows if same_rows else rng.integers(0, len(true_labels), size=len(true_labels))
        first_value = f1_score(true_labels[first_rows], first[first_rows], labels=CLASSES, average="macro", zero_division=0)
        second_value = f1_score(true_labels[second_rows], second[second_rows], labels=CLASSES, average="macro", zero_division=0)
        deltas.append(first_value - second_value)
    return np.array(deltas)


def test_paired_bootstrap_applies_the_same_drawn_rows_to_both_arms():
    true_labels, _ = noisy_predictions()
    first = arm_with_accuracy(true_labels, 0.8, seed=1)
    # The second arm repeats the first arm on most rows, so that paired deltas vary much less than unpaired deltas.
    second = np.where(np.random.default_rng(2).uniform(size=len(true_labels)) < 0.1, arm_with_accuracy(true_labels, 0.0, seed=3), first)
    resamples = 200
    result = paired_bootstrap(true_labels, first, second, metrics.macro_f1_of_classes(range(len(CLASSES))), CLASSES, resamples=resamples, seed=7)
    observed = f1_score(true_labels, first, labels=CLASSES, average="macro", zero_division=0) - f1_score(true_labels, second, labels=CLASSES, average="macro", zero_division=0)
    assert result["delta"] == pytest.approx(observed)
    paired = reference_bootstrap(true_labels, first, second, resamples, seed=7, same_rows=True)
    assert (result["ci95_low"], result["ci95_high"]) == pytest.approx(tuple(np.percentile(paired, [2.5, 97.5])))
    assert result["p_value"] == pytest.approx(metrics.bootstrap_p_value(observed, paired))
    # Separate draws for each arm give a wider interval, so the comparison above detects unpaired draws.
    unpaired = reference_bootstrap(true_labels, first, second, resamples, seed=7, same_rows=False)
    unpaired_low, unpaired_high = np.percentile(unpaired, [2.5, 97.5])
    assert unpaired_high - unpaired_low > 2.0 * (result["ci95_high"] - result["ci95_low"])


def test_paired_bootstrap_rejects_labels_outside_the_label_list():
    with pytest.raises(ValueError):
        paired_bootstrap(np.array(["Benign", "Other"]), np.array(["Benign", "Benign"]), np.array(["Benign", "Benign"]), metrics.balanced_accuracy_of_matrices, ["Benign"])


def test_bootstrap_p_value_counts_resamples_beyond_twice_the_delta():
    resampled = np.array([0.25, 0.15, 0.21, 0.30, 0.20])
    # 2 delta = 0.2; the count is strict, as in Berg-Kirkpatrick et al., EMNLP 2012, Fig. 1.
    assert metrics.bootstrap_p_value(0.1, resampled) == pytest.approx(3 / 5)
    assert metrics.bootstrap_p_value(-0.1, -resampled) == pytest.approx(3 / 5)
    assert metrics.bootstrap_p_value(0.0, resampled) == 1.0


def test_other_day_report_separates_burst_false_alarms_and_detection_per_class():
    labels = ["Benign", "Benign", "Benign", "Benign", "Bot", "Bot", "Infilteration"]
    predicted = ["Benign", "DoS", "DoS", "Benign", "DoS", "Benign", "Benign"]
    burst = [False, False, True, True, False, False, False]
    report = metrics.other_day_report(labels, predicted, burst)
    assert report == {"benign_fpr_outside_burst": 0.5, "benign_fpr_burst": 0.5, "detected::Bot": 0.5, "detected::Infilteration": 0.0}
    assert "benign_fpr_burst" not in metrics.other_day_report(labels, predicted, [False] * len(labels))
