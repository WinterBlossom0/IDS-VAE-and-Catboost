import numpy as np

from idsvae import config
from idsvae.experiments import equal_data
from idsvae.framework import detector


def probabilities_of(attack_scores):
    return np.column_stack([1.0 - attack_scores, attack_scores])


def test_cutoff_gives_the_false_alarm_target_with_equal_day_weights():
    classes = [config.BENIGN, "Bot"]
    rng = np.random.default_rng(0)
    # A large quiet day and a small noisy day: equal day weights make the noisy day set half of the false alarms.
    days = [probabilities_of(rng.uniform(0.0, 0.5, 100_000)), probabilities_of(rng.uniform(0.0, 1.0, 1_000))]
    threshold = equal_data.cutoff_of(classes, days)
    rates = [float((equal_data.flagged_labels(classes, day, threshold) != config.BENIGN).mean()) for day in days]
    assert abs(np.mean(rates) - detector.FALSE_ALARM_TARGET) < 0.002
    assert rates[0] < rates[1]


def test_flags_use_one_minus_the_benign_probability_whatever_the_class_order():
    classes = ["Bot", config.BENIGN]
    probabilities = np.array([[0.9, 0.1], [0.05, 0.95]])
    assert list(equal_data.flagged_labels(classes, probabilities, 0.5)) == [equal_data.FLAGGED, config.BENIGN]
