import numpy as np
import pandas as pd
import pytest

from idsvae.framework import correction

SETTINGS = correction.RuleSettings()
# An arbitrary minimum support for the unit tests; the pipeline computes it from the calibration day.
TEST_SUPPORT = 10


def stream(rows, start_minute=0):
    return np.datetime64("2018-02-15T10:00") + np.arange(rows).astype("timedelta64[s]") + np.timedelta64(start_minute, "m")


def encoded(times, recon_nll=0.0, latent=None):
    frame = pd.DataFrame(np.zeros((len(times), 5)) if latent is None else latent, columns=[f"z{index}" for index in range(5)])
    frame["time"] = times
    frame["recon_nll"] = recon_nll
    return frame


def test_a_burst_in_one_cell_scores_higher_than_steady_traffic():
    steady = np.repeat(np.arange(30), 2)  # two flows per minute in cell 0
    burst_minutes = np.full(40, 30)  # forty flows in minute 30
    minutes = np.concatenate([steady, burst_minutes])
    scores = correction.midas_scores(np.zeros(len(minutes), dtype=np.uint64), minutes)
    assert scores[-1] > 10 * scores[:60].max()


def attributions(seconds, predicted, attack_classes):
    context = correction.AttackContext(attack_classes, SETTINGS.window_seconds)
    result = []
    for second, label in zip(seconds, predicted):
        attack, count = context.top(second)
        result.append(attack if count >= TEST_SUPPORT else "")
        context.add(label)
    return result


def test_attack_context_needs_min_support_and_uses_only_earlier_flows():
    predicted = ["DoS attacks-Slowloris"] * 12 + ["Benign"] * 18
    attributed = attributions(range(30), predicted, ["DoS attacks-Slowloris", "SSH-Bruteforce"])
    assert attributed[0] == "" and attributed[9] == ""  # fewer than TEST_SUPPORT earlier attack predictions
    assert attributed[10] == "DoS attacks-Slowloris" and attributed[29] == "DoS attacks-Slowloris"


def test_attack_context_ignores_flows_of_the_same_second_and_forgets_old_flows():
    seconds = [0] * 12 + [0, 1, 61, 62]
    attributed = attributions(seconds, ["SSH-Bruteforce"] * 16, ["SSH-Bruteforce"])
    assert attributed[12] == ""  # twelve earlier flows, but all in the same second
    assert attributed[13] == "SSH-Bruteforce"
    assert attributed[14] == ""  # second 61: the flows of second 0 are older than the window; second 1 has one flow
    assert attributed[15] == ""


def test_apply_rule_changes_only_novel_bursty_flows_with_attack_context():
    rows = 40
    frame = encoded(stream(rows), np.where(np.arange(rows) >= 30, 100.0, 0.0))
    predicted = np.array(["DoS attacks-Slowloris"] * 15 + ["Benign"] * 25, dtype=object)
    rule_thresholds = correction.RuleThresholds(50.0, -1.0, np.ones(5), TEST_SUPPORT, ["DoS attacks-Slowloris"], SETTINGS)
    changed, override = correction.apply_rule(frame, predicted, rule_thresholds)
    assert not override[:30].any()
    assert override[30:].all() and (changed[30:] == "DoS attacks-Slowloris").all()
    assert (changed[15:30] == "Benign").all()


def test_days_are_separate_streams():
    minutes_day1 = np.datetime64("2018-02-14T10:00") + np.arange(30).astype("timedelta64[m]")
    minutes_day2 = np.datetime64("2018-02-15T10:00") + np.arange(30).astype("timedelta64[m]")
    frame = encoded(np.concatenate([minutes_day1, minutes_day2]))
    both = correction.stream_scores(frame, 5, np.ones(5), SETTINGS.cell_width)
    alone = correction.stream_scores(frame.iloc[30:].reset_index(drop=True), 5, np.ones(5), SETTINGS.cell_width)
    np.testing.assert_allclose(both[30:], alone)


def test_min_support_lies_above_the_false_alarm_counts_of_calibration_benign_flows():
    seconds = np.arange(200)
    calib = encoded(np.datetime64("2018-02-23T10:00") + seconds.astype("timedelta64[s]"))
    calib["label"] = "Benign"
    predicted = np.array(["SSH-Bruteforce" if second % 20 == 0 else "Benign" for second in seconds], dtype=object)
    # One false SSH alarm every 20 s: at most 3 fall in a 60 s window before a flow.
    assert correction.min_support(calib, predicted, ["SSH-Bruteforce"], 5, SETTINGS) == 4


def test_min_support_and_the_online_rule_count_the_same_way():
    rng = np.random.default_rng(0)
    times = np.sort(np.datetime64("2018-02-23T10:00") + rng.integers(0, 300, 400).astype("timedelta64[s]"))
    calib = encoded(times, latent=rng.normal(size=(400, 5)))
    calib["label"] = "Benign"
    predicted = rng.choice(np.array(["Benign", "SSH-Bruteforce"], dtype=object), size=400, p=[0.8, 0.2])
    support = correction.min_support(calib, predicted, ["SSH-Bruteforce"], 5, SETTINGS)
    # With novelty and burst always true, the share of flows the rule relabels is at most 1 - support_quantile.
    rule_thresholds = correction.RuleThresholds(-np.inf, -np.inf, np.ones(5), support, ["SSH-Bruteforce"], SETTINGS)
    _, override = correction.apply_rule(calib, predicted, rule_thresholds)
    assert override.mean() <= 1 - SETTINGS.support_quantile + 1e-9


def test_the_rule_does_not_depend_on_the_order_of_rows_inside_one_second():
    rng = np.random.default_rng(1)
    rows = 600
    times = np.datetime64("2018-02-15T10:00") + np.repeat(np.arange(60), 10).astype("timedelta64[s]")
    frame = encoded(times, rng.normal(size=rows) * 50, rng.normal(size=(rows, 5)))
    predicted = rng.choice(np.array(["Benign", "DoS attacks-Slowloris"], dtype=object), size=rows)
    rule_thresholds = correction.RuleThresholds(0.0, 0.5, np.ones(5), 5, ["DoS attacks-Slowloris"], SETTINGS)
    changed, _ = correction.apply_rule(frame, predicted, rule_thresholds)
    shuffle = rng.permutation(rows)
    shuffled_changed, _ = correction.apply_rule(frame.iloc[shuffle].reset_index(drop=True), predicted[shuffle], rule_thresholds)
    restored = np.empty(rows, dtype=object)
    restored[shuffle] = shuffled_changed
    assert (restored == changed).all()


def test_leave_one_out_stream_fills_the_left_out_rows_and_rejects_rows_that_do_not_match():
    train = encoded(stream(10))
    train["label"] = ["Benign"] * 6 + ["SSH-Bruteforce"] * 4
    holdout = encoded(stream(5, start_minute=1))
    holdout["label"] = "Benign"
    holdout_predicted = np.array(["Benign"] * 5, dtype=object)
    left_out = pd.DataFrame({"label": "SSH-Bruteforce", "time": train["time"].to_numpy()[6:], "split_row": np.arange(6, 10),
                             "predicted": ["DoS attacks-Slowloris", "Benign", "Benign", "Benign"]})
    _, context, threshold_rows = correction.leave_one_out_stream(train, holdout, holdout_predicted, left_out, "SSH-Bruteforce")
    assert context[6] == "DoS attacks-Slowloris" and (context[7:] == "Benign").all()
    assert (threshold_rows["label"] == "Benign").all() and len(threshold_rows) == 6
    for broken in (left_out.assign(split_row=np.arange(5, 9)), left_out.assign(time=left_out["time"] + np.timedelta64(1, "s"))):
        with pytest.raises(ValueError, match="do not match"):
            correction.leave_one_out_stream(train, holdout, holdout_predicted, broken, "SSH-Bruteforce")
