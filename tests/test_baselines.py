import numpy as np

from idsvae.experiments import baselines, deep_baselines, final_run
from idsvae.paper import numbers


def test_scaling_uses_the_training_statistics():
    reference = np.array([[0.0, 10.0], [100.0, 10.0]])
    scaled = baselines.scaled(np.array([[0.0, 10.0]]), reference)
    assert scaled[0, 0] == -1.0 and scaled[0, 1] == 0.0


def test_kang_inputs_hold_out_a_quarter_of_benign_rows_and_attacks_score_higher():
    rng = np.random.default_rng(0)
    # Enough rows for about 100 optimizer steps at the autoencoder batch size of 4096.
    benign = rng.normal(size=(20_000, 6))
    benign[:, 1] = benign[:, 0]
    benign[:, 3] = benign[:, 2]
    attack = rng.normal(size=(2_000, 6))
    attack[:, 1] = -attack[:, 0]
    attack[:, 3] = -attack[:, 2]
    train_x = np.vstack([benign, attack])
    train_y = np.array(["Benign"] * 20_000 + ["Attack"] * 2_000)
    classifier_rows, classifier_scores, other_scores, scorer = baselines.kang_inputs(train_x, train_y, attack[:50], seed=0, threads=1)
    assert classifier_rows[20_000:].all()  # every attack row reaches the classifier
    assert 0.23 < classifier_rows[:20_000].mean() < 0.27  # about 25 % of the benign rows
    assert classifier_scores.shape == (classifier_rows.sum(), 1) and other_scores.shape == (50, 1)
    assert np.array_equal(scorer.score_of_scaled(scorer.scaled(attack[:50])), other_scores)  # the deployment form scores as in training
    is_attack = train_y[classifier_rows] == "Attack"
    assert np.median(classifier_scores[is_attack]) > np.median(classifier_scores[~is_attack])


def test_flow_windows_hold_each_flow_and_its_predecessors_of_the_same_day():
    times = np.array(["2018-02-14T10:00:05", "2018-02-15T09:00:00", "2018-02-14T10:00:01", "2018-02-14T10:00:03"], dtype="datetime64[s]")
    windows = deep_baselines.day_stream_windows(times, times.astype("datetime64[D]"), np.zeros(len(times)))
    assert windows.shape == (4, deep_baselines.WINDOW_SIZE)
    assert windows[0, -3:].tolist() == [2, 3, 0]  # day 02-14 in time order: rows 2, 3, 0
    assert set(windows[0, :-3].tolist()) == {2}  # missing predecessors repeat the earliest flow of the day
    assert set(windows[1].tolist()) == {1}  # the only flow of 02-15 sees no flow of 02-14


def test_log_min_max_scaling_maps_the_training_range_to_zero_one():
    train = np.array([[0.0, 5.0], [99.0, 5.0]])
    minimum, scale = deep_baselines.log_min_max_parameters(train)
    train_scaled, other = (deep_baselines.apply_log_min_max(values, minimum, scale) for values in (train, np.array([[99.0, 7.0]])))
    assert train_scaled[:, 0].tolist() == [0.0, 1.0]
    assert other[0, 0] == 1.0 and other[0, 1] == 0.0  # a constant training column gives 0


def test_flow_windows_order_ties_by_the_key_not_by_the_row_position():
    times = np.array(["2018-02-14T10:00:00"] * 3, dtype="datetime64[s]")
    windows = deep_baselines.day_stream_windows(times, times.astype("datetime64[D]"), np.array([2, 0, 1]))
    assert windows[:, -1].tolist() == [0, 1, 2]
    assert windows[0, -2] == 2 and windows[2, -2] == 1  # order in the second: rows 1, 2, 0


def test_training_windows_hold_only_training_flows_and_evaluation_windows_hold_every_flow():
    # Rows 0-2 are training flows; row 3 is a hold-out flow between them in time, row 4 a later hold-out flow.
    times = np.array(["2018-02-14T10:00:01", "2018-02-14T10:00:03", "2018-02-14T10:00:05", "2018-02-14T10:00:02",
                      "2018-02-14T10:00:06"], dtype="datetime64[s]")
    windows, training_windows = deep_baselines.stream_windows(times, times.astype("datetime64[D]"), np.zeros(len(times)), 3)
    assert training_windows.shape == (3, deep_baselines.WINDOW_SIZE) and training_windows.max() < 3
    assert training_windows[1, -2:].tolist() == [0, 1]  # the predecessor of row 1 is row 0, not the hold-out row 3
    assert windows[1, -2:].tolist() == [3, 1] and windows[4, -2:].tolist() == [2, 4]


def test_fold_files_and_folds_agree_with_the_framework(tmp_path, monkeypatch):
    assert baselines.LEFT_OUT_CLASSES == final_run.LEFT_OUT_CLASSES
    assert baselines.SEEDS == final_run.SEEDS
    monkeypatch.setattr(baselines, "BASELINES_DIR", tmp_path)
    path = baselines.output_path("xgboost", 3, "DoS attacks-GoldenEye")
    assert path.name == "xgboost_s3_loo_dos-attacks-goldeneye.json"
    path.write_text('{"left_out_detection": 0.25}')
    assert numbers.baseline_result("xgboost", 3, "DoS attacks-GoldenEye") == {"left_out_detection": 0.25}
    assert numbers.baseline_result("xgboost", 4, "DoS attacks-GoldenEye") is None


def test_youden_helpers_count_every_benign_flow_and_print_signed_values(monkeypatch):
    import pytest

    monkeypatch.setattr(numbers, "test_benign_counts", lambda: (2_125, 358_621))
    outside_share = 2_125 / (2_125 + 358_621)
    udp, hoic = numbers.test_youden({"benign_fpr_outside_burst": 1.0, "benign_fpr_burst": 0.0,
                                           "detected::DDOS attack-LOIC-UDP": 0.5, "detected::DDOS attack-HOIC": 0.0})
    assert udp == pytest.approx(0.5 - outside_share) and hoic == pytest.approx(-outside_share)
    assert numbers.youden_of_day_reports([{"benign_fpr_outside_burst": 0.1, "detected::A": 0.6, "detected::B": 0.2}]) == pytest.approx([0.5, 0.1])
    assert numbers.signed(-0.004) == "$-$0.004" and numbers.signed(-1.256) == "$-$1.26" and numbers.signed(2.0) == "2.00"
    assert numbers.signed(0.0) == "0" and numbers.exact_text(0.00016, 2) == "0.0002" and numbers.exact_text(0.016, 2) == "0.02"



def test_timing_text_keeps_two_significant_digits_and_drops_empty_flow_groups():
    import pytest

    assert [numbers.time_text(value) for value in (4.6979, 1.5979, 0.0982, 0.010_3, 0.000_774_8)] == ["4.7", "1.6", "0.098", "0.010", "0.00077"]
    assert numbers.time_text(None) == numbers.PENDING
    assert numbers.time_text(0.0996) == "0.10" and numbers.time_text(9.96) == "10"
    with pytest.raises(ValueError):
        numbers.time_text(0.0)
    assert numbers.window_composition(9_928, 0, 72) == "9{,}928 Benign burst and 72 attack flows"
    assert numbers.window_composition(5, 3, 2) == "5 Benign burst, 3 other Benign and 2 attack flows"
    assert numbers.window_composition(0, 10, 0) == "10 other Benign flows"

def youden_test_rows(udp_values, hoic_values):
    # (label, J held-out days, J folds, J test LOIC-UDP, J test HOIC): classifier, framework, then the re-run methods.
    rows = [("Ours, classifier", 0.5, 10.0, 0.0, 0.0), (r"\textbf{Ours, framework}", 30.0, 70.0, 15.0, -60.0)]
    rows += [(label, 20.0, 30.0, udp, hoic) for (label, _), udp, hoic in zip(numbers.BASELINE_ROWS, udp_values, hoic_values)]
    return rows


def test_youden_macros_count_methods_and_name_the_best(monkeypatch):
    udp = [60.0, 40.0, 20.0] + [0.0] * 7
    monkeypatch.setattr(numbers, "youden_rows", lambda: youden_test_rows(udp, [-1.0] * 10))
    macros = numbers.youden_macros()
    assert macros["JUdpAboveCount"] == "three" and macros["JHoicAboveCount"] == "all ten" and macros["JHoicPositiveCount"] == "no"
    assert macros["JBestUdpName"] == "Random forest" and macros["JBestUdp"] == "60.00" and macros["JOursDays"] == "30.00"


def test_youden_macros_need_every_method_and_check_the_claims_of_the_text(monkeypatch):
    import pytest

    rows = youden_test_rows([60.0, 40.0] + [0.0] * 8, [-1.0] * 10)
    monkeypatch.setattr(numbers, "youden_rows", lambda: [*rows[:-1], (rows[-1][0], None, 30.0, 0.0, -1.0)])
    assert numbers.youden_macros() == {}
    monkeypatch.setattr(numbers, "youden_rows", lambda: [*rows[:-1], (rows[-1][0], 35.0, 30.0, 0.0, -1.0)])
    with pytest.raises(ValueError, match="leads"):
        numbers.youden_macros()


def test_count_words_and_class_count_cells():
    assert numbers.count_words(0, 10) == "no" and numbers.count_words(3, 10) == "three" and numbers.count_words(10, 10) == "all ten"
    cell = numbers.class_counts_cell({"Benign": 1000, "Bot": 20, "SQL Injection": 300})
    assert cell == "1{,}320 (Benign 1{,}000; SQL injection 300; Bot 20)"
    assert numbers.class_counts_cell({"Benign": 5, "Bot": 2, "SQL Injection": 3}, "web attacks") == "10 (Benign 5; web attacks 5)"


def test_deployed_flow_transformer_stream_gives_the_windows_of_training():
    import torch

    torch.manual_seed(0)
    network = deep_baselines.FlowTransformerNetwork(3, 2).eval()
    values = np.random.default_rng(0).random((12, 3))
    minimum, scale = deep_baselines.log_min_max_parameters(values)
    times = np.array([f"2018-02-21T10:00:{second:02d}" for second in range(12)], dtype="datetime64[s]")
    windows = deep_baselines.day_stream_windows(times, times.astype("datetime64[D]"), np.zeros(len(times)))
    scaled = torch.as_tensor(deep_baselines.apply_log_min_max(values.copy(), minimum, scale), dtype=torch.float32)
    with torch.no_grad():
        expected = torch.softmax(network(scaled[torch.as_tensor(windows)]), dim=-1).mean(dim=1).numpy()
    deployed = deep_baselines.DeployedFlowTransformer(minimum, scale, network)
    stream = deployed.stream()
    in_batches = np.concatenate([stream(values[start:start + 5]) for start in range(0, 12, 5)])
    assert np.allclose(in_batches, expected, atol=1e-6)
    assert np.allclose(deployed.predict_proba(values), expected, atol=1e-6)


def test_standardizer_and_shallow_deployment_reproduce_the_training_scaling():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(0)
    train, other = rng.random((50, 4)) * 100, rng.random((7, 4)) * 100
    assert np.array_equal(baselines.Standardizer(train).transform(other), baselines.scaled(other, train))
    targets = (train[:, 0] > 50).astype(int)
    standardizer = baselines.Standardizer(train)
    model = LogisticRegression().fit(standardizer.transform(train), targets)
    deployed = baselines.DeployedShallow(standardizer, model)
    assert np.array_equal(deployed.predict_proba(other), model.predict_proba(baselines.scaled(other, train)))
