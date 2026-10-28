import numpy as np
import pandas as pd

from idsvae import config
from idsvae.data import extra_days
from idsvae.experiments import detector_depths, detector_runs, final_run, held_out_days
from idsvae.framework import correction, detector, vae


def test_cell_window_share_matches_a_direct_count_and_ignores_flows_of_the_same_second():
    rng = np.random.default_rng(0)
    seconds = np.sort(rng.integers(0, 400, 300))
    novel = rng.random(300) < 0.4
    window = detector.CellWindow()
    for index, (second, is_novel) in enumerate(zip(seconds, novel)):
        earlier = [j for j in range(index) if seconds[j] < second and seconds[j] >= second - 120]
        expected = novel[earlier].mean() if earlier else 0.0
        assert abs(window.earlier_novel_share(int(second), 120) - expected) < 1e-12
        window.add(int(second), bool(is_novel))


def test_combine_only_relabels_flows_that_stage3_calls_benign():
    stage3 = np.array([config.BENIGN, "SSH-Bruteforce", config.BENIGN, config.BENIGN], dtype=object)
    flags = np.array([True, True, False, True])
    combined = detector.combine(stage3, flags)
    assert list(combined) == [detector.UNKNOWN, "SSH-Bruteforce", config.BENIGN, detector.UNKNOWN]
    assert list(stage3) == [config.BENIGN, "SSH-Bruteforce", config.BENIGN, config.BENIGN]


def test_quantile_of_is_the_share_at_or_below():
    assert list(detector.quantile_of([1.0, 2.0, 3.0, 4.0], np.array([0.5, 2.0, 9.0]))) == [0.0, 0.5, 1.0]


def test_weighted_quantile_gives_each_group_its_weight():
    values = np.array([1.0, 2.0, 3.0, 10.0])
    # The last value carries half of the weight, so the median is 3 and the 0.9 quantile is 10.
    weights = np.array([1.0, 1.0, 1.0, 3.0])
    assert detector.weighted_quantile(values, weights, 0.5) == 3.0
    assert detector.weighted_quantile(values, weights, 0.9) == 10.0


def test_a_day_with_benign_flows_in_one_hour_does_not_cover_the_capture():
    import pandas as pd

    spread = pd.DataFrame({"label": config.BENIGN, "time": pd.date_range("2018-02-20 08:00", "2018-02-20 17:59", periods=1000)})
    one_hour = pd.DataFrame({"label": config.BENIGN, "time": pd.date_range("2018-02-16 13:00", "2018-02-16 13:48", periods=1000)})
    assert extra_days.benign_covers_day(spread)
    assert not extra_days.benign_covers_day(one_hour)


def test_online_detector_equals_the_batch_features_on_two_days(monkeypatch):
    import pandas as pd

    rng = np.random.default_rng(3)
    rows = 3000
    seconds = np.sort(np.r_[rng.integers(0, 3600, rows // 2), 86_400 + rng.integers(0, 3600, rows - rows // 2)])
    frame = pd.DataFrame({f"z{i}": rng.normal(size=rows).round(1) for i in range(2)})
    frame["recon_nll"] = rng.normal(size=rows)
    frame["kl"] = rng.gamma(2.0, size=rows)
    frame["time"] = pd.to_datetime(seconds, unit="s")
    frame["label"] = config.BENIGN
    settings = correction.RuleSettings(cell_width=0.5, window_seconds=60)
    scale = np.ones(2)
    references = [rng.gamma(1.0, size=500), rng.gamma(1.0, size=500), rng.random(500)]
    nll_reference, kl_reference = rng.normal(size=500), rng.gamma(2.0, size=500)
    novelty = 0.5
    # Small reservoir settings make the background median change inside the test.
    monkeypatch.setattr(detector, "BACKGROUND_WARM_UP", 50)
    monkeypatch.setattr(detector, "BACKGROUND_REFRESH", 20)
    monkeypatch.setattr(detector, "BACKGROUND_RESERVOIR", 100)
    relative = detector.relative_nll(frame, 2, scale, settings, 5.0, 0.0)
    stream = detector.stream_values(frame, 2, scale, relative > novelty, settings)
    kl_quantile = detector.quantile_of(kl_reference, frame["kl"].to_numpy())
    kl_tail = 1 - 2 * np.minimum(kl_quantile, 1 - kl_quantile)
    expected = np.column_stack([detector.quantile_of(nll_reference, relative), kl_tail]
                               + [detector.quantile_of(reference, values) for reference, values in zip(references, stream)])
    online = detector.OnlineDetector(["nll_q", "kl_tail_q", *detector.STREAM_FEATURES], scale, settings, novelty, nll_reference, kl_reference,
                                          references, detector.DayBackground(5.0, 0.0, np.random.default_rng(0)))
    order = correction.time_order(frame, 2)[1]
    latent = frame[["z0", "z1"]].to_numpy(np.float64)
    actual = np.zeros_like(expected)
    for row in order:
        actual[row] = online.features(int(seconds[row]), latent[row], frame["recon_nll"][row], frame["kl"][row])
    assert np.array_equal(actual, expected)


def test_training_rows_exclude_held_out_families_and_use_only_the_training_half_of_benign(tmp_path, monkeypatch):
    import pandas as pd

    monkeypatch.setattr(detector_runs, "FEATURE_DIR", tmp_path)
    rng = np.random.default_rng(0)

    def frame(labels):
        values = {name: rng.random(len(labels)) for name in ("nll_q", "kl_tail_q", "burst_q", "novel_burst_q", "novel_share_q")}
        return pd.DataFrame({**values, "label": labels, "time": pd.Timestamp("2018-02-22")})
    day = frame([config.BENIGN] * 400 + ["Bot"] * 30 + ["Infilteration"] * 20)
    day.to_parquet(detector_runs.feature_file("topology", 3, final_run.MAIN, "extra_02-22"))
    calib = frame([config.BENIGN] * 300)
    calib.to_parquet(detector_runs.feature_file("topology", 3, final_run.MAIN, "calib"))
    features, targets, weights = detector_runs.multi_day_training_rows(3, ["02-22"], (), "all", excluded_labels=("Infilteration",))
    assert targets.sum() == 30
    training_half, cutoff_half = detector_runs.benign_halves(day, detector_runs.day_salt("02-22"))
    assert not set(training_half) & set(cutoff_half)
    negatives = features[targets == 0]
    day_values = set(day["nll_q"].iloc[cutoff_half]) | set(calib["nll_q"].iloc[detector_runs.calib_halves(calib)[1]])
    assert not set(negatives["nll_q"]) & day_values
    assert abs(weights[targets == 1].sum() - 0.5) < 1e-9 and abs(weights[targets == 0].sum() - 0.5) < 1e-9


def test_held_out_day_flags_use_the_thread_count_of_the_selection_runs(tmp_path, monkeypatch):
    import pandas as pd

    monkeypatch.setattr(detector_runs, "FEATURE_DIR", tmp_path)
    monkeypatch.setattr(detector, "ITERATIONS", 5)
    rng = np.random.default_rng(0)

    def frame(labels):
        values = {name: rng.random(len(labels)) for name in ("nll_q", "kl_tail_q", "burst_q", "novel_burst_q", "novel_share_q")}
        return pd.DataFrame({**values, "label": labels, "time": pd.Timestamp("2018-02-22")})
    frame([config.BENIGN] * 200 + ["Bot"] * 30).to_parquet(detector_runs.feature_file("topology", 3, final_run.MAIN, "extra_03-02"))
    frame([config.BENIGN] * 200 + ["DDoS attacks-LOIC-HTTP"] * 30).to_parquet(detector_runs.feature_file("topology", 3, final_run.MAIN, "extra_02-20"))
    frame([config.BENIGN] * 200).to_parquet(detector_runs.feature_file("topology", 3, final_run.MAIN, "calib"))
    used_threads = []
    original_fit = detector.fit_detector

    def recording_fit(features, targets, weights, seed, threads=detector.THREADS, depth=detector.DEPTH):
        used_threads.append(threads)
        return original_fit(features, targets, weights, seed, threads, depth)
    monkeypatch.setattr(detector, "fit_detector", recording_fit)
    monkeypatch.setattr(final_run, "LEFT_OUT_CLASSES", ())
    frame_out, flagged = detector_runs.held_out_day_flags(3, "all", "02-20", ["02-20", "03-02"])
    assert used_threads == [detector_runs.SELECTION_THREADS]
    assert len(flagged) == len(frame_out) == 230


def test_held_out_day_rows_enter_neither_the_training_rows_nor_the_cutoff_rows(tmp_path, monkeypatch):
    import pandas as pd

    monkeypatch.setattr(detector_runs, "FEATURE_DIR", tmp_path)
    monkeypatch.setattr(detector, "ITERATIONS", 5)
    monkeypatch.setattr(final_run, "LEFT_OUT_CLASSES", ())
    rng = np.random.default_rng(0)
    feature_names = detector.FEATURE_SETS["all"]

    def frame(labels, offset):
        values = {name: offset + rng.random(len(labels)) for name in feature_names}
        return pd.DataFrame({**values, "label": labels, "time": pd.Timestamp("2018-02-22")})
    # Feature values of the held-out day are at least 10, the values of every other day are below 1.
    held_out = frame([config.BENIGN] * 200 + ["DDoS attacks-LOIC-HTTP"] * 30, 10.0)
    held_out.to_parquet(detector_runs.feature_file("topology", 3, final_run.MAIN, "extra_02-20"))
    frame([config.BENIGN] * 200 + ["Bot"] * 30 + ["DDoS attacks-LOIC-HTTP"] * 20, 0.0).to_parquet(
        detector_runs.feature_file("topology", 3, final_run.MAIN, "extra_03-02"))
    frame([config.BENIGN] * 200, 0.0).to_parquet(detector_runs.feature_file("topology", 3, final_run.MAIN, "calib"))
    training, cutoff_days, cutoff_rows = [], [], []
    original_fit, original_cutoff = detector.fit_detector, detector_runs.multi_day_cutoff

    class RecordingModel:
        def __init__(self, model):
            self.model, self.scored = model, None

        def predict_proba(self, rows):
            if self.scored is not None:
                self.scored.append(rows)
            return self.model.predict_proba(rows)

    def recording_fit(features, targets, weights, seed, threads=detector.THREADS, depth=detector.DEPTH):
        training.append((features, targets))
        return RecordingModel(original_fit(features, targets, weights, seed, threads, depth))

    def recording_cutoff(model, seed, days, *args, **kwargs):
        cutoff_days.append(list(days))
        model.scored = cutoff_rows
        threshold = original_cutoff(model, seed, days, *args, **kwargs)
        model.scored = None
        return threshold
    monkeypatch.setattr(detector, "fit_detector", recording_fit)
    monkeypatch.setattr(detector_runs, "multi_day_cutoff", recording_cutoff)
    frame_out, scores, _ = detector_runs.held_out_day_scores(3, "all", "02-20", ["02-20", "03-02"])
    assert len(training) == 1
    features, targets = training[0]
    assert (features.to_numpy() < 1).all()
    # The family of the held-out day leaves training on every day: only the Bot flows of 03-02 are positives.
    assert targets.sum() == 30
    assert cutoff_days == [["03-02"]]
    assert cutoff_rows and all((rows.to_numpy() < 1).all() for rows in cutoff_rows)
    assert detector_runs.same_rows(frame_out, held_out) and len(scores) == len(held_out)


def test_selection_rule_takes_the_best_qualified_configuration_or_else_the_best_of_all():
    limit = detector_runs.FALSE_ALARM_TOLERANCE * detector.FALSE_ALARM_TARGET
    scores = {"low_alarms": {"mean_detection": 0.6, "mean_benign_false_alarms": limit},
              "high_alarms": {"mean_detection": 0.9, "mean_benign_false_alarms": 2 * limit},
              "lower_alarms": {"mean_detection": 0.5, "mean_benign_false_alarms": limit / 2}}
    assert detector_runs.chosen_configuration(scores) == ("low_alarms", ["low_alarms", "lower_alarms"])
    none_qualified = {name: {**score, "mean_benign_false_alarms": 3 * limit} for name, score in scores.items()}
    assert detector_runs.chosen_configuration(none_qualified) == ("high_alarms", [])


def test_depth_reaches_catboost_without_a_change_of_the_module_depth(tmp_path, monkeypatch):
    import pandas as pd

    monkeypatch.setattr(detector_runs, "FEATURE_DIR", tmp_path)
    monkeypatch.setattr(detector, "ITERATIONS", 5)
    monkeypatch.setattr(final_run, "LEFT_OUT_CLASSES", ())
    monkeypatch.setattr(final_run, "SEEDS", (3,))
    monkeypatch.setattr(detector_runs, "available_days", lambda seed: ["02-20", "03-02"])
    rng = np.random.default_rng(0)

    def frame(labels):
        values = {name: rng.random(len(labels)) for name in detector.FEATURE_SETS["all"]}
        return pd.DataFrame({**values, "label": labels, "time": pd.Timestamp("2018-02-22")})
    frame([config.BENIGN] * 200 + ["Bot"] * 30).to_parquet(detector_runs.feature_file("topology", 3, final_run.MAIN, "extra_03-02"))
    frame([config.BENIGN] * 200 + ["DDoS attacks-LOIC-HTTP"] * 30).to_parquet(detector_runs.feature_file("topology", 3, final_run.MAIN, "extra_02-20"))
    frame([config.BENIGN] * 200).to_parquet(detector_runs.feature_file("topology", 3, final_run.MAIN, "calib"))
    adopted_depth = detector.DEPTH
    tree_depths, module_depths = [], []
    original_fit = detector.fit_detector

    def recording_fit(*args, **kwargs):
        model = original_fit(*args, **kwargs)
        tree_depths.append(model.get_params()["depth"])
        module_depths.append(detector.DEPTH)
        return model
    monkeypatch.setattr(detector, "fit_detector", recording_fit)
    scores = detector_depths.depth_scores(3, "all")
    assert tree_depths == [3, 3]
    assert module_depths == [adopted_depth, adopted_depth] and detector.DEPTH == adopted_depth
    assert set(scores) == {"mean_detection", "mean_benign_false_alarms"}


def test_same_rows_rejects_frames_of_different_length():
    import pandas as pd
    first = pd.DataFrame({"label": ["a", "b"], "time": [1, 2]})
    assert detector_runs.same_rows(first, first.copy())
    assert not detector_runs.same_rows(first, first.iloc[:1])
    assert not detector_runs.same_rows(first, first.assign(time=[1, 3]))


def test_mislabelled_udp_flows_of_02_20_are_dropped():
    import pandas as pd

    frame = pd.DataFrame({"label": ["DDoS attacks-LOIC-HTTP", "DDoS attacks-LOIC-HTTP", config.BENIGN], "Protocol": [6.0, 17.0, 17.0]})
    kept, dropped = extra_days.drop_mislabelled_rows(frame, "02-20")
    assert dropped == 1 and list(kept["Protocol"]) == [6.0, 17.0]
    assert extra_days.drop_mislabelled_rows(frame, "03-02")[1] == 0


def test_detector_score_quality_ranks_each_family_against_benign_only():
    labels = np.array([config.BENIGN] * 4 + ["Bot"] * 2 + ["XSS"] * 2)
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.9, 0.8, 0.35, 0.05])
    quality = held_out_days.score_quality(labels, scores)
    assert quality["Bot"]["roc_auc"] == 1.0 and quality["Bot"]["pr_auc"] == 1.0
    assert quality["XSS"]["roc_auc"] == 0.375  # one XSS score above 3 of 4 Benign, one below all: (3 + 0) / 8


def test_extra_day_inputs_keep_every_column_that_the_encoding_reads(tmp_path, monkeypatch):
    columns = ["Flow IAT Std", "Idle Max", "Fwd Pkt Len Mean"]
    rng = np.random.default_rng(0)
    full = pd.DataFrame({column: rng.integers(0, 3, 50).astype("float32") for column in {*columns, *vae.mask_source_columns(columns), "Unused Column"}})
    full = full.assign(label="Benign", time=pd.Timestamp("2018-03-01 10:00"))
    monkeypatch.setattr(extra_days, "OUTPUT_DIR", tmp_path)
    full.to_parquet(tmp_path / "03-01.parquet")
    subset = detector_runs.extra_day_vae_inputs("03-01", columns)
    assert "Unused Column" not in subset.columns
    assert np.array_equal(vae.undefined_entries(subset, columns), vae.undefined_entries(full, columns))
    pd.testing.assert_frame_equal(subset, full[subset.columns])
