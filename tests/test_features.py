import json

import numpy as np
import pandas as pd
import pytest

from idsvae.framework import features


def noisy_importances(runs, feature_count, noise, seed, subset_fraction=1.0):
    rng = np.random.default_rng(seed)
    truth = np.linspace(feature_count, 1, feature_count)
    values = truth + rng.normal(0, noise, size=(runs, feature_count))
    if subset_fraction < 1.0:
        for run in range(runs):
            dropped = rng.choice(feature_count, int(feature_count * (1 - subset_fraction)), replace=False)
            values[run, dropped] = np.nan
    return values


def test_pairwise_uses_common_features_only():
    first = np.array([5.0, 4.0, 3.0, np.nan, 1.0])
    second = np.array([5.0, 4.0, np.nan, 2.0, 1.0])
    assert features.common_spearman(first, second) == pytest.approx(1.0)
    assert features.common_top_k_jaccard(first, second, size=2) == 1.0


def test_top_set_is_capped_at_common_count():
    first = np.array([3.0, 2.0, np.nan, np.nan])
    second = np.array([3.0, 2.0, 1.0, np.nan])
    assert features.common_top_k_jaccard(first, second, size=10) == 1.0


def test_identical_runs_are_perfectly_stable():
    importances = np.tile(np.arange(10.0), (4, 1))
    report = features.ranking_stability(importances, half=2, size=3)
    assert report["aggregated_halves"]["spearman"] == pytest.approx(1.0)
    assert report["aggregated_halves"]["top_k_jaccard"] == 1.0
    assert report["single_runs"]["mean_spearman"] == pytest.approx(1.0)
    assert report["single_runs"]["mean_top_k_jaccard"] == 1.0


def test_report_compares_equal_runs_per_half():
    subset = noisy_importances(20, 30, noise=2.0, seed=1, subset_fraction=0.7)
    full = noisy_importances(20, 30, noise=2.0, seed=2)
    report = features.stability_report(subset, full, size=8)
    assert report["subset_ranking"]["runs_per_half"] == report["full_feature_ranking"]["runs_per_half"] == 10
    for key in ("subset_ranking", "full_feature_ranking"):
        assert -1.0 <= report[key]["aggregated_halves"]["spearman"] <= 1.0
        assert 0.0 <= report[key]["single_runs"]["mean_top_k_jaccard"] <= 1.0


def test_aggregation_improves_stability_over_single_runs():
    full = noisy_importances(20, 30, noise=6.0, seed=3)
    report = features.ranking_stability(full, half=10, size=8)
    assert report["aggregated_halves"]["spearman"] > report["single_runs"]["mean_spearman"]


def test_baseline_default_matches_subset_runs():
    assert features.SelectionSettings().full_feature_baseline_runs == features.SelectionSettings().runs


def write_fit_train(processed_dir, counts):
    columns = ["f0", "f1", "f2"]
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(sum(counts.values()), len(columns))), columns=columns)
    frame["label"] = np.repeat(list(counts), list(counts.values()))
    processed_dir.mkdir()
    frame.to_parquet(processed_dir / "fit_train.parquet", index=False)
    (processed_dir / "manifest.json").write_text(json.dumps({"feature_columns": columns, "fingerprint_columns": []}))


def test_stage1_never_sees_the_left_out_class(tmp_path, monkeypatch):
    write_fit_train(tmp_path / "processed", {"Benign": 40, "AttackA": 20, "AttackB": 10})
    labels_per_run = []
    threads_per_run = []

    def recording_importances(frame, columns, settings, rng, seed, threads=-1):
        labels_per_run.append(set(frame["label"]))
        threads_per_run.append(threads)
        return np.arange(len(columns), 0, -1, dtype="float64")

    monkeypatch.setattr(features, "mean_abs_shap", recording_importances)
    settings = features.SelectionSettings(runs=2, feature_fraction=1.0, full_feature_baseline_runs=2, left_out_class="AttackA")
    result = features.select_features(settings, tmp_path / "processed", tmp_path / "stage1_loo", threads=3)
    assert labels_per_run == [{"Benign", "AttackB"}] * 4
    assert threads_per_run == [3] * 4
    assert features.load_ranking(tmp_path / "stage1_loo")["settings"]["left_out_class"] == "AttackA"
    assert result["selected_count"] >= 1


@pytest.mark.parametrize("left_out_class", ["Benign", "AttackC"])
def test_left_out_class_must_be_an_attack_class_of_the_fit_data(left_out_class):
    with pytest.raises(ValueError):
        features.left_out_mask(pd.DataFrame({"label": ["Benign", "AttackA"]}), left_out_class)


def test_leave_one_out_ranking_has_its_own_directory_and_tag():
    selection = features.SelectionSettings(left_out_class="DoS attacks-GoldenEye")
    assert features.ranking_dir_for(selection) == features.STAGE1_LEAVE_ONE_OUT_ROOT / "dos-attacks-goldeneye"
    assert features.feature_set_name(selection) == "loo_dos-attacks-goldeneye"
    assert features.ranking_dir_for(features.SelectionSettings()) == features.STAGE1_DIR
    assert features.feature_set_name(features.SelectionSettings(include_fingerprints=True)) == "fingerprints"
    with pytest.raises(ValueError):
        features.SelectionSettings(left_out_class="SSH-Bruteforce", include_fingerprints=True)


def test_shap_importances_do_not_change_with_the_thread_count(tmp_path):
    write_fit_train(tmp_path / "processed", {"Benign": 300, "AttackA": 200, "AttackB": 100})
    frame = pd.read_parquet(tmp_path / "processed" / "fit_train.parquet")
    settings = features.SelectionSettings(rows_per_class=200, shap_rows_per_class=50, iterations=30, depth=4)
    columns = ["f0", "f1", "f2"]
    all_cores = features.mean_abs_shap(frame, columns, settings, np.random.default_rng(0), 0, threads=-1)
    two_threads = features.mean_abs_shap(frame, columns, settings, np.random.default_rng(0), 0, threads=2)
    assert np.array_equal(all_cores, two_threads)


def draw_subsets_before_the_guards(feature_count, settings, rng):
    subset_size = round(settings.feature_fraction * feature_count)
    half = settings.runs // 2
    while True:
        masks = np.zeros((settings.runs, feature_count), dtype=bool)
        for run in range(settings.runs):
            masks[run, rng.choice(feature_count, subset_size, replace=False)] = True
        if masks[:half].any(axis=0).all() and masks[half:].any(axis=0).all():
            return masks


@pytest.mark.parametrize("feature_count", [3, 10, 64, 67])
def test_default_subsets_are_the_same_as_before_the_guards(feature_count):
    settings = features.SelectionSettings()
    expected = draw_subsets_before_the_guards(feature_count, settings, np.random.default_rng(settings.seed))
    assert np.array_equal(features.draw_subsets(feature_count, settings, np.random.default_rng(settings.seed)), expected)


def test_odd_run_count_covers_each_feature_in_the_compared_halves():
    settings = features.SelectionSettings(runs=5, feature_fraction=0.5)
    masks = features.draw_subsets(8, settings, np.random.default_rng(0))
    assert masks[:2].any(axis=0).all() and masks[2:4].any(axis=0).all()


@pytest.mark.parametrize("runs, feature_fraction", [(1, 1.0), (4, 0.1)])
def test_draw_subsets_rejects_settings_without_a_possible_cover(runs, feature_fraction):
    with pytest.raises(ValueError):
        features.draw_subsets(10, features.SelectionSettings(runs=runs, feature_fraction=feature_fraction), np.random.default_rng(0))


def test_draw_subsets_stops_after_the_draw_cap(monkeypatch):
    monkeypatch.setattr(features, "MAX_SUBSET_DRAWS", 3)
    # Two runs of 5 of 10 features cover a half only when they are disjoint, which few draws are.
    with pytest.raises(ValueError, match="no draw"):
        features.draw_subsets(10, features.SelectionSettings(runs=4, feature_fraction=0.5), np.random.default_rng(0))


def test_saved_arrays_are_the_same_as_np_save(tmp_path):
    array = np.random.default_rng(0).normal(size=(4, 3))
    np.save(tmp_path / "direct.npy", array)
    features.save_array_atomic(tmp_path / "atomic.npy", array)
    assert (tmp_path / "atomic.npy").read_bytes() == (tmp_path / "direct.npy").read_bytes()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["atomic.npy", "direct.npy"]
