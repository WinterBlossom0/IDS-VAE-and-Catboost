import json
import os
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import confusion_matrix

from idsvae import config
from idsvae.framework import classify, features

FEATURES = ["f0", "f1", "f2", "f3"]
CLASS_MEANS = {"Benign": 0.0, "AttackA": 3.0, "AttackB": -3.0, "AttackC": 1.5}
FIXED_PARAMS = {"iterations": 30, "depth": 4, "learning_rate": 0.1, "l2_leaf_reg": 3.0}


def synthetic_split(rng, counts, day, in_window_benign=0):
    parts = []
    for label, count in counts.items():
        part = pd.DataFrame(rng.normal(CLASS_MEANS[label], 1.0, size=(count, len(FEATURES))).astype("float32"), columns=FEATURES)
        part["label"] = label
        parts.append(part)
    frame = pd.concat(parts, ignore_index=True)
    frame["day"] = day
    frame["time"] = pd.Timestamp(f"2018-{day} 09:00:00") + pd.to_timedelta(rng.permutation(len(frame)), unit="s")
    frame["in_benign_burst"] = False
    benign_positions = np.flatnonzero(frame["label"].to_numpy() == config.BENIGN)
    frame.loc[benign_positions[:in_window_benign], "in_benign_burst"] = True
    return frame


def synthetic_splits(rng):
    fit_counts = {"Benign": 800, "AttackA": 200, "AttackB": 100}
    return {
        "fit_train": pd.concat([synthetic_split(rng, fit_counts, "02-14"), synthetic_split(rng, fit_counts, "02-15")], ignore_index=True),
        "fit_holdout": synthetic_split(rng, {"Benign": 200, "AttackA": 50, "AttackB": 40}, "02-15"),
        "calib": synthetic_split(rng, {"Benign": 4000, "AttackA": 50}, "02-23"),
        "test": synthetic_split(rng, {"Benign": 1000, "AttackC": 300}, "02-21", in_window_benign=400),
    }


def write_ranking(ranking_dir, left_out_class=""):
    ranking_dir.mkdir(parents=True)
    ranking = {
        "settings": asdict(features.SelectionSettings(left_out_class=left_out_class)),
        "ranking": [{"feature": feature} for feature in FEATURES],
        "selected_count": len(FEATURES),
    }
    (ranking_dir / "feature_ranking.json").write_text(json.dumps(ranking))


def run_with_fixed_params(root, settings, threads=-1):
    """Run Stage 3 on fresh synthetic splits; return (output_dir, result, splits, rows of the final fit)."""
    processed_dir = root / "processed"
    processed_dir.mkdir()
    splits = synthetic_splits(np.random.default_rng(0))
    for split, frame in splits.items():
        frame.to_parquet(processed_dir / f"{split}.parquet", index=False)
    write_ranking(root / "stage1", settings.left_out_class)
    final_fit_rows = []
    original_fit_model = classify.fit_model

    def recording_fit_model(model_name, params, seed, inputs, targets, threads=-1):
        final_fit_rows.append(len(targets))
        return original_fit_model(model_name, params, seed, inputs, targets, threads)

    # Fixed parameters replace the Optuna search, so that the test is fast.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(classify, "tune", lambda *arguments, **keywords: (FIXED_PARAMS, 0.0))
        patch.setattr(classify, "fit_model", recording_fit_model)
        output_dir, result = classify.run_classifier(settings, processed_dir, root / "stage1", root / "stage2", root / "stage3", threads=threads)
    return output_dir, result, splits, final_fit_rows


@pytest.fixture(scope="module")
def main_run(tmp_path_factory):
    settings = classify.ClassifySettings(model="catboost", view="raw", feature_count=len(FEATURES), seed=1)
    return run_with_fixed_params(tmp_path_factory.mktemp("classify_main"), settings)


@pytest.fixture(scope="module")
def leave_one_out_run(tmp_path_factory):
    settings = classify.ClassifySettings(model="catboost", view="raw", feature_count=len(FEATURES), left_out_class="AttackB", seed=1)
    return run_with_fixed_params(tmp_path_factory.mktemp("classify_loo"), settings)


def test_time_block_folds_do_not_overlap_in_time():
    rng = np.random.default_rng(0)
    frame = pd.concat([synthetic_split(rng, {"Benign": 101, "AttackA": 37, "AttackB": 5}, day) for day in ("02-14", "02-15")], ignore_index=True)
    frame = frame.sample(frac=1.0, random_state=0).reset_index(drop=True)
    frame["fold"] = classify.time_block_folds(frame, 3)
    for _, group in frame.groupby(["day", "label"]):
        assert sorted(group["fold"].unique()) == [0, 1, 2]
        sizes = group["fold"].value_counts()
        assert sizes.max() - sizes.min() <= 1
        windows = group.groupby("fold")["time"].agg(["min", "max"]).sort_index()
        assert (windows["max"].to_numpy()[:-1] < windows["min"].to_numpy()[1:]).all()


def test_tuning_rows_sample_each_group_and_keep_rare_classes():
    rng = np.random.default_rng(0)
    frame = pd.concat([synthetic_split(rng, {"Benign": 3000, "AttackA": 1000, "AttackB": 40}, day) for day in ("02-14", "02-15")], ignore_index=True)
    rows = classify.tuning_rows(frame, fraction=0.1, min_class_rows=100, seed=0)
    counts = frame.iloc[rows].groupby(["day", "label"]).size()
    assert counts[("02-14", "Benign")] == 300
    assert counts[("02-15", "AttackA")] == 100
    assert counts[("02-14", "AttackB")] == 40
    assert len(np.unique(rows)) == len(rows)


def test_leave_one_out_run_name_differs_from_the_main_run_name():
    main = classify.ClassifySettings(view="raw", feature_count=44)
    left_out = classify.ClassifySettings(view="raw", feature_count=44, left_out_class="DoS attacks-GoldenEye")
    assert left_out.run_name() == main.run_name() + "_loo-dos-attacks-goldeneye"


def test_leave_one_out_trains_without_the_class(leave_one_out_run):
    output_dir, result, splits, final_fit_rows = leave_one_out_run
    fit_train = splits["fit_train"]
    assert final_fit_rows == [int((fit_train["label"] != "AttackB").sum())]
    assert result["classes"] == ["AttackA", "Benign"]
    holdout = splits["fit_holdout"]
    assert result["seen_multiclass"]["rows"] == int((holdout["label"] != "AttackB").sum())
    assert output_dir.name.endswith("_loo-attackb")


def test_leave_one_out_run_rejects_a_ranking_that_saw_the_class(tmp_path):
    write_ranking(tmp_path / "stage1", left_out_class="")
    settings = classify.ClassifySettings(view="raw", feature_count=len(FEATURES), left_out_class="AttackB")
    with pytest.raises(ValueError, match="leaves out"):
        classify.run_classifier(settings, tmp_path / "processed", tmp_path / "stage1", tmp_path / "stage2", tmp_path / "stage3")


def read_predictions(output_dir, result, part):
    assert result["prediction_files"][part] == f"predictions_{part}.parquet"
    return pd.read_parquet(output_dir / result["prediction_files"][part])


def assert_prediction_columns(predictions, split, classes):
    probability_columns = [f"p_{label}" for label in classes]
    assert predictions.columns.tolist()[:4] == ["label", "day", "time", "in_benign_burst"]
    assert {"predicted", "attack_score", *probability_columns} <= set(predictions.columns)
    assert all(predictions[column].dtype == np.float32 for column in ["attack_score", *probability_columns])
    probabilities = predictions[probability_columns].to_numpy(dtype="float64")
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(predictions["attack_score"], 1.0 - predictions[f"p_{config.BENIGN}"], atol=1e-6)
    assert (np.asarray(classes)[probabilities.argmax(axis=1)] == predictions["predicted"].to_numpy()).all()
    for column in ("label", "day", "time", "in_benign_burst"):
        assert (predictions[column].to_numpy() == split[column].to_numpy()).all()


def test_prediction_files_reproduce_the_reported_confusion_matrices(main_run):
    output_dir, result, splits, _ = main_run
    classes = result["classes"]
    assert sorted(result["prediction_files"]) == ["calib", "fit_holdout", "test"]
    holdout = read_predictions(output_dir, result, "fit_holdout")
    assert_prediction_columns(holdout, splits["fit_holdout"], classes)
    matrix = confusion_matrix(holdout["label"], holdout["predicted"], labels=classes)
    assert matrix.tolist() == result["seen_multiclass"]["confusion_matrix"]["matrix"]

    assert_prediction_columns(read_predictions(output_dir, result, "calib"), splits["calib"], classes)
    assert_prediction_columns(read_predictions(output_dir, result, "test"), splits["test"], classes)


def test_leave_one_out_prediction_files_reproduce_the_reported_confusion_matrices(leave_one_out_run):
    output_dir, result, splits, _ = leave_one_out_run
    classes = result["classes"]
    assert sorted(result["prediction_files"]) == ["fit_holdout", "fit_train_left_out"]
    holdout = read_predictions(output_dir, result, "fit_holdout")
    # The hold-out file keeps the left-out rows, which have no probability column because the classifier never saw their class.
    assert_prediction_columns(holdout, splits["fit_holdout"], classes)
    assert "p_AttackB" not in holdout.columns
    seen = holdout[holdout["label"] != "AttackB"]
    assert confusion_matrix(seen["label"], seen["predicted"], labels=classes).tolist() == result["seen_multiclass"]["confusion_matrix"]["matrix"]

    train_left_out = read_predictions(output_dir, result, "fit_train_left_out")
    fit_train = splits["fit_train"]
    assert train_left_out["split_row"].tolist() == np.flatnonzero(fit_train["label"].to_numpy() == "AttackB").tolist()
    assert_prediction_columns(train_left_out.drop(columns="split_row"), fit_train.iloc[train_left_out["split_row"]], classes)


def test_run_records_training_cost_and_training_rows(main_run, leave_one_out_run):
    for _, result, _, final_fit_rows in (main_run, leave_one_out_run):
        training = result["training_seconds"]
        assert set(training) == {"tuning", "final_fit", "threads"}
        assert training["tuning"] >= 0.0 and training["final_fit"] > 0.0
        assert training["threads"] == os.cpu_count()
        assert result["train_rows"] == final_fit_rows[-1]


def test_catboost_thread_count_does_not_change_predictions():
    rng = np.random.default_rng(0)
    frame = synthetic_split(rng, {"Benign": 400, "AttackA": 200, "AttackB": 100}, "02-14")
    inputs = frame[FEATURES].to_numpy(dtype="float32")
    targets = frame["label"].astype("category").cat.codes.to_numpy()
    all_cores = classify.fit_model("catboost", FIXED_PARAMS, 3, inputs, targets, threads=-1)
    two_threads = classify.fit_model("catboost", FIXED_PARAMS, 3, inputs, targets, threads=2)
    reference = classify.predict_probabilities("catboost", all_cores, inputs, -1)
    assert np.array_equal(reference, classify.predict_probabilities("catboost", two_threads, inputs, 2))
    assert np.array_equal(reference, classify.predict_probabilities("catboost", two_threads, inputs, 1))


def test_run_with_two_catboost_threads_records_them_and_writes_the_same_predictions(tmp_path, main_run):
    main_dir, main_result, _, _ = main_run
    output_dir, result, _, _ = run_with_fixed_params(tmp_path, classify.ClassifySettings(**main_result["settings"]), threads=2)
    assert result["training_seconds"]["threads"] == 2
    for file_name in main_result["prediction_files"].values():
        pd.testing.assert_frame_equal(pd.read_parquet(output_dir / file_name), pd.read_parquet(main_dir / file_name), check_exact=True)
    assert result["seen_multiclass"] == main_result["seen_multiclass"]


def test_lightgbm_and_xgboost_use_and_record_the_fixed_thread_count_whatever_the_thread_argument():
    for model_name in ("lightgbm", "xgboost"):
        assert classify.training_thread_count(model_name, 2) == classify.BOOSTING_ABLATION_THREADS
        assert classify.build_model(model_name, {}, 0, threads=2).get_params()["n_jobs"] == classify.BOOSTING_ABLATION_THREADS
    assert classify.training_thread_count("catboost", -1) == os.cpu_count()
    assert classify.training_thread_count("catboost", 2) == 2


VAE_SETTINGS = {"latent_dim": 5, "left_out_class": "", "feature_set": "main"}


@pytest.mark.parametrize(("vae_left_out", "run_left_out"), [("", "AttackB"), ("AttackB", ""), ("AttackA", "AttackB")])
def test_vae_view_run_rejects_a_vae_with_another_left_out_class(vae_left_out, run_left_out):
    settings = classify.ClassifySettings(view="latent_aug", vae_run="vae", left_out_class=run_left_out)
    with pytest.raises(ValueError, match="leaves out"):
        classify.require_inputs_without_left_out_class(settings, None, {**VAE_SETTINGS, "left_out_class": vae_left_out})


def test_vae_view_run_accepts_a_vae_with_the_same_left_out_class():
    classify.require_inputs_without_left_out_class(classify.ClassifySettings(view="latent_aug", vae_run="vae"), None, VAE_SETTINGS)
    expected = features.feature_set_name(features.SelectionSettings(left_out_class="AttackB"))
    settings = classify.ClassifySettings(view="latent_aug", vae_run="vae", left_out_class="AttackB")
    classify.require_inputs_without_left_out_class(settings, None, {**VAE_SETTINGS, "left_out_class": "AttackB", "feature_set": expected})


def test_leave_one_out_vae_view_run_rejects_a_vae_of_another_feature_set():
    settings = classify.ClassifySettings(view="latent_aug", vae_run="vae", left_out_class="AttackB")
    with pytest.raises(ValueError, match="feature_set"):
        classify.require_inputs_without_left_out_class(settings, None, {**VAE_SETTINGS, "left_out_class": "AttackB"})


def test_concat_view_joins_raw_features_and_vae_outputs_row_by_row(tmp_path):
    raw = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0], "label": ["Benign", "SSH"], "day": ["02-14", "02-14"], "time": [10, 11], "in_benign_burst": [False, False]})
    encoded = pd.DataFrame({"z0": [0.5, 0.6], "recon_nll": [1.0, 2.0], "kl": [0.1, 0.2], "label": ["Benign", "SSH"], "day": ["02-14", "02-14"], "time": [10, 11],
                            "in_benign_burst": [False, False]})
    (tmp_path / "run").mkdir()
    raw.to_parquet(tmp_path / "fit_train.parquet")
    encoded.to_parquet(tmp_path / "run" / "fit_train.parquet")
    settings = SimpleNamespace(view="concat", vae_run="run")
    columns = ["f1", "f2", "z0", "recon_nll", "kl"]
    frame = classify.load_view(settings, "fit_train", tmp_path, tmp_path, columns)
    assert frame[columns].to_numpy().tolist() == [[1.0, 3.0, 0.5, 1.0, 0.1], [2.0, 4.0, 0.6, 2.0, 0.2]]
    encoded.iloc[::-1].reset_index(drop=True).to_parquet(tmp_path / "run" / "fit_train.parquet")
    with pytest.raises(ValueError):
        classify.load_view(settings, "fit_train", tmp_path, tmp_path, columns)
