import pytest
import torch

from idsvae.experiments import final_run, inner_selection
from idsvae.framework import correction, vae


@pytest.fixture(autouse=True)
def chosen_settings(monkeypatch):
    """The settings that inner_selection.py writes; the tests do not depend on its output files."""
    monkeypatch.setattr(inner_selection, "chosen_topology_weight", lambda: 1.0)
    monkeypatch.setattr(inner_selection, "chosen_rule_settings", lambda: correction.RuleSettings())


def test_names_match_the_directories_that_the_steps_write(monkeypatch):
    monkeypatch.setattr(final_run, "feature_count", lambda variant: variant.feature_count or 44)
    assert final_run.vae_settings("topology", 3).run_name() == "student_t_k44_L5_s3_topo1"
    assert final_run.vae_settings("base", 3).run_name() == "student_t_k44_L5_s3"
    assert final_run.classifier_settings("topology", 3).run_name() == "catboost_latent_aug_student_t_k44_L5_s3_topo1_s3"
    assert final_run.classifier_settings("raw", 4).run_name() == "catboost_raw_raw_k44_s4"
    assert final_run.stage2_root("topology") != final_run.stage2_root("base")


def test_topological_loss_is_zero_for_a_scaled_copy_and_positive_for_a_collapsed_latent():
    torch.manual_seed(0)
    inputs = torch.randn(64, 8)
    assert float(vae.topological_loss(inputs, 3.0 * inputs)) < 1e-6
    collapsed = torch.zeros(64, 8)
    collapsed[:, 0] = torch.arange(64.0)
    assert float(vae.topological_loss(inputs, collapsed)) > 0.01



def write_stage_outputs(tmp_path, monkeypatch, rng):
    """A tiny final-run layout: one VAE arm with encoded splits and the Stage 3 prediction files of its classifier."""
    import json

    import numpy as np
    import pandas as pd

    monkeypatch.setattr(final_run, "FINAL_DIR", tmp_path)
    monkeypatch.setattr(final_run, "feature_count", lambda variant: variant.feature_count or 44)
    settings = final_run.classifier_settings("topology", 3)
    encoded_dir = final_run.stage2_root("topology") / settings.vae_run
    run_dir = final_run.stage3_root("topology") / settings.run_name()
    encoded_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    (encoded_dir / "summary.json").write_text(json.dumps({"settings": {"latent_dim": 5}}))
    classes = ["Benign", "DoS attacks-Slowloris"]

    def split(day, rows, attack_rows, novel_attack):
        times = np.datetime64(f"2018-02-{day}T10:00") + np.sort(rng.integers(0, 3600, rows)).astype("timedelta64[s]")
        labels = np.array(["Benign"] * rows, dtype=object)
        labels[attack_rows] = "DoS attacks-Slowloris"
        frame = pd.DataFrame(rng.normal(size=(rows, 5)), columns=[f"z{index}" for index in range(5)])
        frame["recon_nll"] = rng.normal(size=rows)
        frame["kl"] = 1.0
        if novel_attack:
            # The attack flows of this split sit in one latent cell, arrive in one minute and are novel.
            frame.loc[attack_rows, [f"z{index}" for index in range(5)]] = 5.0
            frame.loc[attack_rows, "recon_nll"] = 50.0
            times[attack_rows] = np.datetime64(f"2018-02-{day}T10:30") + np.arange(attack_rows.stop - attack_rows.start).astype("timedelta64[s]")
        frame["label"], frame["time"], frame["in_benign_burst"] = labels, times, False
        return frame

    frames = {
        "fit_train": split(14, 400, slice(0, 0), False),
        "fit_holdout": split(15, 400, slice(100, 160), True),
        "calib": split(23, 300, slice(0, 0), False),
        "test": split(21, 300, slice(0, 0), False),
    }
    prediction_files = {}
    for part, frame in frames.items():
        frame.to_parquet(encoded_dir / f"{part}.parquet", index=False)
        if part == "fit_train":
            continue
        predicted = np.array(["Benign"] * len(frame), dtype=object)
        if part == "fit_holdout":
            # Stage 3 catches the first 20 attack flows (the earliest seconds) and misses the other 40.
            predicted[100:120] = "DoS attacks-Slowloris"
        prediction_files[part] = f"predictions_{part}.parquet"
        pd.DataFrame({"label": frame["label"], "time": frame["time"], "predicted": predicted}).to_parquet(run_dir / prediction_files[part], index=False)
    (run_dir / "result.json").write_text(json.dumps({"classes": classes, "prediction_files": prediction_files}))
    return frames


def test_rule_step_end_to_end_catches_novel_burst_flows_and_writes_results(tmp_path, monkeypatch):
    import json

    import numpy as np

    write_stage_outputs(tmp_path, monkeypatch, np.random.default_rng(0))
    final_run.run_rule("topology", 3)
    output = json.loads((tmp_path / "rule" / "topology_s3.json").read_text())
    base = output["holdout"]["base"]["per_class"]["DoS attacks-Slowloris"]["recall"]
    rule = output["holdout"]["rule"]["per_class"]["DoS attacks-Slowloris"]["recall"]
    assert base == 20 / 60 and rule > base
    assert output["holdout_overrides"]["changed_by_true_class"].get("Benign", 0) == 0
    assert set(output) >= {"calib_in_sample", "test", "novelty_threshold", "burst_threshold"}


def test_rule_step_rejects_prediction_files_that_do_not_match_the_encoded_rows(tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd
    import pytest

    write_stage_outputs(tmp_path, monkeypatch, np.random.default_rng(1))
    settings = final_run.classifier_settings("topology", 3)
    path = final_run.stage3_root("topology") / settings.run_name() / "predictions_calib.parquet"
    shifted = pd.read_parquet(path)
    shifted["time"] = shifted["time"] + np.timedelta64(1, "s")
    shifted.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="do not match"):
        final_run.run_rule("topology", 3)


def test_leave_one_out_rule_counts_the_left_out_training_rows_as_test_rows(tmp_path, monkeypatch):
    import json

    import numpy as np
    import pandas as pd

    from idsvae.framework import classify

    rng = np.random.default_rng(2)
    monkeypatch.setattr(final_run, "FINAL_DIR", tmp_path)
    monkeypatch.setattr(final_run, "feature_count", lambda variant: variant.feature_count or 44)
    # The calibration-day predictions come from the model file; a constant Benign model stands in for it.
    monkeypatch.setattr(classify, "load_probability_function", lambda model, path, threads: (lambda inputs: np.tile([1.0, 0.0], (len(inputs), 1))))
    variant = final_run.Variant(0, "SSH-Bruteforce")
    settings = final_run.classifier_settings("topology", 3, variant)
    encoded_dir = final_run.stage2_root("topology") / settings.vae_run
    run_dir = final_run.stage3_root("topology") / settings.run_name()
    encoded_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    (encoded_dir / "summary.json").write_text(json.dumps({"settings": {"latent_dim": 5}}))
    latent = [f"z{index}" for index in range(5)]

    def split(day, rows):
        frame = pd.DataFrame(rng.normal(size=(rows, 5)), columns=latent)
        frame["recon_nll"], frame["kl"], frame["in_benign_burst"] = rng.normal(size=rows), 1.0, False
        frame["label"] = "Benign"
        frame["time"] = np.datetime64(f"2018-02-{day}T10:00") + np.sort(rng.integers(0, 3600, rows)).astype("timedelta64[s]")
        return frame
    train, holdout, calib = split(14, 400), split(15, 200), split(23, 300)
    left_out = np.arange(200, 260)
    # The left-out SSH flows are novel, share one latent cell and arrive within one minute.
    train.loc[left_out, "label"] = "SSH-Bruteforce"
    train.loc[left_out, latent] = 5.0
    train.loc[left_out, "recon_nll"] = 50.0
    times = train["time"].to_numpy().copy()
    times[left_out] = np.datetime64("2018-02-14T10:30") + np.arange(60).astype("timedelta64[s]")
    train["time"] = times
    for part, frame in (("fit_train", train), ("fit_holdout", holdout), ("calib", calib)):
        frame.to_parquet(encoded_dir / f"{part}.parquet", index=False)
    predicted = np.array(["Benign"] * 60, dtype=object)
    predicted[:20] = "DoS attacks-Slowloris"  # Stage 3 maps the first 20 SSH flows to the other attack class
    pd.DataFrame({"label": "SSH-Bruteforce", "time": train["time"].to_numpy()[left_out], "split_row": np.arange(200, 260), "predicted": predicted}).to_parquet(run_dir / "predictions_fit_train_left_out.parquet", index=False)
    pd.DataFrame({"label": holdout["label"], "time": holdout["time"], "predicted": "Benign"}).to_parquet(run_dir / "predictions_fit_holdout.parquet", index=False)
    (run_dir / "result.json").write_text(json.dumps({"classes": ["Benign", "DoS attacks-Slowloris"], "columns": [*latent, "recon_nll", "kl"],
                                                     "prediction_files": {"fit_holdout": "predictions_fit_holdout.parquet", "fit_train_left_out": "predictions_fit_train_left_out.parquet"}}))
    final_run.run_rule("topology", 3, variant)
    output = json.loads(final_run.rule_output("topology", 3, variant).read_text())
    assert output["base"]["left_out_detection"] == 20 / 60
    assert output["rule"]["left_out_detection"] > output["base"]["left_out_detection"]
    assert output["rule"]["holdout_benign_fpr"] == 0.0
