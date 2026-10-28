import numpy as np
import pandas as pd
import pytest
import torch
from scipy.stats import norm
from scipy.stats import t as student_t

from idsvae.framework import onnx_graph, vae


def decoder_outputs(rows=5, features=23):
    rng = np.random.default_rng(0)
    inputs = rng.standard_t(3, size=(rows, features))
    location = rng.normal(size=(rows, features))
    log_scale = rng.uniform(-3.0, 1.0, size=(rows, features))
    degrees_of_freedom = rng.uniform(1.1, 30.0, size=(rows, features))
    return inputs, location, log_scale, degrees_of_freedom


def test_student_t_terms_equal_scipy_per_feature_negative_log_density():
    inputs, location, log_scale, degrees_of_freedom = decoder_outputs()
    expected = -student_t(df=degrees_of_freedom, loc=location, scale=np.exp(log_scale)).logpdf(inputs)
    actual = vae.student_t_nll_terms(*(torch.tensor(values) for values in (inputs, location, log_scale, degrees_of_freedom))).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-10)


def test_gaussian_terms_equal_scipy_per_feature_negative_log_density():
    inputs, location, log_scale, _ = decoder_outputs()
    expected = -norm(loc=location, scale=np.exp(log_scale)).logpdf(inputs)
    actual = vae.gaussian_nll_terms(*(torch.tensor(values) for values in (inputs, location, log_scale))).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-10)


def test_soft_clip_keeps_the_log_scale_above_the_floor():
    raw = torch.tensor([-1e4, -50.0, -6.0, 0.0, 30.0], dtype=torch.float64)
    clipped = vae.soft_clip_log_scale(raw)
    assert vae.LOG_SCALE_MIN == -6.0
    assert (clipped >= vae.LOG_SCALE_MIN).all()
    # Far above the floor the clip is the identity.
    assert torch.isclose(clipped[-1], raw[-1])


def test_degrees_of_freedom_stay_at_or_above_one():
    degrees_of_freedom = vae.degrees_of_freedom_from(torch.tensor([-1e4, -5.0, 0.0, 5.0], dtype=torch.float64))
    assert (degrees_of_freedom >= 1.0).all()
    assert (degrees_of_freedom[1:] > 1.0).all()


@pytest.mark.parametrize("likelihood", ["gaussian", "student_t"])
def test_reconstruction_nll_sums_the_weighted_terms_of_the_decoder(likelihood):
    torch.manual_seed(0)
    model = vae.Vae(3, 4, 2, likelihood)
    inputs = torch.randn(4, 3)
    latent = torch.randn(4, 2)
    weights = torch.tensor([1.0, 0.0, 1.0])
    with torch.no_grad():
        location, log_scale, degrees_of_freedom = model.decode(latent)
        if likelihood == "gaussian":
            assert degrees_of_freedom is None
            terms = vae.gaussian_nll_terms(inputs, location, log_scale)
        else:
            terms = vae.student_t_nll_terms(inputs, location, log_scale, degrees_of_freedom)
        assert torch.allclose(vae.reconstruction_nll(model, inputs, latent, weights), terms[:, 0] + terms[:, 2])


def test_gaussian_model_has_no_degrees_of_freedom_head():
    assert not hasattr(vae.Vae(3, 4, 2, "gaussian"), "degrees_of_freedom_head")
    with pytest.raises(ValueError, match="unknown likelihood"):
        vae.Vae(3, 4, 2, "laplace")


def test_kl_is_zero_at_the_prior_and_positive_elsewhere():
    zeros = torch.zeros(3, 5)
    assert torch.allclose(vae.kl_per_dimension(zeros, zeros), zeros)
    assert (vae.kl_per_dimension(torch.ones(3, 5), torch.full((3, 5), -1.0)) > 0).all()


def test_gaussian_kl_is_zero_for_equal_posteriors_and_matches_the_closed_form():
    mean, log_variance = torch.randn(4, 3), torch.randn(4, 3)
    assert torch.allclose(vae.gaussian_kl(mean, log_variance, mean, log_variance), torch.zeros(4), atol=1e-6)
    # Against the prior N(0, I), the general formula equals the KL of the ELBO.
    zeros = torch.zeros(4, 3)
    assert torch.allclose(vae.gaussian_kl(mean, log_variance, zeros, zeros), vae.kl_per_dimension(mean, log_variance).sum(dim=1))


def test_product_of_experts_ignores_missing_experts_and_falls_back_to_the_prior():
    torch.manual_seed(0)
    expert_mean, expert_log_variance = torch.randn(2, 3, 2), torch.randn(2, 3, 2)
    observed = torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    mean, log_variance = vae.product_of_experts(expert_mean, expert_log_variance, observed)
    precision = 1.0 + torch.exp(-expert_log_variance[0, 0]) + torch.exp(-expert_log_variance[0, 2])
    expected_mean = (torch.exp(-expert_log_variance[0, 0]) * expert_mean[0, 0] + torch.exp(-expert_log_variance[0, 2]) * expert_mean[0, 2]) / precision
    assert torch.allclose(mean[0], expected_mean, atol=1e-6)
    assert torch.allclose(log_variance[0], -torch.log(precision), atol=1e-6)
    # A row without observed experts gets the N(0, I) prior.
    assert torch.allclose(mean[1], torch.zeros(2))
    assert torch.allclose(log_variance[1], torch.zeros(2))


def test_a_missing_entry_does_not_change_the_posterior():
    torch.manual_seed(0)
    model = vae.Vae(3, 4, 2, "student_t")
    inputs = torch.randn(2, 3)
    observed = torch.tensor([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    changed = inputs.clone()
    changed[:, 1] = 100.0
    with torch.no_grad():
        assert torch.allclose(model.encode(inputs, observed)[0], model.encode(changed, observed)[0])


def flow_frame():
    return pd.DataFrame({
        "Tot Fwd Pkts": [1, 2, 3, 5],
        "Tot Bwd Pkts": [0, 1, 2, 3],
        "Bwd IAT Max": [0.0, 0.0, 7.0, 9.0],
        "Fwd IAT Mean": [0.0, 4.0, 5.0, 6.0],
        "Flow IAT Std": [0.0, 0.0, 2.0, 3.0],
        "Bwd Pkt Len Max": [0.0, 0.0, 30.0, 40.0],
        "Bwd Pkt Len Std": [0.0, 0.0, 1.0, 2.0],
        "Idle Max": [0.0, 5e6, 0.0, 6e6],
        "Init Bwd Win Byts": [-1.0, 0.0, 512.0, -1.0],
        "Fwd Header Len": [20.0, 40.0, 60.0, 80.0],
    })


def test_undefined_entries_follow_the_cicflowmeter_rules():
    frame = flow_frame()
    columns = ["Bwd IAT Max", "Fwd IAT Mean", "Flow IAT Std", "Bwd Pkt Len Max", "Bwd Pkt Len Std", "Idle Max", "Init Bwd Win Byts", "Fwd Header Len"]
    missing = pd.DataFrame(vae.undefined_entries(frame, columns), columns=columns)
    assert missing["Bwd IAT Max"].tolist() == [True, True, False, False]
    assert missing["Fwd IAT Mean"].tolist() == [True, False, False, False]
    assert missing["Flow IAT Std"].tolist() == [True, False, False, False]
    assert missing["Bwd Pkt Len Max"].tolist() == [True, False, False, False]
    assert missing["Bwd Pkt Len Std"].tolist() == [True, True, False, False]
    assert missing["Idle Max"].tolist() == [True, False, True, False]
    assert missing["Init Bwd Win Byts"].tolist() == [True, False, False, True]
    assert not missing["Fwd Header Len"].any()


def test_error_features_are_those_without_an_undefined_rule():
    columns = ["Bwd IAT Max", "Fwd Header Len", "Pkt Len Max", "Init Fwd Win Byts", "Pkt Len Var"]
    assert vae.error_feature_mask(columns).tolist() == [False, True, True, False, False]
    with pytest.raises(ValueError, match="no selected feature"):
        vae.error_feature_mask(["Bwd IAT Max"])


def test_mask_source_columns_add_the_counts_and_the_period_columns():
    assert vae.mask_source_columns(["Idle Mean", "Pkt Len Max"]) == sorted({"Idle Mean", "Pkt Len Max", "Idle Max", "Tot Fwd Pkts", "Tot Bwd Pkts"})


def test_standardizer_uses_only_the_defined_entries():
    values = np.array([[1.0, 10.0], [3.0, 0.0], [5.0, 30.0]])
    missing = np.array([[False, False], [False, True], [False, False]])
    standardizer = vae.Standardizer.fit(values, missing)
    np.testing.assert_allclose(standardizer.mean, [3.0, 20.0])
    np.testing.assert_allclose(standardizer.std, [np.std([1.0, 3.0, 5.0]), 10.0])


def test_subset_masks_stay_inside_the_observed_entries():
    generator = torch.Generator().manual_seed(0)
    observed = (torch.rand(64, 6, generator=generator) > 0.3).float()
    subset = vae.subset_mask(observed, generator)
    assert ((subset == 1) <= (observed == 1)).all()
    assert subset.sum() < observed.sum()


def tiny_setup(feature_count=3, likelihood="student_t"):
    settings = vae.VaeSettings(likelihood=likelihood, latent_dim=2, hidden_dim=4, feature_count=feature_count, batch_size=8, max_epochs=2)
    torch.manual_seed(0)
    return vae.Vae(feature_count, settings.hidden_dim, settings.latent_dim, likelihood), settings


def train_tiny(model, settings, train_inputs, val_inputs):
    return vae.train(model, train_inputs, torch.ones_like(train_inputs), val_inputs, torch.ones_like(val_inputs), settings, torch.Generator().manual_seed(0), torch.Generator().manual_seed(1))


def test_training_loss_is_finite_and_has_gradients():
    model, settings = tiny_setup()
    inputs = torch.randn(16, 3)
    observed = torch.tensor([[1.0, 0.0, 1.0]] * 8 + [[1.0, 1.0, 1.0]] * 8)
    loss = vae.training_loss(model, inputs, observed, settings, torch.Generator().manual_seed(0))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.expert_weight.grad is not None


def test_non_finite_validation_loss_raises_with_epoch_and_likelihood():
    model, settings = tiny_setup()
    with pytest.raises(RuntimeError, match=r"epoch 1.*student_t.*nan"):
        train_tiny(model, settings, torch.randn(16, 3), torch.full((4, 3), float("nan")))


def test_train_history_has_loss_and_learning_rate():
    model, settings = tiny_setup()
    history, _, _ = train_tiny(model, settings, torch.randn(16, 3), torch.randn(8, 3))
    assert set(history[0]) == {"epoch", "val_loss", "learning_rate"}


def test_train_reports_the_epoch_of_the_lowest_validation_loss_and_early_stop():
    model, settings = tiny_setup()
    settings.max_epochs, settings.patience = 6, 2
    history, best_epoch, stopped_early = train_tiny(model, settings, torch.randn(16, 3), torch.randn(8, 3))
    losses = [entry["val_loss"] for entry in history]
    assert best_epoch == losses.index(min(losses)) + 1
    assert stopped_early == (len(history) < settings.max_epochs)


@pytest.mark.parametrize("best_loss", [10.0, -10.0])
def test_improvement_below_relative_tolerance_does_not_reset_patience(best_loss):
    # Each later loss stays inside the margin 2e-4 * |best_loss| below the best loss.
    stopping = vae.EarlyStopping(patience=3, relative_tolerance=2e-4)
    step = 1e-4 * abs(best_loss)
    assert stopping.update(best_loss) == (True, False)
    assert stopping.update(best_loss - step) == (False, False)
    assert stopping.update(best_loss - 1.5 * step) == (False, False)
    assert stopping.update(best_loss - 1.9 * step) == (False, True)
    assert stopping.best_loss == best_loss


@pytest.mark.parametrize("best_loss", [10.0, -10.0])
def test_improvement_above_relative_tolerance_resets_patience(best_loss):
    stopping = vae.EarlyStopping(patience=2, relative_tolerance=2e-4)
    stopping.update(best_loss)
    assert stopping.update(best_loss + 1.0) == (False, False)
    assert stopping.update(best_loss - 3e-4 * abs(best_loss)) == (True, False)
    assert stopping.stale_epochs == 0


def test_scheduler_uses_the_stop_rule_for_a_negative_best_loss():
    optimizer = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
    scheduler = vae.RelativePlateauScheduler(optimizer, mode="min", factor=0.5, patience=0, threshold=2e-4, threshold_mode="rel")
    scheduler.step(-10.0)
    # A slightly worse loss: the plain torch rule (-10.0 * (1 - 2e-4) = -9.998) counts it as better.
    scheduler.step(-9.999)
    assert scheduler.best == -10.0
    assert optimizer.param_groups[0]["lr"] == 0.5


def fit_frame(rows=24):
    rng = np.random.default_rng(0)
    labels = np.array(["Benign", "SSH-Bruteforce", "DoS attacks-GoldenEye"] * (rows // 3))
    # Large values of the left-out class move the standardizer mean visibly if its rows leak into the fit.
    shift = np.where(labels == "SSH-Bruteforce", 1e6, 0.0)
    return pd.DataFrame({
        "Pkt Len Max": rng.uniform(1, 100, size=rows) + shift,
        "Bwd IAT Max": rng.uniform(1, 100, size=rows),
        "Tot Fwd Pkts": rng.integers(1, 5, size=rows),
        "Tot Bwd Pkts": rng.integers(0, 4, size=rows),
        "label": labels, "day": "d", "time": np.arange(rows), "in_benign_burst": False,
    })


def patch_inputs(monkeypatch, frame):
    monkeypatch.setattr(vae.splits, "load_split", lambda split, processed_dir=None, columns=None: frame)
    monkeypatch.setattr(vae.features, "load_ranking", lambda ranking_dir: {"selected_count": 2, "ranking": [{"feature": "Pkt Len Max"}, {"feature": "Bwd IAT Max"}]})


@pytest.mark.parametrize("likelihood", ["gaussian", "student_t"])
def test_run_vae_writes_the_training_and_decoder_summary(tmp_path, monkeypatch, likelihood):
    frame = fit_frame()
    patch_inputs(monkeypatch, frame)
    settings = vae.VaeSettings(likelihood=likelihood, latent_dim=2, hidden_dim=4, batch_size=8, max_epochs=3)
    output_dir, summary = vae.run_vae(settings, output_root=tmp_path)
    training = summary["training"]
    assert set(training) == {"seconds", "device", "epochs_run", "best_epoch", "stopped_early"}
    assert training["seconds"] > 0
    assert training["device"] == "cpu" or training["device"].startswith("cuda: ")
    losses = [entry["val_loss"] for entry in summary["history"]]
    assert training["epochs_run"] == len(losses)
    assert training["best_epoch"] == losses.index(min(losses)) + 1
    assert training["stopped_early"] == (len(losses) < settings.max_epochs)
    expected_decoder_keys = {"rows", "scale_median", "share_scale_near_floor"}
    if likelihood == "student_t":
        expected_decoder_keys.add("degrees_of_freedom_median")
        assert summary["decoder"]["degrees_of_freedom_median"] > 1.0
    assert set(summary["decoder"]) == expected_decoder_keys
    assert summary["decoder"]["rows"] == len(frame)
    assert summary["decoder"]["scale_median"] >= np.exp(vae.LOG_SCALE_MIN)
    assert 0.0 <= summary["decoder"]["share_scale_near_floor"] <= 1.0
    assert len(summary["latent_mean_variance"]) == settings.latent_dim
    assert 0 <= summary["active_units"] <= settings.latent_dim
    assert summary["train_rows"] == len(frame)
    assert summary["error_features"] == ["Pkt Len Max"]
    assert summary["undefined_share_by_feature"]["Bwd IAT Max"] == pytest.approx((frame["Tot Bwd Pkts"] <= 1).mean())
    assert set(pd.read_parquet(output_dir / "fit_train.parquet").columns) >= {"z0", "z1", "recon_nll", "kl", "label"}


def test_run_vae_excludes_the_left_out_class_from_training_rows_and_standardizer(tmp_path, monkeypatch):
    frame = fit_frame()
    patch_inputs(monkeypatch, frame)
    settings = vae.VaeSettings(latent_dim=2, hidden_dim=4, batch_size=8, max_epochs=1, left_out_class="SSH-Bruteforce")
    output_dir, summary = vae.run_vae(settings, output_root=tmp_path)
    kept = frame[frame["label"] != "SSH-Bruteforce"]
    assert summary["train_rows"] == len(kept)
    assert summary["settings"]["left_out_class"] == "SSH-Bruteforce"
    assert summary["standardizer"]["mean"][0] == pytest.approx(vae.signed_log(kept["Pkt Len Max"].to_numpy()).mean())
    # The encoded fit_train split keeps the left-out rows, because the leave-one-out classifier evaluates them.
    assert (pd.read_parquet(output_dir / "fit_train.parquet")["label"] == "SSH-Bruteforce").sum() == 8


def test_vae_rows_use_all_classes_and_drop_only_the_left_out_class():
    frame = pd.DataFrame({"label": ["Benign", "SSH-Bruteforce", "DoS attacks-GoldenEye", "Benign"]})
    assert vae.vae_rows(frame, vae.VaeSettings()).equals(frame)
    kept = vae.vae_rows(frame, vae.VaeSettings(left_out_class="SSH-Bruteforce"))
    assert kept["label"].tolist() == ["Benign", "DoS attacks-GoldenEye", "Benign"]
    with pytest.raises(ValueError, match="Benign"):
        vae.vae_rows(frame, vae.VaeSettings(left_out_class="Benign"))


@pytest.mark.parametrize("likelihood", ["gaussian", "student_t"])
def test_deployed_onnx_graph_gives_the_features_of_encode_frame(likelihood, tmp_path):
    model, settings = tiny_setup(feature_count=2, likelihood=likelihood)
    model.eval()
    columns = ["Pkt Len Max", "Bwd IAT Max"]
    standardizer = vae.Standardizer(np.array([0.1, -0.2]), np.array([1.5, 0.5]))
    frame = fit_frame(6)
    encoded, _ = vae.encode_frame(model, frame, columns, standardizer, settings, "cpu")
    source_columns = vae.mask_source_columns(columns)
    graph = onnx_graph.VaeFeatureGraph(model, standardizer, columns, source_columns)
    run = onnx_graph.graph_function(onnx_graph.session(onnx_graph.vae_onnx(graph, tmp_path / "vae.onnx"), 1), [onnx_graph.FEATURES_NAME])
    np.testing.assert_allclose(run(frame[source_columns].to_numpy("float64"))[0], encoded[["z0", "z1", "recon_nll", "kl"]].to_numpy(), rtol=1e-5, atol=1e-5)


def test_lanczos_log_gamma_matches_torch_above_one_half():
    values = torch.linspace(0.51, 80.0, 500)
    torch.testing.assert_close(onnx_graph.lanczos_log_gamma(values), torch.lgamma(values), rtol=1e-6, atol=1e-5)


def test_encode_frame_copies_in_benign_burst_only_when_present():
    model, settings = tiny_setup(feature_count=2)
    standardizer = vae.Standardizer(np.zeros(2), np.ones(2))
    columns = ["Pkt Len Max", "Bwd IAT Max"]
    frame = fit_frame(6)
    frame["in_benign_burst"] = [False, True] * 3
    encoded, _ = vae.encode_frame(model, frame, columns, standardizer, settings, "cpu")
    assert encoded["in_benign_burst"].tolist() == [False, True] * 3
    encoded, _ = vae.encode_frame(model, frame.drop(columns="in_benign_burst"), columns, standardizer, settings, "cpu")
    assert "in_benign_burst" not in encoded.columns


def test_latent_mean_variance_uses_only_the_vae_training_rows():
    settings = vae.VaeSettings(latent_dim=2, feature_count=3, left_out_class="SSH-Bruteforce")
    encoded = pd.DataFrame({
        "z0": [0.0, 1.0, 0.0, 1.0, 50.0],
        "z1": [0.5, 0.5, 0.5, 0.5, -50.0],
        "label": ["Benign", "Benign", "Benign", "Benign", "SSH-Bruteforce"],
    })
    variance = vae.latent_mean_variance(encoded, settings)
    np.testing.assert_allclose(variance, [0.25, 0.0])
    assert int((variance > vae.ACTIVE_UNIT_VARIANCE_THRESHOLD).sum()) == 1


def test_merged_vae_and_catboost_graph_equals_the_two_steps(tmp_path):
    from catboost import CatBoostClassifier
    model, _ = tiny_setup(feature_count=2, likelihood="student_t")
    model.eval()
    columns = ["Pkt Len Max", "Bwd IAT Max"]
    standardizer = vae.Standardizer(np.array([0.1, -0.2]), np.array([1.5, 0.5]))
    frame = fit_frame(42)
    source_columns = vae.mask_source_columns(columns)
    graph = onnx_graph.VaeFeatureGraph(model, standardizer, columns, source_columns)
    vae_model = onnx_graph.vae_onnx(graph, tmp_path / "vae.onnx")
    raw = frame[source_columns].to_numpy("float64")
    features = onnx_graph.graph_function(onnx_graph.session(vae_model, 1), [onnx_graph.FEATURES_NAME])(raw)[0]
    labels = np.arange(len(features)) % 3
    classifier = CatBoostClassifier(iterations=20, depth=3, verbose=False, random_seed=0, allow_writing_files=False).fit(features, labels)
    classifier.save_model(str(tmp_path / "model.bin"))
    merged = onnx_graph.deployed_onnx(onnx_graph.catboost_onnx(tmp_path / "model.bin", tmp_path / "classifier.onnx"), vae_model)
    merged_features, merged_probabilities = onnx_graph.graph_function(onnx_graph.session(merged, 1), [onnx_graph.FEATURES_NAME, onnx_graph.PROBABILITIES_NAME])(raw)
    np.testing.assert_allclose(merged_features, features, rtol=1e-6)
    np.testing.assert_allclose(merged_probabilities, classifier.predict_proba(features), atol=1e-5)


def reference_tree(distances):
    """Union-find over the stably sorted upper triangle, as in src/topology.py of Moor et al. (topological-autoencoders)."""
    import numpy as np

    matrix = distances.numpy()
    rows, columns = np.triu_indices_from(matrix, 1)
    parent = list(range(len(matrix)))

    def find(node):
        while parent[node] != node:
            node = parent[node]
        return node
    tree = torch.zeros(distances.shape, dtype=torch.bool)
    for edge in np.argsort(matrix[rows, columns], kind="stable"):
        first, second = find(rows[edge]), find(columns[edge])
        if first != second:
            parent[first] = second
            tree[rows[edge], columns[edge]] = True
    return tree


def test_spanning_tree_masks_match_the_reference_union_find_with_ties_and_zero_distances():
    generator = torch.Generator().manual_seed(0)
    points = torch.randn(64, 6, generator=generator)
    # Repeated rows give zero distances and equal weights; rounded points give more ties.
    points = torch.cat([points, points[:16], (points[:32] * 2).round() / 2])
    stacked = torch.stack([vae.distance_matrix(points), vae.distance_matrix(points[:, :2]), vae.distance_matrix((points[:, :1] * 2).round())])
    masks = vae.spanning_tree_masks(stacked)
    for graph in range(len(stacked)):
        assert torch.equal(masks[graph], reference_tree(stacked[graph]))
        assert int(masks[graph].sum()) == len(points) - 1


def test_topological_loss_has_finite_gradients_with_repeated_rows():
    generator = torch.Generator().manual_seed(1)
    inputs = torch.randn(40, 6, generator=generator)
    inputs = torch.cat([inputs, inputs[:20]])
    latent = (inputs[:, :2] * 0.5).clone().requires_grad_(True)
    vae.topological_loss(inputs, latent).backward()
    assert torch.isfinite(latent.grad).all()
