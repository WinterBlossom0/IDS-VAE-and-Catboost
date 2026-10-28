import numpy as np
import onnxruntime
import pytest
import torch
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import QuantileTransformer

from idsvae.experiments import baselines, deep_baselines
from idsvae.timing import baseline_graphs


def run_graph(model, values):
    session = onnxruntime.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return session.run(None, {baseline_graphs.RAW_NAME: values})[0]


def mixed_columns(rows, seed):
    """A column with many repeated values (repeated quantiles) and a heavy-tailed column."""
    generator = np.random.default_rng(seed)
    return np.column_stack([generator.integers(0, 4, rows).astype(float), generator.lognormal(0.0, 2.0, rows) * np.sign(generator.normal(size=rows))])


def test_normal_quantile_matches_scipy_inside_the_clip_range():
    probabilities = np.concatenate([np.geomspace(1e-7, 0.5, 400), 1.0 - np.geomspace(1e-7, 0.5, 400)])
    values = baseline_graphs.normal_quantile(torch.as_tensor(probabilities)).numpy()
    assert np.allclose(values, stats.norm.ppf(probabilities), rtol=2e-9, atol=1e-12)


def test_searchsorted_right_matches_numpy_per_column():
    sorted_values = np.sort(np.random.default_rng(0).integers(0, 5, (7, 3)).astype(float), axis=0)
    values = np.array([[-1.0, 2.0, 5.0], [0.0, 4.0, 2.0], [3.0, 1.0, 9.0]])
    counts = baseline_graphs.searchsorted_right(torch.as_tensor(sorted_values.reshape(-1)), 3, 7, torch.as_tensor(values)).numpy()
    expected = np.column_stack([np.searchsorted(sorted_values[:, column], values[:, column], side="right") for column in range(3)])
    assert (counts == expected).all()


def test_quantile_graph_matches_sklearn_including_repeated_quantiles_and_out_of_range_values():
    train = mixed_columns(3_000, seed=1)
    transformer = QuantileTransformer(n_quantiles=100, output_distribution="normal", subsample=10**9, random_state=0).fit(train)
    values = np.vstack([mixed_columns(500, seed=2), transformer.quantiles_[::7], [[-5.0, -1e6], [9.0, 1e6]]])
    graph = baseline_graphs.QuantileNormalGraph(transformer)
    expected = transformer.transform(values)
    assert np.abs(graph(torch.as_tensor(values)).numpy() - expected).max() < 1e-6
    exported = baseline_graphs.torch_graph(graph, torch.zeros(2, 2, dtype=torch.float64), "normal")
    assert np.abs(run_graph(exported, values) - expected).max() < 1e-6


def test_scaling_graphs_match_their_numpy_forms():
    train = mixed_columns(2_000, seed=3)
    values = mixed_columns(300, seed=4)
    standardizer = baselines.Standardizer(train)
    assert np.allclose(baseline_graphs.StandardizerGraph(standardizer)(torch.as_tensor(values)).numpy(), standardizer.transform(values), rtol=0, atol=1e-12)
    scaler = deep_baselines.CnnScaler(train, seed=0)
    assert np.allclose(baseline_graphs.CnnScalerGraph(scaler)(torch.as_tensor(values)).numpy(), scaler.transform(values), rtol=0, atol=1e-12)
    minimum, scale = deep_baselines.log_min_max_parameters(train)
    windows = np.stack([values, values[::-1]], axis=1)
    expected = deep_baselines.apply_log_min_max(windows.copy(), minimum, scale)
    assert np.allclose(baseline_graphs.LogMinMaxGraph(minimum, scale)(torch.as_tensor(windows)).numpy(), expected, rtol=0, atol=1e-12)


def test_mlp_and_forest_graphs_reproduce_sklearn():
    train = mixed_columns(1_500, seed=5)
    labels = (train[:, 0] + (train[:, 1] > 0)).astype(int)
    standardizer = baselines.Standardizer(train)
    scaled = standardizer.transform(train)
    mlp = MLPClassifier(hidden_layer_sizes=(6,), max_iter=200, random_state=0).fit(scaled, labels)
    mlp_graph = baseline_graphs.method_graph("mlp", baselines.DeployedShallow(standardizer, mlp), 2)
    assert np.abs(run_graph(mlp_graph, train[:200]) - mlp.predict_proba(scaled[:200])).max() < 1e-12
    forest = RandomForestClassifier(n_estimators=5, random_state=0).fit(scaled, labels)
    forest_graph = baseline_graphs.method_graph("random_forest", baselines.DeployedShallow(standardizer, forest), 2)
    assert (run_graph(forest_graph, train).argmax(1) == forest.predict_proba(scaled).argmax(1)).all()


def test_traceable_packages_restore_the_patched_functions():
    import rtdl_revisiting_models
    forward, column_stack = rtdl_revisiting_models.MultiheadAttention.forward, torch.column_stack
    with pytest.raises(RuntimeError), baseline_graphs.traceable_packages():
        assert torch.column_stack([torch.ones(2, 1), torch.zeros(2, 2)]).shape == (2, 3)
        raise RuntimeError("leave the block with an error")
    assert rtdl_revisiting_models.MultiheadAttention.forward is forward and torch.column_stack is column_stack
