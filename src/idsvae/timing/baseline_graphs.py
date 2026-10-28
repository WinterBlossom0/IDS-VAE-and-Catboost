"""ONNX Runtime deployment form of the re-run methods and of the LightGBM and XGBoost classifiers of Stage 3: one graph from
the raw flow features to the class probabilities, preprocessing included, as the framework runs (onnx_graph.py). Tree
ensembles come from their converters (skl2onnx, onnxmltools); networks, the MLP and the preprocessing from torch.onnx.
check_method compares each graph with the evaluated model on the hold-out (CLI: --check)."""
import argparse
import contextlib
import json
import math
import tempfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import onnx
import rtdl_revisiting_models
import torch
import torch.nn.functional as F
from onnx import compose, helper, version_converter
from onnxmltools import convert_lightgbm, convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType as MlFloatTensorType
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import DoubleTensorType
from xgboost import XGBClassifier

from idsvae import files, metrics
from idsvae.data import splits
from idsvae.experiments import baselines, deep_baselines, final_run
from idsvae.framework import onnx_graph, vae

RAW_NAME = "raw"
POSITIONS_NAME = "positions"
SCALED_NAME = "scaled"
TARGET_OPSETS = {"": onnx_graph.ONNX_OPSET, "ai.onnx.ml": 3}
# Highest default-domain opset of the onnxmltools converters; their graphs are raised to onnx_graph.ONNX_OPSET afterwards.
ONNXMLTOOLS_OPSET = 15
CHECK_FILE = final_run.FINAL_DIR / "timing" / "onnx_checks.json"
CHECK_CHUNK = deep_baselines.PREDICT_BATCH
STREAM_CHECK_FLOWS = 50_000


# ---------------------------------------------------------------- preprocessing as torch modules (float64, as numpy)

class StandardizerGraph(torch.nn.Module):
    """baselines.Standardizer: signed log, then (x - mean) / std."""

    def __init__(self, standardizer):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(standardizer.mean, dtype=torch.float64))
        self.register_buffer("std", torch.as_tensor(standardizer.std, dtype=torch.float64))

    def forward(self, raw):
        return (vae.signed_log(raw) - self.mean) / self.std


def searchsorted_right(sorted_flat, feature_count, quantile_count, values):
    """Per element, the count of the column's sorted values <= value, by bisection (ONNX has no searchsorted).

    sorted_flat holds a (quantile_count, feature_count) array in row-major order; the loop unrolls into a fixed number of
    gathers, enough for quantile_count + 1 outcomes.
    """
    feature_offsets = torch.arange(feature_count)
    low = torch.zeros_like(values, dtype=torch.int64)
    high = torch.full_like(low, quantile_count)
    for _ in range(int(np.ceil(np.log2(quantile_count + 1)))):
        middle = torch.div(low + high, 2, rounding_mode="floor")
        active = low < high
        probe = sorted_flat[torch.clamp(middle, max=quantile_count - 1) * feature_count + feature_offsets]
        at_or_below = probe <= values
        low = torch.where(active & at_or_below, middle + 1, low)
        high = torch.where(active & ~at_or_below, middle, high)
    return low


def interpolate(sorted_flat, references, feature_count, quantile_count, values):
    """np.interp(values, quantiles[:, f], references) for every column f, with the end values outside the range."""
    count = searchsorted_right(sorted_flat, feature_count, quantile_count, values)
    left = torch.clamp(count - 1, 0, quantile_count - 2)
    feature_offsets = torch.arange(feature_count)
    x_left = sorted_flat[left * feature_count + feature_offsets]
    x_right = sorted_flat[(left + 1) * feature_count + feature_offsets]
    y_left, y_right = references[left], references[left + 1]
    inside = y_left + (values - x_left) * (y_right - y_left) / (x_right - x_left)
    return torch.where(count == 0, references[0], torch.where(count >= quantile_count, references[-1], inside))


# Acklam's rational approximation of the standard normal quantile (relative error below 1.2e-9), as graph operations;
# P. J. Acklam, "An algorithm for computing the inverse normal cumulative distribution function" (2003).
ACKLAM_A = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02, 1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
ACKLAM_B = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02, 6.680131188771972e01, -1.328068155288572e01)
ACKLAM_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00, -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
ACKLAM_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00)
ACKLAM_LOW = 0.02425


def polynomial(coefficients, values):
    result = torch.full_like(values, coefficients[0])
    for coefficient in coefficients[1:]:
        result = result * values + coefficient
    return result


def normal_quantile(probabilities):
    """Standard normal quantile of probabilities strictly inside (0, 1)."""
    centred = probabilities - 0.5
    squared = centred * centred
    central = centred * polynomial(ACKLAM_A, squared) / (polynomial(ACKLAM_B, squared) * squared + 1.0)
    lower_tail = torch.sqrt(-2.0 * torch.log(probabilities))
    lower = polynomial(ACKLAM_C, lower_tail) / (polynomial(ACKLAM_D, lower_tail) * lower_tail + 1.0)
    upper_tail = torch.sqrt(-2.0 * torch.log(1.0 - probabilities))
    upper = -polynomial(ACKLAM_C, upper_tail) / (polynomial(ACKLAM_D, upper_tail) * upper_tail + 1.0)
    return torch.where(probabilities < ACKLAM_LOW, lower, torch.where(probabilities > 1.0 - ACKLAM_LOW, upper, central))


class QuantileNormalGraph(torch.nn.Module):
    """sklearn QuantileTransformer(output_distribution="normal").transform (sklearn.preprocessing._data._transform_col)."""

    BOUNDS_THRESHOLD = 1e-7

    def __init__(self, transformer):
        super().__init__()
        quantiles = np.asarray(transformer.quantiles_, dtype=np.float64)
        references = np.asarray(transformer.references_, dtype=np.float64)
        self.quantile_count, self.feature_count = quantiles.shape
        self.register_buffer("forward_flat", torch.as_tensor(quantiles.reshape(-1)))
        self.register_buffer("forward_references", torch.as_tensor(references))
        # The second interpolation of sklearn runs on the negated, reversed quantiles (equal for repeated quantiles).
        self.register_buffer("backward_flat", torch.as_tensor((-quantiles[::-1]).reshape(-1).copy()))
        self.register_buffer("backward_references", torch.as_tensor(-references[::-1].copy()))
        self.register_buffer("lower_bound", torch.as_tensor(quantiles[0]))
        self.register_buffer("upper_bound", torch.as_tensor(quantiles[-1]))
        # Clipping before the quantile function equals sklearn's clipping after it, as the function is increasing.
        self.clip = self.BOUNDS_THRESHOLD - np.spacing(1)

    def forward(self, raw):
        shape = (self.forward_flat, self.forward_references, self.feature_count, self.quantile_count)
        forward_value = interpolate(*shape, raw)
        backward_value = interpolate(self.backward_flat, self.backward_references, self.feature_count, self.quantile_count, -raw)
        uniform = 0.5 * (forward_value - backward_value)
        uniform = torch.where(raw + self.BOUNDS_THRESHOLD > self.upper_bound, torch.ones_like(uniform), uniform)
        uniform = torch.where(raw - self.BOUNDS_THRESHOLD < self.lower_bound, torch.zeros_like(uniform), uniform)
        return normal_quantile(torch.clamp(uniform, self.clip, 1.0 - self.clip))


class CnnScalerGraph(torch.nn.Module):
    """deep_baselines.CnnScaler: signed log of the heavy-tailed columns, RobustScaler, clip."""

    def __init__(self, scaler):
        super().__init__()
        self.register_buffer("heavy_tailed", torch.as_tensor(scaler.heavy_tailed))
        self.register_buffer("center", torch.as_tensor(scaler.scaler.center_, dtype=torch.float64))
        self.register_buffer("scale", torch.as_tensor(scaler.scaler.scale_, dtype=torch.float64))

    def forward(self, raw):
        values = torch.where(self.heavy_tailed, vae.signed_log(raw), raw)
        return torch.clamp((values - self.center) / self.scale, -deep_baselines.CNN_CLIP, deep_baselines.CNN_CLIP)


class LogMinMaxGraph(torch.nn.Module):
    """deep_baselines.apply_log_min_max on windows (rows, window, features)."""

    def __init__(self, minimum, scale):
        super().__init__()
        self.register_buffer("minimum", torch.as_tensor(minimum, dtype=torch.float64))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float64))

    def forward(self, raw):
        return torch.log1p(torch.clamp(raw - self.minimum, min=0.0)) / self.scale


class KangScoreGraph(torch.nn.Module):
    """KangScorer.score_of_scaled(KangScorer.scaled(raw)) with the float32 steps of reconstruction_rmse."""

    def __init__(self, scorer):
        super().__init__()
        self.register_buffer("minimum", torch.as_tensor(scorer.minimum, dtype=torch.float64))
        self.register_buffer("safe_range", torch.as_tensor(scorer.safe_range, dtype=torch.float64))
        self.columns = [list(map(int, columns)) for columns, _, _, _ in scorer.groups]
        self.autoencoders = torch.nn.ModuleList(model for _, model, _, _ in scorer.groups)
        # numpy keeps float32 when a float32 array meets a Python float, so the constants are float32 here too.
        self.lows = [float(np.float32(low)) for _, _, low, _ in scorer.groups]
        self.spreads = [float(np.float32(spread)) for _, _, _, spread in scorer.groups]

    def forward(self, raw):
        scaled = (raw - self.minimum) / self.safe_range
        scores = []
        for columns, autoencoder, low, spread in zip(self.columns, self.autoencoders, self.lows, self.spreads):
            inputs = scaled[:, columns].to(torch.float32)
            rmse = torch.sqrt(((autoencoder(inputs) - inputs) ** 2).mean(dim=1))
            scores.append((rmse - low) / spread)
        return torch.stack(scores, dim=0).mean(dim=0)[:, None]


class MlpGraph(torch.nn.Module):
    """sklearn MLPClassifier.predict_proba with ReLU hidden layers and a softmax output, in float64 as sklearn."""

    def __init__(self, standardizer, model):
        super().__init__()
        if model.activation != "relu" or model.out_activation_ != "softmax":
            raise ValueError(f"unsupported MLP activations {model.activation}, {model.out_activation_}")
        self.scale = StandardizerGraph(standardizer)
        self.weights = torch.nn.ParameterList(torch.nn.Parameter(torch.as_tensor(weight, dtype=torch.float64), requires_grad=False) for weight in model.coefs_)
        self.biases = torch.nn.ParameterList(torch.nn.Parameter(torch.as_tensor(bias, dtype=torch.float64), requires_grad=False) for bias in model.intercepts_)

    def forward(self, raw):
        values = self.scale(raw)
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            values = values @ weight + bias
            if index < len(self.weights) - 1:
                values = torch.relu(values)
        return torch.softmax(values, dim=1)


class NetworkGraph(torch.nn.Module):
    """DeployedNetwork.predict_proba: float64 preprocessing, float32 network, softmax mean over the ensemble members."""

    def __init__(self, preprocessing, network):
        super().__init__()
        self.preprocessing, self.network = preprocessing, network.eval()

    def forward(self, raw):
        return torch.softmax(self.network(self.preprocessing(raw).to(torch.float32)), dim=-1).mean(dim=1)


class FlowTransformerGraph(torch.nn.Module):
    """FlowTransformer on a stream: raw context flows (rows, features) and window positions (windows, WINDOW_SIZE) ->
    probabilities; each flow is scaled once, as in DeployedFlowTransformer.stream."""

    def __init__(self, minimum, scale, network):
        super().__init__()
        self.scale, self.network = LogMinMaxGraph(minimum, scale), network.eval()

    def forward(self, context, positions):
        windows = self.scale(context).to(torch.float32)[positions]
        return torch.softmax(self.network(windows), dim=-1).mean(dim=1)


class TabNetGraph(torch.nn.Module):
    """TabNetClassifier.predict_proba: the network's first output, softmax over classes."""

    def __init__(self, preprocessing, network):
        super().__init__()
        self.preprocessing, self.network = preprocessing, network.eval()

    def forward(self, raw):
        return torch.softmax(self.network(self.preprocessing(raw).to(torch.float32))[0], dim=1)


# ---------------------------------------------------------------- graph assembly

def attention_forward(self, x_q, x_kv):
    """rtdl_revisiting_models 0.0.2 MultiheadAttention.forward without key compression (FTTransformer defaults), with the
    batch size read as q.shape[0]: len(q) becomes a constant in a traced graph."""
    if self.key_compression is not None:
        raise ValueError("key compression is not part of this copy of the attention forward")
    q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
    batch_size = q.shape[0]
    d_head_key = k.shape[-1] // self._n_heads
    d_head_value = v.shape[-1] // self._n_heads
    n_q_tokens = q.shape[1]
    q = self._reshape(q)
    k = self._reshape(k)
    attention_probs = F.softmax(q @ k.transpose(1, 2) / math.sqrt(d_head_key), dim=-1)
    if self.dropout is not None:
        attention_probs = self.dropout(attention_probs)
    x = attention_probs @ self._reshape(v)
    x = x.reshape(batch_size, self._n_heads, n_q_tokens, d_head_value).transpose(1, 2).reshape(batch_size, n_q_tokens, self._n_heads * d_head_value)
    if self.W_out is not None:
        x = self.W_out(x)
    return x


@contextlib.contextmanager
def traceable_packages():
    """During an export only: the attention forward above, and torch.column_stack (called by tabm 0.0.3 on 2-D tensors, and
    absent from the TorchScript exporter) as the equal torch.cat on the second axis."""
    attention = rtdl_revisiting_models.MultiheadAttention
    original_forward, original_column_stack = attention.forward, torch.column_stack
    attention.forward = attention_forward
    torch.column_stack = lambda tensors: torch.cat(list(tensors), dim=1)
    try:
        yield
    finally:
        attention.forward, torch.column_stack = original_forward, original_column_stack


def torch_graph(module, example, output_name):
    """ONNX model of module(example) with the input named RAW_NAME and a dynamic first axis; float64 example for raw values."""
    module.eval()
    with torch.no_grad(), traceable_packages(), tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "graph.onnx"
        torch.onnx.export(module, (example,), str(path), input_names=[RAW_NAME], output_names=[output_name],
                          dynamic_axes={RAW_NAME: {0: "rows"}, output_name: {0: "rows"}}, opset_version=onnx_graph.ONNX_OPSET, dynamo=False)
        return onnx.load(str(path))


def probabilities_only(model):
    """The converter's model with its probability tensor as the only output, named onnx_graph.PROBABILITIES_NAME."""
    probability_output = next(output.name for output in model.graph.output if "prob" in output.name)
    if probability_output != onnx_graph.PROBABILITIES_NAME:
        model.graph.node.append(helper.make_node("Identity", [probability_output], [onnx_graph.PROBABILITIES_NAME]))
    del model.graph.output[:]
    model.graph.output.append(helper.make_tensor_value_info(onnx_graph.PROBABILITIES_NAME, onnx.TensorProto.FLOAT, [None, None]))
    if not any(opset.domain in ("", "ai.onnx") for opset in model.opset_import):
        model.opset_import.append(helper.make_opsetid("", onnx_graph.ONNX_OPSET))
    return model


def tree_graph(model, feature_count):
    """Probability graph of a fitted tree ensemble. XGBoost and sklearn round inputs to float32 as their own predict does;
    native LightGBM compares float64 values, so a value next to a split can differ (onnx_checks.json counts the labels)."""
    if isinstance(model, XGBClassifier):
        converted = convert_xgboost(model, initial_types=[(SCALED_NAME, MlFloatTensorType([None, feature_count]))], target_opset=ONNXMLTOOLS_OPSET)
    elif isinstance(model, lgb.Booster) or hasattr(model, "booster_"):
        converted = convert_lightgbm(model, initial_types=[(SCALED_NAME, MlFloatTensorType([None, feature_count]))], zipmap=False,
                                     target_opset=ONNXMLTOOLS_OPSET)
    else:
        # sklearn compares float32 inputs with float64 thresholds; a double graph on float32-rounded inputs does the same.
        converted = convert_sklearn(model, initial_types=[(SCALED_NAME, DoubleTensorType([None, feature_count]))], options={id(model): {"zipmap": False}},
                                    target_opset=TARGET_OPSETS)
    if any(opset.domain in ("", "ai.onnx") and opset.version < onnx_graph.ONNX_OPSET for opset in converted.opset_import):
        converted = version_converter.convert_version(converted, onnx_graph.ONNX_OPSET)
    return probabilities_only(converted)


def chained(first, second, first_output):
    """first, then second on first's output (named first_output); second's names get a prefix to stay unique."""
    second.ir_version = first.ir_version
    second = compose.add_prefix(second, prefix="classifier_")
    return compose.merge_models(first, second, io_map=[(first_output, second.graph.input[0].name)], outputs=[f"classifier_{onnx_graph.PROBABILITIES_NAME}"])


def renamed_output(model, name):
    model.graph.node.append(helper.make_node("Identity", [model.graph.output[0].name], [name]))
    output_type = model.graph.output[0].type
    del model.graph.output[:]
    model.graph.output.append(helper.make_value_info(name, output_type))
    return model


class Float32(torch.nn.Module):
    """The module's output rounded to float32, as the tree libraries read their inputs; as_double keeps that value in float64."""

    def __init__(self, module, as_double=False):
        super().__init__()
        self.module, self.as_double = module, as_double

    def forward(self, raw):
        rounded = self.module(raw).to(torch.float32)
        return rounded.to(torch.float64) if self.as_double else rounded


def method_graph(method, deployed, feature_count):
    """ONNX model raw float64 features (rows, features) -> class probabilities of a re-run method; FlowTransformer takes raw
    windows (rows, WINDOW_SIZE, features)."""
    raw_example = torch.zeros(2, feature_count, dtype=torch.float64)
    if method == "mlp":
        return torch_graph(MlpGraph(deployed.standardizer, deployed.model), raw_example, onnx_graph.PROBABILITIES_NAME)
    if method in ("random_forest", "xgboost", "lightgbm"):
        scaling = torch_graph(Float32(StandardizerGraph(deployed.standardizer), as_double=method == "random_forest"), raw_example, SCALED_NAME)
        return renamed_output(chained(scaling, tree_graph(deployed.model, feature_count), SCALED_NAME), onnx_graph.PROBABILITIES_NAME)
    if method == "kang_idae_xg":
        scoring = torch_graph(KangScoreGraph(deployed.scorer), raw_example, SCALED_NAME)
        return renamed_output(chained(scoring, tree_graph(deployed.model, 1), SCALED_NAME), onnx_graph.PROBABILITIES_NAME)
    if method == "flow_transformer":
        return flow_transformer_graph(deployed, feature_count)
    if method == "tabnet":
        return torch_graph(TabNetGraph(StandardizerGraph(deployed.transform), deployed.model.network), raw_example, onnx_graph.PROBABILITIES_NAME)
    if method == "cnn1d":
        return torch_graph(NetworkGraph(CnnScalerGraph(deployed.transform), deployed.network), raw_example, onnx_graph.PROBABILITIES_NAME)
    module = NetworkGraph(QuantileNormalGraph(deployed.transform.transformer), deployed.network)
    return torch_graph(module, raw_example, onnx_graph.PROBABILITIES_NAME)


def flow_transformer_graph(deployed, feature_count):
    """ONNX model of FlowTransformerGraph with the inputs RAW_NAME (context flows) and POSITIONS_NAME."""
    module = FlowTransformerGraph(deployed.minimum, deployed.scale, deployed.network).eval()
    window = deep_baselines.WINDOW_SIZE
    examples = (torch.zeros(window + 1, feature_count, dtype=torch.float64), torch.stack([torch.arange(window), torch.arange(1, window + 1)]))
    with torch.no_grad(), tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "graph.onnx"
        torch.onnx.export(module, examples, str(path), input_names=[RAW_NAME, POSITIONS_NAME], output_names=[onnx_graph.PROBABILITIES_NAME],
                          dynamic_axes={RAW_NAME: {0: "flows"}, POSITIONS_NAME: {0: "windows"}, onnx_graph.PROBABILITIES_NAME: {0: "windows"}},
                          opset_version=onnx_graph.ONNX_OPSET, dynamo=False)
        return onnx.load(str(path))


def stage3_classifier_graph(model_name, model_path):
    """Probability graph of a Stage 3 LightGBM or XGBoost classifier on the VAE features (float32)."""
    if model_name == "lightgbm":
        booster = lgb.Booster(model_file=str(model_path))
        return tree_graph(booster, booster.num_feature())
    model = XGBClassifier()
    model.load_model(str(model_path))
    return tree_graph(model, model.get_booster().num_features())


# ---------------------------------------------------------------- deployment objects for the timing

class OnnxMethod:
    """A re-run method as one ONNX Runtime graph: raw features -> class probabilities."""

    def __init__(self, model, threads):
        self.run = onnx_graph.graph_function(onnx_graph.session(model, threads), [onnx_graph.PROBABILITIES_NAME])

    def predict_proba(self, values):
        return self.run(values)[0]


class OnnxFlowTransformer:
    """FlowTransformer as one ONNX Runtime graph on a stream of raw flows."""

    def __init__(self, model, threads):
        self.session = onnx_graph.session(model, threads)

    def stream(self):
        """values -> probabilities of a stream that keeps the last WINDOW_SIZE - 1 raw flows between calls; the first flows
        repeat the earliest flow, as DeployedFlowTransformer.stream."""
        history = []
        window = deep_baselines.WINDOW_SIZE

        def predict(values):
            values = np.asarray(values, dtype=np.float64)
            context = np.concatenate([*history, values]) if history else np.concatenate([np.repeat(values[:1], window - 1, axis=0), values])
            positions = np.arange(len(values))[:, None] + np.arange(window)[None, :] + (len(context) - len(values) - window + 1)
            history[:] = [context[-(window - 1):]]
            feeds = {RAW_NAME: np.ascontiguousarray(context), POSITIONS_NAME: positions.astype(np.int64)}
            return self.session.run([onnx_graph.PROBABILITIES_NAME], feeds)[0]
        return predict


def onnx_path(method, seed):
    return baselines.model_path(method, seed).with_suffix(".onnx")


def deployed_method(method, seed, threads):
    """(classes, input columns, ONNX deployment object) of a saved re-run method; the graph is written next to the model."""
    classes, columns, deployed = baselines.load_deployed_model(method, seed)
    model = method_graph(method, deployed, len(columns))
    onnx.checker.check_model(model)
    with files.atomic_path(onnx_path(method, seed)) as temporary:
        onnx.save(model, str(temporary))
    if method == "flow_transformer":
        deployment = OnnxFlowTransformer(model, threads)
    else:
        deployment = OnnxMethod(model, threads)
    return classes, columns, deployment, deployed


# ---------------------------------------------------------------- checks against the evaluated models

def chunked(predict, values):
    return np.concatenate([predict(values[start:start + CHECK_CHUNK]) for start in range(0, len(values), CHECK_CHUNK)])


def check_method(method, seed, threads):
    """Hold-out mean per-class F1 of the ONNX graph against the stored result, and agreement with the native model.

    FlowTransformer: native and ONNX streams over the first STREAM_CHECK_FLOWS flows of the time-ordered test day, as the
    native stream reproduces the stored test-day result.
    """
    classes, columns, deployment, deployed = deployed_method(method, seed, threads)
    if method == "flow_transformer":
        test = splits.load_split("test", columns=columns)
        values = test.iloc[np.argsort(test["time"].to_numpy(), kind="stable")][columns].to_numpy("float64")[:STREAM_CHECK_FLOWS]
        onnx_probabilities, native_probabilities = chunked(deployment.stream(), values), chunked(deployed.stream(), values)
        return {"flows": len(values), "label_agreement": float(np.mean(onnx_probabilities.argmax(1) == native_probabilities.argmax(1))),
                "max_probability_difference": float(np.abs(onnx_probabilities - native_probabilities).max())}
    holdout = splits.load_split("fit_holdout", columns=columns)
    values = holdout[columns].to_numpy("float64")
    onnx_probabilities, native_probabilities = chunked(deployment.predict_proba, values), chunked(deployed.predict_proba, values)
    score = metrics.seen_multiclass_report(holdout["label"].to_numpy(), onnx_probabilities, classes)["macro_f1_min_support"]["value"]
    stored = json.loads(baselines.output_path(method, seed).read_text())["macro_f1_min_support"]
    return {"flows": len(values), "holdout_f1": score, "stored_f1": stored, "f1_equal": score == stored,
            "label_agreement": float(np.mean(onnx_probabilities.argmax(1) == native_probabilities.argmax(1))),
            "max_probability_difference": float(np.abs(onnx_probabilities - native_probabilities).max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="*", default=list(baselines.METHODS))
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    report = json.loads(CHECK_FILE.read_text()) if CHECK_FILE.exists() else {}
    for method in args.methods:
        report[method] = check_method(method, baselines.SEEDS[0], args.threads)
        print(method, json.dumps(report[method]), flush=True)
    CHECK_FILE.parent.mkdir(parents=True, exist_ok=True)
    files.write_json_atomic(CHECK_FILE, report)
    # A graph passes when it gives the evaluated model's labels; float rounding may still move a probability at a split tie.
    failed = [method for method in args.methods if report[method]["label_agreement"] < 1.0 or not report[method].get("f1_equal", True)]
    if failed:
        raise SystemExit(f"ONNX graphs that differ from the evaluated model: {failed}")


if __name__ == "__main__":
    main()
