"""CPU deployment form of a trained pipeline: one ONNX Runtime graph from the raw flow features to the class probabilities.

The VAE arm exports preprocessing, encoder and recon_nll/KL to ONNX and merges the graph with the CatBoost ONNX export,
so that one session call serves a batch. The raw arm uses the CatBoost ONNX export alone.
"""
import math
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch
from catboost import CatBoostClassifier
from onnx import compose, helper

from idsvae import files
from idsvae.framework import vae

ONNX_OPSET = 17
# Lanczos approximation, g = 7, n = 9, coefficients of P. Godfrey (https://en.wikipedia.org/wiki/Lanczos_approximation): ONNX has no lgamma.
LANCZOS_G = 7.0
LANCZOS_COEFFICIENTS = (
    0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
    12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
)
FEATURES_NAME = "vae_features"
PROBABILITIES_NAME = "probabilities"


def lanczos_log_gamma(values):
    """log Gamma(x) for x > 0.5, the range of the Student-t terms (nu > 1 gives arguments above 0.5); float64 inside against cancellation."""
    shifted = values.to(torch.float64) - 1.0
    coefficients = torch.tensor(LANCZOS_COEFFICIENTS[1:], dtype=torch.float64)
    offsets = torch.arange(1, len(LANCZOS_COEFFICIENTS), dtype=torch.float64)
    series = LANCZOS_COEFFICIENTS[0] + (coefficients / (shifted[..., None] + offsets)).sum(dim=-1)
    base = shifted + LANCZOS_G + 0.5
    return (0.5 * math.log(2 * math.pi) + (shifted + 0.5) * torch.log(base) - base + torch.log(series)).to(values.dtype)


def undefined_rule_key(column):
    """Columns with the same key have the same undefined mask: a window column has its own placeholder test."""
    return column if vae.is_window(column) else vae.undefined_rule(column)


class VaeFeatureGraph(torch.nn.Module):
    """Raw source columns (float64) -> the Stage 3 inputs [posterior mean, recon_nll, KL] of vae.encode_batch."""

    def __init__(self, model, standardizer, columns, source_columns):
        super().__init__()
        self.model = model
        self.columns = list(columns)
        self.source_columns = list(source_columns)
        self.register_buffer("mean", torch.as_tensor(standardizer.mean, dtype=torch.float64))
        self.register_buffer("std", torch.as_tensor(standardizer.std, dtype=torch.float64))
        error_mask = vae.error_feature_mask(self.columns)
        # recon_nll sums only the error features, so the decoder heads keep only their rows; the sum is unchanged.
        self.register_buffer("error_index", torch.as_tensor(np.flatnonzero(error_mask)))
        self.feature_index = [self.source_columns.index(column) for column in self.columns]
        # One mask per distinct rule, gathered to the features: fewer graph nodes than one mask per feature.
        keys = [undefined_rule_key(column) for column in self.columns]
        self.rule_columns = list({key: column for key, column in zip(keys, self.columns) if key is not None}.values())
        rule_position = {undefined_rule_key(column): index for index, column in enumerate(self.rule_columns)}
        self.register_buffer("mask_index", torch.as_tensor([rule_position.get(key, len(self.rule_columns)) for key in keys]))

    def forward(self, raw):
        source = {column: raw[:, index] for index, column in enumerate(self.source_columns)}
        values = raw[:, self.feature_index]
        never_undefined = torch.zeros_like(values[:, 0], dtype=torch.bool)
        rule_masks = torch.stack([*(vae.column_undefined(source, column) for column in self.rule_columns), never_undefined], dim=1)
        missing = rule_masks[:, self.mask_index]
        inputs = ((vae.signed_log(values) - self.mean) / self.std).to(torch.float32).masked_fill(missing, 0.0)
        observed = (~missing).to(torch.float32)
        mean, log_variance = self.model.encode(inputs, observed)
        hidden = self.model.decoder(mean)
        location = torch.nn.functional.linear(hidden, self.model.location_head.weight[self.error_index], self.model.location_head.bias[self.error_index])
        log_scale = vae.soft_clip_log_scale(torch.nn.functional.linear(hidden, self.model.log_scale_head.weight[self.error_index], self.model.log_scale_head.bias[self.error_index]))
        error_inputs = inputs[:, self.error_index]
        if self.model.likelihood == "gaussian":
            terms = vae.gaussian_nll_terms(error_inputs, location, log_scale)
        else:
            head = self.model.degrees_of_freedom_head
            degrees_of_freedom = vae.degrees_of_freedom_from(torch.nn.functional.linear(hidden, head.weight[self.error_index], head.bias[self.error_index]))
            terms = vae.student_t_nll_terms(error_inputs, location, log_scale, degrees_of_freedom, log_gamma=lanczos_log_gamma)
        kl = vae.kl_per_dimension(mean, log_variance).sum(dim=1, keepdim=True)
        return torch.cat([mean, terms.sum(dim=1, keepdim=True), kl], dim=1)


def vae_onnx(graph, path):
    graph.eval()
    example = torch.zeros(2, len(graph.source_columns), dtype=torch.float64)
    with files.atomic_path(path) as temporary:
        torch.onnx.export(graph, (example,), str(temporary), input_names=["raw"], output_names=[FEATURES_NAME],
                          dynamic_axes={"raw": {0: "rows"}, FEATURES_NAME: {0: "rows"}}, opset_version=ONNX_OPSET, dynamo=False)
        return onnx.load(str(temporary))


def catboost_onnx(model_path, path):
    """CatBoost ONNX export with one probability tensor output (the ZipMap form gives one dict per row)."""
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    with tempfile.TemporaryDirectory() as scratch:
        raw_export = Path(scratch) / "catboost.onnx"
        model.save_model(str(raw_export), format="onnx")
        exported = onnx.load(str(raw_export))
    zipmap = [node for node in exported.graph.node if node.op_type == "ZipMap"]
    if zipmap:
        probability_output = zipmap[0].input[0]
        exported.graph.node.remove(zipmap[0])
    else:
        probability_output = next(output.name for output in exported.graph.output if "prob" in output.name)
    exported.graph.node.append(helper.make_node("Identity", [probability_output], [PROBABILITIES_NAME]))
    del exported.graph.output[:]
    exported.graph.output.append(helper.make_tensor_value_info(PROBABILITIES_NAME, onnx.TensorProto.FLOAT, [None, None]))
    # The CatBoost export imports only the ai.onnx.ml domain; the added Identity node needs the default domain.
    if not any(opset.domain in ("", "ai.onnx") for opset in exported.opset_import):
        exported.opset_import.append(helper.make_opsetid("", ONNX_OPSET))
    onnx.checker.check_model(exported)
    with files.atomic_path(path) as temporary:
        onnx.save(exported, str(temporary))
    return exported


def deployed_onnx(classifier_model, vae_model=None):
    """CatBoost alone (raw arm), or the VAE graph followed by CatBoost with both the VAE features and the probabilities as outputs."""
    if vae_model is None:
        return classifier_model
    # merge_models needs the same IR version in both models.
    classifier_model.ir_version = vae_model.ir_version
    return compose.merge_models(vae_model, classifier_model, io_map=[(FEATURES_NAME, classifier_model.graph.input[0].name)],
                                outputs=[FEATURES_NAME, PROBABILITIES_NAME])


def session(model, threads):
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Without the memory arena every session frees its memory after each run; with arenas, one process holds the peak of
    # every graph at once (FT-Transformer alone kept 6.8 GB). src/cpp/timing.cpp times all graphs with the same setting.
    options.enable_cpu_mem_arena = False
    return onnxruntime.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])


def graph_function(inference_session, output_names):
    """Callable input array -> list of the named outputs; the input is cast to the type that the graph expects."""
    graph_input = inference_session.get_inputs()[0]
    input_type = np.float64 if graph_input.type == "tensor(double)" else np.float32
    return lambda values: inference_session.run(output_names, {graph_input.name: np.ascontiguousarray(values, dtype=input_type)})
