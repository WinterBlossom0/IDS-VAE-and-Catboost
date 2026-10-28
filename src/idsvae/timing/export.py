"""Export of every timed pipeline for the compiled timing program (src/cpp/timing.cpp): the ONNX graphs, the parameters of the
correction and the detector, the model inputs of the timing sample and of the earlier flows of the day, and the final label
of each sample flow from the Python parts of python_parts.py at batch 1 (pipelines and names as in the paper tables), against which the program checks its own labels."""
import argparse
import gc
import json
import shutil

import numpy as np
import onnx

from idsvae import config, files
from idsvae.data import splits
from idsvae.experiments import baselines, deep_baselines, detector_runs, final_run
from idsvae.framework import detector, features, onnx_graph, vae
from idsvae.timing import baseline_graphs, python_parts

EXPORT = final_run.FINAL_DIR / "timing" / "compiled"
SEED = 3
THREADS = 1
SESSION_SECONDS = 300
COOLDOWN_SECONDS = 120


class Exporter:
    def __init__(self, root):
        self.root = root
        for folder in ("graphs", "params", "inputs", "reference"):
            (root / folder).mkdir(parents=True, exist_ok=True)
        self.inputs = {}
        self.seconds = None
        self.pipelines = []

    def save_array(self, relative, values, dtype):
        path = self.root / relative
        with files.atomic_path(path) as temporary:
            np.ascontiguousarray(values, dtype=dtype).tofile(temporary)
        return relative

    def save_graph(self, name, model):
        relative = f"graphs/{name}.onnx"
        onnx.checker.check_model(model)
        with files.atomic_path(self.root / relative) as temporary:
            onnx.save(model, str(temporary))
        return relative

    def input_id(self, columns, with_earlier):
        """Id of the model inputs of these columns; the sample and earlier flows are written once per column list."""
        key = tuple(columns)
        if key in self.inputs and (self.inputs[key]["earlier"] is not None or not with_earlier):
            return self.inputs[key]["id"]
        earlier, sample = python_parts.split_day(splits.load_split("test", config.PROCESSED_DIR, columns=list(columns)))
        window_values, window_seconds = python_parts.model_inputs(sample, list(columns))
        earlier_values, earlier_seconds = python_parts.model_inputs(earlier, list(columns))
        if self.seconds is None:
            self.seconds = (earlier_seconds, window_seconds)
            self.save_array("seconds_window.i64", window_seconds, np.int64)
            self.save_array("seconds_earlier.i64", earlier_seconds, np.int64)
        if not (np.array_equal(self.seconds[0], earlier_seconds) and np.array_equal(self.seconds[1], window_seconds)):
            raise ValueError(f"the flows of the columns {columns[:3]}... are not in the order of the first input")
        input_id = self.inputs[key]["id"] if key in self.inputs else f"input{len(self.inputs)}"
        entry = {"id": input_id, "columns": len(columns), "window": self.save_array(f"inputs/{input_id}_window.f64", window_values, np.float64),
                 "earlier": self.save_array(f"inputs/{input_id}_earlier.f64", earlier_values, np.float64) if with_earlier else None}
        self.inputs[key] = entry
        return input_id

    def window_values(self, columns):
        return np.fromfile(self.root / self.inputs[tuple(columns)]["window"], dtype=np.float64).reshape(-1, len(columns))

    def earlier_values(self, columns):
        return np.fromfile(self.root / self.inputs[tuple(columns)]["earlier"], dtype=np.float64).reshape(-1, len(columns))

    def save_reference(self, name, indices):
        if len(indices) != python_parts.SAMPLE_COUNT:
            raise ValueError(f"{name}: {len(indices)} reference labels, not {python_parts.SAMPLE_COUNT}")
        return self.save_array(f"reference/{name}.i32", indices, np.int32)

    def add(self, pipeline):
        self.pipelines.append(pipeline)
        self.write_manifest()

    def write_manifest(self):
        manifest = {"sample_count": python_parts.SAMPLE_COUNT, "warmup_flows": python_parts.WARMUP_FLOWS, "warm_batch": python_parts.WARM_BATCH,
                    "batch_size": python_parts.BATCH_SIZE, "session_seconds": SESSION_SECONDS,
                    "cooldown_seconds": COOLDOWN_SECONDS, "sessions": 2, "seed": SEED, "threads": THREADS, "benign_label": config.BENIGN,
                    "seconds": {"window": "seconds_window.i64", "earlier": "seconds_earlier.i64"},
                    "inputs": {entry["id"]: {key: entry[key] for key in ("columns", "window", "earlier")} for entry in self.inputs.values()},
                    "pipelines": self.pipelines}
        files.write_json_atomic(self.root / "manifest.json", manifest)


def batch1_outputs(make_parts, raw_values, seconds):
    """Output of the last part for every flow, one flow per call, as the timed passes run them."""
    parts = make_parts()
    outputs = []
    for batch in python_parts.stream_batches(raw_values, seconds, 1):
        value = batch
        for function in parts:
            value = function(value)
        outputs.append(value)
    return outputs


def argmax_indices(outputs):
    return np.concatenate([np.asarray(probabilities).argmax(axis=1) for probabilities in outputs])


def label_indices(outputs, classes):
    """Final label indices in classes + [unknown]; the rule part returns (labels, VAE features, seconds), the detector part the labels."""
    names = [*classes, config.UNKNOWN_ATTACK]
    return np.array([names.index(label) for output in outputs for label in (output[0] if isinstance(output, tuple) else output)])


def rule_params(rule, classes):
    thresholds = rule.thresholds
    if any(label not in classes for label in thresholds.attack_classes):
        raise ValueError("an attack class of the rule is not a Stage 3 class")
    return {"latent_dim": len(thresholds.scale), "divisor": rule.divisor.tolist(), "novelty": thresholds.novelty, "burst": thresholds.burst,
            "min_support": thresholds.min_support, "window_seconds": thresholds.settings.window_seconds, "attack_classes": thresholds.attack_classes}


def detector_params(exporter, name, online, latent_dim, cutoff, detector_model):
    references = {"nll": online.nll_reference, "kl": online.kl_reference,
                  **{feature: reference for feature, reference in zip(detector.STREAM_FEATURES, online.stream_references)}}
    paths = {key: exporter.save_array(f"params/{name}_{key}_ref.f64", values, np.float64) for key, values in references.items()}
    background = online.background
    return {"latent_dim": latent_dim, "columns": online.columns, "divisor": online.divisor.tolist(), "window_seconds": online.streams.window_seconds,
            "novelty": online.novelty, "nll_reference": paths["nll"], "kl_reference": paths["kl"],
            "stream_references": [paths[feature] for feature in detector.STREAM_FEATURES],
            "background": None if background is None else {
                "burst_cutoff": background.burst_cutoff, "fallback_median": background.fallback_median,
                "reservoir": detector.BACKGROUND_RESERVOIR, "warm_up": detector.BACKGROUND_WARM_UP,
                "refresh": detector.BACKGROUND_REFRESH},
            "graph": exporter.save_graph(f"{name}_detector", detector_model), "cutoff": float(cutoff)}


def deployed_entry(exporter, name, arm, variant, with_rule, with_detector, separate=False):
    """Export one Stage 3 run with the parts of python_parts.make_deployed_parts, and its batch-1 reference labels from them."""
    detector_parts = detector_runs.deployed_detector(arm, SEED, THREADS) if with_detector else None
    run_name = final_run.classifier_settings(arm, SEED, variant).run_name()
    run_dir = final_run.stage3_root(arm) / run_name
    result = json.loads((run_dir / "result.json").read_text())
    settings, classes = result["settings"], result["classes"]
    signal_vae_dir = final_run.rule_inputs(arm, SEED, variant)[0] if (with_rule and arm == "raw") else None
    columns, make_parts, make_states = python_parts.make_deployed_parts(result, run_dir, final_run.stage2_root(arm), THREADS, with_rule, detector_parts, separate, signal_vae_dir)
    input_id = exporter.input_id(columns, with_earlier=make_states is not None)
    window, seconds = exporter.window_values(columns), exporter.seconds[1]
    _, classifier_model = python_parts.classifier_function(settings, run_dir, THREADS)
    entry = {"name": name, "input": input_id, "classes": classes}
    if make_states is None:
        entry.update(kind="graph", graph=exporter.save_graph(name, classifier_model))
        outputs = batch1_outputs(lambda: make_parts(None), window, seconds)
        exporter.add({**entry, "reference": exporter.save_reference(name, argmax_indices(outputs))})
        return
    states = python_parts.warm_states(make_parts, make_states(), exporter.earlier_values(columns), exporter.seconds[0])
    rule, online = states
    entry["rule"] = rule_params(rule, classes)
    latent_dim = entry["rule"]["latent_dim"]
    entry["detector"] = None if online is None else detector_params(exporter, name, online, latent_dim, detector_parts[2],
                                                                onnx.load(str(detector_runs.model_file(arm, SEED).with_suffix(".onnx"))))
    vae_model = onnx.load(str(run_dir / "vae.onnx"))
    if settings["view"] == "raw":
        source_columns = vae.mask_source_columns(vae.load_model(signal_vae_dir)[2])
        entry.update(kind="raw_rule", vae_graph=exporter.save_graph(f"{name}_vae", vae_model), classifier_graph=exporter.save_graph(f"{name}_classifier", classifier_model),
                     vae_columns=[columns.index(column) for column in source_columns], classifier_columns=[columns.index(column) for column in result["columns"]])
    elif separate:
        entry.update(kind="separate_rule", vae_graph=exporter.save_graph(f"{name}_vae", vae_model), classifier_graph=exporter.save_graph(f"{name}_classifier", classifier_model))
    else:
        entry.update(kind="merged_rule", graph=exporter.save_graph(name, onnx_graph.deployed_onnx(classifier_model, vae_model)))
    outputs = batch1_outputs(lambda: make_parts(states), window, seconds)
    exporter.add({**entry, "reference": exporter.save_reference(name, label_indices(outputs, classes))})


def method_entry(exporter, method):
    classes, columns, deployment, _ = baseline_graphs.deployed_method(method, SEED, THREADS)
    input_id = exporter.input_id(columns, with_earlier=False)
    window, seconds = exporter.window_values(columns), exporter.seconds[1]
    if method == "flow_transformer":
        def make_parts():
            predict = deployment.stream()
            return [lambda batch: predict(batch.inputs)]
        kind = {"kind": "flow_transformer", "window_size": deep_baselines.WINDOW_SIZE}
    else:
        def make_parts():
            return [lambda batch: deployment.predict_proba(batch.inputs)]
        kind = {"kind": "graph"}
    graph = f"graphs/{method}.onnx"
    shutil.copyfile(baseline_graphs.onnx_path(method, SEED), exporter.root / graph)
    outputs = batch1_outputs(make_parts, window, seconds)
    exporter.add({"name": method, "input": input_id, "classes": list(classes), **kind, "graph": graph,
                  "reference": exporter.save_reference(method, argmax_indices(outputs))})


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    exporter = Exporter(EXPORT)
    main_variant = final_run.MAIN
    jobs = [("framework", lambda: deployed_entry(exporter, "framework", "topology", main_variant, True, True)),
            ("framework_parts", lambda: deployed_entry(exporter, "framework_parts", "topology", main_variant, True, True, separate=True)),
            ("raw", lambda: deployed_entry(exporter, "raw", "raw", main_variant, False, False)),
            ("raw_correction", lambda: deployed_entry(exporter, "raw_correction", "raw", main_variant, True, False)),
            ("stage3_catboost", lambda: deployed_entry(exporter, "stage3_catboost", "topology", main_variant, True, False)),
            ("stage3_lightgbm", lambda: deployed_entry(exporter, "stage3_lightgbm", "lightgbm", main_variant, True, False)),
            ("stage3_xgboost", lambda: deployed_entry(exporter, "stage3_xgboost", "xgboost", main_variant, True, False)),
            *((method, (lambda method=method: method_entry(exporter, method))) for method in baselines.METHODS)]
    selected = features.load_ranking()["selected_count"]
    for count in sorted(set(final_run.FEATURE_COUNTS) - {selected}):
        variant = final_run.Variant(count)
        jobs.append((f"ours_k{count}", lambda variant=variant, count=count: deployed_entry(exporter, f"ours_k{count}", "topology", variant, True, False)))
        jobs.append((f"raw_k{count}", lambda variant=variant, count=count: deployed_entry(exporter, f"raw_k{count}", "raw", variant, False, False)))
    for name, job in jobs:
        job()
        gc.collect()
        print(f"exported {name}", flush=True)


if __name__ == "__main__":
    main()
