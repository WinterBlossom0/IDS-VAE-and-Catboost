"""Python parts of the timed pipelines: a classifier run in its deployment form (the VAE with its preprocessing and the
Stage 3 classifier as ONNX Runtime graphs), then the novel-burst correction and the unseen-attack detector per flow, with the
stream state warmed on the earlier flows of the test day. export.py exports these parts and their labels for the
compiled timing program (src/cpp/timing.cpp)."""


import numpy as np

from idsvae import config
from idsvae.experiments import inner_selection
from idsvae.framework import correction, onnx_graph, vae
from idsvae.timing import baseline_graphs

# Real-time NIDS pipelines classify each flow as it arrives, or in small micro-batches, so the paper times batch 1.
BATCH_SIZE = 1
# Batch size of the untimed warm state; it changes only the last bits of the model outputs, so timing.cpp uses the same size.
WARM_BATCH = 10_000
SAMPLE_COUNT = 100_000
# The untimed warm-up pass before the sessions primes caches and allocators; a short pass does that.
WARMUP_FLOWS = 1_000


class StreamBatch:
    """Consecutive flows of the time-ordered sample: the model inputs and the epoch seconds of each flow."""

    def __init__(self, inputs, seconds):
        self.inputs = inputs
        self.seconds = seconds

    def __len__(self):
        return len(self.inputs)


def rule_part(rule, classes):
    """(VAE features, probabilities, seconds) -> (labels after the rule, VAE features, seconds); advances the OnlineRule rule."""
    class_names = np.asarray(classes, dtype=object)
    latent_dim = len(rule.thresholds.scale)

    def apply(value):
        features, probabilities, seconds = value
        predicted = class_names[probabilities.argmax(axis=1)]
        labels = [rule.step(int(second), row[:latent_dim], float(row[latent_dim]), label)[0] for second, row, label in zip(seconds, features, predicted)]
        return labels, features, seconds
    return apply


def detector_part(online, score, cutoff, latent_dim):
    """(labels, VAE features, seconds) -> final labels: detector features per flow (advances the OnlineDetector online), then the
    detector model (rows -> attack scores) on the batch."""

    def apply(value):
        labels, features, seconds = value
        rows = [online.features(int(second), row[:latent_dim], float(row[latent_dim]), float(row[latent_dim + 1])) for second, row in zip(seconds, features)]
        flagged = score(np.asarray(rows)) > cutoff
        return [config.UNKNOWN_ATTACK if (label == config.BENIGN and flag) else label for label, flag in zip(labels, flagged)]
    return apply


def split_day(test):
    """(earlier flows, sample) of the time-ordered test day: the sample is the SAMPLE_COUNT consecutive flows in its middle."""
    # A timing sample does not need the within-second order of splits.stream_order; paper/numbers.py counts the classes of the sample.
    ordered = test.iloc[np.argsort(test["time"].to_numpy(), kind="stable")]
    start = (len(ordered) - SAMPLE_COUNT) // 2
    return ordered.iloc[:start], ordered.iloc[start:start + SAMPLE_COUNT]


def sample_window(test):
    return split_day(test)[1]


def model_inputs(frame, columns):
    return frame[columns].to_numpy(dtype="float64"), correction.epoch_seconds(frame["time"].to_numpy())


def rule_thresholds_from(encoded_dir, run_dir, result, latent_dim):
    """Thresholds of the novel-burst correction, computed as in final_run.run_rule."""
    train, _ = correction.load_split(encoded_dir, run_dir, result, "fit_train")
    calib, calib_predicted = correction.load_split(encoded_dir, run_dir, result, "calib")
    attack_classes = [label for label in result["classes"] if label != config.BENIGN]
    return correction.thresholds(train, calib, calib_predicted, attack_classes, latent_dim, inner_selection.chosen_rule_settings())


def classifier_function(settings, run_dir, threads):
    """(features -> probabilities, ONNX model) of a Stage 3 classifier in ONNX Runtime (CatBoost, LightGBM or XGBoost)."""
    if settings["model"] == "catboost":
        model = onnx_graph.catboost_onnx(run_dir / "model.bin", run_dir / "classifier.onnx")
    else:
        model = baseline_graphs.stage3_classifier_graph(settings["model"], run_dir / "model.bin")
    run = onnx_graph.graph_function(onnx_graph.session(model, threads), [onnx_graph.PROBABILITIES_NAME])
    return (lambda features: run(features)[0]), model


def raw_rule_parts(result, run_dir, signal_vae_dir, classify_features, threads):
    """(input columns, stream states -> parts, factory of empty stream states) of raw CatBoost with the correction: the
    design VAE gives the correction its signals, as in final_run.rule_inputs."""
    model, standardizer, vae_columns = vae.load_model(signal_vae_dir)
    source_columns = vae.mask_source_columns(vae_columns)
    columns = list(dict.fromkeys([*result["columns"], *source_columns]))
    raw_index, source_index = ([columns.index(column) for column in names] for names in (result["columns"], source_columns))
    vae_model = onnx_graph.vae_onnx(onnx_graph.VaeFeatureGraph(model, standardizer, vae_columns, source_columns), run_dir / "vae.onnx")
    encode = onnx_graph.graph_function(onnx_graph.session(vae_model, threads), [onnx_graph.FEATURES_NAME])
    rule_thresholds = rule_thresholds_from(signal_vae_dir, run_dir, result, model.latent_dim)

    def parts(states):
        rule, _ = states
        return [lambda batch: (batch.inputs, encode(batch.inputs[:, source_index])[0], batch.seconds),
                lambda value: (value[1], classify_features(value[0][:, raw_index]), value[2]),
                rule_part(rule, result["classes"])]
    return columns, parts, lambda: (correction.OnlineRule(rule_thresholds), None)


def make_deployed_parts(result, run_dir, stage2_root, threads, with_rule, deployed_detector=None, separate=False, signal_vae_dir=None):
    """(input columns, stream states -> deployed parts, factory of empty stream states or None) of one Stage 3 run; the stream
    states are the OnlineRule and OnlineDetector (or None) that the rule and the detector advance. separate keeps the VAE and
    CatBoost graphs apart, so that each part has its own time; signal_vae_dir is the VAE whose signals the correction of a raw
    run uses."""
    settings, classes = result["settings"], result["classes"]
    classify_features, classifier_model = classifier_function(settings, run_dir, threads)
    if settings["view"] == "raw" and with_rule:
        return raw_rule_parts(result, run_dir, signal_vae_dir, classify_features, threads)
    if settings["view"] == "raw":
        def raw_parts(_states):
            return [lambda batch: classify_features(batch.inputs)]
        return result["columns"], raw_parts, None
    vae_dir = stage2_root / settings["vae_run"]
    model, standardizer, vae_columns = vae.load_model(vae_dir)
    source_columns = vae.mask_source_columns(vae_columns)
    vae_model = onnx_graph.vae_onnx(onnx_graph.VaeFeatureGraph(model, standardizer, vae_columns, source_columns), run_dir / "vae.onnx")
    rule_thresholds = rule_thresholds_from(vae_dir, run_dir, result, model.latent_dim) if with_rule else None

    def empty_states():
        return correction.OnlineRule(rule_thresholds), (None if deployed_detector is None else deployed_detector[0]())

    def rule_parts(states):
        if not with_rule:
            return []
        rule, online = states
        parts = [rule_part(rule, classes)]
        if online is not None:
            parts.append(detector_part(online, deployed_detector[1], deployed_detector[2], model.latent_dim))
        return parts
    make_states = empty_states if with_rule else None
    if separate:
        encode = onnx_graph.graph_function(onnx_graph.session(vae_model, threads), [onnx_graph.FEATURES_NAME])

        def separate_parts(states):
            return [lambda batch: (encode(batch.inputs)[0], batch.seconds),
                    lambda value: (value[0], classify_features(value[0]), value[1]), *rule_parts(states)]
        return source_columns, separate_parts, make_states
    deployed = onnx_graph.graph_function(onnx_graph.session(onnx_graph.deployed_onnx(classifier_model, vae_model), threads),
                                        [onnx_graph.FEATURES_NAME, onnx_graph.PROBABILITIES_NAME])

    def deployed_parts(states):
        return [lambda batch: (*deployed(batch.inputs), batch.seconds), *rule_parts(states)]
    return source_columns, deployed_parts, make_states


def warm_states(make_parts, states, raw_values, seconds):
    """Advance the stream states over the earlier flows of the day, untimed, in batches of WARM_BATCH flows."""
    parts = make_parts(states)
    for batch in stream_batches(raw_values, seconds, WARM_BATCH):
        value = batch
        for function in parts:
            value = function(value)
    return states


def stream_batches(raw_values, seconds, batch_size):
    """Batches of batch_size consecutive flows."""
    return [StreamBatch(raw_values[start:start + batch_size], seconds[start:start + batch_size]) for start in range(0, len(raw_values), batch_size)]
