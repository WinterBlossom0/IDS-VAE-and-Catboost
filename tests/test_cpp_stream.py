"""The C++ correction and detector features (src/cpp/timing.cpp, stream mode) against the Python classes on a synthetic
stream of three days with bursts, overrides and many background refreshes: branches that the timing sample of one day
rarely reaches. Needs the export (python -m idsvae.timing.export) and the build (src/cpp/build.sh)."""
import json
import subprocess

import numpy as np
import pytest

from idsvae import config
from idsvae.framework import correction, detector

EXPORT = config.PROJECT_ROOT / "data" / "final" / "timing" / "compiled"
PROGRAM = config.PROJECT_ROOT / "build" / "timing"
DAY_START = 1_519_171_200  # 2018-02-21 00:00:00 UTC
DAYS = 3

pytestmark = pytest.mark.skipif(not (EXPORT / "manifest.json").exists() or not PROGRAM.exists(), reason="needs the compiled export and build")


def rule_pipelines():
    if not (EXPORT / "manifest.json").exists():
        return []
    return [spec for spec in json.loads((EXPORT / "manifest.json").read_text())["pipelines"] if "rule" in spec]


def synthetic_stream(spec, seed):
    """(seconds, float32 model outputs [latent, recon_nll, kl], predicted class indices) in time order."""
    generator = np.random.default_rng(seed)
    divisor = np.asarray(spec["rule"]["divisor"])
    latent_dim = len(divisor)
    attack_indices = [spec["classes"].index(label) for label in spec["rule"]["attack_classes"]]
    seconds, latent, predicted = [], [], []
    for day in range(DAYS):
        # A new cell scores minute - 1 under MIDAS-R, so the flows of a day stay within its first hour, where many flows are
        # below the background burst cut-off and refresh the background median.
        clock = DAY_START + day * 86_400 + int(generator.integers(0, 300))
        for _ in range(40):
            length = int(generator.integers(20, 90))
            count = int(generator.integers(50, 400))
            hot_cell = generator.integers(-3, 4, latent_dim) if generator.random() < 0.5 else None
            attack = attack_indices[int(generator.integers(len(attack_indices)))] if generator.random() < 0.5 else None
            segment_seconds = np.sort(clock + generator.integers(0, length, count))
            for second in segment_seconds:
                in_hot_cell = hot_cell is not None and generator.random() < 0.8
                cells = hot_cell + generator.uniform(0.05, 0.95, latent_dim) if in_hot_cell else generator.normal(0.0, 3.0, latent_dim)
                seconds.append(int(second))
                latent.append(cells * divisor)
                predicted.append(attack if attack is not None and generator.random() < 0.7 else spec["classes"].index(config.BENIGN))
            clock += length + int(generator.integers(0, 30))
    flows = len(seconds)
    detector_spec = spec["detector"]
    kl_reference = np.fromfile(EXPORT / detector_spec["kl_reference"]) if detector_spec else np.abs(generator.normal(5.0, 2.0, 1_000))
    nll = generator.uniform(-150.0, 150.0, flows)
    kl = generator.choice(kl_reference, flows) * generator.uniform(0.5, 1.5, flows)
    outputs = np.column_stack([np.asarray(latent), nll, kl]).astype(np.float32)
    return np.asarray(seconds, dtype=np.int64), outputs, np.asarray(predicted, dtype=np.int32)


def python_stream(spec, seconds, outputs, predicted):
    """Rule labels and detector rows of the Python classes, as python_parts.rule_part and python_parts.detector_part call them."""
    rule_spec = spec["rule"]
    latent_dim = rule_spec["latent_dim"]
    thresholds = correction.RuleThresholds(rule_spec["novelty"], rule_spec["burst"], np.ones(latent_dim), rule_spec["min_support"], rule_spec["attack_classes"],
                                             correction.RuleSettings(window_seconds=rule_spec["window_seconds"]))
    rule = correction.OnlineRule(thresholds)
    rule.divisor = np.asarray(rule_spec["divisor"])
    online = None
    detector_spec = spec["detector"]
    if detector_spec is not None:
        background = None if detector_spec["background"] is None else detector.DayBackground(detector_spec["background"]["burst_cutoff"], detector_spec["background"]["fallback_median"],
                                                                                               np.random.default_rng(0))
        online = detector.OnlineDetector(detector_spec["columns"], np.ones(latent_dim), correction.RuleSettings(window_seconds=detector_spec["window_seconds"]),
                                              detector_spec["novelty"], np.fromfile(EXPORT / detector_spec["nll_reference"]), np.fromfile(EXPORT / detector_spec["kl_reference"]),
                                              [np.fromfile(EXPORT / path) for path in detector_spec["stream_references"]], background)
        online.divisor = np.asarray(detector_spec["divisor"])
    classes = spec["classes"]
    labels, rows = [], []
    for second, row, label in zip(seconds, outputs, predicted):
        labels.append(classes.index(rule.step(int(second), row[:latent_dim], float(row[latent_dim]), classes[label])[0]))
        if online is not None:
            rows.append(online.features(int(second), row[:latent_dim], float(row[latent_dim]), float(row[latent_dim + 1])))
    return np.asarray(labels, dtype=np.int32), np.asarray(rows, dtype=np.float64).reshape(len(seconds), -1), online


@pytest.mark.parametrize("spec", rule_pipelines(), ids=lambda spec: spec["name"])
def test_cpp_stream_state_equals_python_on_a_three_day_stream(spec, tmp_path):
    seconds, outputs, predicted = synthetic_stream(spec, seed=len(spec["name"]))
    prefix = tmp_path / "stream"
    seconds.tofile(f"{prefix}_seconds.i64")
    outputs.tofile(f"{prefix}_features.f32")
    predicted.tofile(f"{prefix}_predicted.i32")
    subprocess.run([str(PROGRAM), str(EXPORT), "stream", str(prefix), spec["name"]], check=True)
    expected_labels, expected_rows, online = python_stream(spec, seconds, outputs, predicted)
    labels = np.fromfile(f"{prefix}_cpp_labels.i32", dtype=np.int32)
    rows = np.fromfile(f"{prefix}_cpp_features.f64").reshape(expected_rows.shape)
    assert len(np.unique(seconds // 86_400)) == DAYS
    assert (expected_labels != predicted).sum() > 50, "the stream must make the rule override Stage 3"
    assert np.array_equal(labels, expected_labels)
    assert np.array_equal(rows, expected_rows)
    if online is not None and online.background is not None:
        # At least 6 refreshes of the background median on the last day (warm-up 1,000, then every 500 non-burst flows).
        assert online.background.seen >= 3_500
