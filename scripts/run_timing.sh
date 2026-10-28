#!/usr/bin/env bash
# CPU inference time of every timing in the paper, after run_detector.sh and run_baselines.sh: one compiled C++ program
# (src/cpp/timing.cpp) runs every pipeline with the ONNX Runtime C++ API, the correction and the detector in C++, batch 1, one
# thread, seed 3, two sessions of 5 minutes with a 2-minute idle pause. Run it alone on an idle machine: other CPU jobs change
# the times.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/final/logs
export IDSVAE_NOVELTY_REFERENCE
IDSVAE_NOVELTY_REFERENCE=$(uv run --frozen python -c "import json; print(json.load(open('data/final/unseen/choice.json'))['novelty_reference'])")
# Each re-run method as one ONNX graph must reproduce its evaluated model on the hold-out (data/final/timing/onnx_checks.json).
uv run --frozen python -m idsvae.timing.baseline_graphs >> data/final/logs/timing.log 2>&1
# Graphs, parameters, inputs and the Python final label of each timed flow (data/final/timing/compiled/).
uv run --frozen python -m idsvae.timing.export >> data/final/logs/timing.log 2>&1
src/cpp/build.sh
# Every C++ final label must equal the Python label (data/final/timing/compiled_check.json); the timed passes check it again.
build/timing data/final/timing/compiled check data/final/timing/compiled_check.json >> data/final/logs/timing.log 2>&1
# Two full runs with the same program; the paper reports the medians of the second, the framework difference to the first,
# and that both runs order the main-table pipelines the same.
build/timing data/final/timing/compiled time data/final/timing/compiled_threads1_run1.json >> data/final/logs/timing.log 2>&1
build/timing data/final/timing/compiled time data/final/timing/compiled_threads1.json >> data/final/logs/timing.log 2>&1
echo "timing done"
