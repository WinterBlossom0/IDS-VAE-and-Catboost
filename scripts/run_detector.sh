#!/usr/bin/env bash
# The unseen-attack detector (src/idsvae/framework/detector.py; experiments in src/idsvae/experiments/detector_runs.py), after run_final_pipeline.sh:
# 1. extra capture days, streamed from the public dataset bucket and cleaned (src/idsvae/data/extra_days.py);
# 2. features of every VAE for both novelty references, the multi-day selection, and the choice (pre-set rule);
# 3. the detector with the chosen configuration on the main arms, the leave-one-attack-out report, the LOIC-HTTP template report, the
# held-out scores of deeper detector trees, the raw-feature detector control, the full framework on the held-out days, and
# the test-day detector scores. The data and feature steps skip finished outputs; all later steps always recompute. The VAEs
# encode the extra days under the shared GPU lock. Run no other job at the same time: the steps that encode day 02-20
# (5 million flows) need up to about 12 GB of RAM.
set -euo pipefail
cd "$(dirname "$0")/.."
export IDSVAE_GPU_LOCK="${IDSVAE_GPU_LOCK:-/tmp/idsvae-gpu.lock}"
mkdir -p "$(dirname "$IDSVAE_GPU_LOCK")"
for day in 02-16 02-20 02-22 03-01 03-02; do
  [ -f "data/processed_extra/$day.parquet" ] || uv run --frozen python -m idsvae.data.extra_days "$day"
done
for reference in day calib; do
  export IDSVAE_NOVELTY_REFERENCE=$reference
  uv run --frozen python -m idsvae.experiments.detector_runs features
  uv run --frozen python -m idsvae.experiments.detector_runs extra-features
  uv run --frozen python -m idsvae.experiments.detector_runs select-multi
done
uv run --frozen python -m idsvae.experiments.detector_runs choose-multi
export IDSVAE_NOVELTY_REFERENCE
IDSVAE_NOVELTY_REFERENCE=$(uv run --frozen python -c "import json; print(json.load(open('data/final/unseen/choice.json'))['novelty_reference'])")
uv run --frozen python -m idsvae.experiments.detector_runs apply
uv run --frozen python -m idsvae.experiments.detector_runs leave-one-out
uv run --frozen python -m idsvae.experiments.detector_runs templates
uv run --frozen python -m idsvae.experiments.detector_depths
uv run --frozen python -m idsvae.experiments.raw_detector
# The full framework on the held-out days; the encode step peaks at about 12 GB of RAM, so it runs alone.
uv run --frozen python -m idsvae.experiments.held_out_days encode
uv run --frozen python -m idsvae.experiments.held_out_days report
uv run --frozen python -m idsvae.experiments.held_out_days test
echo "detector done (novelty reference $IDSVAE_NOVELTY_REFERENCE)"
