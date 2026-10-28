#!/usr/bin/env bash
# Comparison methods (src/idsvae/experiments/baselines.py) on seeds 3-5, after run_detector.sh (which writes the further capture days): the
# main runs (hold-out, test day, held-out days of the detector), the leave-one-attack-out folds, then the saved deployment
# form of seed 3 for the CPU timing; finished jobs are skipped.
# One job at a time: one main job needs 6-13 GB of RAM (measured peak). Deep methods (experiments/deep_baselines.py) run on the GPU under
# the shared GPU lock. Arguments select the methods (default: all, shallow first, TabNet last).
set -uo pipefail
cd "$(dirname "$0")/.."
export IDSVAE_GPU_LOCK="${IDSVAE_GPU_LOCK:-/tmp/idsvae-gpu.lock}"
mkdir -p "$(dirname "$IDSVAE_GPU_LOCK")" data/final/logs
methods=("$@")
failures=0
if [ "${#methods[@]}" -eq 0 ]; then
  methods=(xgboost lightgbm random_forest mlp kang_idae_xg cnn1d flow_transformer tabm ft_transformer tabnet)
fi
for method in "${methods[@]}"; do
  for seed in 3 4 5; do
    if ! uv run --frozen python -m idsvae.experiments.baselines --method "$method" --seed "$seed" --threads 4 >> "data/final/logs/baseline_${method}_${seed}.log" 2>&1; then
      echo "failed: $method $seed"
      failures=$((failures + 1))
    fi
  done
done
for method in "${methods[@]}"; do
  for left_out in "SSH-Bruteforce" "DoS attacks-GoldenEye" "DoS attacks-Slowloris"; do
    for seed in 3 4 5; do
      if ! uv run --frozen python -m idsvae.experiments.baselines --method "$method" --seed "$seed" --threads 4 --left-out-class "$left_out" \
        >> "data/final/logs/baseline_${method}_${seed}_loo.log" 2>&1; then
        echo "failed: $method $seed $left_out"
        failures=$((failures + 1))
      fi
    done
  done
done
# Seed 3 of each method trained again with its deployment form saved, for the CPU timing (run_timing.sh).
for method in "${methods[@]}"; do
  if ! uv run --frozen python -m idsvae.experiments.baselines --method "$method" --seed 3 --threads 4 --save-model >> "data/final/logs/save_model_${method}.log" 2>&1; then
    echo "failed: $method save-model"
    failures=$((failures + 1))
  fi
done
if [ "$failures" -gt 0 ]; then
  echo "baselines done with $failures failed jobs"
  exit 1
fi
echo "baselines done"
