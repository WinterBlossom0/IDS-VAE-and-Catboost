#!/usr/bin/env bash
# The detector on each VAE arm (src/idsvae/experiments/detector_ablation.py), then the shallow comparison methods and Isolation Forest
# trained also on the further days and flagged at the detector's cut-off (src/idsvae/experiments/equal_data.py), after
# run_detector.sh and run_baselines.sh. One job at a time: one equal-data job peaks at 8.3 GB of RAM (measured). Finished jobs are skipped.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/final/logs
uv run --frozen python -m idsvae.experiments.detector_ablation >> data/final/logs/detector_ablation.log 2>&1 || echo "failed: detector_ablation"
for seed in 3 4 5; do
  for method in isolation_forest kang_idae_xg xgboost lightgbm random_forest mlp; do
    for day in 02-28 02-20 02-22 03-01 03-02; do
      uv run --frozen python -m idsvae.experiments.equal_data --method "$method" --seed "$seed" --mode day --target "$day" >> "data/final/logs/equal_${method}_${seed}.log" 2>&1 || echo "failed: $method $seed $day"
    done
    for left_out in "SSH-Bruteforce" "DoS attacks-GoldenEye" "DoS attacks-Slowloris"; do
      uv run --frozen python -m idsvae.experiments.equal_data --method "$method" --seed "$seed" --mode fold --target "$left_out" >> "data/final/logs/equal_${method}_${seed}.log" 2>&1 || echo "failed: $method $seed $left_out"
    done
    uv run --frozen python -m idsvae.experiments.equal_data --method "$method" --seed "$seed" --mode test >> "data/final/logs/equal_${method}_${seed}.log" 2>&1 || echo "failed: $method $seed test"
  done
done
