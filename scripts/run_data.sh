#!/usr/bin/env bash
# Data and Stage 1: clean the protocol days and write the splits (src/idsvae/data/splits.py, data/processed), then the
# Stage 1 ranking of the main split and of each leave-one-attack-out fold (src/idsvae/framework/features.py).
set -euo pipefail
cd "$(dirname "$0")/.."
uv run --frozen python -m idsvae.data.splits
uv run --frozen python -m idsvae.framework.features
for class in "SSH-Bruteforce" "DoS attacks-GoldenEye" "DoS attacks-Slowloris"; do
  uv run --frozen python -m idsvae.framework.features --left-out-class "$class"
done
echo "data done"
