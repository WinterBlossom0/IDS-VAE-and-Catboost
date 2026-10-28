#!/usr/bin/env bash
# Paper figures and numbers from the finished runs: the SHAP figure (src/idsvae/paper/figures.py), the tables and inline numbers
# (src/idsvae/paper/numbers.py, paper/generated/). Then build the PDF with: tectonic paper/main.tex
set -euo pipefail
cd "$(dirname "$0")/.."
uv run --frozen python -m idsvae.paper.figures
uv run --frozen python -m idsvae.paper.numbers
echo "paper inputs written to paper/generated and paper/figures"
