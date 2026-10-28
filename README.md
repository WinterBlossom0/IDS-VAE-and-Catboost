# VAE-CatBoost Network Intrusion Detection

Code for the paper *A Practical Hybrid Robust VAE-CatBoost Framework for Efficient Multiclass Network Intrusion Detection*
(ICDEC 2026, Springer LNNS). The paper is [`paper/main.pdf`](paper/main.pdf); its LaTeX source is in `paper/`
(build: `tectonic paper/main.tex`).

The framework has three stages:

1. CatBoost with SHAP feature selection.
2. A robust variational autoencoder (VAE).
3. A CatBoost classifier of the VAE outputs, with a correction for bursts of new attacks and a detector of unseen attack types.

## Layout

```
src/idsvae/framework/    the method: features.py (Stage 1), vae.py (Stage 2), classify.py (Stage 3),
                         correction.py, detector.py, onnx_graph.py (deployment as one ONNX graph)
src/idsvae/data/         CSE-CIC-IDS2018 reading, cleaning and day splits
src/idsvae/experiments/  the paper's experiments and the ten comparison methods
src/idsvae/timing/       CPU timing; src/cpp/timing.cpp is the C++ timing program
src/idsvae/paper/        tables, numbers and figures of the paper
scripts/                 the full pipeline, one script per step
```

## Setup

You need Linux or WSL2, Python 3.12, [uv](https://docs.astral.sh/uv/) and an NVIDIA GPU with CUDA 12.8.

```bash
uv sync --frozen
CUDA_VISIBLE_DEVICES="" uv run --frozen pytest -q
```

The scripts stream the [CSE-CIC-IDS2018](https://www.unb.ca/cic/datasets/ids-2018.html) CSV files from the public bucket.
To use local copies, set `IDSVAE_RAW_DIR` to the folder that holds them.

## Run

Run the scripts in this order. The results go to `data/`.

```bash
scripts/run_data.sh                    # clean the data, make the splits, Stage 1 ranking
setsid scripts/run_final_pipeline.sh   # VAEs, classifiers, correction, ablations
scripts/run_detector.sh                # unseen-attack detector
scripts/run_baselines.sh               # the ten comparison methods
scripts/run_timing.sh                  # CPU inference time (C++, about 1.5 h; needs g++)
scripts/run_paper.sh                   # tables, numbers and figures of the paper
```

The comment at the top of each script says what it does. We used an RTX 5080 Laptop GPU and 15 GB of RAM. Run one script at a
time, because some steps need up to 13 GB of RAM.

CatBoost models of the detector change with the thread count, so each step uses a fixed count: Stage 1 all cores, Stage 3
five threads, detector selection and held-out days four, final detector and folds eight, the LightGBM and XGBoost ablations
16, comparison methods four, timing one.

## License

MIT
