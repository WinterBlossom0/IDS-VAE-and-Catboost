"""Stage 3: gradient-boosting classifiers, Optuna tuning, the seen-class evaluation, and the prediction files of the evaluation splits."""

import gc
import json
import os
import time
from dataclasses import asdict, dataclass

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from idsvae import config, files, metrics
from idsvae.data import splits
from idsvae.framework import features, vae

PROBABILITY_PREFIX = "p_"
# LightGBM and XGBoost results depend on the thread count; the stored runs used 16 threads, one per logical CPU of the run machine.
BOOSTING_ABLATION_THREADS = 16


@dataclass
class ClassifySettings:
    model: str = "catboost"
    view: str = "latent_aug"
    vae_run: str = ""
    feature_count: int = 0
    trials: int = 20
    folds: int = 3
    tuning_fraction: float = 0.1
    tuning_min_class_rows: int = 2000
    # An attack class that no fitted stage sees; the run tests if the classifier flags its rows as attacks (leave-one-attack-out).
    left_out_class: str = ""
    seed: int = 0

    def __post_init__(self):
        if self.folds < 2:
            raise ValueError(f"folds must be at least 2, got {self.folds}")
        if not 0.0 < self.tuning_fraction <= 1.0:
            raise ValueError(f"tuning_fraction must be in (0, 1], got {self.tuning_fraction}")

    def run_name(self):
        source = self.vae_run if self.view != "raw" else f"raw_k{self.feature_count}"
        name = f"{self.model}_{self.view}_{source}_s{self.seed}"
        if not self.left_out_class:
            return name
        # The VAE run name also carries the left-out class, but a raw run has no VAE run, so the classifier name keeps its own tag.
        return f"{name}_loo-{features.class_slug(self.left_out_class)}"


def view_columns(view, feature_count, ranking_dir, latent_dim):
    latent = vae.latent_columns(latent_dim)
    if view == "latent":
        return latent
    if view == "latent_aug":
        return latent + ["recon_nll", "kl"]
    if view == "raw":
        return features.top_features(features.load_ranking(ranking_dir), feature_count)
    if view == "concat":
        return features.top_features(features.load_ranking(ranking_dir), feature_count) + latent + ["recon_nll", "kl"]
    raise ValueError(f"unknown view: {view}")


def load_view(settings, split, processed_dir, stage2_root, columns):
    if settings.view == "raw":
        return splits.load_split(split, processed_dir, columns=columns)
    if settings.view != "concat":
        return splits.load_split(split, stage2_root / settings.vae_run, columns=columns)
    encoded = splits.load_split(split, stage2_root / settings.vae_run)
    raw_columns = [column for column in columns if column not in encoded.columns]
    raw = splits.load_split(split, processed_dir, columns=raw_columns)
    # The VAE encodes each split in its stored order, so both files hold the same flows row by row.
    if not all(np.array_equal(raw[column].to_numpy(), encoded[column].to_numpy()) for column in ("label", "time")):
        raise ValueError(f"{split}: the raw split and the VAE encodings of {settings.vae_run} hold different rows")
    return pd.concat([raw[raw_columns], encoded], axis=1)


def suggest_params(model_name, trial):
    if model_name == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 200, 1000),
            # Depths 9-10 gave no better time-blocked CV macro-F1 than depth 8 on the real tuning rows but cost 4-5x the fit time; CatBoost docs give 4-10 as the usual optimum.
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
        }
    if model_name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 200, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }
    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }
    raise ValueError(f"unknown model: {model_name}")


def training_thread_count(model_name, threads):
    if model_name != "catboost":
        return BOOSTING_ABLATION_THREADS
    if threads > 0:
        return threads
    # thread_count=-1 starts one thread per logical CPU.
    return os.cpu_count()


def build_model(model_name, params, seed, threads=-1):
    if model_name == "catboost":
        # CatBoost documents that thread_count does not change results; allow_writing_files=False keeps parallel runs from sharing catboost_info/.
        return CatBoostClassifier(loss_function="MultiClass", boosting_type="Plain", random_seed=seed, thread_count=threads, allow_writing_files=False, verbose=0, **params)
    # Known limit: LightGBM picks its row-wise or column-wise histogram mode by a timing test, so a run on other hardware can differ.
    if model_name == "lightgbm":
        return LGBMClassifier(objective="multiclass", random_state=seed, n_jobs=BOOSTING_ABLATION_THREADS, verbose=-1, **params)
    if model_name == "xgboost":
        return XGBClassifier(objective="multi:softprob", tree_method="hist", random_state=seed, n_jobs=BOOSTING_ABLATION_THREADS, **params)
    raise ValueError(f"unknown model: {model_name}")


def fit_model(model_name, params, seed, inputs, targets, threads=-1):
    model = build_model(model_name, params, seed, threads)
    # Balanced weights w_i = N / (C * n_i), Eq. 13 of the paper.
    model.fit(inputs, targets, sample_weight=compute_sample_weight("balanced", targets))
    return model


def predict_probabilities(model_name, model, inputs, threads):
    if model_name == "catboost":
        return model.predict_proba(inputs, thread_count=threads)
    return model.predict_proba(inputs)


def tuning_rows(frame, fraction, min_class_rows, seed):
    rng = np.random.default_rng(seed)
    selected = (frame.groupby("label")["label"].transform("size") < min_class_rows).to_numpy(copy=True)
    for positions in frame.groupby(["day", "label"]).indices.values():
        if selected[positions[0]]:
            continue
        selected[rng.choice(positions, size=round(fraction * len(positions)), replace=False)] = True
    return np.flatnonzero(selected)


def time_block_folds(frame, fold_count):
    # Contiguous time blocks make each validation fold a separate time span, as the chronological hold-out is.
    by_group = frame.groupby(["day", "label"])["time"]
    position = by_group.rank(method="first").to_numpy(dtype="int64") - 1
    group_size = by_group.transform("size").to_numpy()
    return (position * fold_count) // group_size


def print_trial(study, trial):
    # Optuna logs only warnings here, and one tuning run can take an hour, so this line shows that the run progresses.
    print(f"trial {trial.number + 1}: cv macro-F1 {trial.value:.4f}, best {study.best_value:.4f}", flush=True)


def tune(settings, inputs, targets, fold_ids, threads=-1):
    folds_per_class = pd.Series(fold_ids).groupby(targets).nunique()
    single_fold_classes = folds_per_class.index[folds_per_class < 2].tolist()
    if single_fold_classes:
        # A class in one fold only is absent from that fold's training rows, and the class columns of predict_proba then shift.
        raise ValueError(f"classes {single_fold_classes} lie in one fold only; tuning needs each class in two or more folds")
    folds = [(np.flatnonzero(fold_ids != fold), np.flatnonzero(fold_ids == fold)) for fold in range(settings.folds)]

    def objective(trial):
        params = suggest_params(settings.model, trial)
        scores = []
        for train_index, valid_index in folds:
            model = fit_model(settings.model, params, settings.seed, inputs[train_index], targets[train_index], threads)
            predicted = predict_probabilities(settings.model, model, inputs[valid_index], threads).argmax(axis=1)
            scores.append(f1_score(targets[valid_index], predicted, average="macro"))
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=settings.seed))
    study.optimize(objective, n_trials=settings.trials, callbacks=[print_trial])
    return study.best_params, float(study.best_value)


def tuned_params(settings, train_frame, inputs, targets, threads=-1):
    rows = tuning_rows(train_frame, settings.tuning_fraction, settings.tuning_min_class_rows, settings.seed)
    fold_ids = time_block_folds(train_frame.iloc[rows], settings.folds)
    best_params, best_cv_macro_f1 = tune(settings, inputs[rows], targets[rows], fold_ids, threads)
    return {"best_params": best_params, "best_cv_macro_f1": best_cv_macro_f1, "rows": len(rows)}


def save_model(model_name, model, path):
    if model_name == "lightgbm":
        model.booster_.save_model(str(path))
    else:
        model.save_model(str(path))
    return path.stat().st_size


def load_probability_function(model_name, path, threads):
    if model_name == "catboost":
        catboost_model = CatBoostClassifier()
        catboost_model.load_model(str(path))
        return lambda inputs: catboost_model.predict_proba(inputs, thread_count=threads)
    if model_name == "lightgbm":
        booster = lgb.Booster(model_file=str(path))
        return lambda inputs: booster.predict(inputs, num_threads=threads)
    if model_name == "xgboost":
        xgboost_model = XGBClassifier(n_jobs=threads)
        xgboost_model.load_model(str(path))
        return xgboost_model.predict_proba
    raise ValueError(f"unknown model: {model_name}")


def predicted_labels(probabilities, classes):
    return np.asarray(classes)[probabilities.argmax(axis=1)]


def prediction_file_name(part):
    return f"predictions_{part}.parquet"


def save_predictions(output_dir, part, frame, probabilities, classes, split_rows=None):
    """Write one row per row of the frame, in the frame order: label, metadata, predicted class, attack score and class probabilities."""
    columns = {column: frame[column].to_numpy() for column in splits.METADATA_COLUMNS if column in frame.columns}
    if split_rows is not None:
        columns["split_row"] = split_rows
    columns["predicted"] = predicted_labels(probabilities, classes)
    # 1 - P(Benign) in float32 keeps the order of rows with P(Benign) near 1, which a float32 P(Benign) rounds to ties.
    columns["attack_score"] = attack_score(probabilities, classes).astype("float32")
    for index, label in enumerate(classes):
        columns[f"{PROBABILITY_PREFIX}{label}"] = probabilities[:, index].astype("float32")
    file_name = prediction_file_name(part)
    files.write_parquet_atomic(pd.DataFrame(columns), output_dir / file_name, index=False)
    return file_name


def attack_score(probabilities, classes):
    """1 - P(Benign), which ranks the rows for the AUCs."""
    return 1.0 - probabilities[:, classes.index(config.BENIGN)]


def require_inputs_without_left_out_class(settings, ranking_dir, vae_settings):
    # The ranking and the VAE are fitted stages too, so a leave-one-out run must use the ones fitted without the left-out class.
    if settings.view == "raw":
        ranking_class = features.load_ranking(ranking_dir)["settings"]["left_out_class"]
        if ranking_class != settings.left_out_class:
            raise ValueError(f"ranking in {ranking_dir} leaves out {ranking_class!r}, this run leaves out {settings.left_out_class!r}")
        return
    if vae_settings["left_out_class"] != settings.left_out_class:
        raise ValueError(f"VAE run {settings.vae_run} leaves out {vae_settings['left_out_class']!r}, this run leaves out {settings.left_out_class!r}")
    if not settings.left_out_class:
        return
    expected = features.feature_set_name(features.SelectionSettings(left_out_class=settings.left_out_class))
    if vae_settings["feature_set"] != expected:
        raise ValueError(f"VAE run {settings.vae_run} has feature_set {vae_settings['feature_set']!r}, this run needs {expected!r}")


def run_classifier(settings, processed_dir, ranking_dir, stage2_root, output_root, threads=-1):
    if settings.view == "raw" and settings.feature_count == 0:
        settings.feature_count = features.load_ranking(ranking_dir)["selected_count"]
    vae_settings = {"latent_dim": 0}
    if settings.view != "raw":
        vae_settings = json.loads((stage2_root / settings.vae_run / "summary.json").read_text())["settings"]
    require_inputs_without_left_out_class(settings, ranking_dir, vae_settings)
    columns = view_columns(settings.view, settings.feature_count, ranking_dir, vae_settings["latent_dim"])

    def model_inputs(frame):
        return frame[columns].to_numpy(dtype="float32")

    def split_probabilities(frame):
        return predict_probabilities(settings.model, model, model_inputs(frame), threads)

    train_frame = load_view(settings, "fit_train", processed_dir, stage2_root, columns)
    if settings.left_out_class:
        train_left_out = features.left_out_mask(train_frame, settings.left_out_class)
        # No fitted stage sees the left-out class, so its rows in both fit splits are test rows.
        train_left_out_rows = np.flatnonzero(train_left_out)
        train_left_out_frame = train_frame.iloc[train_left_out_rows].copy()
        train_frame = train_frame[~train_left_out].reset_index(drop=True)
    classes = sorted(train_frame["label"].unique())
    class_index = {label: index for index, label in enumerate(classes)}
    inputs = model_inputs(train_frame)
    targets = train_frame["label"].map(class_index).to_numpy()
    train_rows = len(targets)

    tuning_start = time.perf_counter()
    tuning = tuned_params(settings, train_frame, inputs, targets, threads)
    tuning_seconds = time.perf_counter() - tuning_start
    fit_start = time.perf_counter()
    model = fit_model(settings.model, tuning["best_params"], settings.seed, inputs, targets, threads)
    final_fit_seconds = time.perf_counter() - fit_start
    # The evaluation splits load after the training data is freed, so the peak memory of a run is lower.
    del train_frame, inputs, targets
    gc.collect()

    output_dir = output_root / settings.run_name()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_bytes = save_model(settings.model, model, output_dir / "model.bin")

    # The prediction files let figures, paired bootstraps and per-class analyses run without new inference; all but fit_train_left_out cover a whole split in split order.
    holdout = load_view(settings, "fit_holdout", processed_dir, stage2_root, columns)
    holdout_labels = holdout["label"].to_numpy()
    holdout_seen = np.ones(len(holdout_labels), dtype=bool)
    if settings.left_out_class:
        holdout_seen = holdout_labels != settings.left_out_class
    holdout_probabilities = split_probabilities(holdout)
    prediction_files = {"fit_holdout": save_predictions(output_dir, "fit_holdout", holdout, holdout_probabilities, classes)}
    del holdout
    result = {
        "settings": asdict(settings),
        "columns": columns,
        "classes": classes,
        "tuning": tuning,
        "train_rows": train_rows,
        "training_seconds": {"tuning": tuning_seconds, "final_fit": final_fit_seconds, "threads": training_thread_count(settings.model, threads)},
        "model_file_bytes": model_bytes,
        "seen_multiclass": metrics.seen_multiclass_report(holdout_labels[holdout_seen], holdout_probabilities[holdout_seen], classes),
    }

    if settings.left_out_class:
        # Evaluation 2 (day 02-21) is not repeated: the leave-one-out classifiers only test the left-out class.
        train_left_out_probabilities = split_probabilities(train_left_out_frame)
        # Only the left-out rows of fit_train are test rows, so split_row gives their positions in the fit_train split.
        prediction_files["fit_train_left_out"] = save_predictions(output_dir, "fit_train_left_out", train_left_out_frame, train_left_out_probabilities, classes, split_rows=train_left_out_rows)
    else:
        for part in ("calib", "test"):
            frame = load_view(settings, part, processed_dir, stage2_root, columns)
            prediction_files[part] = save_predictions(output_dir, part, frame, split_probabilities(frame), classes)
            del frame

    result["prediction_files"] = prediction_files
    files.write_json_atomic(output_dir / "result.json", result)
    return output_dir, result
