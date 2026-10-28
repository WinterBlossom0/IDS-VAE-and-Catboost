"""Comparison methods of related work, rerun on this study's protocol: the Stage 1 features (in a leave-one-attack-out fold,
those of the fold's ranking), fit_train -> fit_holdout, seeds 3-5. The papers do not state their hyperparameters, so each
model keeps its library defaults except where a method below says otherwise; balanced class weights are used where the
library supports them, as in Stage 3.

- random_forest: Sakr et al., Computers 2026 (random forest).
- mlp: Ferreira and Antunes, RECPAD 2020 (back-propagation MLP; WEKA's default hidden layer of (attributes + classes) / 2 units,
  plus scikit-learn's early stopping; WEKA trains without it).
- xgboost, lightgbm: the gradient-boosting baselines of Stage 3.
- kang_idae_xg: Kang et al., J. Phys.: Conf. Ser. 2022: min-max inputs, random-forest feature ranking, affinity-propagation
  feature groups, one three-layer autoencoder per group trained on 75 % of the Benign flows, and XGBoost on the mean
  normalized RMSE of the other flows. The paper is a binary detector: one anomaly score cannot separate attack classes.
- The deep methods (TabM, FT-Transformer, CNN-1D, FlowTransformer, TabNet) are in deep_baselines.py; they train on the GPU
  under the shared lock.
"""
import argparse
import importlib
import json

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from sklearn.cluster import AffinityPropagation
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from idsvae import config, files, metrics
from idsvae.data import extra_days, splits
from idsvae.framework import features, vae

BASELINES_DIR = config.PROJECT_ROOT / "data" / "final" / "baselines"
SHALLOW_METHODS = ("random_forest", "mlp", "xgboost", "lightgbm", "kang_idae_xg")
DEEP_METHODS = ("tabm", "ft_transformer", "cnn1d", "flow_transformer", "tabnet")
METHODS = SHALLOW_METHODS + DEEP_METHODS
SEEDS = (3, 4, 5)
# The leave-one-attack-out folds of the framework (final_run.LEFT_OUT_CLASSES; a test checks that both agree).
LEFT_OUT_CLASSES = ("SSH-Bruteforce", "DoS attacks-GoldenEye", "DoS attacks-Slowloris")
# Kang et al. 2022 give no training settings for the autoencoders; 20 epochs, batch 4096 and Adam 1e-3 are our choice.
AUTOENCODER_EPOCHS = 20
AUTOENCODER_BATCH = 4096
# Kang et al. 2022, Sect. 2.2.2: the autoencoders train on 75 % of the normal rows; detection uses the other 25 %.
KANG_AUTOENCODER_SHARE = 0.75


def selected_features(left_out_class=""):
    """The Stage 1 features of the main ranking, or of the ranking without the left-out class of a fold."""
    ranking = features.load_ranking(features.ranking_dir_for(features.SelectionSettings(left_out_class=left_out_class)))
    return features.top_features(ranking, ranking["selected_count"])


class Standardizer:
    """Signed log, then standardization with the mean and standard deviation of the reference rows (all entries, unlike
    Stage 2, which standardizes on defined entries only)."""

    def __init__(self, reference):
        reference_transformed = vae.signed_log(reference)
        std = reference_transformed.std(axis=0)
        self.mean, self.std = reference_transformed.mean(axis=0), np.where(std > 0, std, 1.0)

    def transform(self, values):
        return (vae.signed_log(values) - self.mean) / self.std


def scaled(values, reference):
    return Standardizer(reference).transform(values)


def train_autoencoder(benign_rows, seed, threads):
    """A 3-layer autoencoder (input, hidden of half the width, input) trained on benign_rows."""
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    width = benign_rows.shape[1]
    model = torch.nn.Sequential(torch.nn.Linear(width, max(1, width // 2)), torch.nn.ReLU(), torch.nn.Linear(max(1, width // 2), width))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    benign = torch.from_numpy(benign_rows.astype("float32"))
    generator = torch.Generator().manual_seed(seed)
    for _ in range(AUTOENCODER_EPOCHS):
        for rows in torch.randperm(len(benign), generator=generator).split(AUTOENCODER_BATCH):
            loss = torch.nn.functional.mse_loss(model(benign[rows]), benign[rows])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def reconstruction_rmse(model, inputs):
    with torch.no_grad():
        values = torch.from_numpy(inputs.astype("float32"))
        return torch.sqrt(((model(values) - values) ** 2).mean(dim=1)).numpy()


class KangScorer:
    """Mean normalized autoencoder RMSE of Kang et al. 2022 on raw inputs: min-max scaling, then one autoencoder per group."""

    def __init__(self, minimum, safe_range):
        self.minimum, self.safe_range = minimum, safe_range
        self.groups = []

    def add_group(self, columns, model, low, spread):
        self.groups.append((columns, model, low, spread))

    def scaled(self, values):
        return (values - self.minimum) / self.safe_range

    def score_of_scaled(self, scaled_values):
        return np.mean([(reconstruction_rmse(model, scaled_values[:, columns]) - low) / spread for columns, model, low, spread in self.groups], axis=0)[:, None]


def kang_inputs(train_values, train_y, other_values, seed, threads):
    """(classifier-row mask of fit_train, mean normalized RMSE of those rows, of the other rows); Kang et al. 2022.

    Sect. 2.2.1: min-max inputs in [0, 1]. Sect. 2.1: random-forest ranking, affinity-propagation groups, one autoencoder
    per group on normal rows. Sect. 2.2.2: the classifier sees 25 % of the normal rows that the autoencoders did not train
    on, plus all attack rows. Tables 1-2: each autoencoder's RMSE is normalized, then the mean goes to XGBoost.
    """
    minimum = train_values.min(axis=0)
    value_range = train_values.max(axis=0) - minimum
    scorer = KangScorer(minimum, np.where(value_range > 0, value_range, 1.0))
    train_x, other_x = (scorer.scaled(values) for values in (train_values, other_values))
    ranking = RandomForestClassifier(n_estimators=100, n_jobs=threads, random_state=seed, class_weight="balanced").fit(train_x, train_y)
    # The paper gives neither the number of "best features" nor the autoencoder widths; half of the features is our choice.
    best = np.argsort(ranking.feature_importances_)[::-1][: train_x.shape[1] // 2]
    correlation = np.nan_to_num(np.corrcoef(train_x[:, best].T))
    groups = AffinityPropagation(affinity="precomputed", random_state=seed).fit_predict(correlation)
    benign = train_y == config.BENIGN
    autoencoder_rows = benign & (np.random.default_rng(seed).random(len(train_y)) < KANG_AUTOENCODER_SHARE)
    classifier_rows = ~autoencoder_rows
    for group in np.unique(groups):
        columns = best[groups == group]
        model = train_autoencoder(train_x[autoencoder_rows][:, columns], seed, threads)
        reference = reconstruction_rmse(model, train_x[autoencoder_rows][:, columns])
        scorer.add_group(columns, model, reference.min(), max(float(reference.max() - reference.min()), 1e-12))
    return classifier_rows, scorer.score_of_scaled(train_x[classifier_rows]), scorer.score_of_scaled(other_x), scorer


class DeployedShallow:
    """A shallow method in its deployment form: raw flow features -> class probabilities (scaling included)."""

    def __init__(self, standardizer, model):
        self.standardizer, self.model = standardizer, model

    def single_thread(self):
        if "n_jobs" in self.model.get_params():
            self.model.set_params(n_jobs=1)

    def predict_proba(self, values):
        values = values if self.standardizer is None else self.standardizer.transform(values)
        return self.model.predict_proba(values)


class DeployedKang(DeployedShallow):
    def __init__(self, scorer, model):
        super().__init__(None, model)
        self.scorer = scorer

    def single_thread(self):
        super().single_thread()
        torch.set_num_threads(1)

    def predict_proba(self, values):
        return self.model.predict_proba(self.scorer.score_of_scaled(self.scorer.scaled(values)))


def fit_shallow(method, seed, threads, train_x, train_y, targets, evaluation_x, class_count):
    """(probabilities for each matrix of evaluation_x, deployed model) from one fit of a scikit-learn or boosting method;
    Kang IDAE-XG takes raw values, the others values scaled with the Standardizer of train_x."""
    weights = compute_sample_weight("balanced", targets)
    if method == "kang_idae_xg":
        sizes = np.cumsum([len(values) for values in evaluation_x])[:-1]
        classifier_rows, classifier_x, other_x, scorer = kang_inputs(train_x, train_y, np.concatenate(evaluation_x), seed, threads)
        model = XGBClassifier(objective="multi:softprob", tree_method="hist", n_jobs=threads, random_state=seed)
        model.fit(classifier_x, targets[classifier_rows], sample_weight=compute_sample_weight("balanced", targets[classifier_rows]))
        return [model.predict_proba(values) for values in np.split(other_x, sizes)], DeployedKang(scorer, model)
    standardizer = Standardizer(train_x)
    train_x = standardizer.transform(train_x)
    evaluation_x = [standardizer.transform(values) for values in evaluation_x]
    if method == "random_forest":
        model = RandomForestClassifier(n_jobs=threads, random_state=seed, class_weight="balanced").fit(train_x, targets)
    elif method == "mlp":
        hidden = (train_x.shape[1] + class_count) // 2
        model = MLPClassifier(hidden_layer_sizes=(hidden,), early_stopping=True, random_state=seed).fit(train_x, targets, sample_weight=weights)
    elif method == "xgboost":
        model = XGBClassifier(objective="multi:softprob", tree_method="hist", n_jobs=threads, random_state=seed).fit(train_x, targets, sample_weight=weights)
    else:
        model = LGBMClassifier(objective="multiclass", n_jobs=threads, random_state=seed, verbose=-1).fit(train_x, targets, sample_weight=weights)
    return [model.predict_proba(values) for values in evaluation_x], DeployedShallow(standardizer, model)


def held_out_day_parts(columns):
    """(day, float32 values, frame of label and time) of each day that the unseen-attack detector holds out in turn."""
    # Imported here: the detector module loads CatBoost and the VAE code, which the shallow methods do not need.
    from idsvae.experiments import detector_runs
    for day in detector_runs.qualified_days():
        if day == "02-28":
            frame = splits.load_split("vae_val", columns=columns)
        else:
            frame = pd.read_parquet(extra_days.OUTPUT_DIR / f"{day}.parquet", columns=[*columns, "label", "time"])
        # float32 halves the RAM of these millions of rows; the hold-out and test-day parts stay float64.
        yield day, frame[columns].to_numpy("float32"), frame[["label", "time"]].reset_index(drop=True)


def predict_parts(method, seed, threads, train, columns, evaluation):
    """(classes, probabilities for each (values, frame) part of evaluation, method details, deployed model) from one model
    trained on train."""
    train_y = train["label"].to_numpy()
    train_values = train[columns].to_numpy("float64")
    classes = sorted(np.unique(train_y))
    class_index = {label: index for index, label in enumerate(classes)}
    targets = np.array([class_index[label] for label in train_y])
    if method in DEEP_METHODS:
        # Imported here: deep_baselines imports this module for scaled, and the GPU packages load only for the deep methods.
        from idsvae.experiments import deep_baselines
        probabilities, details, deployed = deep_baselines.run_on_gpu(method, seed, train_values, targets, train, evaluation, len(classes), lambda line: print(line, flush=True))
        return classes, probabilities, details, deployed
    probabilities, deployed = fit_shallow(method, seed, threads, train_values, train_y, targets, [values for values, _ in evaluation], len(classes))
    return classes, probabilities, {}, deployed


def run(method, seed, threads):
    columns = selected_features()
    train = splits.load_split("fit_train", columns=columns)
    holdout = splits.load_split("fit_holdout", columns=columns)
    # The test day (02-21, unseen classes) is predicted by the same model; its Benign burst flag splits the false alarms.
    test = splits.load_split("test", columns=[*columns, "in_benign_burst"])
    days = list(held_out_day_parts(columns))
    # Hold-out, test day, then the held-out days of the detector: one trained model predicts every part.
    evaluation = [(frame[columns].to_numpy("float64"), frame) for frame in (holdout, test)] + [(values, frame) for _, values, frame in days]
    classes, (probabilities, test_probabilities, *day_probabilities), details, _ = predict_parts(method, seed, threads, train, columns, evaluation)
    report = metrics.seen_multiclass_report(holdout["label"].to_numpy(), probabilities, classes)
    test_predicted = np.asarray(classes)[test_probabilities.argmax(axis=1)]
    return {
        "method": method, "seed": seed, "macro_f1_min_support": report["macro_f1_min_support"]["value"],
        "balanced_accuracy": report["balanced_accuracy"], "per_class_f1": {label: values["f1"] for label, values in report["per_class"].items()},
        "test": metrics.other_day_report(test["label"].to_numpy(), test_predicted, test["in_benign_burst"].to_numpy(dtype=bool)),
        # No held-out day has a Benign burst, so benign_fpr_outside_burst is the Benign false-positive rate of the day.
        "held_out_days": {day: metrics.other_day_report(frame["label"].to_numpy(), np.asarray(classes)[day_values.argmax(axis=1)], np.zeros(len(frame), dtype=bool))
                          for (day, _, frame), day_values in zip(days, day_probabilities)},
        **details,
    }


def run_fold(method, seed, threads, left_out_class):
    """Leave-one-attack-out fold, as for the framework: the model trains without the class on the fold's own Stage 1
    features; the report gives the share of the class flows of days 02-14 and 02-15 predicted as any attack and the hold-out
    Benign false-positive rate."""
    columns = selected_features(left_out_class)
    train_all = splits.load_split("fit_train", columns=columns)
    holdout = splits.load_split("fit_holdout", columns=columns)
    is_left_out = features.left_out_mask(train_all, left_out_class)
    train, left_out_rows = train_all[~is_left_out].reset_index(drop=True), train_all[is_left_out].reset_index(drop=True)
    evaluation = [(frame[columns].to_numpy("float64"), frame) for frame in (left_out_rows, holdout)]
    classes, (left_out_probabilities, holdout_probabilities), details, _ = predict_parts(method, seed, threads, train, columns, evaluation)
    left_out_attack = np.asarray(classes)[left_out_probabilities.argmax(axis=1)] != config.BENIGN
    holdout_labels = holdout["label"].to_numpy()
    holdout_attack = np.asarray(classes)[holdout_probabilities.argmax(axis=1)] != config.BENIGN
    detected = np.concatenate([left_out_attack, holdout_attack[holdout_labels == left_out_class]])
    return {"method": method, "seed": seed, "left_out_class": left_out_class, "feature_count": len(columns),
            "left_out_detection": float(detected.mean()), "holdout_benign_false_alarms": float(holdout_attack[holdout_labels == config.BENIGN].mean()),
            **details}


def model_path(method, seed):
    return BASELINES_DIR / "models" / f"{method}_s{seed}.pt"


def save_deployed_model(method, seed, threads):
    """Train the main run of a method again with the hold-out as the only evaluation part and save its deployment form for
    CPU timing; the hold-out F1 must equal that of the stored main run, so the saved model is the evaluated one."""
    if __name__ == "__main__":
        raise RuntimeError("a pickle written from __main__ names its classes __main__.*, which other programs cannot load")
    stored = json.loads(output_path(method, seed).read_text())
    columns = selected_features()
    train = splits.load_split("fit_train", columns=columns)
    holdout = splits.load_split("fit_holdout", columns=columns)
    classes, (probabilities,), _, deployed = predict_parts(method, seed, threads, train, columns, [(holdout[columns].to_numpy("float64"), holdout)])
    score = metrics.seen_multiclass_report(holdout["label"].to_numpy(), probabilities, classes)["macro_f1_min_support"]["value"]
    if score != stored["macro_f1_min_support"]:
        raise ValueError(f"{method} seed {seed}: hold-out F1 {score} differs from the stored main run ({stored['macro_f1_min_support']})")
    path = model_path(method, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with files.atomic_path(path) as temporary:
        torch.save({"classes": classes, "columns": columns, "deployed": deployed}, temporary)


def load_deployed_model(method, seed):
    """(classes, input columns, deployed model) saved by save_deployed_model; the file holds pickled objects of this package."""
    # TabNet keeps its optimizer state and early-stopping copy on the training device; prediction needs neither, nor a GPU.
    saved = torch.load(model_path(method, seed), weights_only=False, map_location="cpu")
    return saved["classes"], saved["columns"], saved["deployed"]


def output_path(method, seed, left_out_class=""):
    suffix = f"_loo_{features.class_slug(left_out_class)}" if left_out_class else ""
    return BASELINES_DIR / f"{method}_s{seed}{suffix}.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--left-out-class", default="", choices=("", *LEFT_OUT_CLASSES), help="leave-one-attack-out fold")
    parser.add_argument("--save-model", action="store_true", help="retrain the main run and save its deployment form for CPU timing")
    args = parser.parse_args()
    if args.save_model:
        if not model_path(args.method, args.seed).exists():
            save_deployed_model(args.method, args.seed, args.threads)
        return
    output = output_path(args.method, args.seed, args.left_out_class)
    if output.exists():
        return
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    result = run_fold(args.method, args.seed, args.threads, args.left_out_class) if args.left_out_class else run(args.method, args.seed, args.threads)
    files.write_json_atomic(output, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    # The importable module runs the CLI, so that saved models name their classes idsvae.experiments.baselines.*.
    importlib.import_module("idsvae.experiments.baselines").main()
