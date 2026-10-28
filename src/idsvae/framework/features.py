"""Stage 1: rank features by mean |SHAP| over CatBoost runs on random feature subsets."""

import argparse
import json
import re
from dataclasses import asdict, dataclass

import numpy as np
from catboost import CatBoostClassifier, Pool
from scipy.stats import spearmanr

from idsvae import config, files
from idsvae.data import splits

STAGE1_DIR = config.PROJECT_ROOT / "data" / "stage1"
STAGE1_WITH_FINGERPRINTS_DIR = config.PROJECT_ROOT / "data" / "stage1_with_fingerprints"
STAGE1_LEAVE_ONE_OUT_ROOT = config.PROJECT_ROOT / "data" / "stage1_loo"


@dataclass
class SelectionSettings:
    runs: int = 20
    feature_fraction: float = 0.7
    rows_per_class: int = 50_000
    shap_rows_per_class: int = 1_000
    iterations: int = 300
    depth: int = 6
    learning_rate: float = 0.1
    cumulative_share: float = 0.95
    full_feature_baseline_runs: int = 20
    include_fingerprints: bool = False
    # An attack class whose rows Stage 1 must not see (leave-one-attack-out evaluation); empty for the main runs.
    left_out_class: str = ""
    seed: int = 0

    def __post_init__(self):
        if self.left_out_class and self.include_fingerprints:
            raise ValueError("a leave-one-out ranking uses the main feature set, without the fingerprint columns")


def class_slug(label):
    return re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")


def ranking_dir_for(selection):
    if selection.left_out_class:
        return STAGE1_LEAVE_ONE_OUT_ROOT / class_slug(selection.left_out_class)
    return STAGE1_WITH_FINGERPRINTS_DIR if selection.include_fingerprints else STAGE1_DIR


def feature_set_name(selection):
    """Value of VaeSettings.feature_set for a VAE on this ranking."""
    if selection.left_out_class:
        return f"loo_{class_slug(selection.left_out_class)}"
    return "fingerprints" if selection.include_fingerprints else "main"


def require_attack_class(left_out_class):
    if left_out_class == config.BENIGN:
        raise ValueError("the left-out class must be an attack class, not Benign")


def left_out_mask(frame, left_out_class):
    require_attack_class(left_out_class)
    mask = (frame["label"] == left_out_class).to_numpy()
    if not mask.any():
        raise ValueError(f"left-out class {left_out_class!r} has no rows in the fit data")
    return mask


def stratified_sample(frame, rows_per_class, rng):
    parts = [group.sample(min(len(group), rows_per_class), random_state=int(rng.integers(2**31))) for _, group in frame.groupby("label")]
    return frame.loc[np.concatenate([part.index.to_numpy() for part in parts])]


def mean_abs_shap(frame, columns, settings, rng, seed, threads=-1):
    fit_rows = stratified_sample(frame, settings.rows_per_class, rng)
    model = CatBoostClassifier(
        loss_function="MultiClass",
        iterations=settings.iterations,
        depth=settings.depth,
        learning_rate=settings.learning_rate,
        auto_class_weights="Balanced",
        random_seed=seed,
        # CatBoost documents that thread_count does not change results; allow_writing_files=False keeps parallel runs from sharing catboost_info/.
        thread_count=threads,
        allow_writing_files=False,
        verbose=0,
    )
    model.fit(fit_rows[columns], fit_rows["label"])
    shap_rows = stratified_sample(frame, settings.shap_rows_per_class, rng)
    # Multiclass ShapValues have shape (rows, classes, features + 1); the last entry is the expected value.
    shap = model.get_feature_importance(Pool(shap_rows[columns], shap_rows["label"]), type="ShapValues", thread_count=threads)
    return np.abs(shap[:, :, :-1]).mean(axis=(0, 1))


# A draw that misses a feature in a half is drawn again; the cap stops the loop at settings where a cover is possible but rare.
MAX_SUBSET_DRAWS = 1000


def draw_subsets(feature_count, settings, rng):
    subset_size = round(settings.feature_fraction * feature_count)
    half = settings.runs // 2
    if settings.runs < 2:
        raise ValueError(f"the split-half stability report needs at least 2 runs, got {settings.runs}")
    if half * subset_size < feature_count:
        raise ValueError(f"{half} runs of {subset_size} features cannot contain all {feature_count} features in each half")
    for _ in range(MAX_SUBSET_DRAWS):
        masks = np.zeros((settings.runs, feature_count), dtype=bool)
        for run in range(settings.runs):
            masks[run, rng.choice(feature_count, subset_size, replace=False)] = True
        # split_half_stability compares runs [0, half) with runs [half, 2 half), so each of these halves must contain each feature.
        if masks[:half].any(axis=0).all() and masks[half : 2 * half].any(axis=0).all():
            return masks
    raise ValueError(f"no draw of {MAX_SUBSET_DRAWS} contained all {feature_count} features in each half; increase runs or feature_fraction")


def subset_importances(frame, feature_columns, settings, rng, threads=-1):
    masks = draw_subsets(len(feature_columns), settings, rng)
    importances = np.full(masks.shape, np.nan)
    for run in range(settings.runs):
        chosen = np.flatnonzero(masks[run])
        columns = [feature_columns[index] for index in chosen]
        importances[run, chosen] = mean_abs_shap(frame, columns, settings, rng, settings.seed + run, threads)
        print(f"subset run {run + 1}/{settings.runs} done", flush=True)
    return importances


def aggregate(importances):
    # A run that excludes a feature gives no evidence about the feature, so the mean skips that run.
    return np.nanmean(importances, axis=0)


def ranking_order(scores):
    return np.argsort(-scores, kind="stable")


def cumulative_cut(sorted_scores, share):
    cumulative = np.cumsum(sorted_scores) / sorted_scores.sum()
    return int(np.searchsorted(cumulative, share) + 1)


def top_set(scores, size):
    return set(ranking_order(scores)[:size].tolist())


def jaccard(first, second):
    return len(first & second) / len(first | second)


def common_spearman(first, second):
    common = np.isfinite(first) & np.isfinite(second)
    return float(spearmanr(first[common], second[common]).statistic)


def common_top_k_jaccard(first, second, size):
    common = np.flatnonzero(np.isfinite(first) & np.isfinite(second))
    # A pair of subset runs shares only part of the features; the same top fraction keeps the overlap from being 1 by construction.
    scaled = max(1, round(size * len(common) / len(first)))
    return jaccard(top_set(first[common], scaled), top_set(second[common], scaled))


def mean_pairwise_stability(importances, size):
    """Mean over all run pairs, each pair compared on the features that both runs contain."""
    pairs = [(first, second) for first in range(len(importances)) for second in range(first + 1, len(importances))]
    return {
        "mean_spearman": float(np.mean([common_spearman(importances[first], importances[second]) for first, second in pairs])),
        "mean_top_k_jaccard": float(np.mean([common_top_k_jaccard(importances[first], importances[second], size) for first, second in pairs])),
    }


def split_half_stability(importances, half, size):
    first_half, second_half = aggregate(importances[:half]), aggregate(importances[half : 2 * half])
    return {
        "spearman": common_spearman(first_half, second_half),
        "top_k_jaccard": common_top_k_jaccard(first_half, second_half, size),
    }


def ranking_stability(importances, half, size):
    return {
        "runs": len(importances),
        "runs_per_half": half,
        "aggregated_halves": split_half_stability(importances, half, size),
        "single_runs": mean_pairwise_stability(importances, size),
    }


def stability_report(importances, baseline, size):
    # Both rankings use the same number of runs per half, so the comparison is like for like.
    half = min(len(importances), len(baseline)) // 2
    return {
        "top_k": size,
        "subset_ranking": ranking_stability(importances, half, size),
        "full_feature_ranking": ranking_stability(baseline, half, size),
    }


def save_array_atomic(path, array):
    # np.save adds ".npy" to a file name without that suffix, so it writes into an open file of the temporary path.
    with files.atomic_path(path) as temporary, open(temporary, "wb") as handle:
        np.save(handle, array)


def select_features(settings, processed_dir=config.PROCESSED_DIR, output_dir=None, threads=-1):
    if output_dir is None:
        output_dir = ranking_dir_for(settings)
    manifest = splits.load_manifest(processed_dir)
    feature_columns = manifest["feature_columns"] + (manifest["fingerprint_columns"] if settings.include_fingerprints else [])
    frame = splits.load_split("fit_train", processed_dir, columns=feature_columns)
    if settings.left_out_class:
        frame = frame[~left_out_mask(frame, settings.left_out_class)]
    rng = np.random.default_rng(settings.seed)

    importances = subset_importances(frame, feature_columns, settings, rng, threads)
    scores = aggregate(importances)
    order = ranking_order(scores)
    selected_count = cumulative_cut(scores[order], settings.cumulative_share)

    baseline = np.array([mean_abs_shap(frame, feature_columns, settings, rng, settings.seed + 10_000 + run, threads) for run in range(settings.full_feature_baseline_runs)])

    result = {
        "settings": asdict(settings),
        "ranking": [{"feature": feature_columns[index], "mean_abs_shap": float(scores[index]), "runs_with_feature": int(np.isfinite(importances[:, index]).sum())} for index in order],
        "selected_count": selected_count,
        "selected_features": [feature_columns[index] for index in order[:selected_count]],
        "stability": stability_report(importances, baseline, selected_count),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    save_array_atomic(output_dir / "subset_importances.npy", importances)
    save_array_atomic(output_dir / "full_feature_baseline_importances.npy", baseline)
    files.write_json_atomic(output_dir / "feature_ranking.json", result)
    return result


def top_features(ranking, count):
    """The first count features of a stored ranking."""
    if count > len(ranking["ranking"]):
        raise ValueError(f"feature_count {count} exceeds the {len(ranking['ranking'])} ranked features")
    return [entry["feature"] for entry in ranking["ranking"][:count]]


def load_ranking(input_dir=STAGE1_DIR):
    return json.loads((input_dir / "feature_ranking.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=SelectionSettings.runs)
    parser.add_argument("--rows-per-class", type=int, default=SelectionSettings.rows_per_class)
    parser.add_argument("--iterations", type=int, default=SelectionSettings.iterations)
    parser.add_argument("--baseline-runs", type=int, default=SelectionSettings.full_feature_baseline_runs)
    parser.add_argument("--include-fingerprints", action="store_true", help="rank the TCP fingerprint columns too (comparison arm)")
    parser.add_argument("--left-out-class", default="", help="attack class whose rows Stage 1 must not see (leave-one-attack-out)")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    settings = SelectionSettings(
        runs=args.runs,
        rows_per_class=args.rows_per_class,
        iterations=args.iterations,
        full_feature_baseline_runs=args.baseline_runs,
        include_fingerprints=args.include_fingerprints,
        left_out_class=args.left_out_class,
    )
    input_dir = config.project_path(args.input_dir, config.PROCESSED_DIR)
    output_dir = config.project_path(args.output_dir, None)
    result = select_features(settings, input_dir, output_dir)
    print(json.dumps({key: result[key] for key in ("selected_count", "selected_features", "stability")}, indent=1))


if __name__ == "__main__":
    main()
