"""Clean the CSE-CIC-IDS2018 day files and write one parquet file per split."""

import argparse
import contextlib
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from idsvae import config, files
from idsvae.data import download

# Columns of each split parquet that are not model inputs.
METADATA_COLUMNS = ("label", "day", "time", "in_benign_burst")
FEATURE_DTYPE = "float32"


@contextlib.contextmanager
def official_csv(file_name):
    """Source for pd.read_csv of one official CSV file: the local copy in config.RAW_DIR, otherwise a stream from the public dataset bucket."""
    path = config.RAW_DIR / file_name
    if not path.exists():
        with download.stream(file_name) as source:
            yield source
        return
    download.require_size(path, path.stat().st_size, config.SOURCE_FILE_BYTES[file_name])
    yield path


def read_day(day, sample_fraction=None):
    with official_csv(config.DAY_FILES[day]) as source:
        if sample_fraction is None:
            return pd.read_csv(source, low_memory=False)
        rng = np.random.default_rng(0)
        return pd.read_csv(source, low_memory=False, skiprows=lambda row: row > 0 and rng.random() > sample_fraction)


def parse_capture_time(timestamps):
    parsed = pd.to_datetime(timestamps, format=config.TIME_FORMAT)
    afternoon = parsed.dt.hour < config.CAPTURE_FIRST_HOUR
    return parsed.where(~afternoon, parsed + pd.Timedelta(hours=12))


def attack_windows(day_frame):
    attacks = day_frame.loc[day_frame["label"] != config.BENIGN]
    bounds = attacks.groupby("label")["time"].agg(["min", "max"])
    return {label: (start, end) for label, start, end in bounds.itertuples(name=None)}


def benign_rate_per_minute(benign_times, start, end):
    minutes = max((end - start).total_seconds() / 60.0, 1.0)
    return float(benign_times.between(start, end).sum()) / minutes


def top_template_share(rows):
    columns = [column for column in config.BURST_TEMPLATE_COLUMNS if column in rows.columns]
    if rows.empty or not columns:
        return 0.0
    return float(rows[columns].value_counts(normalize=True).iloc[0])


def union_minutes(intervals):
    """Length of the union of (start, end) intervals, so that overlapping windows count once."""
    total, current_start, current_end = 0.0, None, None
    for start, end in sorted(intervals):
        if current_end is None or start > current_end:
            if current_end is not None:
                total += (current_end - current_start).total_seconds() / 60.0
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    if current_end is not None:
        total += (current_end - current_start).total_seconds() / 60.0
    return total


def benign_burst_windows(day_frame, windows):
    """Mark each attack window whose benign rate is at least BENIGN_BURST_RATIO times the rate outside all windows and whose benign rows mostly share one template."""
    benign = day_frame.loc[day_frame["label"] == config.BENIGN]
    benign_times = benign["time"]
    outside = np.ones(len(benign_times), dtype=bool)
    for start, end in windows.values():
        outside &= ~benign_times.between(start, end).to_numpy()
    day_minutes = max((benign_times.max() - benign_times.min()).total_seconds() / 60.0, 1.0)
    outside_rate = float(outside.sum()) / max(day_minutes - union_minutes(windows.values()), 1.0)
    report = {}
    for label, (start, end) in windows.items():
        inside_rate = benign_rate_per_minute(benign_times, start, end)
        template_share = top_template_share(benign.loc[benign_times.between(start, end)])
        report[label] = {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "benign_per_minute_inside": inside_rate,
            "benign_per_minute_outside_windows": outside_rate,
            "benign_top_template_share_inside": template_share,
            "benign_burst": inside_rate >= config.BENIGN_BURST_RATIO * outside_rate and template_share >= config.BURST_TEMPLATE_SHARE,
        }
    return report


def benign_in_burst_window(day_frame, windows, burst_report):
    times = day_frame["time"]
    inside_burst = np.zeros(len(day_frame), dtype=bool)
    for label, (start, end) in windows.items():
        if burst_report[label]["benign_burst"]:
            inside_burst |= times.between(start, end).to_numpy()
    return inside_burst & (day_frame["label"].to_numpy() == config.BENIGN)


def clean_day(frame, day):
    stats = {"rows_raw": len(frame)}
    # Some official CSV files repeat the header line inside the data.
    header_rows = frame[config.LABEL_COLUMN] == config.LABEL_COLUMN
    frame = frame.loc[~header_rows]
    stats["header_rows"] = int(header_rows.sum())
    stats["raw_class_counts"] = {label: int(count) for label, count in frame[config.LABEL_COLUMN].value_counts().items()}

    time = parse_capture_time(frame[config.TIME_COLUMN])
    # Some rows carry dates outside the capture (1970 on 02-14), so their position in the day is unknown.
    capture_year = (time.dt.year == config.CAPTURE_YEAR).to_numpy()
    stats["rows_bad_timestamp"] = int((~capture_year).sum())
    frame = frame.loc[capture_year]
    time = time.loc[capture_year]

    labels = frame[config.LABEL_COLUMN].astype(str)
    features = frame.drop(columns=config.EXCLUDED_COLUMNS + [config.LABEL_COLUMN])
    # The cast comes before the finite check and all row hashes, so that each check sees the stored model-input values.
    features = features.apply(pd.to_numeric, errors="raise").astype(FEATURE_DTYPE)

    finite = np.isfinite(features.to_numpy()).all(axis=1)
    stats["rows_nonfinite"] = int((~finite).sum())
    cleaned = features.loc[finite].reset_index(drop=True)
    cleaned["label"] = labels.loc[finite].to_numpy()
    cleaned["day"] = day
    cleaned["time"] = time.loc[finite].to_numpy()

    windows = attack_windows(cleaned)
    stats["attack_windows"] = benign_burst_windows(cleaned, windows)
    cleaned["in_benign_burst"] = benign_in_burst_window(cleaned, windows, stats["attack_windows"])
    stats["rows_clean"] = len(cleaned)
    return cleaned, stats


def feature_columns_of(frame):
    return [column for column in frame.columns if column not in METADATA_COLUMNS]


def row_hashes(frame, columns):
    return pd.util.hash_pandas_object(frame[columns], index=False).to_numpy()


def stream_order(seconds, tie_key):
    """Row order by time; the timestamps have one-second resolution, so tie_key orders the flows of one second.

    tie_key must not depend on the split of a flow: an order by split would put every hold-out flow after all training
    flows of its second, and real traffic arrives mixed.
    """
    return np.lexsort((tie_key, seconds))


def remove_label_conflicts(frame, columns):
    """Return the kept frame, removed rows by class and, per class, the distinct removed feature vectors and how many of them equal a Benign vector."""
    hashes = row_hashes(frame, columns)
    labels = pd.Series(frame["label"].to_numpy())
    labels_per_row = labels.groupby(hashes).nunique()
    conflicted = labels_per_row.index[labels_per_row > 1]
    keep = ~np.isin(hashes, conflicted)
    removed = pd.DataFrame({"hash": hashes[~keep], "label": labels.to_numpy()[~keep]})
    benign_hashes = set(removed.loc[removed["label"] == config.BENIGN, "hash"])
    distinct_vectors = {}
    for label, group in removed.groupby("label"):
        vectors = set(group["hash"])
        distinct_vectors[label] = {"distinct_vectors": len(vectors), "distinct_vectors_equal_to_benign": len(vectors & benign_hashes)}
    return frame.loc[keep].reset_index(drop=True), class_counts(frame.loc[~keep]), distinct_vectors


def remove_duplicates(frame, columns):
    # The earliest copy stays so that the chronological hold-out sees each flow at its first time.
    frame = frame.sort_values("time", kind="stable").reset_index(drop=True)
    keep = ~pd.Series(row_hashes(frame, columns + ["label"])).duplicated().to_numpy()
    return frame.loc[keep].reset_index(drop=True), class_counts(frame.loc[~keep])


def remove_rows_in_fit(frame, columns, fit_hashes):
    overlap = np.isin(row_hashes(frame, columns), fit_hashes)
    return frame.loc[~overlap].reset_index(drop=True), class_counts(frame.loc[overlap])


def assign_seen_holdout(frame):
    order = frame.groupby(["day", "label"])["time"].rank(method="first", ascending=False)
    group_size = frame.groupby(["day", "label"])["time"].transform("size")
    return (order <= np.ceil(config.SEEN_HOLDOUT_FRACTION * group_size)).to_numpy()


def class_counts(frame):
    return {label: int(count) for label, count in frame["label"].value_counts().items()}


def split_summary(frame):
    return {"rows_final": len(frame), "class_counts": class_counts(frame), "benign_burst_rows": int(frame["in_benign_burst"].sum())}


def remove_conflicts_and_duplicates(frame, columns):
    record = {"rows_after_cleaning": len(frame)}
    frame, record["label_conflict_rows_removed_by_class"], record["label_conflict_distinct_vectors_by_class"] = remove_label_conflicts(frame, columns)
    frame, record["duplicate_rows_removed_by_class"] = remove_duplicates(frame, columns)
    record["duplicate_rows_removed"] = sum(record["duplicate_rows_removed_by_class"].values())
    return frame, record


def build_splits(sample_fraction=None, output_dir=config.PROCESSED_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"sample_fraction": sample_fraction, "days": {}, "splits": {}}

    split_frames = {}
    for split, days in config.SPLIT_DAYS.items():
        parts = []
        for day in days:
            cleaned, manifest["days"][day] = clean_day(read_day(day, sample_fraction), day)
            parts.append(cleaned)
        split_frames[split] = pd.concat(parts, ignore_index=True)

    fit = split_frames.pop("fit")
    all_columns = feature_columns_of(fit)
    # A column with one value on all fit rows cannot tell two rows apart, so these hashes find the same conflicts and
    # duplicates as hashes without the constant columns, which are known only after the hold-out split.
    main_columns = [column for column in all_columns if column not in config.FINGERPRINT_COLUMNS]
    fit, fit_record = remove_conflicts_and_duplicates(fit, main_columns)
    holdout = assign_seen_holdout(fit)
    # The hold-out rows must not choose the model inputs, so the constant columns come from the training rows only.
    constant_columns = [column for column in all_columns if fit.loc[~holdout, column].nunique() <= 1]
    # The fingerprint columns stay in the split files for the comparison arm, but the hashes use the main features so that both arms see the same rows.
    feature_columns = [column for column in all_columns if column not in constant_columns and column not in config.FINGERPRINT_COLUMNS]
    manifest["constant_columns_dropped"] = constant_columns
    manifest["feature_columns"] = feature_columns
    manifest["fingerprint_columns"] = [column for column in config.FINGERPRINT_COLUMNS if column in all_columns and column not in constant_columns]

    fit = fit.drop(columns=constant_columns)
    fit_record.update(split_summary(fit))
    manifest["splits"]["fit"] = fit_record
    fit_hashes = row_hashes(fit, feature_columns)
    for split, frame in split_frames.items():
        frame, record = remove_conflicts_and_duplicates(frame.drop(columns=constant_columns), feature_columns)
        frame, record["rows_also_in_fit_removed_by_class"] = remove_rows_in_fit(frame, feature_columns, fit_hashes)
        record["rows_also_in_fit_removed"] = sum(record["rows_also_in_fit_removed_by_class"].values())
        split_frames[split] = frame
        record.update(split_summary(frame))
        manifest["splits"][split] = record

    split_frames["fit_train"] = fit.loc[~holdout].reset_index(drop=True)
    split_frames["fit_holdout"] = fit.loc[holdout].reset_index(drop=True)
    for split in ("fit_train", "fit_holdout"):
        manifest["splits"][split] = split_summary(split_frames[split])

    # A crash during the writes must not leave new split files next to the manifest of an older build.
    (output_dir / "manifest.json").unlink(missing_ok=True)
    for split, frame in split_frames.items():
        files.write_parquet_atomic(frame, output_dir / f"{split}.parquet", index=False)
    files.write_json_atomic(output_dir / "manifest.json", manifest, default=str)
    return manifest


def load_split(split, input_dir=config.PROCESSED_DIR, columns=None):
    path = input_dir / f"{split}.parquet"
    if columns is None:
        return pd.read_parquet(path)
    available = set(pq.read_schema(path).names)
    metadata = [column for column in METADATA_COLUMNS if column in available and column not in columns]
    return pd.read_parquet(path, columns=[*columns, *metadata])


def load_manifest(input_dir=config.PROCESSED_DIR):
    return json.loads((input_dir / "manifest.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-fraction", type=float, default=None, help="read a random fraction of rows (cheap check)")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    output_dir = config.project_path(args.output_dir, config.PROCESSED_DIR)
    manifest = build_splits(args.sample_fraction, output_dir)
    print(json.dumps({"constant_columns_dropped": manifest["constant_columns_dropped"], "n_features": len(manifest["feature_columns"]), "days": manifest["days"], "splits": manifest["splits"]}, indent=1))


if __name__ == "__main__":
    main()
