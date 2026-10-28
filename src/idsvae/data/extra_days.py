"""Capture days of CSE-CIC-IDS2018 outside the protocol splits, for the unseen-attack detector.

These days add attack families that no protocol split contains (training and validation of the detector) and more Benign days
(false-alarm checks). Each day is cleaned as the protocol days are: splits.clean_day, the constant columns of the fit split
dropped, label conflicts and duplicates removed, and rows equal to a fit-day row removed. The days come from the
"Processed Traffic Data for ML Algorithms" folder of the public dataset bucket.
"""
import argparse
import json

import pandas as pd

from idsvae import config, files
from idsvae.data import splits

OUTPUT_DIR = config.PROJECT_ROOT / "data" / "processed_extra"
# The 02-20 file name has the spelling "Thuesday" in the bucket.
EXTRA_DAY_FILES = {
    "02-16": "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv",
    "02-20": "Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv",
    "02-22": "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv",
    "03-01": "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv",
    "03-02": "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",
}
# Only the 02-20 file has these columns; they identify hosts, and no protocol model has them as inputs.
HOST_COLUMNS = ["Flow ID", "Src IP", "Src Port", "Dst IP"]
CHUNK_ROWS = 1_000_000


def read_extra_day(day):
    """The raw rows of one day, read in chunks: the 02-20 file has 8 million rows, and float32 per chunk keeps the memory low."""
    chunks = []
    with splits.official_csv(EXTRA_DAY_FILES[day]) as source:
        for chunk in pd.read_csv(source, low_memory=False, chunksize=CHUNK_ROWS):
            chunk = chunk.drop(columns=[column for column in HOST_COLUMNS if column in chunk.columns])
            # Some official CSV files repeat the header line inside the data.
            chunk = chunk.loc[chunk[config.LABEL_COLUMN] != config.LABEL_COLUMN]
            numeric = [column for column in chunk.columns if column not in (config.LABEL_COLUMN, config.TIME_COLUMN)]
            chunk[numeric] = chunk[numeric].apply(pd.to_numeric, errors="raise").astype(splits.FEATURE_DTYPE)
            chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)


# A day serves the detector only when its Benign flows cover the capture: at least this many hours that each hold at least 1 % of
# them. The truncated 02-16 file keeps all of its Benign flows in one hour, 45 % of them with one flow template, which is a
# burst like the 02-21 Benign anomaly and not normal traffic.
MINIMUM_BENIGN_HOURS = 6
MINIMUM_HOUR_SHARE = 0.01


def benign_hours(frame):
    """Count of capture hours that hold at least MINIMUM_HOUR_SHARE of the Benign flows of a day."""
    benign_times = pd.to_datetime(frame.loc[frame["label"] == config.BENIGN, "time"])
    shares = benign_times.dt.hour.value_counts(normalize=True)
    return int((shares >= MINIMUM_HOUR_SHARE).sum())


def benign_covers_day(frame):
    return benign_hours(frame) >= MINIMUM_BENIGN_HOURS


# The 02-20 CSV labels UDP flows of 13:14-13:29 "DDoS attacks-LOIC-HTTP", but LOIC-HTTP uses TCP only; the CIC schedule of
# that day has the LOIC-UDP attack at 13:13-13:32. LOIC-UDP is a test-day (02-21) class, so these flows must not train the detector.
UDP_PROTOCOL = 17
MISLABELLED_UDP = {"02-20": "DDoS attacks-LOIC-HTTP"}


def drop_mislabelled_rows(frame, day):
    """(frame without the mislabelled LOIC-UDP flows of a day, count of dropped rows)."""
    label = MISLABELLED_UDP.get(day)
    if label is None:
        return frame, 0
    mislabelled = ((frame["label"] == label) & (frame["Protocol"] == UDP_PROTOCOL)).to_numpy()
    return frame.loc[~mislabelled].reset_index(drop=True), int(mislabelled.sum())


def fit_hashes(feature_columns):
    fit = pd.concat([splits.load_split("fit_train"), splits.load_split("fit_holdout")], ignore_index=True)
    return splits.row_hashes(fit, feature_columns)


def process_day(day, manifest, hashes):
    """(cleaned frame of one extra day, its manifest record)."""
    cleaned, stats = splits.clean_day(read_extra_day(day), day)
    frame = cleaned.drop(columns=[column for column in manifest["constant_columns_dropped"] if column in cleaned.columns])
    missing = [column for column in manifest["feature_columns"] if column not in frame.columns]
    if missing:
        raise ValueError(f"{day}: columns of the protocol splits are missing: {missing}")
    feature_columns = manifest["feature_columns"]
    record = {"cleaning": stats, "rows_after_cleaning": len(frame)}
    frame, record["label_conflict_rows_removed_by_class"], _ = splits.remove_label_conflicts(frame, feature_columns)
    frame, record["duplicate_rows_removed_by_class"] = splits.remove_duplicates(frame, feature_columns)
    frame, record["rows_also_in_fit_removed_by_class"] = splits.remove_rows_in_fit(frame, feature_columns, hashes)
    frame, record["mislabelled_udp_rows_removed"] = drop_mislabelled_rows(frame, day)
    record.update(splits.split_summary(frame))
    return frame, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("days", nargs="*", default=list(EXTRA_DAY_FILES))
    args = parser.parse_args()
    manifest = splits.load_manifest()
    hashes = fit_hashes(manifest["feature_columns"])
    report_path = OUTPUT_DIR / "manifest.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for day in args.days:
        frame, report[day] = process_day(day, manifest, hashes)
        # run_detector.sh skips a day whose parquet exists, so the record is written before the parquet gets its final name:
        # a crash in between leaves a record without a parquet, and the next run processes the day again.
        with files.atomic_path(OUTPUT_DIR / f"{day}.parquet") as temporary:
            frame.to_parquet(temporary, index=False)
            files.write_json_atomic(report_path, report, default=str)
        print(day, json.dumps(report[day]["class_counts"] if "class_counts" in report[day] else report[day], default=str)[:400], flush=True)


if __name__ == "__main__":
    main()
