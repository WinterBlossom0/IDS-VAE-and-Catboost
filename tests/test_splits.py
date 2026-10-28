import threading
import time

import numpy as np
import pandas as pd
import pytest

from idsvae import config
from idsvae.data import download, splits

RAW_COLUMNS = [config.TIME_COLUMN, "Dst Port", "f1", "f2", config.LABEL_COLUMN]
# 2**24 + 1 has no float32 representation and rounds to 2**24, so the two values differ only in float64.
FLOAT32_TWIN_LOW = 16777216.0
FLOAT32_TWIN_HIGH = 16777217.0


def raw_frame():
    rows = [
        # Timestamp, Dst Port, f1, f2, Label
        ("14/02/2018 10:00:02", 80, 1.0, 2.0, "Benign"),
        ("14/02/2018 10:00:01", 80, 1.0, 2.0, "Benign"),
        ("Timestamp", "Dst Port", "f1", "f2", "Label"),
        ("14/02/2018 10:00:03", 21, np.inf, 2.0, "FTP-BruteForce"),
        ("14/02/2018 10:00:04", 21, 5.0, 6.0, "FTP-BruteForce"),
        ("14/02/2018 10:00:05", 22, 5.0, 6.0, "SSH-Bruteforce"),
        ("14/02/2018 10:00:06", 22, 7.0, 8.0, "SSH-Bruteforce"),
    ]
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def test_clean_day_removes_header_rows_nonfinite_rows_and_excluded_columns():
    cleaned, stats = splits.clean_day(raw_frame(), "02-14")
    assert stats["header_rows"] == 1
    assert stats["rows_nonfinite"] == 1
    assert stats["rows_bad_timestamp"] == 0
    assert len(cleaned) == 5
    assert "Dst Port" not in cleaned.columns
    assert splits.feature_columns_of(cleaned) == ["f1", "f2"]
    assert (cleaned[["f1", "f2"]].dtypes == np.float32).all()
    assert cleaned["in_benign_burst"].dtype == bool


def test_twelve_hour_timestamps_map_hours_below_eight_to_the_afternoon():
    raw = pd.Series(["14/02/2018 02:01:50", "14/02/2018 10:33:00", "14/02/2018 12:15:00", "14/02/2018 08:00:00", "14/02/2018 05:59:59"])
    corrected = splits.parse_capture_time(raw)
    expected = pd.to_datetime(["2018-02-14 14:01:50", "2018-02-14 10:33:00", "2018-02-14 12:15:00", "2018-02-14 08:00:00", "2018-02-14 17:59:59"])
    assert (corrected.to_numpy() == expected.to_numpy()).all()


def test_clean_day_drops_rows_outside_the_capture_year():
    rows = [
        ("14/02/2018 09:00:00", 80, 1.0, 2.0, "Benign"),
        ("05/01/1970 03:01:17", 80, 3.0, 4.0, "Benign"),
        ("14/02/2018 02:00:00", 22, 5.0, 6.0, "SSH-Bruteforce"),
    ]
    cleaned, stats = splits.clean_day(pd.DataFrame(rows, columns=RAW_COLUMNS), "02-14")
    assert stats["rows_bad_timestamp"] == 1
    assert stats["rows_clean"] == 2
    assert (cleaned["time"].dt.year == config.CAPTURE_YEAR).all()
    assert 3.0 not in cleaned["f1"].to_numpy()


def test_only_benign_rows_of_a_burst_window_are_flagged(monkeypatch):
    monkeypatch.setattr(config, "BURST_TEMPLATE_COLUMNS", ["f2"])
    # SSH window 14:00-14:10 holds 30 benign rows (3 per minute); 60 background rows give about 0.12 per minute outside the windows.
    burst_rows = [(f"14/02/2018 02:{minute:02d}:{second:02d}", 80, 100.0 + 3 * minute + second // 20, 1.0, "Benign") for minute in range(10) for second in (0, 20, 40)]
    background_rows = [(f"14/02/2018 {8 + minute // 60:02d}:{minute % 60:02d}:30", 80, 200.0 + minute, 1.0, "Benign") for minute in range(0, 120, 2)]
    rows = burst_rows + background_rows + [
        ("14/02/2018 02:00:00", 22, 2.0, 2.0, "SSH-Bruteforce"),
        ("14/02/2018 02:10:00", 22, 3.0, 3.0, "SSH-Bruteforce"),
        ("14/02/2018 08:00:00", 80, 4.0, 4.0, "Benign"),
        ("14/02/2018 10:30:00", 21, 5.0, 5.0, "FTP-BruteForce"),
        ("14/02/2018 10:40:00", 21, 6.0, 6.0, "FTP-BruteForce"),
        ("14/02/2018 10:35:00", 80, 7.0, 7.0, "Benign"),
        ("14/02/2018 05:00:00", 80, 8.0, 8.0, "Benign"),
    ]
    cleaned, stats = splits.clean_day(pd.DataFrame(rows, columns=RAW_COLUMNS), "02-14")
    windows = stats["attack_windows"]
    assert (windows["SSH-Bruteforce"]["start"], windows["SSH-Bruteforce"]["end"]) == ("2018-02-14T14:00:00", "2018-02-14T14:10:00")
    assert windows["SSH-Bruteforce"]["benign_burst"]
    assert not windows["FTP-BruteForce"]["benign_burst"]
    flagged = cleaned.set_index("f1")["in_benign_burst"]
    assert flagged.loc[[100.0 + 3 * minute + offset for minute in range(10) for offset in range(3)]].all()
    assert not flagged.loc[[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]].any()


def test_conflicts_and_duplicates_are_removed_and_earliest_copy_stays():
    cleaned, _ = splits.clean_day(raw_frame(), "02-14")
    no_conflicts, conflicts_by_class, _ = splits.remove_label_conflicts(cleaned, ["f1", "f2"])
    assert conflicts_by_class == {"FTP-BruteForce": 1, "SSH-Bruteforce": 1}
    deduplicated, duplicates_by_class = splits.remove_duplicates(no_conflicts, ["f1", "f2"])
    assert duplicates_by_class == {"Benign": 1}
    benign = deduplicated[deduplicated["label"] == "Benign"]
    assert benign["time"].iloc[0] == pd.Timestamp("2018-02-14 10:00:01")


def test_rows_equal_in_float32_count_as_duplicates():
    rows = [
        ("14/02/2018 09:00:01", 80, FLOAT32_TWIN_LOW, 2.0, "Benign"),
        ("14/02/2018 09:00:02", 80, FLOAT32_TWIN_HIGH, 2.0, "Benign"),
    ]
    cleaned, _ = splits.clean_day(pd.DataFrame(rows, columns=RAW_COLUMNS), "02-14")
    deduplicated, duplicates_by_class = splits.remove_duplicates(cleaned, ["f1", "f2"])
    assert duplicates_by_class == {"Benign": 1}
    assert len(deduplicated) == 1


def test_removal_counts_by_class_and_distinct_conflict_vectors():
    rows = [
        ("Benign", 1.0, 1.0), ("Benign", 1.0, 1.0), ("Bot", 1.0, 1.0),
        ("Bot", 2.0, 2.0), ("Bot", 2.0, 2.0), ("Infiltration", 2.0, 2.0),
        ("Bot", 3.0, 3.0), ("Infiltration", 3.0, 3.0), ("Infiltration", 3.0, 3.0), ("Infiltration", 4.0, 4.0),
        ("Bot", 5.0, 5.0), ("Bot", 5.0, 5.0),
    ]
    frame = pd.DataFrame(rows, columns=["label", "f1", "f2"])
    frame["time"] = pd.to_datetime("2018-02-14") + pd.to_timedelta(range(len(frame)), unit="s")
    kept, by_class, vectors = splits.remove_label_conflicts(frame, ["f1", "f2"])
    assert by_class == {"Bot": 4, "Infiltration": 3, "Benign": 2}
    assert vectors == {
        "Benign": {"distinct_vectors": 1, "distinct_vectors_equal_to_benign": 1},
        "Bot": {"distinct_vectors": 3, "distinct_vectors_equal_to_benign": 1},
        "Infiltration": {"distinct_vectors": 2, "distinct_vectors_equal_to_benign": 0},
    }
    kept, duplicates = splits.remove_duplicates(kept, ["f1", "f2"])
    assert duplicates == {"Bot": 1}
    fit_hashes = splits.row_hashes(kept.iloc[:1], ["f1", "f2"])
    kept, in_fit = splits.remove_rows_in_fit(kept, ["f1", "f2"], fit_hashes)
    assert in_fit == {"Infiltration": 1}


def test_seen_holdout_takes_latest_fraction_of_each_group():
    times = pd.date_range("2018-02-14", periods=10, freq="min")
    frame = pd.DataFrame({"day": "02-14", "label": ["Benign"] * 10, "time": times})
    holdout = splits.assign_seen_holdout(frame)
    assert holdout.sum() == int(np.ceil(config.SEEN_HOLDOUT_FRACTION * 10))
    assert frame.loc[holdout, "time"].min() > frame.loc[~holdout, "time"].max()


def synthetic_day(rows):
    # The constant column checks that build_splits drops columns that are constant on the fit days.
    records = [(timestamp, 80, f1, f2, 1.0, label) for timestamp, f1, f2, label in rows]
    return pd.DataFrame(records, columns=[config.TIME_COLUMN, "Dst Port", "f1", "f2", "f_constant", config.LABEL_COLUMN])


SYNTHETIC_DAYS = {
    "02-14": synthetic_day([
        ("14/02/2018 09:00:00", FLOAT32_TWIN_LOW, 1.0, "Benign"),
        ("14/02/2018 10:00:00", 1.0, 1.0, "Benign"),
        ("14/02/2018 02:30:00", 2.0, 2.0, "Benign"),
        ("14/02/2018 05:00:00", 3.0, 3.0, "Benign"),
        ("14/02/2018 02:00:00", 4.0, 4.0, "SSH-Bruteforce"),
        ("14/02/2018 03:00:00", 5.0, 5.0, "SSH-Bruteforce"),
        ("05/01/1970 03:01:17", 50.0, 50.0, "Benign"),
    ]),
    "02-15": synthetic_day([
        ("15/02/2018 09:00:00", 6.0, 6.0, "Benign"),
        ("15/02/2018 09:30:00", 7.0, 7.0, "DoS attacks-GoldenEye"),
        ("15/02/2018 09:45:00", 8.0, 8.0, "Benign"),
        ("15/02/2018 10:00:00", 9.0, 9.0, "DoS attacks-GoldenEye"),
    ]),
    "02-28": synthetic_day([
        ("28/02/2018 09:00:00", 30.0, 30.0, "Benign"),
        ("28/02/2018 10:50:00", 31.0, 31.0, "Infilteration"),
    ]),
    "02-23": synthetic_day([
        ("23/02/2018 09:00:00", 40.0, 40.0, "Benign"),
        ("23/02/2018 01:00:00", 41.0, 41.0, "Brute Force -XSS"),
    ]),
    "02-21": synthetic_day([
        ("21/02/2018 08:40:00", FLOAT32_TWIN_HIGH, 1.0, "Benign"),
        ("21/02/2018 02:11:08", 20.0, 20.0, "DDOS attack-HOIC"),
        ("21/02/2018 02:20:00", 22.0, 22.0, "Benign"),
        ("21/02/2018 02:21:00", 24.0, 22.0, "Benign"),
        ("21/02/2018 02:22:00", 25.0, 22.0, "Benign"),
        ("21/02/2018 02:33:29", 21.0, 21.0, "DDOS attack-HOIC"),
        ("21/02/2018 10:00:00", 23.0, 23.0, "Benign"),
    ]),
}


def test_build_splits_writes_the_split_interface_and_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BURST_TEMPLATE_COLUMNS", ["f2"])
    monkeypatch.setattr(splits, "read_day", lambda day, sample_fraction=None: SYNTHETIC_DAYS[day].copy())
    manifest = splits.build_splits(output_dir=tmp_path)
    assert manifest["constant_columns_dropped"] == ["f_constant"]
    assert manifest["feature_columns"] == ["f1", "f2"]
    assert manifest["days"]["02-14"]["rows_bad_timestamp"] == 1
    hoic_window = manifest["days"]["02-21"]["attack_windows"]["DDOS attack-HOIC"]
    assert (hoic_window["start"], hoic_window["end"], hoic_window["benign_burst"]) == ("2018-02-21T14:11:08", "2018-02-21T14:33:29", True)
    assert not manifest["days"]["02-14"]["attack_windows"]["SSH-Bruteforce"]["benign_burst"]

    frames = {split: splits.load_split(split, tmp_path) for split in ("fit_train", "fit_holdout", "vae_val", "calib", "test")}
    for split, frame in frames.items():
        assert list(frame.columns) == ["f1", "f2", *splits.METADATA_COLUMNS]
        assert (frame[["f1", "f2"]].dtypes == np.float32).all()
        assert frame["in_benign_burst"].dtype == bool
        assert pd.api.types.is_datetime64_any_dtype(frame["time"])
        assert splits.feature_columns_of(frame) == manifest["feature_columns"]
        assert manifest["splits"][split]["benign_burst_rows"] == int(frame["in_benign_burst"].sum())

    # The latest Benign flow of 02-14 is at 17:00 after the 12-hour correction, so it forms the hold-out.
    holdout_benign = frames["fit_holdout"].query("day == '02-14' and label == 'Benign'")
    assert holdout_benign["time"].tolist() == [pd.Timestamp("2018-02-14 17:00:00")]
    assert frames["fit_train"]["in_benign_burst"].sum() == 0
    assert frames["fit_holdout"]["in_benign_burst"].sum() == 0

    # The first test Benign row equals a fit row only in float32, so the fit-overlap check removes it.
    assert manifest["splits"]["test"]["rows_also_in_fit_removed"] == 1
    assert manifest["splits"]["test"]["rows_also_in_fit_removed_by_class"] == {"Benign": 1}
    for record in manifest["splits"].values():
        if "duplicate_rows_removed_by_class" in record:
            assert record["duplicate_rows_removed"] == sum(record["duplicate_rows_removed_by_class"].values())
            assert set(record["label_conflict_distinct_vectors_by_class"]) == set(record["label_conflict_rows_removed_by_class"])
    test_benign = frames["test"].query("label == 'Benign'").set_index("f1")["in_benign_burst"].to_dict()
    assert test_benign == {22.0: True, 24.0: True, 25.0: True, 23.0: False}
    assert manifest["splits"]["test"]["benign_burst_rows"] == 3
    assert manifest["splits"]["calib"]["benign_burst_rows"] == 0


def test_overlapping_windows_count_once():
    start = pd.Timestamp("2018-02-23 10:00:00")
    windows = [(start, start + pd.Timedelta(minutes=60)), (start + pd.Timedelta(minutes=30), start + pd.Timedelta(minutes=90)), (start + pd.Timedelta(minutes=120), start + pd.Timedelta(minutes=130))]
    assert splits.union_minutes(windows) == 100.0


def test_load_split_with_columns_adds_the_metadata_columns_that_the_file_has(tmp_path):
    frame = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [4.0, 5.0, 6.0], "f3": [7.0, 8.0, 9.0], "label": ["Benign", "A", "Benign"], "day": ["02-14"] * 3}).astype({"f1": "float32", "f2": "float32", "f3": "float32"})
    frame.to_parquet(tmp_path / "test.parquet", index=False)
    loaded = splits.load_split("test", tmp_path, columns=["f3", "f1"])
    assert list(loaded.columns) == ["f3", "f1", "label", "day"]
    pd.testing.assert_frame_equal(loaded, frame[["f3", "f1", "label", "day"]], check_exact=True)
    pd.testing.assert_frame_equal(splits.load_split("test", tmp_path), frame, check_exact=True)


def write_local_day(raw_dir, monkeypatch, day="02-14", rows=200):
    """A small local copy of the CSV of a day; downloads fail, so a read must use the local copy."""
    frame = pd.DataFrame({config.TIME_COLUMN: ["14/02/2018 09:00:00"] * rows, "f1": np.arange(rows, dtype=float), config.LABEL_COLUMN: "Benign"})
    path = raw_dir / config.DAY_FILES[day]
    frame.to_csv(path, index=False)
    monkeypatch.setattr(config, "RAW_DIR", raw_dir)
    monkeypatch.setitem(config.SOURCE_FILE_BYTES, config.DAY_FILES[day], path.stat().st_size)

    def no_download(file_name):
        raise AssertionError("the local copy must be used")
    monkeypatch.setattr(download, "stream", no_download)
    return frame


def test_read_day_uses_the_local_copy_and_samples_the_same_rows_each_time(tmp_path, monkeypatch):
    frame = write_local_day(tmp_path, monkeypatch)
    pd.testing.assert_frame_equal(splits.read_day("02-14"), frame)
    first, second = splits.read_day("02-14", sample_fraction=0.3), splits.read_day("02-14", sample_fraction=0.3)
    pd.testing.assert_frame_equal(first, second)
    assert 0 < len(first) < len(frame)
    assert set(first["f1"]) <= set(frame["f1"])


def test_local_copy_with_an_unexpected_size_is_rejected(tmp_path, monkeypatch):
    write_local_day(tmp_path, monkeypatch)
    monkeypatch.setitem(config.SOURCE_FILE_BYTES, config.DAY_FILES["02-14"], 1)
    with pytest.raises(OSError, match="expected 1"):
        splits.read_day("02-14")


PAYLOAD_SIZE = 10
PAYLOAD = bytes(range(PAYLOAD_SIZE))


def fake_remote(monkeypatch, size=PAYLOAD_SIZE):
    """Ranges of 4 bytes and 2 workers; the first range returns only after the last range, so the stream must reorder them."""
    monkeypatch.setattr(download, "RANGE_BYTES", 4)
    monkeypatch.setattr(download, "RANGE_WORKERS", 2)
    monkeypatch.setattr(download, "remote_size", lambda url: size)
    requested = []
    last_range_fetched = threading.Event()

    def fetch(url, start, end):
        requested.append((start, end))
        if start == 0:
            last_range_fetched.wait(timeout=5)
        if end == size - 1:
            last_range_fetched.set()
        return PAYLOAD[start:end + 1]
    monkeypatch.setattr(download, "fetch_range", fetch)
    return requested


def test_parallel_range_stream_returns_the_ranges_in_file_order(monkeypatch):
    requested = fake_remote(monkeypatch)
    raw = download.ParallelRangeStream("https://example.invalid/day.csv", len(PAYLOAD))
    with download.io.BufferedReader(raw, buffer_size=3) as stream:
        assert stream.read() == PAYLOAD
        assert stream.read() == b""
    assert sorted(requested) == [(0, 3), (4, 7), (8, 9)]


def test_parallel_range_stream_rejects_an_unexpected_size(monkeypatch):
    fake_remote(monkeypatch, size=11)
    with pytest.raises(OSError, match="expected 10"):
        download.ParallelRangeStream("https://example.invalid/day.csv", len(PAYLOAD))


class FakeResponse:
    def __init__(self, status, body):
        self.status, self.body = status, body

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        return False

    def read(self):
        return self.body


def test_fetch_range_rejects_a_response_that_is_not_partial_content(monkeypatch):
    monkeypatch.setattr(download.urllib.request, "urlopen", lambda request, timeout: FakeResponse(200, PAYLOAD))
    with pytest.raises(RuntimeError, match="status 200"):
        download.fetch_range("https://example.invalid/day.csv", 0, 3)


def test_fetch_range_retries_with_exponential_backoff(monkeypatch):
    answers = iter([OSError("reset"), FakeResponse(206, PAYLOAD[:2]), FakeResponse(206, PAYLOAD[:4])])

    def urlopen(request, timeout):
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer
    sleeps = []
    monkeypatch.setattr(download.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(time, "sleep", sleeps.append)
    assert download.fetch_range("https://example.invalid/day.csv", 0, 3) == PAYLOAD[:4]
    assert sleeps == [2, 4]


def test_a_crash_during_the_split_writes_leaves_no_old_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BURST_TEMPLATE_COLUMNS", ["f2"])
    monkeypatch.setattr(splits, "read_day", lambda day, sample_fraction=None: SYNTHETIC_DAYS[day].copy())
    (tmp_path / "manifest.json").write_text("{}")

    def crash(frame, path, **options):
        raise OSError("disk full")
    monkeypatch.setattr(splits.files, "write_parquet_atomic", crash)
    with pytest.raises(OSError):
        splits.build_splits(output_dir=tmp_path)
    assert not (tmp_path / "manifest.json").exists()


def test_build_splits_drops_a_column_that_is_constant_on_fit_train_but_not_on_the_hold_out(tmp_path, monkeypatch):
    days = {day: frame.copy() for day, frame in SYNTHETIC_DAYS.items()}
    for frame in days.values():
        frame["f_train_constant"] = 1.0
    # The 02-14 Benign flow at 05:00 (17:00 after the 12-hour correction) is the latest of its group, so it is a hold-out row.
    days["02-14"].loc[days["02-14"][config.TIME_COLUMN] == "14/02/2018 05:00:00", "f_train_constant"] = 2.0
    monkeypatch.setattr(config, "BURST_TEMPLATE_COLUMNS", ["f2"])
    monkeypatch.setattr(splits, "read_day", lambda day, sample_fraction=None: days[day].copy())
    manifest = splits.build_splits(output_dir=tmp_path)
    assert manifest["constant_columns_dropped"] == ["f_constant", "f_train_constant"]
    assert manifest["feature_columns"] == ["f1", "f2"]
