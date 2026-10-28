import json

from idsvae import files


def test_write_json_atomic_leaves_no_temporary_file(tmp_path):
    path = tmp_path / "result.json"
    files.write_json_atomic(path, {"value": 1})
    assert json.loads(path.read_text()) == {"value": 1}
    assert list(tmp_path.iterdir()) == [path]


def test_write_parquet_atomic_round_trips_and_leaves_no_temporary_file(tmp_path):
    import pandas as pd
    path = tmp_path / "frame.parquet"
    frame = pd.DataFrame({"value": [1, 2, 3]})
    files.write_parquet_atomic(frame, path, index=False)
    assert pd.read_parquet(path).equals(frame)
    assert list(tmp_path.iterdir()) == [path]


def test_a_failed_write_keeps_the_old_file_and_leaves_no_temporary_file(tmp_path):
    import pytest
    path = tmp_path / "result.json"
    files.write_json_atomic(path, {"value": 1})
    with pytest.raises(RuntimeError), files.atomic_path(path) as temporary:
        temporary.write_text('{"value": 2')
        raise RuntimeError("stopped during the write")
    assert json.loads(path.read_text()) == {"value": 1}
    assert list(tmp_path.iterdir()) == [path]


def test_atomic_writes_give_the_usual_file_mode(tmp_path):
    path = tmp_path / "result.json"
    files.write_json_atomic(path, {"value": 1})
    assert path.stat().st_mode & 0o777 == 0o666 & ~files.UMASK
