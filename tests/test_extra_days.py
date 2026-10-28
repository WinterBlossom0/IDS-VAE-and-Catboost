import pandas as pd

from idsvae import config
from idsvae.data import download, extra_days


def test_every_source_file_has_an_expected_size():
    assert set(config.SOURCE_FILE_BYTES) == {*config.DAY_FILES.values(), *extra_days.EXTRA_DAY_FILES.values()}


def test_only_the_udp_flows_with_the_mislabelled_class_are_dropped():
    frame = pd.DataFrame({
        "label": ["DDoS attacks-LOIC-HTTP", "DDoS attacks-LOIC-HTTP", config.BENIGN, "DDoS attacks-LOIC-HTTP"],
        "Protocol": [extra_days.UDP_PROTOCOL, 6, extra_days.UDP_PROTOCOL, extra_days.UDP_PROTOCOL],
        "f1": [1.0, 2.0, 3.0, 4.0],
    })
    kept, dropped = extra_days.drop_mislabelled_rows(frame, "02-20")
    assert dropped == 2
    assert kept["f1"].tolist() == [2.0, 3.0]
    assert kept.index.tolist() == [0, 1]
    unchanged, none_dropped = extra_days.drop_mislabelled_rows(frame, "03-01")
    assert none_dropped == 0
    pd.testing.assert_frame_equal(unchanged, frame)


def test_read_extra_day_uses_the_local_copy(tmp_path, monkeypatch):
    file_name = extra_days.EXTRA_DAY_FILES["02-20"]
    lines = [
        "Flow ID,Src IP,Src Port,Dst IP,Dst Port,Protocol,Timestamp,Flow Duration,Label",
        "a,1.1.1.1,5,2.2.2.2,80,6,20/02/2018 09:00:00,10,Benign",
        "Flow ID,Src IP,Src Port,Dst IP,Dst Port,Protocol,Timestamp,Flow Duration,Label",
        "b,1.1.1.2,6,2.2.2.3,53,17,20/02/2018 01:15:00,20,DDoS attacks-LOIC-HTTP",
    ]
    (tmp_path / file_name).write_text("\n".join(lines) + "\n")
    monkeypatch.setattr(config, "RAW_DIR", tmp_path)
    monkeypatch.setitem(config.SOURCE_FILE_BYTES, file_name, (tmp_path / file_name).stat().st_size)

    def no_download(name):
        raise AssertionError("the local copy must be used")
    monkeypatch.setattr(download, "stream", no_download)
    frame = extra_days.read_extra_day("02-20")
    assert list(frame.columns) == ["Dst Port", "Protocol", "Timestamp", "Flow Duration", "Label"]
    assert frame["Label"].tolist() == ["Benign", "DDoS attacks-LOIC-HTTP"]
    assert frame["Flow Duration"].dtype == "float32"
    assert frame["Protocol"].tolist() == [6.0, 17.0]
