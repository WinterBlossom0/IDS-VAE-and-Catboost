
import numpy as np
import pandas as pd

from idsvae import config
from idsvae.timing import python_parts


def test_detector_part_relabels_only_benign_flows_that_the_detector_flags():
    class FakeOnline:
        def features(self, second, latent_row, nll, kl):
            return [nll]

    apply = python_parts.detector_part(FakeOnline(), lambda rows: np.asarray(rows)[:, 0], 0.5, latent_dim=1)
    features = np.array([[0.0, 0.9, 1.0], [0.0, 0.9, 1.0], [0.0, 0.1, 1.0]])
    labels = [config.BENIGN, "SSH-Bruteforce", config.BENIGN]
    assert apply((labels, features, [10, 11, 12])) == [config.UNKNOWN_ATTACK, "SSH-Bruteforce", config.BENIGN]


def test_sample_window_takes_the_middle_flows_in_stable_time_order(monkeypatch):
    monkeypatch.setattr(python_parts, "SAMPLE_COUNT", 3)
    seconds = [5, 1, 3, 3, 2, 3, 4]
    test = pd.DataFrame({"time": pd.to_datetime("2018-02-21") + pd.to_timedelta(seconds, unit="s"), "row": range(len(seconds))})
    # Stable time order: rows 1, 4, 2, 3, 5, 6, 0; the middle three start at position (7 - 3) // 2 = 2.
    assert python_parts.sample_window(test)["row"].tolist() == [2, 3, 5]
    assert python_parts.split_day(test)[0]["row"].tolist() == [1, 4]


def test_warm_states_run_every_part_over_all_earlier_flows_in_warm_batches():
    def make_parts(states):
        def first(batch):
            states.append(("first", len(batch)))
            return batch

        def second(batch):
            states.append(("second", len(batch)))
            return batch
        return [first, second]
    states = python_parts.warm_states(make_parts, [], np.zeros((25_000, 3)), np.arange(25_000))
    assert states == [("first", 10_000), ("second", 10_000), ("first", 10_000), ("second", 10_000), ("first", 5_000), ("second", 5_000)]
