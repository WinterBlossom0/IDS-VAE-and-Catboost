import numpy as np

from idsvae import config
from idsvae.timing import export


def test_label_indices_reads_rule_tuples_and_detector_lists():
    classes = [config.BENIGN, "DoS attacks-GoldenEye"]
    outputs = [(["DoS attacks-GoldenEye"], np.zeros((1, 3)), np.zeros(1)), [config.UNKNOWN_ATTACK], [config.BENIGN]]
    assert export.label_indices(outputs, classes).tolist() == [1, 2, 0]


def test_argmax_indices_take_the_first_largest_probability():
    outputs = [np.array([[0.2, 0.4, 0.4]], dtype=np.float32), np.array([[0.9, 0.1, 0.0]])]
    assert export.argmax_indices(outputs).tolist() == [1, 0]
