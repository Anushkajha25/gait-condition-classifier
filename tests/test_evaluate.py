import numpy as np
import pandas as pd
import pytest

from gaitcond.evaluate import evaluate, majority_baseline, sequence_predictions


@pytest.fixture
def meta():
    # two sequences of 3 frames each: one normal (label 0), one bag (label 1)
    return pd.DataFrame({
        "sequence_id": ["a"] * 3 + ["b"] * 3,
        "label": [0, 0, 0, 1, 1, 1],
        "angle": [0, 0, 0, 90, 90, 90],
    })


def test_majority_baseline():
    assert majority_baseline(np.array([0, 0, 0, 1]), np.array([0, 1, 1, 1])) == 0.25


def test_sequence_vote_fixes_a_single_wrong_frame(meta):
    proba = np.array([
        [0.9, 0.1, 0.0], [0.2, 0.8, 0.0], [0.8, 0.2, 0.0],   # one wrong frame in seq a
        [0.1, 0.9, 0.0], [0.2, 0.8, 0.0], [0.3, 0.7, 0.0],
    ])
    seq = sequence_predictions(meta, proba)
    assert seq["pred"].tolist() == [0, 1]

    res = evaluate(meta, proba, y_train=np.array([0, 0, 1]))
    assert res["frames"]["accuracy"] == pytest.approx(5 / 6)
    assert res["sequences"]["accuracy"] == 1.0
    assert res["sequences"]["n"] == 2
    assert res["frames"]["per_angle_accuracy"] == {0: pytest.approx(2 / 3), 90: 1.0}
    assert res["majority_baseline"] == 0.5


def test_balanced_accuracy_exposes_majority_guessing(meta):
    proba = np.tile([1.0, 0.0, 0.0], (6, 1))  # always predicts "normal"
    res = evaluate(meta, proba)
    assert res["frames"]["accuracy"] == 0.5
    # coat never appears in this toy test set, so balanced accuracy averages nm and bg
    assert res["frames"]["balanced_accuracy"] == 0.5
    assert res["frames"]["recall"]["bg"] == 0.0
