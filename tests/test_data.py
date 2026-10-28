import numpy as np
import pytest

from gaitcond.data import build_index, frame_split, parse_path, subject_split


def test_parse_filename():
    meta = parse_path("001/bg-01/090/001-bg-01-090-042.png")
    assert meta == {"subject": 1, "condition": "bg", "seq": 1, "angle": 90, "frame": 42}


def test_parse_directory_fallback():
    meta = parse_path("data/017/cl-02/180/frame_7.png")
    assert meta == {"subject": 17, "condition": "cl", "seq": 2, "angle": 180, "frame": 7}


def test_parse_rejects_unrelated_paths():
    assert parse_path("notes/readme.png") is None


def test_build_index_counts(fake_casia):
    index = build_index(fake_casia)
    # 6 subjects x 10 sequences x 3 angles x 4 frames
    assert len(index) == 6 * 10 * 3 * 4
    assert index["sequence_id"].nunique() == 6 * 10 * 3
    assert set(index["condition"]) == {"nm", "bg", "cl"}
    assert (index.groupby("condition")["label"].first().to_dict()) == {"nm": 0, "bg": 1, "cl": 2}


def test_frame_stride_keeps_every_nth_frame(fake_casia):
    full = build_index(fake_casia)
    half = build_index(fake_casia, frame_stride=2)
    assert len(half) == len(full) // 2
    assert half["sequence_id"].nunique() == full["sequence_id"].nunique()


def test_subject_split_is_disjoint(fake_casia):
    index = build_index(fake_casia)
    split = subject_split(index, "1-4", "5", "6")
    subjects = {k: set(index.iloc[v]["subject"]) for k, v in split.as_dict().items()}
    assert subjects == {"train": {1, 2, 3, 4}, "val": {5}, "test": {6}}
    sequences = {k: set(index.iloc[v]["sequence_id"]) for k, v in split.as_dict().items()}
    assert not sequences["train"] & sequences["test"]


def test_subject_split_rejects_overlap(fake_casia):
    index = build_index(fake_casia)
    with pytest.raises(ValueError, match="appear in both"):
        subject_split(index, "1-4", "4-5", "6")


def test_frame_split_leaks_sequences(fake_casia):
    """Documents the bug in the original notebooks: sequences straddle train and test."""
    index = build_index(fake_casia)
    split = frame_split(index)
    train_seq = set(index.iloc[split.train]["sequence_id"])
    test_seq = set(index.iloc[split.test]["sequence_id"])
    assert len(train_seq & test_seq) > 0
    assert len(np.intersect1d(split.train, split.test)) == 0
