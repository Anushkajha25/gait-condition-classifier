from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gaitcond.data import build_index

SEQS = {"nm": 6, "bg": 2, "cl": 2}  # CASIA-B has 6 normal, 2 bag, 2 coat sequences per subject


def make_fake_casia(
    root: Path, subjects=(1, 2, 3, 4, 5, 6), angles=(0, 90, 180), frames=4
) -> Path:
    """Write a tiny CASIA-B-shaped tree of 8x8 grayscale PNGs."""
    from PIL import Image  # installed with matplotlib

    rng = np.random.default_rng(0)
    for s in subjects:
        for cond, n_seq in SEQS.items():
            for q in range(1, n_seq + 1):
                for a in angles:
                    d = root / f"{s:03d}" / f"{cond}-{q:02d}" / f"{a:03d}"
                    d.mkdir(parents=True, exist_ok=True)
                    for f in range(1, frames + 1):
                        img = (rng.random((8, 8)) > 0.5).astype(np.uint8) * 255
                        Image.fromarray(img).save(d / f"{s:03d}-{cond}-{q:02d}-{a:03d}-{f:03d}.png")
    return root


@pytest.fixture
def fake_casia(tmp_path) -> Path:
    return make_fake_casia(tmp_path / "CASIA-B")


@pytest.fixture
def fake_features(tmp_path, fake_casia) -> Path:
    """A feature directory like extract_features.py writes, with a learnable class signal."""
    index = build_index(fake_casia)
    rng = np.random.default_rng(1)
    out = tmp_path / "features"
    out.mkdir()
    for name, dim in (("resnet101", 32), ("efficientnetb2", 16)):
        X = rng.normal(size=(len(index), dim)).astype(np.float16)
        X[:, 0] += index["label"].to_numpy() * 4.0  # class signal
        np.save(out / f"{name}.npy", X)
    index.to_csv(out / "index.csv", index=False)
    return out


def subject_ranges_for(index: pd.DataFrame) -> dict:
    return {"kind": "subject", "train_subjects": "1-4", "val_subjects": "5",
            "test_subjects": "6"}
