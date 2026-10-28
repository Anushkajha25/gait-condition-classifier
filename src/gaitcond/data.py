"""Index CASIA-B silhouette frames and split them without leakage.

CASIA-B silhouettes are laid out as::

    <root>/<subject>/<condition>-<seq>/<angle>/<subject>-<condition>-<seq>-<angle>-<frame>.png
    e.g. 001/bg-01/090/001-bg-01-090-042.png

Every frame gets: subject, condition, seq, angle, frame and a ``sequence_id`` that
identifies the walking sequence it came from (subject + condition + seq + angle).
Frames within one sequence are near-duplicates, so they must never be split across
train and test. The default split is by subject.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gaitcond import CLASS_TO_ID, CLASSES

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

_FILENAME_RE = re.compile(
    r"(?P<subject>\d{3})-(?P<condition>nm|bg|cl)-(?P<seq>\d{2})-(?P<angle>\d{3})"
    r"(?:-(?P<frame>\d+))?"
)
_COND_DIR_RE = re.compile(r"^(?P<condition>nm|bg|cl)-(?P<seq>\d{2})$")


def parse_path(path: str | Path) -> dict | None:
    """Return frame metadata parsed from a CASIA-B path, or None if it doesn't match.

    The filename is tried first; the directory layout is the fallback.
    """
    p = Path(path)
    m = _FILENAME_RE.search(p.stem)
    if m:
        frame = m.group("frame")
        return {
            "subject": int(m.group("subject")),
            "condition": m.group("condition"),
            "seq": int(m.group("seq")),
            "angle": int(m.group("angle")),
            "frame": int(frame) if frame is not None else -1,
        }

    parts = p.parts
    for i, part in enumerate(parts):
        c = _COND_DIR_RE.match(part)
        if c and 0 < i < len(parts) - 1:
            subject, angle = parts[i - 1], parts[i + 1]
            if subject.isdigit() and angle.isdigit():
                digits = re.findall(r"\d+", p.stem)
                return {
                    "subject": int(subject),
                    "condition": c.group("condition"),
                    "seq": int(c.group("seq")),
                    "angle": int(angle),
                    "frame": int(digits[-1]) if digits else -1,
                }
    return None


def build_index(root: str | Path, frame_stride: int = 1) -> pd.DataFrame:
    """Walk ``root`` and return one row per silhouette frame.

    Args:
        root: dataset directory.
        frame_stride: keep every n-th frame of each sequence (1 keeps all). Consecutive
            frames are nearly identical, so a stride of 2-4 cuts extraction time with
            little information loss.
    """
    root = Path(root)
    rows = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in IMAGE_EXTENSIONS or not path.is_file():
            continue
        if path.stat().st_size == 0 or "__MACOSX" in path.parts:
            continue
        meta = parse_path(path.relative_to(root))
        if meta is None:
            continue
        rows.append({"path": str(path), **meta})

    if not rows:
        raise FileNotFoundError(f"No CASIA-B style silhouette frames found under {root}")

    df = pd.DataFrame(rows)
    df["label"] = df["condition"].map(CLASS_TO_ID).astype(int)
    df["sequence_id"] = (
        df["subject"].map("{:03d}".format)
        + "-"
        + df["condition"]
        + "-"
        + df["seq"].map("{:02d}".format)
        + "-"
        + df["angle"].map("{:03d}".format)
    )
    df = df.sort_values(["sequence_id", "frame", "path"]).reset_index(drop=True)

    if frame_stride > 1:
        rank = df.groupby("sequence_id").cumcount()
        df = df[rank % frame_stride == 0].reset_index(drop=True)
    return df


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {"train": self.train, "val": self.val, "test": self.test}


def _expand(spec: Sequence[int] | str) -> list[int]:
    """Accept [1, 74] (inclusive range) or "1-62,70" style specs."""
    if isinstance(spec, str):
        out: list[int] = []
        for chunk in spec.split(","):
            chunk = chunk.strip()
            if "-" in chunk:
                a, b = chunk.split("-")
                out.extend(range(int(a), int(b) + 1))
            elif chunk:
                out.append(int(chunk))
        return out
    if len(spec) == 2:
        return list(range(int(spec[0]), int(spec[1]) + 1))
    return [int(s) for s in spec]


def subject_split(
    df: pd.DataFrame,
    train_subjects: Iterable[int] | str = (1, 62),
    val_subjects: Iterable[int] | str = (63, 74),
    test_subjects: Iterable[int] | str = (75, 124),
) -> Split:
    """Split frames so that no subject appears in more than one partition.

    Defaults follow the common CASIA-B large-training protocol (subjects 1-74 for
    development, 75-124 for test), holding out 63-74 of the development subjects
    for validation.
    """
    groups = {
        name: set(_expand(list(s) if not isinstance(s, str) else s))
        for name, s in (("train", train_subjects), ("val", val_subjects), ("test", test_subjects))
    }
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = groups[a] & groups[b]
        if overlap:
            raise ValueError(f"Subjects {sorted(overlap)} appear in both {a} and {b}")

    idx = {k: np.flatnonzero(df["subject"].isin(v).to_numpy()) for k, v in groups.items()}
    for k, v in idx.items():
        if len(v) == 0:
            raise ValueError(f"The {k} split is empty; check the subject ranges against the data")
    return Split(**idx)


def frame_split(
    df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1, seed: int = 42
) -> Split:
    """Random stratified split over individual frames.

    This reproduces the original notebooks. It leaks: frames from one walking
    sequence land in both train and test. Kept only to measure the size of that leak.
    """
    from sklearn.model_selection import train_test_split

    all_idx = np.arange(len(df))
    y = df["label"].to_numpy()
    trval, test = train_test_split(all_idx, test_size=test_size, stratify=y, random_state=seed)
    train, val = train_test_split(
        trval, test_size=val_size / (1 - test_size), stratify=y[trval], random_state=seed
    )
    return Split(np.sort(train), np.sort(val), np.sort(test))


def make_split(df: pd.DataFrame, cfg: dict) -> Split:
    """Build the split described by the ``split`` section of the config."""
    kind = cfg.get("kind", "subject")
    if kind == "subject":
        return subject_split(
            df,
            cfg.get("train_subjects", (1, 62)),
            cfg.get("val_subjects", (63, 74)),
            cfg.get("test_subjects", (75, 124)),
        )
    if kind == "frame":
        return frame_split(
            df, cfg.get("test_size", 0.2), cfg.get("val_size", 0.1), cfg.get("seed", 42)
        )
    raise ValueError(f"Unknown split kind {kind!r}; use 'subject' or 'frame'")


def describe(df: pd.DataFrame, split: Split) -> pd.DataFrame:
    """Frame counts per split and class, plus subject and sequence counts."""
    rows = []
    for name, idx in split.as_dict().items():
        part = df.iloc[idx]
        row = {"split": name, "frames": len(part), "subjects": part["subject"].nunique(),
               "sequences": part["sequence_id"].nunique()}
        for c in CLASSES:
            row[c] = int((part["condition"] == c).sum())
        rows.append(row)
    return pd.DataFrame(rows)
