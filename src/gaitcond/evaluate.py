"""Metrics that don't hide behind class imbalance or near-duplicate frames.

Beyond accuracy this reports balanced accuracy, macro F1, the majority-class baseline,
per-class recall, per-angle accuracy, and sequence-level accuracy (average the frame
probabilities over each walking sequence, then classify the sequence once).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)

from gaitcond import CLASSES


def majority_baseline(y_train: np.ndarray, y_test: np.ndarray) -> float:
    """Accuracy of always predicting the most common training class."""
    majority = np.bincount(y_train).argmax()
    return float(np.mean(y_test == majority))


def frame_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = list(range(len(CLASSES)))
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        "recall": {c: float(r) for c, r in zip(CLASSES, recalls, strict=True)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def per_angle_accuracy(meta: pd.DataFrame, y_pred: np.ndarray) -> dict[int, float]:
    correct = meta["label"].to_numpy() == y_pred
    s = pd.Series(correct).groupby(meta["angle"].to_numpy()).mean()
    return {int(k): float(v) for k, v in s.items()}


def sequence_predictions(meta: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
    """Average frame probabilities within each walking sequence."""
    cols = [f"p_{c}" for c in CLASSES]
    df = pd.DataFrame(proba, columns=cols)
    df["sequence_id"] = meta["sequence_id"].to_numpy()
    df["label"] = meta["label"].to_numpy()
    df["angle"] = meta["angle"].to_numpy()
    seq = df.groupby("sequence_id").agg({**{c: "mean" for c in cols}, "label": "first",
                                        "angle": "first"})
    seq["pred"] = seq[cols].to_numpy().argmax(axis=1)
    return seq.reset_index()


def evaluate(
    meta_test: pd.DataFrame, proba_test: np.ndarray, y_train: np.ndarray | None = None
) -> dict:
    """Full evaluation of test-set probabilities. ``meta_test`` rows align with ``proba_test``."""
    meta_test = meta_test.reset_index(drop=True)
    y_true = meta_test["label"].to_numpy()
    y_pred = np.asarray(proba_test).argmax(axis=1)

    out = {"frames": frame_metrics(y_true, y_pred)}
    out["frames"]["per_angle_accuracy"] = per_angle_accuracy(meta_test, y_pred)
    out["frames"]["n"] = int(len(y_true))

    seq = sequence_predictions(meta_test, proba_test)
    out["sequences"] = frame_metrics(seq["label"].to_numpy(), seq["pred"].to_numpy())
    out["sequences"]["n"] = int(len(seq))

    if y_train is not None:
        out["majority_baseline"] = majority_baseline(np.asarray(y_train), y_true)
    return out


def summary_row(name: str, result: dict) -> dict:
    f, s = result["frames"], result["sequences"]
    row = {
        "model": name,
        "frame_acc": f["accuracy"],
        "frame_balanced_acc": f["balanced_accuracy"],
        "frame_macro_f1": f["macro_f1"],
        "seq_acc": s["accuracy"],
        "seq_balanced_acc": s["balanced_accuracy"],
        "majority_baseline": result.get("majority_baseline"),
    }
    for c in CLASSES:
        row[f"recall_{c}"] = f["recall"][c]
    return row
