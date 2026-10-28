"""Figures for a single evaluation run."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from gaitcond import CLASS_NAMES, CLASSES  # noqa: E402

INK = "#1f2933"
MUTED = "#7b8794"
ACCENT = "#2f6fb2"


def confusion_matrix_plot(cm, path: str | Path, title: str = "") -> Path:
    cm = np.asarray(cm, dtype=float)
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    labels = [CLASS_NAMES[c] for c in CLASSES]

    fig, ax = plt.subplots(figsize=(4.2, 3.8), dpi=150)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            color = "white" if norm[i, j] > 0.6 else INK
            ax.text(j, i, f"{norm[i, j]:.1%}\n{int(cm[i, j]):,}", ha="center", va="center",
                    fontsize=8, color=color)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted", color=MUTED)
    ax.set_ylabel("True", color=MUTED)
    if title:
        ax.set_title(title, fontsize=10, color=INK, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path)
    plt.close(fig)
    return path


def per_angle_plot(per_angle: dict, path: str | Path, title: str = "") -> Path:
    angles = sorted(int(a) for a in per_angle)
    acc = [per_angle[a] if a in per_angle else per_angle[str(a)] for a in angles]

    fig, ax = plt.subplots(figsize=(5.5, 3), dpi=150)
    ax.plot(angles, acc, marker="o", color=ACCENT, linewidth=1.8)
    ax.set_xticks(angles)
    ax.set_xlabel("Camera angle (degrees)", color=MUTED)
    ax.set_ylabel("Frame accuracy", color=MUTED)
    ax.set_ylim(max(0.0, min(acc) - 0.1), 1.0)
    ax.grid(axis="y", color="#e4e7eb", linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if title:
        ax.set_title(title, fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path)
    plt.close(fig)
    return path
