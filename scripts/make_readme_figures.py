"""Regenerate the README figures from results/original_frame_split/all_runs.csv."""

from __future__ import annotations

import ast
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gaitcond.plots import confusion_matrix_plot  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "original_frame_split" / "all_runs.csv"
OUT = ROOT / "docs" / "figures"

VGG, RES = "vgg16+inceptionv3", "resnet101+efficientnetb2"
C_VGG, C_RES = "#eb6834", "#2a78d6"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

# Heads that were run with identical architecture on both backbone pairs.
LABELS = {
    "Enhanced_CNN": "Residual MLP",
    "BiLSTM": "BiLSTM (length-1 input)",
    "Enhanced_LSTM": "LSTM (length-1 input)",
    "GaitNet": "MLP 512-256-128",
    "MLP_Recognition": "MLP",
    "VisionTransformer": "MLP (\"ViT\", no attention)",
    "CNN_LSTM": "Conv1D + LSTM over PCA dims",
    "CNN_BiLSTM": "Conv1D + BiLSTM over PCA dims",
}


def backbone_comparison(df: pd.DataFrame, path: Path) -> None:
    wide = df.pivot_table(index="original_name", columns="backbones", values="accuracy")
    wide = wide.loc[[k for k in LABELS if k in wide.index]].sort_values(RES)
    y = np.arange(len(wide))
    baseline = float(df["majority_baseline"].iloc[0])

    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axvline(baseline * 100, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(baseline * 100 + 0.4, 0.5, f"majority class\n{baseline:.1%}",
            color=MUTED, fontsize=8, va="center")
    for i, (_, row) in enumerate(wide.iterrows()):
        ax.plot([row[VGG] * 100, row[RES] * 100], [i, i], color=GRID, lw=2, zorder=2)
        gap = (row[RES] - row[VGG]) * 100
        ax.text(row[RES] * 100 + 0.8, i, f"+{gap:.0f} pts", va="center", fontsize=8, color=MUTED)
    ax.scatter(wide[VGG] * 100, y, s=64, color=C_VGG, edgecolor=SURFACE, lw=2, zorder=3,
               label="VGG16 + InceptionV3")
    ax.scatter(wide[RES] * 100, y, s=64, color=C_RES, edgecolor=SURFACE, lw=2, zorder=3,
               label="ResNet101 + EfficientNetB2")

    ax.set_yticks(y, [LABELS[k] for k in wide.index], color=INK, fontsize=9)
    ax.set_xlim(55, 100)
    ax.set_xlabel("Test accuracy (%), original frame-level split", color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Same classifier heads, different frozen backbones", loc="left", color=INK,
                 fontsize=11, pad=24)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=8.5,
              labelcolor=INK, handletextpad=0.3, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RUNS)
    backbone_comparison(df, OUT / "backbone_comparison.png")

    best = df.loc[df["accuracy"].idxmax()]
    cm = np.array(ast.literal_eval(best["confusion_matrix"]))
    confusion_matrix_plot(
        cm, OUT / "original_confusion_matrix.png",
        title=f"Original best run (leaky frame split), acc {best['accuracy']:.1%}",
    )
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
