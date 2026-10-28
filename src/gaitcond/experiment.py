"""End-to-end experiment: saved features -> split -> reduce -> train -> evaluate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from gaitcond.data import describe, make_split
from gaitcond.evaluate import evaluate, summary_row
from gaitcond.features import load_features
from gaitcond.plots import confusion_matrix_plot, per_angle_plot
from gaitcond.reduce import explained_variance, fit_reducer, transform
from gaitcond.train import train_and_predict


def load_config(path: str | Path, overrides: dict | None = None) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    for key, value in (overrides or {}).items():
        if value is None:
            continue
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = value
    return cfg


def run_experiment(cfg: dict) -> pd.DataFrame:
    """Train and evaluate every model in ``cfg['models']``; returns the summary table."""
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    backbones = cfg["backbones"]
    X, index = load_features(cfg["features_dir"], backbones)
    split = make_split(index, cfg.get("split", {}))
    split_table = describe(index, split)
    split_table.to_csv(out_dir / "split.csv", index=False)
    print(split_table.to_string(index=False))

    train_subjects = set(index.iloc[split.train]["subject"])
    test_subjects = set(index.iloc[split.test]["subject"])
    shared = sorted(train_subjects & test_subjects)
    print(f"Subjects shared between train and test: {len(shared)}")

    seed = cfg.get("seed", 42)
    reducer = fit_reducer(
        X, split.train, cfg.get("pca_components", 256), cfg.get("pca_fit_samples", 100_000), seed
    )
    Xs = {k: transform(reducer, X, v) for k, v in split.as_dict().items()}
    ys = {k: index.iloc[v]["label"].to_numpy() for k, v in split.as_dict().items()}
    meta_test = index.iloc[split.test].reset_index(drop=True)

    info = {
        "backbones": backbones,
        "split": cfg.get("split", {}).get("kind", "subject"),
        "input_dim": int(X.shape[1]),
        "reduced_dim": int(Xs["train"].shape[1]),
        "pca_explained_variance": explained_variance(reducer),
        "subjects_shared_train_test": len(shared),
    }

    tcfg = cfg.get("train", {})
    rows, all_results = [], {}
    for name in cfg["models"]:
        print(f"\n=== {name} ===")
        fit = train_and_predict(
            name, Xs["train"], ys["train"], Xs["val"], ys["val"], Xs["test"], out_dir / "models",
            epochs=tcfg.get("epochs", 40), batch_size=tcfg.get("batch_size", 256),
            patience=tcfg.get("patience", 6), seed=seed,
        )
        result = evaluate(meta_test, fit["proba_test"], ys["train"])
        result["train_seconds"] = fit["train_seconds"]
        all_results[name] = result
        np.save(out_dir / f"proba_test_{name}.npy", fit["proba_test"].astype(np.float32))

        row = summary_row(name, result)
        row["train_seconds"] = round(fit["train_seconds"], 1)
        rows.append(row)
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})

    summary = pd.DataFrame(rows).sort_values("frame_balanced_acc", ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps({"info": info, "results": all_results},
                                                     indent=2))

    best = summary.iloc[0]["model"]
    label = f"{'+'.join(backbones)} · {info['split']} split · {best}"
    confusion_matrix_plot(all_results[best]["frames"]["confusion_matrix"],
                          out_dir / "confusion_matrix.png", label)
    per_angle_plot(all_results[best]["frames"]["per_angle_accuracy"],
                   out_dir / "per_angle_accuracy.png", label)
    print(f"\nResults written to {out_dir}")
    return summary
