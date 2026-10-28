"""Train one classifier on reduced features and return test-set probabilities."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from gaitcond.models import KERAS_MODELS, SKLEARN_MODELS, build_keras_model, build_sklearn_model


def class_weights(y: np.ndarray, num_classes: int = 3) -> dict[int, float]:
    counts = np.bincount(y, minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0
    return {i: float(len(y) / (num_classes * c)) for i, c in enumerate(counts)}


def train_and_predict(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    out_dir: str | Path,
    epochs: int = 40,
    batch_size: int = 256,
    patience: int = 6,
    lr: float | None = None,
    seed: int = 42,
) -> dict:
    """Fit ``name``, pick the best epoch on validation loss, predict on test.

    Returns ``{"proba_val", "proba_test", "train_seconds", "history"}``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    if name in SKLEARN_MODELS:
        model = build_sklearn_model(name, seed)
        model.fit(X_train, y_train)
        result = {
            "proba_val": model.predict_proba(X_val),
            "proba_test": model.predict_proba(X_test),
            "history": {},
        }

    elif name in KERAS_MODELS:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
        model = build_keras_model(name, X_train.shape[1], lr=lr)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=patience, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.3, patience=max(1, patience // 2), min_lr=1e-6
            ),
        ]
        hist = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights(y_train),
            callbacks=callbacks,
            verbose=2,
        )
        model.save(out_dir / f"{name}.keras")
        result = {
            "proba_val": model.predict(X_val, batch_size=4096, verbose=0),
            "proba_test": model.predict(X_test, batch_size=4096, verbose=0),
            "history": {k: [float(v) for v in vals] for k, vals in hist.history.items()},
        }
    else:
        raise ValueError(f"Unknown model {name!r}")

    result["train_seconds"] = time.time() - start
    (out_dir / f"{name}_history.json").write_text(json.dumps(result["history"], indent=2))
    return result
