"""End-to-end run on synthetic features using the scikit-learn baseline (no TensorFlow)."""

import json

import numpy as np

from gaitcond.data import subject_split
from gaitcond.features import load_features
from gaitcond.reduce import fit_reducer, transform

from .conftest import subject_ranges_for


def test_reducer_is_fitted_on_train_rows_only(fake_features):
    X, index = load_features(fake_features, ["resnet101", "efficientnetb2"])
    split = subject_split(index, "1-4", "5", "6")
    reducer = fit_reducer(X, split.train, n_components=8)
    train_mean = np.asarray(X[split.train], dtype=np.float32).mean(axis=0)
    all_mean = np.asarray(X[np.arange(len(index))], dtype=np.float32).mean(axis=0)
    np.testing.assert_allclose(reducer.named_steps["scale"].mean_, train_mean, rtol=1e-4, atol=1e-4)
    assert not np.allclose(reducer.named_steps["scale"].mean_, all_mean)
    assert transform(reducer, X, split.test).shape == (len(split.test), 8)


def test_lazy_concat_matches_eager(fake_features):
    X, _ = load_features(fake_features, ["resnet101", "efficientnetb2"])
    a = np.load(fake_features / "resnet101.npy")
    b = np.load(fake_features / "efficientnetb2.npy")
    rows = np.array([0, 5, 9])
    np.testing.assert_array_equal(X[rows], np.concatenate([a[rows], b[rows]], axis=1))
    assert X.shape == (a.shape[0], a.shape[1] + b.shape[1])


def test_run_experiment_end_to_end(fake_features, tmp_path):
    from gaitcond.experiment import run_experiment

    out = tmp_path / "run"
    cfg = {
        "features_dir": str(fake_features),
        "output_dir": str(out),
        "backbones": ["resnet101", "efficientnetb2"],
        "pca_components": 8,
        "models": ["logreg"],
        "split": subject_ranges_for(None),
    }
    summary = run_experiment(cfg)

    assert summary.loc[0, "model"] == "logreg"
    # the injected signal is strong, so the baseline should be clearly beaten
    assert summary.loc[0, "frame_balanced_acc"] > 0.8
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["info"]["subjects_shared_train_test"] == 0
    for name in ("summary.csv", "split.csv", "confusion_matrix.png", "per_angle_accuracy.png",
                 "proba_test_logreg.npy", "config.yaml"):
        assert (out / name).exists(), name
