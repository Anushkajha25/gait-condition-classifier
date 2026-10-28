"""TensorFlow-dependent tests. Skipped automatically when TensorFlow isn't installed."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from gaitcond.data import build_index  # noqa: E402
from gaitcond.features import extract_features, load_features  # noqa: E402
from gaitcond.models import KERAS_MODELS, build_keras_model  # noqa: E402
from gaitcond.train import train_and_predict  # noqa: E402


@pytest.mark.parametrize("name", KERAS_MODELS)
def test_keras_models_output_probabilities(name):
    model = build_keras_model(name, input_dim=16)
    out = model.predict(np.random.rand(4, 16).astype("float32"), verbose=0)
    assert out.shape == (4, 3)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, rtol=1e-5)


def test_train_and_predict_runs(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 8)).astype("float32")
    y = np.repeat([0, 1, 2], 20)
    res = train_and_predict("mlp", X, y, X, y, X, tmp_path, epochs=2, batch_size=16)
    assert res["proba_test"].shape == (60, 3)
    assert (tmp_path / "mlp.keras").exists()


def test_extract_features_shapes(fake_casia, tmp_path):
    index = build_index(fake_casia).head(10)
    extract_features(index, ["efficientnetb2"], tmp_path / "f", image_size=32, batch_size=4,
                     weights=None)
    X, idx = load_features(tmp_path / "f", ["efficientnetb2"])
    assert X.shape == (10, 1408)
    assert len(idx) == 10
    assert np.isfinite(np.asarray(X)).all()
