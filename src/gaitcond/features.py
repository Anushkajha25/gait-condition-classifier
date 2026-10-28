"""Frozen ImageNet backbones as feature extractors.

Fixes relative to the original notebooks:
* each backbone gets its own ``preprocess_input`` on 0-255 RGB input (the notebooks
  fed every model [0, 1] images, which is wrong for ResNet, VGG and Inception);
* silhouettes are padded to square before resizing instead of being stretched;
* no augmentation is applied at extraction time (the notebooks augmented test images too);
* features are written straight to disk with a memory map, so the full dataset never
  has to sit in RAM.

The backbones are never trained here; this is a linear-probe style setup on frozen
features.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

# name -> (keras.applications class name, preprocess module, feature dim)
BACKBONES: dict[str, tuple[str, str, int]] = {
    "resnet101": ("ResNet101", "resnet", 2048),
    "efficientnetb2": ("EfficientNetB2", "efficientnet", 1408),
    "vgg16": ("VGG16", "vgg16", 512),
    "inceptionv3": ("InceptionV3", "inception_v3", 2048),
}


def feature_dim(name: str) -> int:
    return BACKBONES[name][2]


def build_extractor(name: str, image_size: int = 224, weights: str | None = "imagenet"):
    """Return ``(model, preprocess_fn)``; the model maps preprocessed images to pooled features."""
    import tensorflow as tf

    if name not in BACKBONES:
        raise ValueError(f"Unknown backbone {name!r}; choose from {sorted(BACKBONES)}")
    cls_name, module_name, _ = BACKBONES[name]
    apps = tf.keras.applications
    model = getattr(apps, cls_name)(
        weights=weights, include_top=False, pooling="avg", input_shape=(image_size, image_size, 3)
    )
    model.trainable = False
    preprocess = getattr(apps, module_name).preprocess_input
    return model, preprocess


def _make_dataset(paths: Sequence[str], image_size: int, batch_size: int):
    import tensorflow as tf

    def load(path):
        raw = tf.io.read_file(path)
        img = tf.io.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize_with_pad(tf.cast(img, tf.float32), image_size, image_size)
        return img  # float32 in [0, 255]

    ds = tf.data.Dataset.from_tensor_slices(list(paths))
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def extract_features(
    index: pd.DataFrame,
    backbones: Sequence[str],
    out_dir: str | Path,
    image_size: int = 224,
    batch_size: int = 128,
    weights: str | None = "imagenet",
    dtype: str = "float16",
) -> dict[str, Path]:
    """Extract pooled features for every frame in ``index``.

    Writes ``<out_dir>/<backbone>.npy`` (one row per index row, same order),
    ``index.csv`` and ``meta.json``. Each image is decoded once and shared by all
    backbones. Returns the paths of the feature files.
    """
    import tensorflow as tf
    from tqdm.auto import tqdm

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(index)

    extractors = {b: build_extractor(b, image_size, weights) for b in backbones}
    outputs = {
        b: np.lib.format.open_memmap(
            out_dir / f"{b}.npy", mode="w+", dtype=dtype, shape=(n, feature_dim(b))
        )
        for b in backbones
    }

    @tf.function(reduce_retracing=True)
    def run(model, x):
        return model(x, training=False)

    ds = _make_dataset(index["path"].tolist(), image_size, batch_size)
    start = 0
    for batch in tqdm(ds, total=int(np.ceil(n / batch_size)), desc="Extracting"):
        stop = start + int(batch.shape[0])
        for b, (model, preprocess) in extractors.items():
            feats = run(model, preprocess(tf.identity(batch)))
            outputs[b][start:stop] = feats.numpy().astype(dtype)
        start = stop

    for arr in outputs.values():
        arr.flush()
    index.to_csv(out_dir / "index.csv", index=False)
    meta = {
        "backbones": list(backbones),
        "image_size": image_size,
        "weights": weights,
        "dtype": dtype,
        "num_frames": n,
        "dims": {b: feature_dim(b) for b in backbones},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return {b: out_dir / f"{b}.npy" for b in backbones}


def load_features(
    feature_dir: str | Path, backbones: Sequence[str]
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load and concatenate saved features. Arrays are memory-mapped, not read into RAM."""
    feature_dir = Path(feature_dir)
    index = pd.read_csv(feature_dir / "index.csv")
    arrays = [np.load(feature_dir / f"{b}.npy", mmap_mode="r") for b in backbones]
    for b, a in zip(backbones, arrays, strict=True):
        if a.shape[0] != len(index):
            raise ValueError(f"{b}.npy has {a.shape[0]} rows but index.csv has {len(index)}")
    if len(arrays) == 1:
        return arrays[0], index
    return _LazyConcat(arrays), index


class _LazyConcat:
    """Column-wise concatenation of memory-mapped arrays, materialised only on row indexing."""

    def __init__(self, arrays: Sequence[np.ndarray]):
        self.arrays = list(arrays)
        self.shape = (arrays[0].shape[0], sum(a.shape[1] for a in arrays))
        self.dtype = arrays[0].dtype

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, rows) -> np.ndarray:
        return np.concatenate([np.asarray(a[rows]) for a in self.arrays], axis=1)
