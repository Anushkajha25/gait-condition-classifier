"""Classifiers that sit on top of the (reduced) frozen features.

Every model here sees one frame's feature vector, so they are all feed-forward. The
original notebooks also had LSTM/BiLSTM models fed a length-1 "sequence" (equivalent
to an MLP) and Conv1D/LSTM hybrids that slid over PCA components (which have no
spatial or temporal order). Those are dropped; see README "What changed".

Mapping to the original notebooks:
* ``residual_mlp`` = "Enhanced_CNN" in notebook 3 (best original model)
* ``mlp``          = "GaitNet" / "VisionTransformer" / "MLP_Recognition" in notebook 4
* ``linear``       = new: a linear probe, the baseline any deeper head has to beat
* ``logreg``       = new: scikit-learn logistic regression, runs without TensorFlow
"""

from __future__ import annotations

KERAS_MODELS = ("linear", "mlp", "residual_mlp")
SKLEARN_MODELS = ("logreg",)
ALL_MODELS = KERAS_MODELS + SKLEARN_MODELS


def build_keras_model(name: str, input_dim: int, num_classes: int = 3, lr: float | None = None):
    import tensorflow as tf

    layers = tf.keras.layers
    inputs = tf.keras.Input(shape=(input_dim,))

    if name == "linear":
        outputs = layers.Dense(num_classes, activation="softmax")(inputs)
        default_lr = 1e-3

    elif name == "mlp":
        x = inputs
        for units, drop in ((512, 0.3), (256, 0.3), (128, 0.2)):
            x = layers.Dense(units, activation="relu")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(drop)(x)
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        default_lr = 2e-4

    elif name == "residual_mlp":
        x = layers.Dense(2048, activation="relu")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        for units, drop in ((1024, 0.3), (512, 0.25)):
            shortcut = layers.Dense(units)(x)
            x = layers.Dense(units, activation="relu")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(drop)(x)
            x = layers.Dense(units, activation="relu")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(layers.Add()([shortcut, x]))
        x = layers.Dense(256, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        default_lr = 2e-4

    else:
        raise ValueError(f"Unknown Keras model {name!r}; choose from {KERAS_MODELS}")

    model = tf.keras.Model(inputs, outputs, name=name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr or default_lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_sklearn_model(name: str, seed: int = 42):
    if name == "logreg":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    raise ValueError(f"Unknown scikit-learn model {name!r}; choose from {SKLEARN_MODELS}")
