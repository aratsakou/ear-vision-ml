from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import tensorflow as tf


@dataclass
class TrainResult:
    history: tf.keras.callbacks.History


def make_optimizer(cfg: Any) -> tf.keras.optimizers.Optimizer:
    lr = float(cfg.training.learning_rate)
    opt = str(cfg.training.optimizer).lower()
    if opt == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr)
    raise ValueError(f"Unsupported optimizer: {opt}")


def fit_model(
    model: tf.keras.Model,
    cfg: Any,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset | None,
    callbacks: list[tf.keras.callbacks.Callback],
) -> TrainResult:
    epochs = int(cfg.training.epochs)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )
    return TrainResult(history=history)
