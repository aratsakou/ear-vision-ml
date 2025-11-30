from typing import Any

import tensorflow as tf

from src.core.interfaces import Trainer
from src.core.training.callbacks import make_callbacks
from src.core.training.losses import classification_loss, cropper_loss, segmentation_loss
from src.core.training.metrics import classification_metrics, segmentation_metrics
from src.core.training.trainer_base import fit_model, make_optimizer


class StandardTrainer(Trainer):
    def train(self, model: tf.keras.Model, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, cfg: Any) -> Any:
        task_name = str(cfg.task.name).lower()
        
        if task_name == "classification":
            loss = classification_loss()
            metrics = classification_metrics()
        elif task_name == "segmentation":
            loss = segmentation_loss()
            metrics = segmentation_metrics()
        elif task_name == "cropper":
            loss = cropper_loss()
            metrics = [] # Cropper usually just minimizes loss, or we can add IoU metric if available
        else:
            # Default or fallback
            loss = "mse"
            metrics = ["accuracy"]

        model.compile(
            optimizer=make_optimizer(cfg), 
            loss=loss, 
            metrics=metrics
        )
        
        callbacks = make_callbacks(cfg, str(cfg.run.artifacts_dir))
        
        result = fit_model(model, cfg, train_ds, val_ds, callbacks)
        return result
