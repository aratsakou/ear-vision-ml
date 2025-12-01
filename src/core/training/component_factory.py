from typing import Any, List, Union, Callable
import tensorflow as tf

from src.core.training.losses import classification_loss, cropper_loss, segmentation_loss
from src.core.training.metrics import classification_metrics, segmentation_metrics
from src.core.training.trainer_base import make_optimizer
from src.core.training.callbacks import make_callbacks

class TrainingComponentFactory:
    """
    Factory for creating training components based on configuration.
    """
    
    def create_loss(self, cfg: Any) -> Union[tf.keras.losses.Loss, str]:
        task_name = str(cfg.task.name).lower()
        
        # Check for explicit loss configuration
        loss_type = cfg.get("training", {}).get("loss", {}).get("type", None)
        
        if loss_type == "focal":
            from src.core.training.losses import focal_loss
            alpha = cfg.training.loss.get("alpha", 0.25)
            gamma = cfg.training.loss.get("gamma", 2.0)
            return focal_loss(alpha=alpha, gamma=gamma)
        
        if task_name == "classification":
            return classification_loss()
        elif task_name == "segmentation":
            return segmentation_loss()
        elif task_name == "cropper":
            return cropper_loss()
        else:
            return "mse"

    def create_metrics(self, cfg: Any) -> List[Union[tf.keras.metrics.Metric, str]]:
        task_name = str(cfg.task.name).lower()
        
        if task_name == "classification":
            return classification_metrics()
        elif task_name == "segmentation":
            return segmentation_metrics()
        elif task_name == "cropper":
            return []
        else:
            return ["accuracy"]

    def create_optimizer(self, cfg: Any) -> tf.keras.optimizers.Optimizer:
        return make_optimizer(cfg)

    def create_callbacks(self, cfg: Any, artifacts_dir: str) -> List[tf.keras.callbacks.Callback]:
        return make_callbacks(cfg, artifacts_dir)
