from typing import Any, List, Union, Callable
import tensorflow as tf

from src.core.training.losses import get_loss
from src.core.training.metrics import get_metrics
from src.core.training.trainer_base import make_optimizer
from src.core.training.callbacks import make_callbacks

class TrainingComponentFactory:
    """
    Factory for creating training components based on configuration.
    """
    
    def create_loss(self, cfg: Any) -> Union[tf.keras.losses.Loss, str]:
        task_name = str(cfg.task.name).lower()
        
        # Check for explicit loss configuration
        training_cfg = cfg.get("training", {})
        loss_cfg = training_cfg.get("loss", {})
        
        loss_name = None
        loss_kwargs = {}
        
        if isinstance(loss_cfg, dict):
            loss_name = loss_cfg.get("type", None)
            # Extract other kwargs
            loss_kwargs = {k: v for k, v in loss_cfg.items() if k != "type"}
        elif isinstance(loss_cfg, str):
            loss_name = loss_cfg
            
        if loss_name:
            try:
                return get_loss(loss_name, **loss_kwargs)
            except ValueError:
                # Fallback or re-raise? Let's try to map task defaults if name is unknown?
                # Actually get_loss raises ValueError if unknown, which is good.
                pass

        # Default defaults if no explicit config
        if task_name == "classification":
            return get_loss("categorical_crossentropy")
        elif task_name == "segmentation":
            return get_loss("categorical_crossentropy")
        elif task_name == "cropper":
            return get_loss("huber")
        else:
            return "mse"

    def create_metrics(self, cfg: Any) -> List[Union[tf.keras.metrics.Metric, str]]:
        task_name = str(cfg.task.name).lower()
        
        # Check for explicit metrics configuration (future proofing)
        # For now, we just use the task default from get_metrics
        # But we could allow passing kwargs if needed
        
        return get_metrics(task_name)

    def create_optimizer(self, cfg: Any) -> tf.keras.optimizers.Optimizer:
        return make_optimizer(cfg)

    def create_callbacks(self, cfg: Any, artifacts_dir: str) -> List[tf.keras.callbacks.Callback]:
        return make_callbacks(cfg, artifacts_dir)
