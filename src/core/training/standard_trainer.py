from typing import Any, Optional
import logging

import tensorflow as tf

from src.core.interfaces import Trainer, Component
from src.core.training.component_factory import TrainingComponentFactory
from src.core.training.distillation import Distiller

log = logging.getLogger(__name__)

class StandardTrainer(Trainer, Component):
    def initialize(self) -> None:
        log.info("StandardTrainer initialized")

    def cleanup(self) -> None:
        log.info("StandardTrainer cleaned up")
    def __init__(self, component_factory: Optional[TrainingComponentFactory] = None):
        self.component_factory = component_factory or TrainingComponentFactory()

    def train(self, model: tf.keras.Model, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, cfg: Any) -> Any:
        from src.core.training.trainer_base import fit_model # Keep this import to avoid circular dep if any, or move it
        
        loss = self.component_factory.create_loss(cfg)
        metrics = self.component_factory.create_metrics(cfg)
        optimizer = self.component_factory.create_optimizer(cfg)
        
        log.debug(f"Created loss: {loss}")
        log.debug(f"Created metrics: {metrics}")
        log.debug(f"Created optimizer: {optimizer}")
        
        # Distillation support
        if cfg.training.get("distillation", {}).get("enabled", False):
            teacher_path = cfg.training.distillation.teacher_model_path
            if not teacher_path:
                raise ValueError("Distillation enabled but teacher_model_path not provided")
            
            log.info(f"Loading teacher model from {teacher_path}...")
            teacher_model = tf.keras.models.load_model(teacher_path)
            
            # Wrap student in Distiller
            # Distiller needs the student loss function
            student_loss_fn = loss if not isinstance(loss, str) else tf.keras.losses.get(loss)
            model = Distiller(student=model, teacher=teacher_model, cfg=cfg, student_loss_fn=student_loss_fn)
            
            # Compile Distiller
            model.compile(
                optimizer=optimizer,
                metrics=metrics,
                student_loss_fn=student_loss_fn
            )
        else:
            model.compile(
                optimizer=optimizer, 
                loss=loss, 
                metrics=metrics
            )
        
        # Ensure artifacts directory exists
        import os
        os.makedirs(cfg.run.artifacts_dir, exist_ok=True)
        
        callbacks = self.component_factory.create_callbacks(cfg, str(cfg.run.artifacts_dir))
        
        # Add Hyperparameter Tuning Callback if enabled
        if cfg.training.get("tuning", {}).get("enabled", False):
            from src.core.tuning.vertex_vizier import VertexVizierTuner
            tuner = VertexVizierTuner(project=cfg.get("project", "local"), location=cfg.get("location", "local"))
            
            def report_tuning_metrics(epoch, logs):
                metric_name = "val_accuracy" if "val_accuracy" in logs else "val_loss"
                if metric_name in logs:
                     tuner.report_metrics("current_trial", {metric_name: logs[metric_name]}, step=epoch)

            callbacks.append(tf.keras.callbacks.LambdaCallback(on_epoch_end=report_tuning_metrics))

        result = fit_model(model, cfg, train_ds, val_ds, callbacks)
        
        # Explainability Integration
        explain_cfg = cfg.get("explainability", {})
        if explain_cfg.get("enabled", False):
            try:
                from src.core.explainability.registry import run_explainability
                log.info("\nRunning Explainability Framework...")
                
                run_ctx = {
                    "run_id": cfg.run.name,
                    "artifacts_dir": cfg.run.artifacts_dir
                }
                
                datasets = {
                    "train": train_ds,
                    "val": val_ds
                }
                
                run_explainability(cfg, run_ctx, model, datasets)
            except Exception as e:
                log.error(f"Explainability failed: {e}")
                import traceback
                traceback.print_exc() # Keep traceback printing for now or use log.exception

        return result
