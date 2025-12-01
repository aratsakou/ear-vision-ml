from typing import Any

import tensorflow as tf

from src.core.interfaces import Trainer
from src.core.training.callbacks import make_callbacks
from src.core.training.losses import classification_loss, cropper_loss, segmentation_loss
from src.core.training.metrics import classification_metrics, segmentation_metrics
from src.core.training.trainer_base import fit_model, make_optimizer


from src.core.training.distillation import Distiller

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

        # Distillation support
        if cfg.training.get("distillation", {}).get("enabled", False):
            teacher_path = cfg.training.distillation.teacher_model_path
            if not teacher_path:
                raise ValueError("Distillation enabled but teacher_model_path not provided")
            
            print(f"Loading teacher model from {teacher_path}...")
            teacher_model = tf.keras.models.load_model(teacher_path)
            
            # Wrap student in Distiller
            model = Distiller(student=model, teacher=teacher_model, cfg=cfg, student_loss_fn=loss if not isinstance(loss, str) else tf.keras.losses.get(loss))
            
            # Compile Distiller
            # Note: Distiller.compile takes student_loss_fn, not loss
            model.compile(
                optimizer=make_optimizer(cfg),
                metrics=metrics,
                student_loss_fn=loss if not isinstance(loss, str) else tf.keras.losses.get(loss)
            )
        else:
            model.compile(
                optimizer=make_optimizer(cfg), 
                loss=loss, 
                metrics=metrics
            )
        
        callbacks = make_callbacks(cfg, str(cfg.run.artifacts_dir))
        
        # Add Hyperparameter Tuning Callback if enabled
        # This is a simplified integration. In a real scenario, we might use a dedicated callback class.
        if cfg.get("tuning", {}).get("enabled", False):
            from src.core.tuning.vertex_vizier import VertexVizierTuner
            # We use a simple LambdaCallback for MVP to report metrics
            # In production, use a robust callback that handles frequency and metric selection
            tuner = VertexVizierTuner(project=cfg.get("project", "local"), location=cfg.get("location", "local"))
            
            def report_tuning_metrics(epoch, logs):
                # Report validation accuracy or loss
                metric_name = "val_accuracy" if "val_accuracy" in logs else "val_loss"
                if metric_name in logs:
                     tuner.report_metrics("current_trial", {metric_name: logs[metric_name]}, step=epoch)

            callbacks.append(tf.keras.callbacks.LambdaCallback(on_epoch_end=report_tuning_metrics))

        result = fit_model(model, cfg, train_ds, val_ds, callbacks)
        
        # Explainability Integration
        if cfg.get("explainability", {}).get("enabled", False):
            try:
                from src.core.explainability.registry import run_explainability
                print("\nRunning Explainability Framework...")
                
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
                print(f"Explainability failed: {e}")
                import traceback
                traceback.print_exc()

        return result
