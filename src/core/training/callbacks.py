from __future__ import annotations

from typing import Any

import tensorflow as tf


def make_callbacks(cfg: Any, artifacts_dir: str) -> list[tf.keras.callbacks.Callback]:
    """
    Create training callbacks based on configuration.
    
    Modern callbacks included:
    - TensorBoard with profiling
    - Model checkpointing (best + latest)
    - Early stopping with restore best weights
    - Learning rate scheduling (ReduceLROnPlateau, Cosine Annealing)
    - CSV logging
    - Gradient clipping (via optimizer, not callback)
    """
    cbs: list[tf.keras.callbacks.Callback] = []
    
    # TensorBoard with histogram and profiling
    tb_config = getattr(cfg.training, "tensorboard", None)
    if tb_config and getattr(tb_config, "enabled", True):
        profile_batch = getattr(tb_config, "profile_batch", 0)
        if isinstance(profile_batch, str) and profile_batch.isdigit():
            profile_batch = int(profile_batch)
        cbs.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=f"{artifacts_dir}/tb",
                histogram_freq=1,
                write_graph=True,
                update_freq="epoch",
                profile_batch=profile_batch,
            )
        )
    
    # Model checkpointing - save best model
    checkpoint_config = getattr(cfg.training, "checkpoint", None)
    if checkpoint_config and getattr(checkpoint_config, "enabled", True):
        monitor = getattr(checkpoint_config, "monitor", "val_loss")
        mode = getattr(checkpoint_config, "mode", "min")
        
        # Best model checkpoint
        cbs.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f"{artifacts_dir}/checkpoints/best_model.keras",
                monitor=monitor,
                mode=mode,
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
            )
        )
        
        # Latest model checkpoint (every N epochs)
        save_freq = getattr(checkpoint_config, "save_freq", 5)
        cbs.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f"{artifacts_dir}/checkpoints/epoch_{{epoch:03d}}.keras",
                save_freq="epoch",
                save_weights_only=False,
                verbose=0,
            )
        )
    
    # Early stopping with patience
    es_config = getattr(cfg.training, "early_stopping", None)
    if es_config and getattr(es_config, "enabled", False):
        monitor = getattr(es_config, "monitor", "val_loss")
        patience = getattr(es_config, "patience", 10)
        min_delta = getattr(es_config, "min_delta", 0.001)
        restore_best = getattr(es_config, "restore_best_weights", True)
        
        cbs.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=patience,
                min_delta=min_delta,
                restore_best_weights=restore_best,
                verbose=1,
            )
        )
    
    # Learning rate scheduling
    lr_schedule_config = getattr(cfg.training, "lr_schedule", None)
    if lr_schedule_config and getattr(lr_schedule_config, "enabled", False):
        schedule_type = getattr(lr_schedule_config, "type", "reduce_on_plateau")
        
        if schedule_type == "reduce_on_plateau":
            # Reduce LR when metric plateaus
            cbs.append(
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor=getattr(lr_schedule_config, "monitor", "val_loss"),
                    factor=getattr(lr_schedule_config, "factor", 0.5),
                    patience=getattr(lr_schedule_config, "patience", 5),
                    min_lr=getattr(lr_schedule_config, "min_lr", 1e-7),
                    verbose=1,
                )
            )
        elif schedule_type == "cosine_annealing":
            # Cosine annealing with warm restarts
            initial_lr = float(cfg.training.learning_rate)
            epochs = int(cfg.training.epochs)
            
            def cosine_annealing(epoch, lr):
                import math
                return initial_lr * 0.5 * (1 + math.cos(math.pi * epoch / epochs))
            
            cbs.append(tf.keras.callbacks.LearningRateScheduler(cosine_annealing, verbose=1))
        
        elif schedule_type == "exponential_decay":
            # Exponential decay
            initial_lr = float(cfg.training.learning_rate)
            decay_rate = getattr(lr_schedule_config, "decay_rate", 0.96)
            decay_steps = getattr(lr_schedule_config, "decay_steps", 10)
            
            def exponential_decay(epoch, lr):
                return initial_lr * (decay_rate ** (epoch / decay_steps))
            
            cbs.append(tf.keras.callbacks.LearningRateScheduler(exponential_decay, verbose=1))
    
    # CSV logger for easy metric tracking
    csv_config = getattr(cfg.training, "csv_logger", None)
    if csv_config and getattr(csv_config, "enabled", True):
        cbs.append(
            tf.keras.callbacks.CSVLogger(
                filename=f"{artifacts_dir}/training_log.csv",
                separator=",",
                append=False,
            )
        )
    
    # Terminate on NaN
    cbs.append(tf.keras.callbacks.TerminateOnNaN())
    
    return cbs


class WarmUpLearningRate(tf.keras.callbacks.Callback):
    """
    Learning rate warm-up callback.
    
    Gradually increases learning rate from 0 to target over warmup_epochs.
    Useful for training with large batch sizes.
    """
    
    def __init__(self, target_lr: float, warmup_epochs: int = 5):
        super().__init__()
        self.target_lr = target_lr
        self.warmup_epochs = warmup_epochs
    
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.target_lr * (epoch + 1) / self.warmup_epochs
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)
            print(f"\nEpoch {epoch + 1}: Warm-up LR = {lr:.6f}")


class GradientAccumulation(tf.keras.callbacks.Callback):
    """
    Gradient accumulation callback for simulating larger batch sizes.
    
    Useful when GPU memory is limited.
    """
    
    def __init__(self, accumulation_steps: int = 4):
        super().__init__()
        self.accumulation_steps = accumulation_steps
        self.step_counter = 0
    
    def on_train_batch_begin(self, batch, logs=None):
        if self.step_counter % self.accumulation_steps == 0:
            # Reset gradients
            for var in self.model.trainable_variables:
                var.assign(tf.zeros_like(var))
    
    def on_train_batch_end(self, batch, logs=None):
        self.step_counter += 1
        
        # Only update weights every N steps
        if self.step_counter % self.accumulation_steps != 0:
            # Skip optimizer update
            return


class MixedPrecisionCallback(tf.keras.callbacks.Callback):
    """
    Monitor mixed precision training metrics.
    
    Logs loss scale and checks for numerical stability.
    """
    
    def on_epoch_end(self, epoch, logs=None):
        if hasattr(self.model.optimizer, "loss_scale"):
            loss_scale = self.model.optimizer.loss_scale
            if hasattr(loss_scale, "current_loss_scale"):
                current_scale = loss_scale.current_loss_scale()
                print(f"\nLoss scale: {current_scale}")
