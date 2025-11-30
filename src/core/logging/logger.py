"""
Multi-layered logging system for ear-vision-ml.

Logging Layers:
1. Console logging (INFO+) - User-facing messages
2. File logging (DEBUG+) - Detailed logs for debugging
3. Structured JSON logging - Machine-readable logs
4. Performance logging - Timing and metrics
5. Experiment tracking - ML-specific logging (Vertex, TensorBoard)

Features:
- Hierarchical loggers (root, module, task)
- Configurable log levels per layer
- Automatic log rotation
- Structured logging with context
- Performance profiling
- Integration with Vertex AI
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ANSI color codes for console
COLORS = {
    "DEBUG": "\033[36m",  # Cyan
    "INFO": "\033[32m",  # Green
    "WARNING": "\033[33m",  # Yellow
    "ERROR": "\033[31m",  # Red
    "CRITICAL": "\033[35m",  # Magenta
    "RESET": "\033[0m",
}


@dataclass
class LogContext:
    """Context information for structured logging."""
    task_name: str | None = None
    run_id: str | None = None
    experiment_id: str | None = None
    model_name: str | None = None
    dataset_id: str | None = None
    epoch: int | None = None
    step: int | None = None


class ColoredFormatter(logging.Formatter):
    """Formatter that adds colors to console output."""
    
    def format(self, record: logging.LogRecord) -> str:
        levelname = record.levelname
        if levelname in COLORS:
            record.levelname = f"{COLORS[levelname]}{levelname}{COLORS['RESET']}"
        return super().format(record)


class StructuredFormatter(logging.Formatter):
    """Formatter that outputs structured JSON logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra context if available
        if hasattr(record, "context"):
            log_data["context"] = asdict(record.context)
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


class PerformanceLogger:
    """Logger for performance metrics and profiling."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.timers: dict[str, float] = {}
    
    def start_timer(self, name: str) -> None:
        """Start a named timer."""
        self.timers[name] = time.perf_counter()
    
    def stop_timer(self, name: str, log_level: int = logging.INFO) -> float:
        """Stop a timer and log the elapsed time."""
        if name not in self.timers:
            self.logger.warning(f"Timer '{name}' was not started")
            return 0.0
        
        elapsed = time.perf_counter() - self.timers[name]
        self.logger.log(log_level, f"⏱️  {name}: {elapsed:.4f}s")
        del self.timers[name]
        return elapsed
    
    def log_metric(self, name: str, value: float, unit: str = "") -> None:
        """Log a performance metric."""
        unit_str = f" {unit}" if unit else ""
        self.logger.info(f"📊 {name}: {value:.4f}{unit_str}")


def setup_logging(
    log_dir: str | Path | None = None,
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    structured_logging: bool = True,
    log_to_file: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> logging.Logger:
    """
    Setup multi-layered logging system.
    
    Args:
        log_dir: Directory for log files (None = artifacts/logs)
        console_level: Console logging level
        file_level: File logging level
        structured_logging: Enable JSON structured logging
        log_to_file: Enable file logging
        max_bytes: Max size per log file before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        Root logger instance
    """
    # Get root logger
    root_logger = logging.getLogger("ear_vision_ml")
    root_logger.setLevel(logging.DEBUG)  # Capture everything, filter at handlers
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # 1. Console Handler (colored, INFO+)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, console_level.upper()))
    console_formatter = ColoredFormatter(
        fmt="%(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # 2. File Handler (detailed, DEBUG+)
    if log_to_file:
        log_dir = Path("artifacts/logs") if log_dir is None else Path(log_dir)
        
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_dir / "ear_vision_ml.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(getattr(logging, file_level.upper()))
        file_formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # 3. Structured JSON Handler (machine-readable)
    if structured_logging and log_to_file:
        json_handler = logging.handlers.RotatingFileHandler(
            filename=log_dir / "ear_vision_ml.json",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        json_handler.setLevel(logging.DEBUG)
        json_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(json_handler)
    
    # Prevent propagation to root
    root_logger.propagate = False
    
    return root_logger


def get_logger(name: str, context: LogContext | None = None) -> logging.Logger:
    """
    Get a logger with optional context.
    
    Args:
        name: Logger name (typically __name__)
        context: Optional context information
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(f"ear_vision_ml.{name}")
    
    # Attach context if provided
    if context:
        # Create a custom LoggerAdapter that adds context
        class ContextAdapter(logging.LoggerAdapter):
            def process(self, msg, kwargs):
                # Add context to extra
                if "extra" not in kwargs:
                    kwargs["extra"] = {}
                kwargs["extra"]["context"] = context
                return msg, kwargs
        
        return ContextAdapter(logger, {})
    
    return logger


def log_experiment_start(
    logger: logging.Logger,
    config: dict[str, Any],
    run_id: str,
) -> None:
    """Log experiment start with configuration."""
    logger.info("=" * 80)
    logger.info(f"🚀 Starting experiment: {run_id}")
    logger.info("=" * 80)
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 80)


def log_experiment_end(
    logger: logging.Logger,
    run_id: str,
    metrics: dict[str, float],
    duration: float,
) -> None:
    """Log experiment end with final metrics."""
    logger.info("=" * 80)
    logger.info(f"✅ Experiment completed: {run_id}")
    logger.info(f"⏱️  Total duration: {duration:.2f}s")
    logger.info("📊 Final metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    logger.info("=" * 80)


def log_model_summary(
    logger: logging.Logger,
    model: Any,
    input_shape: tuple[int, ...],
) -> None:
    """Log model architecture summary."""
    try:
        import tensorflow as tf
        
        if isinstance(model, tf.keras.Model):
            logger.info("=" * 80)
            logger.info("🏗️  Model Architecture")
            logger.info("=" * 80)
            
            # Count parameters
            total_params = model.count_params()
            trainable_params = sum(tf.size(w).numpy() for w in model.trainable_weights)
            non_trainable_params = sum(tf.size(w).numpy() for w in model.non_trainable_weights)
            
            logger.info(f"Input shape: {input_shape}")
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,}")
            logger.info(f"Non-trainable parameters: {non_trainable_params:,}")
            logger.info(f"Number of layers: {len(model.layers)}")
            logger.info("=" * 80)
    except Exception as e:
        logger.warning(f"Could not log model summary: {e}")


def log_dataset_info(
    logger: logging.Logger,
    dataset_id: str,
    splits: dict[str, int],
    num_classes: int | None = None,
) -> None:
    """Log dataset information."""
    logger.info("=" * 80)
    logger.info(f"📁 Dataset: {dataset_id}")
    logger.info("=" * 80)
    
    total_samples = sum(splits.values())
    logger.info(f"Total samples: {total_samples:,}")
    
    for split, count in splits.items():
        percentage = (count / total_samples * 100) if total_samples > 0 else 0
        logger.info(f"  {split}: {count:,} ({percentage:.1f}%)")
    
    if num_classes:
        logger.info(f"Number of classes: {num_classes}")
    
    logger.info("=" * 80)


def log_training_progress(
    logger: logging.Logger,
    epoch: int,
    total_epochs: int,
    metrics: dict[str, float],
    learning_rate: float | None = None,
) -> None:
    """Log training progress for an epoch."""
    logger.info(f"📈 Epoch {epoch}/{total_epochs}")
    
    if learning_rate:
        logger.info(f"  Learning rate: {learning_rate:.6f}")
    
    for metric_name, value in metrics.items():
        logger.info(f"  {metric_name}: {value:.4f}")


class ExperimentLogger:
    """
    High-level logger for ML experiments.
    
    Combines multiple logging layers:
    - Standard logging
    - Performance tracking
    - Experiment tracking (Vertex AI)
    """
    
    def __init__(
        self,
        name: str,
        run_id: str,
        log_dir: str | Path | None = None,
        context: LogContext | None = None,
    ):
        self.name = name
        self.run_id = run_id
        self.context = context or LogContext(run_id=run_id)
        
        # Setup logging if not already done
        if not logging.getLogger("ear_vision_ml").handlers:
            setup_logging(log_dir=log_dir)
        
        # Get logger
        self.logger = get_logger(name, context=self.context)
        
        # Performance logger
        self.perf = PerformanceLogger(self.logger)
        
        # Vertex AI logger (optional)
        self.vertex_logger = None
        try:
            from src.core.logging.vertex_experiments import VertexExperimentLogger
            self.vertex_logger = VertexExperimentLogger(
                enabled=False,  # Will be enabled via config
                experiment_name="",
                run_name=run_id,
            )
        except ImportError:
            pass
    
    def log_config(self, config: dict[str, Any]) -> None:
        """Log experiment configuration."""
        log_experiment_start(self.logger, config, self.run_id)
        
        if self.vertex_logger and self.vertex_logger.enabled:
            self.vertex_logger.log_params(config)
    
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics to all layers."""
        for name, value in metrics.items():
            self.logger.info(f"📊 {name}: {value:.4f}")
        
        if self.vertex_logger and self.vertex_logger.enabled:
            self.vertex_logger.log_metrics(metrics)
    
    def log_artifact(self, artifact_path: str | Path, artifact_type: str = "model") -> None:
        """Log artifact path."""
        self.logger.info(f"💾 Saved {artifact_type}: {artifact_path}")
    
    def start_timer(self, name: str) -> None:
        """Start a performance timer."""
        self.perf.start_timer(name)
    
    def stop_timer(self, name: str) -> float:
        """Stop a performance timer."""
        return self.perf.stop_timer(name)
    
    def finalize(self, final_metrics: dict[str, float], duration: float) -> None:
        """Finalize experiment logging."""
        log_experiment_end(self.logger, self.run_id, final_metrics, duration)


# Convenience functions for common logging patterns
def debug(msg: str, **kwargs) -> None:
    """Log debug message."""
    logging.getLogger("ear_vision_ml").debug(msg, **kwargs)


def info(msg: str, **kwargs) -> None:
    """Log info message."""
    logging.getLogger("ear_vision_ml").info(msg, **kwargs)


def warning(msg: str, **kwargs) -> None:
    """Log warning message."""
    logging.getLogger("ear_vision_ml").warning(msg, **kwargs)


def error(msg: str, **kwargs) -> None:
    """Log error message."""
    logging.getLogger("ear_vision_ml").error(msg, **kwargs)


def critical(msg: str, **kwargs) -> None:
    """Log critical message."""
    logging.getLogger("ear_vision_ml").critical(msg, **kwargs)
