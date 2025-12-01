import logging
import sys
import warnings
import os
import json
from typing import Optional, Dict, Any

# Import rich and json logger if available, otherwise fallback
try:
    from rich.logging import RichHandler
    from pythonjsonlogger import jsonlogger
    HAS_RICH = True
    HAS_JSON = True
except ImportError:
    HAS_RICH = False
    HAS_JSON = False

class RunContextFilter(logging.Filter):
    """
    Injects run context into every log record.
    """
    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.context = context or {}
        
        # Auto-detect Vertex AI environment variables
        if "AIP_MODEL_DIR" in os.environ:
            self.context["vertex_job_id"] = os.environ.get("CLOUD_ML_JOB_ID", "unknown")
            self.context["vertex_task_index"] = os.environ.get("CLOUD_ML_TASK_INDEX", "0")

    def filter(self, record):
        for k, v in self.context.items():
            setattr(record, k, v)
        return True

def setup_logging(level=logging.INFO, context: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """
    Configures the root logger with a standard format.
    
    - Local (Interactive): Uses RichHandler for pretty output.
    - Cloud (Non-Interactive): Uses JsonFormatter for structured logging.
    """
    # Suppress TensorFlow logs (must be done before importing TF)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ERROR and WARNING
    
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers to avoid duplication
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
        
    # Determine environment
    # We assume cloud if AIP_MODEL_DIR is set or if not a TTY
    is_cloud = "AIP_MODEL_DIR" in os.environ
    is_tty = sys.stdout.isatty()
    
    handler = None
    
    if is_cloud or not is_tty:
        # JSON Logging for Cloud/Non-Interactive
        if HAS_JSON:
            handler = logging.StreamHandler(sys.stdout)
            formatter = jsonlogger.JsonFormatter(
                fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                rename_fields={"levelname": "severity", "asctime": "time"}
            )
            handler.setFormatter(formatter)
        else:
            # Fallback to standard if json logger missing
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            handler.setFormatter(formatter)
            
    else:
        # Rich Logging for Local Interactive
        if HAS_RICH:
            handler = RichHandler(
                rich_tracebacks=True,
                show_time=True,
                show_level=True,
                show_path=True
            )
        else:
            # Fallback
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            handler.setFormatter(formatter)

    # Add Context Filter
    context_filter = RunContextFilter(context)
    handler.addFilter(context_filter)
    
    root_logger.addHandler(handler)
    
    # Capture Python warnings
    logging.captureWarnings(True)
    
    # Set library log levels
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    logging.getLogger("absl").setLevel(logging.WARNING)
    logging.getLogger("h5py").setLevel(logging.WARNING)
    logging.getLogger("py.warnings").setLevel(logging.WARNING)
    
    return root_logger
