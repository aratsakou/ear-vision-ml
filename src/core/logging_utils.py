import logging
import sys
import warnings
import os

def setup_logging(level=logging.INFO):
    """
    Configures the root logger with a standard format.
    """
    # Suppress TensorFlow logs (must be done before importing TF)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ERROR and WARNING
    
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers to avoid duplication (e.g. from Hydra)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
        
    root_logger.addHandler(handler)
    
    # Capture Python warnings and route them through logging
    logging.captureWarnings(True)
    
    # Set library log levels
    # We keep them at WARNING to avoid flooding with INFO/DEBUG from internals
    # but ensure Warnings and Errors are shown.
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    logging.getLogger("absl").setLevel(logging.WARNING)
    logging.getLogger("h5py").setLevel(logging.WARNING)
    logging.getLogger("py.warnings").setLevel(logging.WARNING)
    
    return root_logger
