"""
Constants for data loading and processing to avoid magic numbers.
"""

# Dataset Loading Constants
SHUFFLE_BUFFER_SIZE = 1000  # Balance between randomness and memory usage

# Dataset Oversampling
OVERSAMPLE_CHUNK_SIZE = 10000  # Number of rows to process per chunk when oversampling

# Export & Benchmarking Constants
QUANTIZATION_CALIBRATION_SAMPLES = 100  # Number of samples for TFLite quantization calibration
BENCHMARK_WARMUP_RUNS = 10  # Number of warm-up runs before benchmarking
BENCHMARK_MEASUREMENT_RUNS = 100  # Number of runs to measure for benchmarking

# ROI Validation
ROI_BBOX_EPSILON = 1e-6  # Minimum bbox dimension to avoid degenerate boxes

# Git Subprocess
GIT_COMMAND_TIMEOUT = 5  # Timeout in seconds for git subprocess commands
