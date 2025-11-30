# Advanced Logging and Reporting System

## Overview
Implemented a comprehensive multi-layered logging and reporting system for the ear-vision-ml repository, providing detailed tracking of experiments from setup through training to final results.

## 🎯 Logging Layers

### 1. Console Logging (INFO+)
- **Purpose**: User-facing messages
- **Features**:
  - Colored output for better readability
  - Clean, concise messages
  - Real-time progress updates
  - Emoji indicators for different message types

### 2. File Logging (DEBUG+)
- **Purpose**: Detailed debugging information
- **Features**:
  - Rotating file handler (10MB max, 5 backups)
  - Detailed timestamps and context
  - Function names and line numbers
  - Full stack traces for errors

### 3. Structured JSON Logging
- **Purpose**: Machine-readable logs
- **Features**:
  - JSON format for easy parsing
  - Structured context information
  - Integration with log analysis tools
  - Separate JSON log file

### 4. Performance Logging
- **Purpose**: Timing and profiling
- **Features**:
  - Named timers
  - Automatic elapsed time calculation
  - Metric logging with units
  - Performance bottleneck identification

### 5. Experiment Tracking
- **Purpose**: ML-specific logging
- **Features**:
  - Vertex AI Experiments integration
  - TensorBoard logging
  - Parameter and metric tracking
  - Artifact logging

## 📊 Reporting System

### Setup Reports
Comprehensive experiment setup documentation including:
- **Configuration**: All hyperparameters and settings
- **Dataset Information**: Splits, class distribution, preprocessing
- **Model Architecture**: Parameters, layers, optimizer, loss
- **Environment**: Git commit, Python version, dependencies

### Training Progress Reports
Real-time training updates with:
- **Epoch Progress**: Current epoch, total epochs
- **Metrics**: All tracked metrics per epoch
- **Learning Rate**: Current LR value
- **Time Estimates**: Time per epoch, ETA

### Results Reports
Final experiment results with:
- **Final Metrics**: End-of-training performance
- **Best Metrics**: Best performance achieved
- **Training History**: Complete metric history
- **Artifacts**: Paths to saved models and logs
- **Status**: Completed, failed, or stopped

### Report Formats
- **HTML**: Beautiful, interactive reports with styling
- **Markdown**: Version-control friendly documentation
- **JSON**: Machine-readable structured data

## 🚀 Features

### ExperimentLogger
High-level logger combining all layers:
```python
from src.core.logging import ExperimentLogger, LogContext

# Create logger with context
context = LogContext(
    task_name="classification",
    run_id="exp_001",
    model_name="mobilenetv3",
)

logger = ExperimentLogger(
    name="my_experiment",
    run_id="exp_001",
    context=context,
)

# Log configuration
logger.log_config(config_dict)

# Track performance
logger.start_timer("training")
# ... training code ...
duration = logger.stop_timer("training")

# Log metrics
logger.log_metrics({"accuracy": 0.95, "loss": 0.15})

# Finalize
logger.finalize(final_metrics, duration)
```

### ExperimentReporter
Generate comprehensive reports:
```python
from src.core.logging import ExperimentReporter, create_experiment_report

# Generate all reports at once
reports = create_experiment_report(
    run_id="exp_001",
    config=config_dict,
    dataset_info=dataset_dict,
    model_info=model_dict,
    results=results_dict,
    output_dir="artifacts/reports",
)

# Returns paths to:
# - setup_html, setup_md, setup_json
# - results_html, results_md, results_json
```

### Convenience Functions
Quick logging for common patterns:
```python
from src.core.logging import (
    log_experiment_start,
    log_experiment_end,
    log_model_summary,
    log_dataset_info,
    log_training_progress,
)

# Log experiment start
log_experiment_start(logger, config, run_id)

# Log dataset info
log_dataset_info(logger, dataset_id, splits, num_classes)

# Log model summary
log_model_summary(logger, model, input_shape)

# Log training progress
log_training_progress(logger, epoch, total_epochs, metrics, lr)

# Log experiment end
log_experiment_end(logger, run_id, final_metrics, duration)
```

## 📈 Example Output

### Console Output
```
INFO | ear_vision_ml.training | ================================================================================
INFO | ear_vision_ml.training | 🚀 Starting experiment: exp_001
INFO | ear_vision_ml.training | ================================================================================
INFO | ear_vision_ml.training | 📁 Dataset: otoscopy_v1
INFO | ear_vision_ml.training |   train: 7,000 (70.0%)
INFO | ear_vision_ml.training |   val: 2,000 (20.0%)
INFO | ear_vision_ml.training |   test: 1,000 (10.0%)
INFO | ear_vision_ml.training | 🏗️  Model Architecture
INFO | ear_vision_ml.training |   Total parameters: 1,234,567
INFO | ear_vision_ml.training | ⏱️  training: 3600.1234s
INFO | ear_vision_ml.training | 📊 accuracy: 0.9500
INFO | ear_vision_ml.training | ✅ Experiment completed: exp_001
```

### HTML Report
Beautiful, styled reports with:
- Gradient headers
- Metric cards with large values
- Responsive tables
- Color-coded status badges
- Professional styling

### JSON Log Entry
```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "level": "INFO",
  "logger": "ear_vision_ml.training",
  "message": "Epoch 10/50 completed",
  "module": "trainer",
  "function": "train_epoch",
  "line": 123,
  "context": {
    "task_name": "classification",
    "run_id": "exp_001",
    "epoch": 10,
    "step": 1000
  }
}
```

## ✅ Testing

**7 comprehensive tests** covering:
- Logging setup and configuration
- Logger creation with context
- Experiment logger functionality
- Setup report generation (HTML, MD, JSON)
- Results report generation (HTML, MD, JSON)
- Convenience function

**All tests passing**: ✅

## 📁 Files Created

1. `src/core/logging/logger.py` (450 lines) - Multi-layered logging system
2. `src/core/logging/reporting.py` (600 lines) - Advanced reporting system
3. `src/core/logging/__init__.py` - Package exports
4. `tests/unit/test_logging.py` (200 lines) - Comprehensive tests

## 🎯 Integration

The logging system integrates with:
- ✅ Vertex AI Experiments
- ✅ TensorBoard
- ✅ BigQuery (optional)
- ✅ File system (rotating logs)
- ✅ Console output
- ✅ JSON structured logs

## 📊 Benefits

1. **Comprehensive Tracking**: Every aspect of experiments is logged
2. **Multiple Formats**: HTML, Markdown, JSON for different use cases
3. **Performance Monitoring**: Built-in timing and profiling
4. **Beautiful Reports**: Professional HTML reports for sharing
5. **Machine Readable**: JSON logs for automated analysis
6. **Debugging**: Detailed file logs with full context
7. **User Friendly**: Clean console output with colors
8. **Scalable**: Rotating logs prevent disk space issues

## 🎉 Summary

The repository now has a **production-grade logging and reporting system** with:
- **5 logging layers** (console, file, JSON, performance, experiment)
- **3 report formats** (HTML, Markdown, JSON)
- **Comprehensive coverage** (setup, progress, results)
- **7 passing tests**
- **Easy integration** with existing code

Total test count: **34 tests (33 passed, 1 skipped)** ✅
