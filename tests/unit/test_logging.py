"""
Tests for logging and reporting system.
"""

from __future__ import annotations

from pathlib import Path

from src.core.logging import (
    DatasetInfo,
    ExperimentLogger,
    ExperimentReporter,
    ExperimentResults,
    ExperimentSetup,
    LogContext,
    ModelInfo,
    create_experiment_report,
    get_logger,
    setup_logging,
)


def test_setup_logging(tmp_path: Path) -> None:
    """Test logging setup."""
    log_dir = tmp_path / "logs"
    
    logger = setup_logging(
        log_dir=log_dir,
        console_level="INFO",
        file_level="DEBUG",
    )
    
    assert logger is not None
    assert log_dir.exists()
    assert (log_dir / "ear_vision_ml.log").exists()
    assert (log_dir / "ear_vision_ml.json").exists()


def test_get_logger() -> None:
    """Test logger creation."""
    logger = get_logger("test_module")
    assert logger is not None
    assert "test_module" in logger.name


def test_logger_with_context() -> None:
    """Test logger with context."""
    context = LogContext(
        task_name="classification",
        run_id="test_run_001",
        model_name="mobilenetv3",
    )
    
    logger = get_logger("test_module", context=context)
    assert logger is not None


def test_experiment_logger(tmp_path: Path) -> None:
    """Test experiment logger."""
    exp_logger = ExperimentLogger(
        name="test_experiment",
        run_id="test_run_001",
        log_dir=tmp_path / "logs",
    )
    
    # Test config logging
    config = {"model": "mobilenetv3", "epochs": 10}
    exp_logger.log_config(config)
    
    # Test metrics logging
    metrics = {"accuracy": 0.95, "loss": 0.15}
    exp_logger.log_metrics(metrics)
    
    # Test timer
    exp_logger.start_timer("test_operation")
    duration = exp_logger.stop_timer("test_operation")
    assert duration >= 0
    
    # Test finalize
    final_metrics = {"final_accuracy": 0.96}
    exp_logger.finalize(final_metrics, duration=100.0)


def test_experiment_reporter_setup(tmp_path: Path) -> None:
    """Test experiment setup report generation."""
    reporter = ExperimentReporter(output_dir=tmp_path)
    
    setup = ExperimentSetup(
        run_id="test_run_001",
        task_name="classification",
        model_name="mobilenetv3",
        dataset_id="test_dataset",
        created_at="2024-01-01T00:00:00Z",
        created_by="test_user",
        git_commit="abc123",
        config={"epochs": 10},
        environment={"python": "3.10"},
    )
    
    dataset = DatasetInfo(
        dataset_id="test_dataset",
        total_samples=1000,
        splits={"train": 700, "val": 200, "test": 100},
        num_classes=3,
        class_distribution={"class_0": 400, "class_1": 300, "class_2": 300},
        preprocessing_pipeline="full_frame_v1",
        augmentation="mixup",
    )
    
    model = ModelInfo(
        model_name="mobilenetv3",
        architecture="MobileNetV3Small",
        total_params=1000000,
        trainable_params=950000,
        input_shape=(224, 224, 3),
        output_shape=(3,),
        optimizer="adam",
        loss_function="categorical_crossentropy",
        metrics=["accuracy", "f1"],
    )
    
    # Generate HTML report
    html_path = reporter.generate_setup_report(setup, dataset, model, format="html")
    assert html_path.exists()
    assert html_path.suffix == ".html"
    
    # Generate Markdown report
    md_path = reporter.generate_setup_report(setup, dataset, model, format="markdown")
    assert md_path.exists()
    assert md_path.suffix == ".md"
    
    # Generate JSON report
    json_path = reporter.generate_setup_report(setup, dataset, model, format="json")
    assert json_path.exists()
    assert json_path.suffix == ".json"


def test_experiment_reporter_results(tmp_path: Path) -> None:
    """Test experiment results report generation."""
    reporter = ExperimentReporter(output_dir=tmp_path)
    
    setup = ExperimentSetup(
        run_id="test_run_001",
        task_name="classification",
        model_name="mobilenetv3",
        dataset_id="test_dataset",
        created_at="2024-01-01T00:00:00Z",
        created_by="test_user",
        git_commit="abc123",
        config={},
        environment={},
    )
    
    results = ExperimentResults(
        run_id="test_run_001",
        status="completed",
        total_duration=3600.0,
        best_epoch=25,
        final_metrics={"accuracy": 0.95, "loss": 0.15},
        best_metrics={"accuracy": 0.96, "loss": 0.14},
        training_history={"accuracy": [0.8, 0.9, 0.95], "loss": [0.3, 0.2, 0.15]},
        artifacts={"model": "models/best_model.keras", "logs": "logs/training.csv"},
        notes="Experiment completed successfully",
    )
    
    # Generate HTML report
    html_path = reporter.generate_results_report(results, setup, format="html")
    assert html_path.exists()
    assert html_path.suffix == ".html"
    
    # Generate Markdown report
    md_path = reporter.generate_results_report(results, setup, format="markdown")
    assert md_path.exists()
    assert md_path.suffix == ".md"
    
    # Generate JSON report
    json_path = reporter.generate_results_report(results, setup, format="json")
    assert json_path.exists()
    assert json_path.suffix == ".json"


def test_create_experiment_report(tmp_path: Path) -> None:
    """Test convenience function for creating complete reports."""
    reports = create_experiment_report(
        run_id="test_run_001",
        config={
            "task": {"name": "classification"},
            "model": {"name": "mobilenetv3"},
            "created_by": "test_user",
        },
        dataset_info={
            "dataset_id": "test_dataset",
            "total_samples": 1000,
            "splits": {"train": 700, "val": 200, "test": 100},
            "num_classes": 3,
            "class_distribution": None,
            "preprocessing_pipeline": "full_frame_v1",
            "augmentation": None,
        },
        model_info={
            "model_name": "mobilenetv3",
            "architecture": "MobileNetV3Small",
            "total_params": 1000000,
            "trainable_params": 950000,
            "input_shape": (224, 224, 3),
            "output_shape": (3,),
            "optimizer": "adam",
            "loss_function": "categorical_crossentropy",
            "metrics": ["accuracy"],
        },
        results={
            "run_id": "test_run_001",
            "status": "completed",
            "total_duration": 3600.0,
            "best_epoch": 25,
            "final_metrics": {"accuracy": 0.95},
            "best_metrics": {"accuracy": 0.96},
            "training_history": {"accuracy": [0.95]},
            "artifacts": {"model": "models/best_model.keras"},
            "notes": None,
        },
        output_dir=tmp_path,
    )
    
    # Check all reports were generated
    assert "setup_html" in reports
    assert "setup_md" in reports
    assert "setup_json" in reports
    assert "results_html" in reports
    assert "results_md" in reports
    assert "results_json" in reports
    
    # Verify files exist
    for report_path in reports.values():
        assert report_path.exists()
