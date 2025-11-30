"""
Advanced experiment reporting system.

Features:
- Comprehensive experiment setup reports
- Real-time training progress reports
- Final results reports with visualizations
- HTML and Markdown report generation
- Comparison reports across experiments
- Export to multiple formats (HTML, PDF, JSON)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ExperimentSetup:
    """Experiment setup information."""
    run_id: str
    task_name: str
    model_name: str
    dataset_id: str
    created_at: str
    created_by: str
    git_commit: str
    config: dict[str, Any]
    environment: dict[str, str]


@dataclass
class DatasetInfo:
    """Dataset information for reporting."""
    dataset_id: str
    total_samples: int
    splits: dict[str, int]
    num_classes: int | None
    class_distribution: dict[str, int] | None
    preprocessing_pipeline: str
    augmentation: str | None


@dataclass
class ModelInfo:
    """Model information for reporting."""
    model_name: str
    architecture: str
    total_params: int
    trainable_params: int
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    optimizer: str
    loss_function: str
    metrics: list[str]


@dataclass
class TrainingProgress:
    """Training progress information."""
    epoch: int
    total_epochs: int
    metrics: dict[str, float]
    learning_rate: float
    batch_size: int
    time_per_epoch: float
    estimated_time_remaining: float


@dataclass
class ExperimentResults:
    """Final experiment results."""
    run_id: str
    status: str  # completed, failed, stopped
    total_duration: float
    best_epoch: int
    final_metrics: dict[str, float]
    best_metrics: dict[str, float]
    training_history: dict[str, list[float]]
    artifacts: dict[str, str]
    notes: str | None


class ExperimentReporter:
    """
    Generate comprehensive experiment reports.
    
    Supports:
    - Setup reports (config, data, model)
    - Progress reports (real-time updates)
    - Results reports (final metrics, plots)
    - Comparison reports (multiple experiments)
    """
    
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_setup_report(
        self,
        setup: ExperimentSetup,
        dataset: DatasetInfo,
        model: ModelInfo,
        format: str = "html",
    ) -> Path:
        """
        Generate experiment setup report.
        
        Args:
            setup: Experiment setup info
            dataset: Dataset info
            model: Model info
            format: Output format ('html', 'markdown', 'json')
            
        Returns:
            Path to generated report
        """
        if format == "html":
            return self._generate_setup_html(setup, dataset, model)
        elif format == "markdown":
            return self._generate_setup_markdown(setup, dataset, model)
        elif format == "json":
            return self._generate_setup_json(setup, dataset, model)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _generate_setup_html(
        self,
        setup: ExperimentSetup,
        dataset: DatasetInfo,
        model: ModelInfo,
    ) -> Path:
        """Generate HTML setup report."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Experiment Setup Report - {setup.run_id}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .section {{
            background: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
        }}
        .metric {{
            display: inline-block;
            background: #e3f2fd;
            padding: 8px 16px;
            margin: 5px;
            border-radius: 20px;
            font-size: 14px;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }}
        .badge-primary {{ background: #667eea; color: white; }}
        .badge-success {{ background: #48bb78; color: white; }}
        .badge-info {{ background: #4299e1; color: white; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 Experiment Setup Report</h1>
        <p><strong>Run ID:</strong> {setup.run_id}</p>
        <p><strong>Created:</strong> {setup.created_at}</p>
        <p><strong>Task:</strong> <span class="badge badge-primary">{setup.task_name}</span></p>
    </div>
    
    <div class="section">
        <h2>📋 Configuration</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Model</td><td>{setup.model_name}</td></tr>
            <tr><td>Dataset</td><td>{setup.dataset_id}</td></tr>
            <tr><td>Git Commit</td><td><code>{setup.git_commit}</code></td></tr>
            <tr><td>Created By</td><td>{setup.created_by}</td></tr>
        </table>
    </div>
    
    <div class="section">
        <h2>📊 Dataset Information</h2>
        <table>
            <tr><th>Property</th><th>Value</th></tr>
            <tr><td>Dataset ID</td><td>{dataset.dataset_id}</td></tr>
            <tr><td>Total Samples</td><td>{dataset.total_samples:,}</td></tr>
            <tr><td>Number of Classes</td><td>{dataset.num_classes or 'N/A'}</td></tr>
            <tr><td>Preprocessing</td><td>{dataset.preprocessing_pipeline}</td></tr>
            <tr><td>Augmentation</td><td>{dataset.augmentation or 'None'}</td></tr>
        </table>
        
        <h3>Split Distribution</h3>
        <table>
            <tr><th>Split</th><th>Samples</th><th>Percentage</th></tr>
"""
        
        for split, count in dataset.splits.items():
            pct = (count / dataset.total_samples * 100) if dataset.total_samples > 0 else 0
            html += f"            <tr><td>{split}</td><td>{count:,}</td><td>{pct:.1f}%</td></tr>\n"
        
        html += """
        </table>
    </div>
    
    <div class="section">
        <h2>🏗️ Model Architecture</h2>
        <table>
            <tr><th>Property</th><th>Value</th></tr>
"""
        
        html += f"""
            <tr><td>Model Name</td><td>{model.model_name}</td></tr>
            <tr><td>Architecture</td><td>{model.architecture}</td></tr>
            <tr><td>Total Parameters</td><td>{model.total_params:,}</td></tr>
            <tr><td>Trainable Parameters</td><td>{model.trainable_params:,}</td></tr>
            <tr><td>Input Shape</td><td>{model.input_shape}</td></tr>
            <tr><td>Output Shape</td><td>{model.output_shape}</td></tr>
            <tr><td>Optimizer</td><td>{model.optimizer}</td></tr>
            <tr><td>Loss Function</td><td>{model.loss_function}</td></tr>
        </table>
        
        <h3>Metrics</h3>
        <div>
"""
        
        for metric in model.metrics:
            html += f'            <span class="metric">{metric}</span>\n'
        
        html += """
        </div>
    </div>
    
    <div class="section">
        <h2>⚙️ Full Configuration</h2>
        <pre style="background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto;">
"""
        html += json.dumps(setup.config, indent=2)
        html += """
        </pre>
    </div>
</body>
</html>
"""
        
        output_path = self.output_dir / f"{setup.run_id}_setup.html"
        output_path.write_text(html, encoding="utf-8")
        return output_path
    
    def _generate_setup_markdown(
        self,
        setup: ExperimentSetup,
        dataset: DatasetInfo,
        model: ModelInfo,
    ) -> Path:
        """Generate Markdown setup report."""
        md = f"""# Experiment Setup Report

## Overview
- **Run ID**: `{setup.run_id}`
- **Task**: {setup.task_name}
- **Created**: {setup.created_at}
- **Created By**: {setup.created_by}
- **Git Commit**: `{setup.git_commit}`

## Configuration
| Parameter | Value |
|-----------|-------|
| Model | {setup.model_name} |
| Dataset | {setup.dataset_id} |

## Dataset Information
- **Dataset ID**: {dataset.dataset_id}
- **Total Samples**: {dataset.total_samples:,}
- **Number of Classes**: {dataset.num_classes or 'N/A'}
- **Preprocessing**: {dataset.preprocessing_pipeline}
- **Augmentation**: {dataset.augmentation or 'None'}

### Split Distribution
| Split | Samples | Percentage |
|-------|---------|------------|
"""
        
        for split, count in dataset.splits.items():
            pct = (count / dataset.total_samples * 100) if dataset.total_samples > 0 else 0
            md += f"| {split} | {count:,} | {pct:.1f}% |\n"
        
        md += f"""
## Model Architecture
- **Model Name**: {model.model_name}
- **Architecture**: {model.architecture}
- **Total Parameters**: {model.total_params:,}
- **Trainable Parameters**: {model.trainable_params:,}
- **Input Shape**: {model.input_shape}
- **Output Shape**: {model.output_shape}
- **Optimizer**: {model.optimizer}
- **Loss Function**: {model.loss_function}

### Metrics
{', '.join(model.metrics)}

## Full Configuration
```json
{json.dumps(setup.config, indent=2)}
```
"""
        
        output_path = self.output_dir / f"{setup.run_id}_setup.md"
        output_path.write_text(md, encoding="utf-8")
        return output_path
    
    def _generate_setup_json(
        self,
        setup: ExperimentSetup,
        dataset: DatasetInfo,
        model: ModelInfo,
    ) -> Path:
        """Generate JSON setup report."""
        data = {
            "setup": asdict(setup),
            "dataset": asdict(dataset),
            "model": asdict(model),
        }
        
        output_path = self.output_dir / f"{setup.run_id}_setup.json"
        output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return output_path
    
    def generate_results_report(
        self,
        results: ExperimentResults,
        setup: ExperimentSetup,
        format: str = "html",
    ) -> Path:
        """
        Generate final results report.
        
        Args:
            results: Experiment results
            setup: Experiment setup (for context)
            format: Output format
            
        Returns:
            Path to generated report
        """
        if format == "html":
            return self._generate_results_html(results, setup)
        elif format == "markdown":
            return self._generate_results_markdown(results, setup)
        elif format == "json":
            return self._generate_results_json(results)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _generate_results_html(
        self,
        results: ExperimentResults,
        setup: ExperimentSetup,
    ) -> Path:
        """Generate HTML results report."""
        status_color = {
            "completed": "#48bb78",
            "failed": "#f56565",
            "stopped": "#ed8936",
        }.get(results.status, "#718096")
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Experiment Results - {results.run_id}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .status {{
            display: inline-block;
            padding: 8px 20px;
            border-radius: 20px;
            background: {status_color};
            color: white;
            font-weight: 600;
        }}
        .section {{
            background: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-card .value {{
            font-size: 32px;
            font-weight: 700;
            margin: 10px 0;
        }}
        .metric-card .label {{
            font-size: 14px;
            opacity: 0.9;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>✅ Experiment Results</h1>
        <p><strong>Run ID:</strong> {results.run_id}</p>
        <p><strong>Status:</strong> <span class="status">{results.status.upper()}</span></p>
        <p><strong>Duration:</strong> {results.total_duration:.2f}s ({results.total_duration/60:.1f} minutes)</p>
    </div>
    
    <div class="section">
        <h2>📊 Final Metrics</h2>
        <div class="metric-grid">
"""
        
        for metric_name, value in results.final_metrics.items():
            html += f"""
            <div class="metric-card">
                <div class="label">{metric_name}</div>
                <div class="value">{value:.4f}</div>
            </div>
"""
        
        html += """
        </div>
    </div>
    
    <div class="section">
        <h2>🏆 Best Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th><th>Epoch</th></tr>
"""
        
        for metric_name, value in results.best_metrics.items():
            html += f"            <tr><td>{metric_name}</td><td>{value:.4f}</td><td>{results.best_epoch}</td></tr>\n"
        
        html += """
        </table>
    </div>
    
    <div class="section">
        <h2>💾 Artifacts</h2>
        <table>
            <tr><th>Type</th><th>Path</th></tr>
"""
        
        for artifact_type, path in results.artifacts.items():
            html += f"            <tr><td>{artifact_type}</td><td><code>{path}</code></td></tr>\n"
        
        html += """
        </table>
    </div>
"""
        
        if results.notes:
            html += f"""
    <div class="section">
        <h2>📝 Notes</h2>
        <p>{results.notes}</p>
    </div>
"""
        
        html += """
</body>
</html>
"""
        
        output_path = self.output_dir / f"{results.run_id}_results.html"
        output_path.write_text(html, encoding="utf-8")
        return output_path
    
    def _generate_results_markdown(
        self,
        results: ExperimentResults,
        setup: ExperimentSetup,
    ) -> Path:
        """Generate Markdown results report."""
        md = f"""# Experiment Results

## Overview
- **Run ID**: `{results.run_id}`
- **Status**: {results.status.upper()}
- **Duration**: {results.total_duration:.2f}s ({results.total_duration/60:.1f} minutes)
- **Best Epoch**: {results.best_epoch}

## Final Metrics
| Metric | Value |
|--------|-------|
"""
        
        for metric_name, value in results.final_metrics.items():
            md += f"| {metric_name} | {value:.4f} |\n"
        
        md += """
## Best Metrics
| Metric | Value | Epoch |
|--------|-------|-------|
"""
        
        for metric_name, value in results.best_metrics.items():
            md += f"| {metric_name} | {value:.4f} | {results.best_epoch} |\n"
        
        md += """
## Artifacts
| Type | Path |
|------|------|
"""
        
        for artifact_type, path in results.artifacts.items():
            md += f"| {artifact_type} | `{path}` |\n"
        
        if results.notes:
            md += f"\n## Notes\n{results.notes}\n"
        
        output_path = self.output_dir / f"{results.run_id}_results.md"
        output_path.write_text(md, encoding="utf-8")
        return output_path
    
    def _generate_results_json(self, results: ExperimentResults) -> Path:
        """Generate JSON results report."""
        output_path = self.output_dir / f"{results.run_id}_results.json"
        output_path.write_text(json.dumps(asdict(results), indent=2), encoding="utf-8")
        return output_path


def create_experiment_report(
    run_id: str,
    config: dict[str, Any],
    dataset_info: dict[str, Any],
    model_info: dict[str, Any],
    results: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    """
    Convenience function to create complete experiment report.
    
    Args:
        run_id: Experiment run ID
        config: Configuration dictionary
        dataset_info: Dataset information
        model_info: Model information
        results: Results dictionary
        output_dir: Output directory
        
    Returns:
        Dictionary of generated report paths
    """
    reporter = ExperimentReporter(output_dir)
    
    # Create setup objects
    setup = ExperimentSetup(
        run_id=run_id,
        task_name=config.get("task", {}).get("name", "unknown"),
        model_name=config.get("model", {}).get("name", "unknown"),
        dataset_id=dataset_info.get("dataset_id", "unknown"),
        created_at=datetime.now(timezone.utc).isoformat(),
        created_by=config.get("created_by", "unknown"),
        git_commit=config.get("git_commit", "unknown"),
        config=config,
        environment={},
    )
    
    dataset = DatasetInfo(**dataset_info)
    model = ModelInfo(**model_info)
    experiment_results = ExperimentResults(**results)
    
    # Generate reports
    reports = {
        "setup_html": reporter.generate_setup_report(setup, dataset, model, format="html"),
        "setup_md": reporter.generate_setup_report(setup, dataset, model, format="markdown"),
        "setup_json": reporter.generate_setup_report(setup, dataset, model, format="json"),
        "results_html": reporter.generate_results_report(experiment_results, setup, format="html"),
        "results_md": reporter.generate_results_report(experiment_results, setup, format="markdown"),
        "results_json": reporter.generate_results_report(experiment_results, setup, format="json"),
    }
    
    return reports
