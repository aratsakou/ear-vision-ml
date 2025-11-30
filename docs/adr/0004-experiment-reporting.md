# ADR 0004: Advanced Experiment Reporting

## Status
Accepted

## Context
ML experiments generate significant amounts of data:
- Configuration parameters
- Dataset statistics
- Model architecture details
- Training metrics over time
- Final results and artifacts

This information needs to be:
- Easily shareable with stakeholders
- Version-control friendly
- Machine-readable for automation
- Human-readable for analysis

## Decision
Implement a comprehensive reporting system that generates three formats:

1. **HTML Reports**: Beautiful, styled reports for sharing
2. **Markdown Reports**: Version-control friendly documentation
3. **JSON Reports**: Machine-readable structured data

Reports cover three phases:
- **Setup**: Configuration, dataset, model architecture
- **Progress**: Real-time training updates (via logging)
- **Results**: Final metrics, best performance, artifacts

## Consequences

### Positive
- **Shareability**: HTML reports can be emailed or hosted
- **Version Control**: Markdown reports track changes over time
- **Automation**: JSON reports enable automated analysis
- **Completeness**: All experiment details in one place
- **Reproducibility**: Full configuration captured
- **Comparison**: Easy to compare multiple experiments

### Negative
- **Storage**: Multiple report formats per experiment
- **Maintenance**: Need to keep formats in sync
- **Generation Time**: Small overhead to create reports

### Mitigations
- Reports are optional (can disable if not needed)
- Convenience function generates all formats at once
- Reports are relatively small (< 1MB typically)

## Alternatives Considered

### Single Format (HTML Only)
- **Rejected**: Not version-control friendly
- Cannot be easily parsed by scripts

### Single Format (JSON Only)
- **Rejected**: Not human-friendly
- Requires tools to view

### External Reporting Service
- **Rejected**: Adds dependency
- May not work offline
- Vendor lock-in

### No Structured Reporting
- **Rejected**: Information scattered across logs
- Difficult to share and compare
- No standard format

## Implementation Notes
- `ExperimentReporter` class handles all report generation
- Dataclasses define report structure (`ExperimentSetup`, `ExperimentResults`)
- HTML reports use inline CSS for portability
- Markdown reports use tables for structured data
- JSON reports use dataclass serialization

## Integration
- Works with `ExperimentLogger` for complete tracking
- Can be called at experiment start and end
- Convenience function `create_experiment_report()` for ease of use

## References
- Jupyter Notebook HTML export
- MLflow experiment tracking
- Weights & Biases reporting
