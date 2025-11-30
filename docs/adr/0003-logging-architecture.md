# ADR 0003: Multi-Layered Logging Architecture

## Status
Accepted

## Context
The repository requires comprehensive logging for:
- Development debugging
- Production monitoring
- Experiment tracking
- Performance profiling
- Compliance and audit trails

A single logging approach cannot satisfy all these needs effectively.

## Decision
Implement a multi-layered logging system with five distinct layers:

1. **Console Logging (INFO+)**: User-facing messages with color coding
2. **File Logging (DEBUG+)**: Detailed debugging with rotation
3. **Structured JSON Logging**: Machine-readable logs for analysis
4. **Performance Logging**: Timing and profiling metrics
5. **Experiment Tracking**: ML-specific logging (Vertex AI, TensorBoard)

Each layer serves a specific purpose and can be configured independently.

## Consequences

### Positive
- **Flexibility**: Different log levels for different use cases
- **Debugging**: Detailed file logs without cluttering console
- **Analysis**: JSON logs enable automated log analysis
- **Performance**: Built-in profiling without external tools
- **Integration**: Seamless Vertex AI and TensorBoard integration
- **Maintainability**: Rotating logs prevent disk space issues

### Negative
- **Complexity**: More configuration options to manage
- **Disk Usage**: Multiple log files (mitigated by rotation)
- **Learning Curve**: Users need to understand different layers

### Mitigations
- Sensible defaults for all layers
- Single `setup_logging()` function for initialization
- Clear documentation of each layer's purpose
- Automatic log rotation to manage disk space

## Alternatives Considered

### Single-Layer Logging
- **Rejected**: Cannot satisfy all use cases effectively
- Console output would be too verbose or file logs too sparse

### Third-Party Logging Service
- **Rejected**: Adds external dependency and cost
- May not work in air-gapped environments
- Vendor lock-in concerns

### Python's Standard Logging Only
- **Rejected**: Lacks ML-specific features
- No built-in performance profiling
- No experiment tracking integration

## Implementation Notes
- Uses Python's standard `logging` module as foundation
- Custom formatters for colored console and JSON output
- `PerformanceLogger` wrapper for timing operations
- `ExperimentLogger` combines all layers for ML workflows
- Rotating file handlers prevent unbounded growth

## References
- Python logging documentation
- Twelve-Factor App logging principles
- Vertex AI Experiments API
