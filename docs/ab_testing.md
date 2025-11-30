# A/B Testing Framework

Compare model performance using statistical significance tests.

## Implementation
- **Analyzer**: `src/core/evaluation/ab_test.py`
- **Tests**: `tests/unit/test_ab_test.py`

## Features
- **T-Test**: For continuous metrics (e.g., Latency, IoU).
- **Z-Test**: For proportion metrics (e.g., Accuracy, Conversion Rate).
- **Sample Size Calculation**: Estimate required samples for a given effect size.

## Usage
Use the `ABTestAnalyzer` to compare results from two models (Champion vs Challenger).

```python
from src.core.evaluation.ab_test import ABTestAnalyzer

analyzer = ABTestAnalyzer()

# Compare Accuracy (Proportions)
result = analyzer.compare_proportions(
    control_success=850, control_total=1000,
    treatment_success=880, treatment_total=1000
)
print(f"Significant: {result['significant']}, Lift: {result['lift_percent']}%")

# Compare Latency (Means)
result = analyzer.compare_means(
    control_data=latencies_v1,
    treatment_data=latencies_v2
)
```
