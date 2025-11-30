# Devlog 0016: A/B Testing Framework

**Date:** 2025-11-30
**Author:** System
**Status:** ✅ Complete

## Summary
Implemented a statistical framework for A/B testing to rigorously compare model performance (e.g., Champion vs Challenger). This allows for data-driven decisions when promoting new models to production.

## Changes Made

### 1. A/B Test Analyzer (`src/core/evaluation/ab_test.py`)
-   **`ABTestAnalyzer`**: Implements:
    -   **T-Test**: For comparing means of continuous metrics (e.g., Latency, IoU).
    -   **Z-Test**: For comparing proportions of binary metrics (e.g., Accuracy, Conversion).
    -   **Sample Size Calculator**: Estimates required sample size for a desired effect size and power.

### 2. Configuration (`configs/evaluation/ab_test.yaml`)
-   Added configuration for defining A/B tests, including metrics, minimum effect sizes, and significance levels.

### 3. Documentation
-   Created `docs/ab_testing.md` with usage instructions and examples.

## Testing
-   **Unit Tests**: Created `tests/unit/test_ab_test.py` covering:
    -   Significant and non-significant results for both T-tests and Z-tests.
    -   Sample size calculation.
-   **Status**: All tests passed.

## Usage
```python
from src.core.evaluation.ab_test import ABTestAnalyzer

analyzer = ABTestAnalyzer()
result = analyzer.compare_proportions(
    control_success=850, control_total=1000,
    treatment_success=880, treatment_total=1000
)
if result['significant']:
    print(f"Challenger wins with {result['lift_percent']}% lift!")
```
