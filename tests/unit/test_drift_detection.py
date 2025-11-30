import numpy as np
import pytest
from src.core.monitoring.drift_detector import DriftDetector

def test_psi_no_drift():
    detector = DriftDetector()
    # Same distribution
    data1 = np.random.normal(0, 1, 1000)
    data2 = np.random.normal(0, 1, 1000)
    
    psi = detector.calculate_psi(data1, data2)
    assert psi < 0.1

def test_psi_significant_drift():
    detector = DriftDetector()
    # Different distribution
    data1 = np.random.normal(0, 1, 1000)
    data2 = np.random.normal(2, 1, 1000)
    
    psi = detector.calculate_psi(data1, data2)
    assert psi >= 0.2

def test_ks_statistic():
    detector = DriftDetector()
    data1 = np.random.normal(0, 1, 1000)
    data2 = np.random.normal(0, 1, 1000)
    
    stat, p_value = detector.calculate_ks_statistic(data1, data2)
    assert p_value > 0.05 # Cannot reject null hypothesis (same distribution)

def test_detect_drift_full():
    np.random.seed(42)
    detector = DriftDetector()
    baseline = {
        "feature1": np.random.normal(0, 1, 2000),
        "feature2": np.random.normal(0, 1, 2000)
    }
    current = {
        "feature1": np.random.normal(0, 1, 2000), # No drift
        "feature2": np.random.normal(3, 1, 2000)  # Drift
    }
    
    results = detector.detect_drift(baseline, current)
    print(f"DEBUG: {results}")
    
    assert not results["feature1"]["drift_detected"]
    assert results["feature2"]["drift_detected"]
