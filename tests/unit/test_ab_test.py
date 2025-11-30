import numpy as np
import pytest
from src.core.evaluation.ab_test import ABTestAnalyzer

def test_compare_means_significant():
    analyzer = ABTestAnalyzer()
    # Distinct distributions
    control = np.random.normal(0.5, 0.1, 1000)
    treatment = np.random.normal(0.6, 0.1, 1000)
    
    result = analyzer.compare_means(control, treatment)
    assert result["significant"]
    assert result["lift_percent"] > 0

def test_compare_means_not_significant():
    analyzer = ABTestAnalyzer()
    # Same distributions
    control = np.random.normal(0.5, 0.1, 1000)
    treatment = np.random.normal(0.5, 0.1, 1000)
    
    result = analyzer.compare_means(control, treatment)
    assert not result["significant"]

def test_compare_proportions_significant():
    analyzer = ABTestAnalyzer()
    # 50% vs 60% accuracy
    result = analyzer.compare_proportions(500, 1000, 600, 1000)
    assert result["significant"]
    assert result["lift_percent"] > 0

def test_compare_proportions_not_significant():
    analyzer = ABTestAnalyzer()
    # 50% vs 51% accuracy (small sample)
    result = analyzer.compare_proportions(50, 100, 51, 100)
    assert not result["significant"]

def test_calculate_sample_size():
    analyzer = ABTestAnalyzer()
    n = analyzer.calculate_sample_size(effect_size=0.05)
    assert n > 0
