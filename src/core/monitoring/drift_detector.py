import numpy as np
from typing import Dict, Any, Tuple
from scipy.stats import ks_2samp

class DriftDetector:
    """
    Detects data drift using statistical tests.
    """
    
    @staticmethod
    def calculate_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI).
        PSI < 0.1: No significant drift
        PSI < 0.2: Moderate drift
        PSI >= 0.2: Significant drift
        """
        def scale_range(input, min, max):
            input += -(np.min(input))
            input /= np.max(input) / (max - min)
            input += min
            return input

        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
        breakpoints = np.percentile(expected, breakpoints)
        
        expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
        actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)
        
        # Avoid division by zero
        expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
        actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
        
        psi_value = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
        return psi_value

    @staticmethod
    def calculate_ks_statistic(expected: np.ndarray, actual: np.ndarray) -> Tuple[float, float]:
        """
        Calculate Kolmogorov-Smirnov statistic.
        Returns (statistic, p-value).
        Small p-value indicates distributions are different.
        """
        return ks_2samp(expected, actual)

    def detect_drift(self, baseline_data: Dict[str, np.ndarray], current_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Detect drift for multiple features.
        """
        results = {}
        for feature_name, baseline_values in baseline_data.items():
            if feature_name not in current_data:
                continue
                
            current_values = current_data[feature_name]
            
            # Ensure numerical
            if not np.issubdtype(baseline_values.dtype, np.number) or not np.issubdtype(current_values.dtype, np.number):
                continue

            psi = self.calculate_psi(baseline_values, current_values)
            ks_stat, ks_p_value = self.calculate_ks_statistic(baseline_values, current_values)
            
            results[feature_name] = {
                "psi": float(psi),
                "ks_statistic": float(ks_stat),
                "ks_p_value": float(ks_p_value),
                "drift_detected": bool(psi >= 0.2 or ks_p_value < 0.05)
            }
            
        return results
