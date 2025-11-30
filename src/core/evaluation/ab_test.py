import numpy as np
from typing import Dict, Any, Tuple
from scipy import stats

class ABTestAnalyzer:
    """
    Analyzes A/B test results to determine if the challenger model is significantly better than the champion.
    """
    
    @staticmethod
    def calculate_sample_size(effect_size: float = 0.05, power: float = 0.8, alpha: float = 0.05) -> int:
        """
        Calculate required sample size per variant for a given effect size (Cohen's h for proportions).
        Simplified approximation for binary metrics (e.g. accuracy, conversion).
        """
        # Standard normal quantiles
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        # For proportions p1=0.5, p2=0.5+effect_size (worst case variance)
        p = 0.5
        n = 2 * p * (1-p) * ((z_alpha + z_beta) / effect_size)**2
        return int(np.ceil(n))

    @staticmethod
    def compare_means(control_data: np.ndarray, treatment_data: np.ndarray) -> Dict[str, Any]:
        """
        Compare means using t-test (for continuous metrics like IoU, loss).
        """
        t_stat, p_value = stats.ttest_ind(control_data, treatment_data, equal_var=False)
        
        mean_diff = np.mean(treatment_data) - np.mean(control_data)
        lift = (mean_diff / np.mean(control_data)) * 100 if np.mean(control_data) != 0 else 0
        
        return {
            "test_type": "t_test",
            "statistic": float(t_stat),
            "p_value": float(p_value),
            "mean_diff": float(mean_diff),
            "lift_percent": float(lift),
            "significant": p_value < 0.05
        }

    @staticmethod
    def compare_proportions(control_success: int, control_total: int, 
                          treatment_success: int, treatment_total: int) -> Dict[str, Any]:
        """
        Compare proportions using Z-test (for binary metrics like Accuracy).
        """
        p1 = control_success / control_total
        p2 = treatment_success / treatment_total
        
        p_pooled = (control_success + treatment_success) / (control_total + treatment_total)
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/control_total + 1/treatment_total))
        
        z_stat = (p2 - p1) / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat))) # Two-tailed
        
        lift = ((p2 - p1) / p1) * 100 if p1 != 0 else 0
        
        return {
            "test_type": "z_test",
            "statistic": float(z_stat),
            "p_value": float(p_value),
            "diff": float(p2 - p1),
            "lift_percent": float(lift),
            "significant": p_value < 0.05
        }
