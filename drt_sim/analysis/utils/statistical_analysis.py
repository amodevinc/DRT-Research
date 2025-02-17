import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

@dataclass
class StatTestResult:
    """Container for statistical test results."""
    test_name: str
    statistic: float
    p_value: float
    interpretation: str
    additional_info: Optional[Dict] = None

class StatisticalAnalyzer:
    """Performs statistical analysis on metrics data."""
    
    def __init__(self, alpha: float = 0.05):
        """Initialize with significance level."""
        self.alpha = alpha
        
    def compare_groups(self,
                      df: pd.DataFrame,
                      group_column: str,
                      value_column: str = 'value') -> List[StatTestResult]:
        """Compare groups using appropriate statistical tests."""
        results = []
        groups = df.groupby(group_column)[value_column].apply(list).to_dict()
        
        if len(groups) == 2:
            # Two groups: Use t-test and Mann-Whitney U test
            group_values = list(groups.values())
            
            # T-test
            t_stat, t_p = stats.ttest_ind(group_values[0], group_values[1])
            results.append(StatTestResult(
                test_name="Independent t-test",
                statistic=t_stat,
                p_value=t_p,
                interpretation=self._interpret_p_value(t_p),
                additional_info={
                    'group_means': {k: np.mean(v) for k, v in groups.items()},
                    'group_stds': {k: np.std(v) for k, v in groups.items()}
                }
            ))
            
            # Mann-Whitney U test
            u_stat, u_p = stats.mannwhitneyu(group_values[0], group_values[1])
            results.append(StatTestResult(
                test_name="Mann-Whitney U test",
                statistic=u_stat,
                p_value=u_p,
                interpretation=self._interpret_p_value(u_p)
            ))
        else:
            # Multiple groups: Use ANOVA and Kruskal-Wallis
            # One-way ANOVA
            f_stat, f_p = stats.f_oneway(*groups.values())
            results.append(StatTestResult(
                test_name="One-way ANOVA",
                statistic=f_stat,
                p_value=f_p,
                interpretation=self._interpret_p_value(f_p),
                additional_info={
                    'group_means': {k: np.mean(v) for k, v in groups.items()},
                    'group_stds': {k: np.std(v) for k, v in groups.items()}
                }
            ))
            
            # Kruskal-Wallis H-test
            h_stat, h_p = stats.kruskal(*groups.values())
            results.append(StatTestResult(
                test_name="Kruskal-Wallis H-test",
                statistic=h_stat,
                p_value=h_p,
                interpretation=self._interpret_p_value(h_p)
            ))
            
        return results
        
    def test_normality(self,
                      data: Union[List[float], np.ndarray],
                      test_name: str = 'shapiro') -> StatTestResult:
        """Test for normality using various methods."""
        if test_name == 'shapiro':
            stat, p = stats.shapiro(data)
            test_name = "Shapiro-Wilk"
        elif test_name == 'kstest':
            stat, p = stats.kstest(data, 'norm')
            test_name = "Kolmogorov-Smirnov"
        else:
            raise ValueError(f"Unknown normality test: {test_name}")
            
        return StatTestResult(
            test_name=f"{test_name} normality test",
            statistic=stat,
            p_value=p,
            interpretation=self._interpret_p_value(p, reverse=True),
            additional_info={
                'skewness': stats.skew(data),
                'kurtosis': stats.kurtosis(data)
            }
        )
        
    def compute_confidence_interval(self,
                                  data: Union[List[float], np.ndarray],
                                  confidence: float = 0.95) -> Tuple[float, float]:
        """Compute confidence interval for the mean."""
        mean = np.mean(data)
        sem = stats.sem(data)
        ci = stats.t.interval(confidence, len(data)-1, loc=mean, scale=sem)
        return ci
        
    def detect_outliers(self,
                       data: Union[List[float], np.ndarray],
                       method: str = 'zscore',
                       threshold: float = 3.0) -> np.ndarray:
        """Detect outliers using various methods."""
        if method == 'zscore':
            z_scores = np.abs(stats.zscore(data))
            return z_scores > threshold
        elif method == 'iqr':
            q1, q3 = np.percentile(data, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            return (data < lower_bound) | (data > upper_bound)
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
            
    def analyze_trend(self,
                     df: pd.DataFrame,
                     value_column: str = 'value') -> StatTestResult:
        """Analyze trend using Mann-Kendall test."""
        values = df[value_column].values
        result = stats.kendalltau(np.arange(len(values)), values)
        
        trend_strength = abs(result.correlation)
        trend_direction = "positive" if result.correlation > 0 else "negative"
        
        interpretation = f"{'Strong' if trend_strength > 0.5 else 'Weak'} {trend_direction} trend"
        if result.pvalue > self.alpha:
            interpretation += " (not statistically significant)"
            
        return StatTestResult(
            test_name="Mann-Kendall trend test",
            statistic=result.correlation,
            p_value=result.pvalue,
            interpretation=interpretation,
            additional_info={
                'trend_strength': trend_strength,
                'trend_direction': trend_direction
            }
        )
        
    def _interpret_p_value(self, p_value: float, reverse: bool = False) -> str:
        """Interpret p-value result."""
        if reverse:
            # For normality tests, high p-value indicates normality
            if p_value > self.alpha:
                return "Data appears to be normally distributed"
            else:
                return "Data appears to be non-normal"
        else:
            if p_value <= self.alpha:
                return f"Statistically significant (p={p_value:.4f})"
            else:
                return f"Not statistically significant (p={p_value:.4f})" 