"""
Metrics computation for simulation results.
"""
from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

def _extract_metrics(result: Dict[str, Any]) -> Tuple[int, int, Dict[str, List[float]]]:
    """Extract metrics and success/failure counts from a result."""
    metrics = {}
    successful_runs = 0
    failed_runs = 0

    if result.get("status") in ["FAILED", "failed"]:
        failed_runs = 1
    else:
        successful_runs = 1
        if isinstance(result.get("metrics"), dict):
            for metric_type, metric_values in result["metrics"].items():
                if isinstance(metric_values, dict):
                    for name, value in metric_values.items():
                        if isinstance(value, (int, float)):
                            metric_key = f"mean_{metric_type}.{name}"
                            metrics.setdefault(metric_key, []).append(value)

    return successful_runs, failed_runs, metrics

def _compute_summary_metrics(
    results: List[Dict[str, Any]],
    prefix: str = ""
) -> Dict[str, float]:
    """Compute summary metrics from a list of results."""
    try:
        metrics = {}
        successful_runs = 0
        failed_runs = 0

        for result in results:
            if isinstance(result, dict):
                if "results" in result and isinstance(result["results"], list):
                    # Handle nested results (study level)
                    for run_result in result["results"]:
                        succ, fail, run_metrics = _extract_metrics(run_result)
                        successful_runs += succ
                        failed_runs += fail
                        for key, values in run_metrics.items():
                            metrics.setdefault(key, []).extend(values)
                else:
                    # Handle direct results (parameter set level)
                    succ, fail, run_metrics = _extract_metrics(result)
                    successful_runs += succ
                    failed_runs += fail
                    for key, values in run_metrics.items():
                        metrics.setdefault(key, []).extend(values)

        summary = {
            f"{prefix}successful_runs": successful_runs,
            f"{prefix}failed_runs": failed_runs,
            f"{prefix}success_rate": successful_runs / (successful_runs + failed_runs) 
                if (successful_runs + failed_runs) > 0 else 0
        }
        
        for metric_name, values in metrics.items():
            if values:
                summary[metric_name] = sum(values) / len(values)
                
        return summary

    except Exception as e:
        logger.error(f"Error computing metrics: {str(e)}")
        return {}

def compute_parameter_set_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute aggregate metrics for a parameter set."""
    return _compute_summary_metrics(results, prefix="replication_")

def compute_study_summary_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute summary metrics for the entire study."""
    return _compute_summary_metrics(results) 