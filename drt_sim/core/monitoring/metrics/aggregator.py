from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
from drt_sim.core.monitoring.metrics.collector import MetricsStorage
import logging

logger = logging.getLogger(__name__)

class MetricsAggregator:
    """Handles basic metric aggregation and DataFrame conversion."""
    
    def __init__(self, storage: MetricsStorage):
        self.storage = storage
    
    def to_dataframe(self,
                    metric_names: Optional[List[str]] = None,
                    start_time: Optional[datetime] = None,
                    end_time: Optional[datetime] = None) -> pd.DataFrame:
        """Convert metrics to a DataFrame with optional filtering.
        
        Args:
            metric_names: Optional list of metrics to include
            start_time: Optional start time filter
            end_time: Optional end time filter
            
        Returns:
            DataFrame containing the requested metrics with columns:
            - metric_name: Name of the metric
            - value: Metric value
            - timestamp: Time the metric was recorded
            - Additional columns for any tags associated with the metric
        """
        # Collect all metrics into a list
        all_metrics = []
        for chunk in self.storage.iter_chunks():
            for metric in chunk:
                # Apply filters
                if metric_names and metric.name not in metric_names:
                    continue
                if start_time and metric.timestamp < start_time:
                    continue
                if end_time and metric.timestamp > end_time:
                    continue
                    
                # Convert metric to dict format
                metric_dict = {
                    'metric_name': metric.name,
                    'value': metric.value,
                    'timestamp': metric.timestamp
                }
                # Add any tags
                metric_dict.update(metric.tags)
                all_metrics.append(metric_dict)
        
        # Convert to DataFrame
        if not all_metrics:
            return pd.DataFrame()
            
        df = pd.DataFrame(all_metrics)
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        return df