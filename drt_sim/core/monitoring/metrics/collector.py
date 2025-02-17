from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union, List
import pandas as pd
import logging
from pydantic import ValidationError
from drt_sim.core.monitoring.types.metrics import MetricName, MetricPoint
from drt_sim.core.monitoring.metrics.mlflow_adapter import MLflowAdapter
from drt_sim.core.monitoring.metrics.storage import MetricsStorage
from drt_sim.core.monitoring.metrics.aggregator import MetricsAggregator
import traceback
logger = logging.getLogger(__name__)



class MetricsCollector:
    """Main metrics collection system with metric registry integration."""
    
    def __init__(self,
                 base_path: Union[str, Path],
                 replication_id: str,
                 chunk_size: int = 1000):
        """Initialize a new metrics collector with metric registry integration."""
        self.storage = MetricsStorage(base_path, replication_id, chunk_size)
        self.mlflow_adapter = MLflowAdapter()
        self.replication_id = replication_id
        self.aggregator = MetricsAggregator(self.storage)
        self.base_path = Path(base_path)
        
        # Set run-level tags
        self.mlflow_adapter.set_tag("replication_id", replication_id)
        self.mlflow_adapter.set_tag("schema_version", "1.0.0")
        
    def log(self,
            metric_name: Union[str, MetricName],
            value: float,
            timestamp: Optional[datetime] = None,
            tags: Optional[Dict[str, str]] = None):
        """Log a metric with validation against the metric registry."""
        if isinstance(metric_name, MetricName):
            metric_name = metric_name.value
            
        try:
            # Create and validate metric point
            metric = MetricPoint(
                name=metric_name,
                value=value,
                timestamp=timestamp or datetime.now(),
                tags=tags or {}
            )
            
            # Validate required context
            if not metric.validate_context():
                logger.error(f"Missing required context for metric {metric_name}")
                return
                
            # Store locally
            self.storage.add_metric(metric)
            
            # Queue for MLflow logging
            self.mlflow_adapter.log_metric(
                metric.name,
                metric.value,
                metric.timestamp,
                metric.tags
            )
            
        except ValidationError as e:
            logger.error(f"Invalid metric data: {e}")
        except Exception as e:
            logger.error(f"Error logging metric: {traceback.format_exc()}")
    def cleanup(self):
        """Clean up all stored data."""
        self.storage.shutdown_storage()
        self.storage.cleanup()
        self.mlflow_adapter.shutdown()

    def get_metrics_df(self,
                      metric_names: Optional[List[str]] = None,
                      start_time: Optional[datetime] = None,
                      end_time: Optional[datetime] = None) -> pd.DataFrame:
        """Convert metrics to a DataFrame with optional filtering and derived metrics.
        
        Args:
            metric_names: Optional list of metrics to include
            start_time: Optional start time filter
            end_time: Optional end time filter
            include_derived: Whether to compute and include derived metrics
            
        Returns:
            DataFrame containing the requested metrics with columns:
            - metric_name: Name of the metric
            - value: Metric value
            - timestamp: Time the metric was recorded
            - Additional columns for any tags associated with the metric
        """
        return self.aggregator.to_dataframe(
            metric_names=metric_names,
            start_time=start_time,
            end_time=end_time,
        )
