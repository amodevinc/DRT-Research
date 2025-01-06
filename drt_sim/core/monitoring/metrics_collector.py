# drt_sim/core/monitoring/metrics_collector.py

from typing import Dict, Any, List, Optional, DefaultDict
from datetime import datetime
import logging
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy import stats

from drt_sim.models.simulation import SimulationState
from drt_sim.models.request import RequestStatus
from drt_sim.models.vehicle import VehicleStatus
from drt_sim.config.config import MetricsConfig, MetricType

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Collects and manages simulation metrics according to configuration"""
    
    def __init__(self, metrics_config: MetricsConfig):
        logger.info("Initializing MetricsCollector with configuration: %s", metrics_config)
        self.config = metrics_config
        self.validate_config()
        
        # Metric storage by type
        self.temporal_metrics: DefaultDict[str, List[float]] = defaultdict(list)
        self.temporal_timestamps: List[datetime] = []
        self.aggregate_metrics: DefaultDict[str, float] = defaultdict(float)
        self.metric_counts: DefaultDict[str, int] = defaultdict(int)
        
        # Analysis storage
        self.raw_data: DefaultDict[str, List[float]] = defaultdict(list)
        self.batch_metrics: List[Dict[str, float]] = []
        
        self.initialized = False
        self.last_save_time: Optional[datetime] = None
        self.initialize()
        
    def validate_config(self) -> None:
        """Validate metrics configuration"""
        if not self.config.validate_metrics():
            raise ValueError("Invalid metrics configuration: some metrics are undefined")
        
        if self.config.collect_interval <= 0:
            raise ValueError("Collection interval must be positive")
        
        if self.config.save_interval <= 0:
            raise ValueError("Save interval must be positive")
    
    def initialize(self) -> None:
        """Initialize metrics collector"""
        if self.initialized:
            return
            
        # Initialize storage for active metrics
        active_metrics = self.config.get_active_metrics()
        for metric in active_metrics:
            definition = self.config.definitions[metric]
            if definition.type == MetricType.TEMPORAL:
                self.temporal_metrics[metric] = []
            elif definition.type == MetricType.AGGREGATE:
                self.aggregate_metrics[metric] = 0.0
                
        self.initialized = True
        logger.info(f"Metrics collector initialized with {len(active_metrics)} active metrics")
    
    def collect(self, state: SimulationState) -> None:
        """Collect metrics from current simulation state"""
        if not self.initialized:
            raise RuntimeError("Metrics collector not initialized")
            
        try:
            current_metrics = self._calculate_metrics(state)
            self._store_metrics(current_metrics, state.current_time)
            
            # Handle batch processing and saving
            self.batch_metrics.append(current_metrics)
            if len(self.batch_metrics) >= self.config.batch_size:
                self._process_batch()
            
            # Check if we need to save
            if (not self.last_save_time or 
                (state.current_time - self.last_save_time).total_seconds() >= self.config.save_interval):
                self.save_metrics()
                self.last_save_time = state.current_time
                
        except Exception as e:
            import traceback
            logger.error(f"Failed to collect metrics: {str(e)}\n{traceback.format_exc()}")
            raise
    
    def _calculate_metrics(self, state: SimulationState) -> Dict[str, float]:
        """Calculate current values for all active metrics"""
        metrics: Dict[str, float] = {}
        
        # Vehicle-based metrics
        vehicles = state.vehicles
        total_vehicles = len(vehicles)
        active_vehicles = sum(1 for v in vehicles.values() 
                            if v['current_state']['status'] != VehicleStatus.OFF_DUTY.value)
        
        # Request-based metrics
        requests = state.requests
        total_requests = len(requests)
        completed_requests = sum(1 for r in requests.values() 
                               if r['status'] == RequestStatus.COMPLETED.value)
        
        # Calculate configured metrics
        for metric in self.config.get_active_metrics():
            definition = self.config.definitions[metric]
            
            if metric == "waiting_time":
                wait_times = [
                    (r['pickup_time'] - r['request_time']).total_seconds()
                    for r in requests.values()
                    if r['status'] == RequestStatus.COMPLETED.value 
                    and r.get('pickup_time')
                ]
                metrics[metric] = np.mean(wait_times) if wait_times else 0.0
                
            elif metric == "service_rate":
                metrics[metric] = (completed_requests / max(total_requests, 1)) * 100
                
            elif metric == "vehicle_utilization":
                active_time = sum(
                    v['current_state'].get('service_time', 0)
                    for v in vehicles.values()
                )
                total_time = sum(
                    v['current_state'].get('total_time', 0)
                    for v in vehicles.values()
                )
                metrics[metric] = (active_time / max(total_time, 1)) * 100
                
            elif metric == "total_distance":
                metrics[metric] = sum(
                    v['current_state']['distance_traveled']
                    for v in vehicles.values()
                )
                
            elif metric == "occupancy_rate":
                total_occupancy = sum(
                    v['current_state']['current_occupancy']
                    for v in vehicles.values()
                )
                metrics[metric] = (total_occupancy / max(active_vehicles, 1)) * 100
                
        return metrics
    
    def _store_metrics(self, metrics: Dict[str, float], timestamp: datetime) -> None:
        """Store calculated metrics according to their types"""
        # First handle temporal metrics to ensure consistency
        temporal_metrics_updated = False
        
        for metric_name, value in metrics.items():
            definition = self.config.definitions[metric_name]
            
            if definition.type == MetricType.TEMPORAL:
                # Only append timestamp once after we've processed all temporal metrics
                if not temporal_metrics_updated:
                    temporal_metrics_updated = True
                    self.temporal_timestamps.append(timestamp)
                
                # Ensure the metric has an entry for every timestamp
                current_length = len(self.temporal_metrics[metric_name])
                expected_length = len(self.temporal_timestamps)
                
                # If we somehow missed some entries, pad with NaN
                if current_length < expected_length - 1:
                    self.temporal_metrics[metric_name].extend(
                        [float('nan')] * (expected_length - 1 - current_length)
                    )
                
                # Add the current value
                self.temporal_metrics[metric_name].append(value)
                
            elif definition.type == MetricType.AGGREGATE:
                current_count = self.metric_counts[metric_name]
                current_value = self.aggregate_metrics[metric_name]
                
                # Update running average
                self.aggregate_metrics[metric_name] = (
                    (current_value * current_count + value) / (current_count + 1)
                )
                self.metric_counts[metric_name] += 1
            
            # Store raw data for analysis
            self.raw_data[metric_name].append(value)
        
        # Verify all temporal metrics have the same length as timestamps
        if temporal_metrics_updated:
            expected_length = len(self.temporal_timestamps)
            for metric_name, values in self.temporal_metrics.items():
                if len(values) != expected_length:
                    logger.warning(
                        f"Metric {metric_name} length ({len(values)}) does not match "
                        f"timestamp length ({expected_length}). Padding with NaN."
                    )
                    # Pad with NaN if necessary
                    if len(values) < expected_length:
                        self.temporal_metrics[metric_name].extend(
                            [float('nan')] * (expected_length - len(values))
                        )
                    # Trim if somehow we got too many values
                    elif len(values) > expected_length:
                        self.temporal_metrics[metric_name] = values[:expected_length]
    
    def _process_batch(self) -> None:
        """Process and clear current batch of metrics"""
        if not self.batch_metrics:
            return
            
        # Calculate batch statistics
        batch_stats = defaultdict(dict)
        for metric in self.config.get_active_metrics():
            values = [m[metric] for m in self.batch_metrics if metric in m]
            if values:
                batch_stats[metric].update({
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                })
        
        # Store batch statistics if needed
        # TODO: Implement batch statistics storage
        
        # Clear batch
        self.batch_metrics.clear()
    
    def save_metrics(self) -> None:
        """Save current metrics to storage"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save temporal metrics
        temporal_df = pd.DataFrame({
            'timestamp': self.temporal_timestamps,
            **self.temporal_metrics
        })
        
        if self.config.storage_format == "parquet":
            temporal_df.to_parquet(
                f"metrics_temporal_{timestamp}.parquet",
                compression=self.config.compression
            )
        else:
            temporal_df.to_csv(f"metrics_temporal_{timestamp}.csv")
        
        # Save aggregate metrics
        aggregate_df = pd.DataFrame([self.aggregate_metrics])
        if self.config.storage_format == "parquet":
            aggregate_df.to_parquet(
                f"metrics_aggregate_{timestamp}.parquet",
                compression=self.config.compression
            )
        else:
            aggregate_df.to_csv(f"metrics_aggregate_{timestamp}.csv")
    
    def get_analysis(self, metric_name: str) -> Dict[str, Any]:
        """Get statistical analysis for a metric"""
        if metric_name not in self.raw_data:
            raise ValueError(f"No data available for metric {metric_name}")
            
        values = self.raw_data[metric_name]
        
        analysis = {
            'count': len(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values)
        }
        
        # Add confidence interval
        if self.config.analysis['confidence_level']:
            ci = stats.t.interval(
                self.config.analysis['confidence_level'],
                len(values) - 1,
                loc=np.mean(values),
                scale=stats.sem(values)
            )
            analysis['confidence_interval'] = ci
        
        return analysis
    
    def cleanup(self) -> None:
        """Clean up resources"""
        self._process_batch()  # Process any remaining batch data
        self.save_metrics()    # Final save
        
        # Clear all data
        self.temporal_metrics.clear()
        self.temporal_timestamps.clear()
        self.aggregate_metrics.clear()
        self.metric_counts.clear()
        self.raw_data.clear()
        self.batch_metrics.clear()
        
        self.initialized = False
        logger.info("Metrics collector cleaned up")