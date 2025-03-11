from pathlib import Path
from typing import Optional, Union
import pandas as pd
import mlflow
import logging
from plotly import express as px
import plotly.graph_objects as go
import traceback

from drt_sim.core.monitoring.metrics.collector import MetricsCollector
from drt_sim.analysis.visualizations.metric_plotter import MetricPlotter

logger = logging.getLogger(__name__)

class MetricsManager:
    """Streamlined metrics manager for critical simulation monitoring."""
    
    def __init__(self, metrics_collector: MetricsCollector, output_dir: Path, replication_id: str):
        # Initialize as before but with simplified outputs
        self.metrics_collector = metrics_collector
        self.output_dir = Path(output_dir)
        self.replication_id = replication_id
        
        # Create directories
        self.analysis_dir = self.output_dir / "metrics" / replication_id
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.plotter = MetricPlotter()
    
    def save_figure(self, fig, filename, subdir=None):
        # Same as before but simplified
        save_dir = self.analysis_dir
        if subdir:
            save_dir = self.analysis_dir / subdir
            save_dir.mkdir(parents=True, exist_ok=True)
            
        filepath = save_dir / f"{filename}.png"
        fig.write_image(str(filepath))
        
        # Simplified MLflow logging
        try:
            if mlflow.active_run():
                artifact_path = "metrics" if not subdir else f"metrics/{subdir}"
                mlflow.log_artifact(str(filepath), artifact_path=artifact_path)
        except Exception as e:
            logger.error(f"Failed to log to MLflow: {str(e)}")
    
    def generate_essential_analysis(self):
        """Generate only essential analysis during simulation."""
        metrics_df = self.metrics_collector.get_metrics_df()
        if metrics_df.empty:
            return
        
        # Create and export pivot tables for post-analysis
        self._export_pivots(metrics_df)
        
        # Generate only critical visualizations
        self._generate_key_service_metrics(metrics_df)
        self._generate_key_vehicle_metrics(metrics_df)
        self._generate_key_passenger_metrics(metrics_df)
    
    def _export_pivots(self, metrics_df):
        """Export key pivot tables for post-analysis."""
        # Directory for exports
        export_dir = self.analysis_dir / "exports"
        export_dir.mkdir(exist_ok=True)
        
        # 1. Request metrics pivot
        request_metrics = metrics_df[
            metrics_df['metric_name'].isin(['request.received', 'request.assigned', 'request.rejected'])
        ]
        if not request_metrics.empty:
            try:
                request_counts = request_metrics.groupby(['timestamp', 'metric_name']).size().reset_index(name='count')
                pivot_counts = request_counts.pivot_table(
                    index='timestamp',
                    columns='metric_name',
                    values='count',
                    fill_value=0
                ).reset_index()
                
                pivot_counts.to_csv(export_dir / "requests_pivot.csv", index=False)
                
                # Also log to MLflow as a table
                if mlflow.active_run():
                    mlflow.log_table(pivot_counts, "exports/requests_pivot.json")
            except Exception as e:
                logger.error(f"Error exporting request pivot: {str(e)}")
        
        # 2. Vehicle metrics pivot
        try:
            vehicle_metrics = metrics_df[metrics_df['metric_name'].str.startswith('vehicle.')]
            if not vehicle_metrics.empty:
                # Aggregate metrics by vehicle
                vehicle_pivot = vehicle_metrics.pivot_table(
                    index='vehicle_id',
                    columns='metric_name',
                    values='value',
                    aggfunc='mean'
                ).reset_index()
                
                vehicle_pivot.to_csv(export_dir / "vehicle_metrics_pivot.csv", index=False)
        except Exception as e:
            logger.error(f"Error exporting vehicle pivot: {str(e)}")
        
        # 3. Passenger metrics pivot
        try:
            passenger_metrics = metrics_df[metrics_df['metric_name'].str.startswith('passenger.')]
            if not passenger_metrics.empty:
                # Create summary statistics
                passenger_summary = passenger_metrics.groupby('metric_name')['value'].agg([
                    'count', 'mean', 'median', 'std', 'min', 'max'
                ]).reset_index()
                
                passenger_summary.to_csv(export_dir / "passenger_metrics_summary.csv", index=False)
        except Exception as e:
            logger.error(f"Error exporting passenger summary: {str(e)}")
    
    def _generate_key_service_metrics(self, metrics_df):
        """Generate only essential service metric visualizations."""
        # Request success/failure pie chart
        request_metrics = metrics_df[
            metrics_df['metric_name'].isin(['request.assigned', 'request.rejected'])
        ]
        
        if not request_metrics.empty:
            try:
                # Simple count by status
                status_counts = request_metrics.groupby('metric_name')['metric_name'].count()
                
                # Create pie chart
                labels = {'request.assigned': 'Accepted', 'request.rejected': 'Rejected'}
                values = [
                    status_counts.get('request.assigned', 0),
                    status_counts.get('request.rejected', 0)
                ]
                
                fig = go.Figure(data=[go.Pie(
                    labels=list(labels.values()),
                    values=values,
                    hole=.3
                )])
                fig.update_layout(title_text="Request Acceptance Rate")
                
                self.save_figure(fig, "request_acceptance_rate", "service")
                
                # Log the actual rate to MLflow
                total = sum(values)
                if total > 0:
                    acceptance_rate = values[0] / total
                    mlflow.log_metric("service.acceptance_rate", acceptance_rate)
            except Exception as e:
                logger.error(f"Error generating service metrics: {str(e)}")
    
    def _generate_key_vehicle_metrics(self, metrics_df):
        """Generate only essential vehicle metric visualizations."""
        vehicle_metrics = metrics_df[metrics_df['metric_name'].str.startswith('vehicle.')]
        
        if not vehicle_metrics.empty:
            try:
                # Just a simple boxplot of key metrics
                key_metrics = [
                    'vehicle.utilization',
                    'vehicle.occupancy_ratio',
                    'vehicle.passengers_served'
                ]
                
                for metric in key_metrics:
                    metric_data = vehicle_metrics[vehicle_metrics['metric_name'] == metric]
                    if not metric_data.empty:
                        fig = px.box(
                            metric_data,
                            y='value',
                            title=f"{metric} Distribution"
                        )
                        self.save_figure(fig, f"{metric.replace('.', '_')}_box", "vehicle")
                        
                        # Log simple statistics
                        mlflow.log_metric(f"{metric}.mean", float(metric_data['value'].mean()))
                        mlflow.log_metric(f"{metric}.median", float(metric_data['value'].median()))
            except Exception as e:
                logger.error(f"Error generating vehicle metrics: {str(e)}")
    
    def _generate_key_passenger_metrics(self, metrics_df):
        """Generate only essential passenger metric visualizations."""
        passenger_metrics = metrics_df[metrics_df['metric_name'].str.startswith('passenger.')]
        
        if not passenger_metrics.empty:
            try:
                # Just a simple histogram of wait times
                wait_times = passenger_metrics[
                    passenger_metrics['metric_name'] == 'passenger.wait_time'
                ]
                
                if not wait_times.empty:
                    fig = px.histogram(
                        wait_times,
                        x='value',
                        title='Passenger Wait Time Distribution',
                        labels={'value': 'Wait Time (minutes)'},
                        nbins=20
                    )
                    self.save_figure(fig, "passenger_wait_time_hist", "passenger")
                    
                    # Log simple statistics
                    mlflow.log_metric("passenger.wait_time.mean", float(wait_times['value'].mean()))
                    mlflow.log_metric("passenger.wait_time.median", float(wait_times['value'].median()))
            except Exception as e:
                logger.error(f"Error generating passenger metrics: {str(e)}")

    def cleanup(self) -> None:
        """Clean up all metrics resources and ensure proper data export."""
        try:
            # Ensure all data is properly stored
            logger.debug(f"Starting metrics cleanup for replication {self.replication_id}")
            
            # 1. Save consolidated metrics data through the collector's storage
            if hasattr(self.metrics_collector, 'storage') and hasattr(self.metrics_collector.storage, 'save_consolidated_data'):
                try:
                    archive_paths = self.metrics_collector.storage.save_consolidated_data()
                    if archive_paths:
                        logger.debug(f"Saved consolidated metrics data to {archive_paths}")
                        
                        # Log to MLflow if available
                        try:
                            active_run = mlflow.active_run()
                            if active_run:
                                mlflow.log_artifact(str(archive_paths), "archive")
                        except Exception as e:
                            logger.error(f"Failed to log consolidated data to MLflow: {str(e)}")
                except Exception as e:
                    logger.error(f"Error saving consolidated metrics data: {str(e)}")
            
            # 2. Final export of metrics summary
            try:
                # Get final metrics state and export summary stats
                metrics_df = self.metrics_collector.get_metrics_df()
                if not metrics_df.empty:
                    # Create summary statistics for all metrics
                    summary = metrics_df.groupby('metric_name')['value'].agg([
                        'count', 'mean', 'median', 'std', 'min', 'max'
                    ]).reset_index()
                    
                    # Save to CSV
                    summary_path = self.analysis_dir / "exports" / "metrics_summary.csv"
                    summary_path.parent.mkdir(exist_ok=True)
                    summary.to_csv(summary_path, index=False)
                    
                    # Log to MLflow
                    if mlflow.active_run():
                        mlflow.log_artifact(str(summary_path), "exports")
            except Exception as e:
                logger.error(f"Error exporting final metrics summary: {str(e)}")
            
            # 3. Clean up the metrics collector
            try:
                if hasattr(self.metrics_collector, 'cleanup'):
                    self.metrics_collector.cleanup()
            except Exception as e:
                logger.error(f"Error during metrics collector cleanup: {str(e)}")
            
            
            logger.debug("Metrics cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during metrics cleanup: {str(e)}\n{traceback.format_exc()}")