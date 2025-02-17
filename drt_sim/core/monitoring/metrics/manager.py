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
from drt_sim.core.monitoring.metrics.registry import metric_registry

logger = logging.getLogger(__name__)

class MetricsManager:
    """Manages analysis and visualization of collected metrics with focused, practical plots."""
    
    def __init__(self,
                 metrics_collector: MetricsCollector,
                 output_dir: Union[str, Path],
                 replication_id: str):
        """Initialize the metrics manager.
        
        Args:
            metrics_collector: The metrics collector containing the data to analyze
            output_dir: Base directory for analysis outputs
            replication_id: ID of the current simulation replication
        """
        self.metrics_collector = metrics_collector
        self.output_dir = Path(output_dir)
        self.replication_id = replication_id
        
        # Initialize analysis directory
        self.analysis_dir = self.output_dir / "analysis" / replication_id
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize visualization and reporting components
        self.plotter = MetricPlotter()
        
        # Create subdirectories for different analysis types
        self.vehicle_dir = self.analysis_dir / "vehicle_analysis"
        self.passenger_dir = self.analysis_dir / "passenger_analysis"
        self.service_dir = self.analysis_dir / "service_analysis"
        self.system_dir = self.analysis_dir / "system_analysis"
        
        for directory in [self.vehicle_dir, self.passenger_dir, self.service_dir, self.system_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def save_figure(self, fig, filename: str, subdir: Optional[str] = None) -> None:
        """Save a plotly figure to HTML and optionally to MLflow.
        
        Args:
            fig: The plotly figure to save
            filename: Name of the file without extension
            subdir: Optional subdirectory within analysis dir
        """
        save_dir = self.analysis_dir
        if subdir:
            save_dir = self.analysis_dir / subdir
            save_dir.mkdir(parents=True, exist_ok=True)
            
        filepath = save_dir / f"{filename}.html"
        fig.write_html(str(filepath))
        
        # Log to MLflow if available
        try:
            # Check if we're in an active run
            active_run = mlflow.active_run()
            if not active_run:
                logger.error("No active MLflow run found when trying to save figure")
                return
                
            logger.info(f"Current MLflow run ID: {active_run.info.run_id}")
            logger.info(f"Saving figure to local path: {filepath}")
            
            # Use the same directory structure for MLflow artifacts
            artifact_path = "analysis"
            if subdir:
                artifact_path = str(Path("analysis") / subdir)
                
            logger.info(f"Using MLflow artifact path: {artifact_path}")
            
            # Log the artifact
            mlflow.log_artifact(str(filepath), artifact_path=artifact_path)
            logger.info(f"Successfully logged figure to MLflow: {filename}")
            
        except Exception as e:
            logger.error(f"Failed to log figure to MLflow: {str(e)}\nTraceback: {traceback.format_exc()}")

    def generate_all_analysis(self) -> None:
        """Generate all analysis plots and metrics."""
        logger.info("Starting comprehensive metrics analysis")
        
        # Get the complete metrics dataset
        metrics_df = self.metrics_collector.get_metrics_df()
        
        if metrics_df.empty:
            logger.warning("No metrics data available for analysis")
            return
        
        # Add derived metrics
        metrics_df = self._add_derived_metrics(metrics_df)
        
        # Generate all analyses
        self.analyze_vehicle_performance(metrics_df)
        self.analyze_passenger_experience(metrics_df)
        # self.analyze_service_efficiency(metrics_df)
        # self.analyze_system_performance(metrics_df)
        
        logger.info("Completed comprehensive metrics analysis")

    def _add_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived metrics to the DataFrame."""
        # Create a copy to avoid modifying the original
        df = df.copy()
        
        # 1. Passenger Journey Time Components
        passenger_metrics = df[df['metric_name'].str.startswith('passenger.')]
        if not passenger_metrics.empty:
            journey_components = [
                'passenger.wait_time',
                'passenger.ride_time',
                'passenger.walk_time_to_origin_stop',
                'passenger.walk_time_from_destination_stop'
            ]
            
            # Filter for available components only
            available_components = [
                comp for comp in journey_components
                if comp in passenger_metrics['metric_name'].unique()
            ]
            
            if available_components:
                try:
                    # Ensure we have numeric values
                    passenger_metrics.loc[:, 'value'] = pd.to_numeric(passenger_metrics['value'], errors='coerce')
                    passenger_metrics = passenger_metrics.dropna(subset=['value'])
                    
                    # Pivot to get all components
                    journey_df = passenger_metrics[
                        passenger_metrics['metric_name'].isin(available_components)
                    ].pivot_table(
                        index=['passenger_id', 'request_id'],
                        columns='metric_name',
                        values='value',
                        aggfunc='first'
                    ).reset_index()
                    
                    # Get actual columns from pivot table (excluding index columns)
                    actual_columns = [col for col in journey_df.columns 
                                    if col in available_components]
                    
                    if actual_columns:  # Only proceed if we have actual metric columns
                        # Calculate total journey time from available components
                        journey_df['passenger.total_journey_time'] = journey_df[actual_columns].fillna(0).sum(axis=1)
                        
                        # Convert back to long format and append
                        journey_long = pd.DataFrame({
                            'passenger_id': journey_df['passenger_id'],
                            'request_id': journey_df['request_id'],
                            'metric_name': 'passenger.total_journey_time',
                            'value': journey_df['passenger.total_journey_time']
                        })
                        
                        df = pd.concat([df, journey_long], ignore_index=True)
                        
                        if len(actual_columns) < len(journey_components):
                            logger.warning(f"Some journey components were missing. Using only: {actual_columns}")
                    else:
                        logger.warning("No journey components available after pivot")
                except Exception as e:
                    logger.warning(f"Error calculating journey metrics: {str(e)}")
        
        # 2. Vehicle Performance Metrics
        vehicle_metrics = df[df['metric_name'].str.startswith('vehicle.')]
        if not vehicle_metrics.empty:
            try:
                # Ensure numeric values
                vehicle_metrics.loc[:, 'value'] = pd.to_numeric(vehicle_metrics['value'], errors='coerce')
                vehicle_metrics = vehicle_metrics.dropna(subset=['value'])
                
                # Get distance metrics
                distance_metrics = ['vehicle.occupied_distance', 'vehicle.empty_distance']
                available_distance_metrics = [
                    metric for metric in distance_metrics
                    if metric in vehicle_metrics['metric_name'].unique()
                ]
                
                if len(available_distance_metrics) == len(distance_metrics):
                    try:
                        distance_df = vehicle_metrics[
                            vehicle_metrics['metric_name'].isin(available_distance_metrics)
                        ].pivot_table(
                            index=['vehicle_id', 'timestamp'],
                            columns='metric_name',
                            values='value',
                            aggfunc='sum'
                        ).reset_index()
                        
                        if not distance_df.empty and all(metric in distance_df.columns for metric in available_distance_metrics):
                            # Calculate total distance
                            distance_df['total_distance'] = (
                                distance_df['vehicle.occupied_distance'] + 
                                distance_df['vehicle.empty_distance']
                            )
                            
                            # Calculate occupancy ratio
                            distance_df['vehicle.occupancy_ratio'] = (
                                distance_df['vehicle.occupied_distance'] / 
                                distance_df['total_distance']
                            ).fillna(0)  # Handle division by zero
                            
                            # Get passengers served if available
                            if 'vehicle.passengers_served' in vehicle_metrics['metric_name'].unique():
                                try:
                                    passengers_df = vehicle_metrics[
                                        vehicle_metrics['metric_name'] == 'vehicle.passengers_served'
                                    ].pivot_table(
                                        index=['vehicle_id', 'timestamp'],
                                        values='value',
                                        aggfunc='sum'
                                    ).reset_index()
                                    
                                    if not passengers_df.empty:
                                        # Merge with distance data
                                        combined_df = distance_df.merge(
                                            passengers_df,
                                            on=['vehicle_id', 'timestamp'],
                                            how='left'
                                        )
                                        
                                        # Calculate passengers per km
                                        combined_df['vehicle.passengers_per_km'] = (
                                            combined_df['value'] / (combined_df['total_distance'] / 1000)
                                        ).fillna(0)  # Handle division by zero
                                        
                                        # Convert derived metrics to DataFrame format and append
                                        derived_metrics = []
                                        
                                        # Add occupancy ratio
                                        derived_metrics.append(pd.DataFrame({
                                            'vehicle_id': combined_df['vehicle_id'],
                                            'timestamp': combined_df['timestamp'],
                                            'metric_name': 'vehicle.occupancy_ratio',
                                            'value': combined_df['vehicle.occupancy_ratio']
                                        }))
                                        
                                        # Add passengers per km
                                        derived_metrics.append(pd.DataFrame({
                                            'vehicle_id': combined_df['vehicle_id'],
                                            'timestamp': combined_df['timestamp'],
                                            'metric_name': 'vehicle.passengers_per_km',
                                            'value': combined_df['vehicle.passengers_per_km']
                                        }))
                                        
                                        # Combine all derived metrics
                                        if derived_metrics:
                                            derived_vehicle_metrics = pd.concat(derived_metrics, ignore_index=True)
                                            df = pd.concat([df, derived_vehicle_metrics], ignore_index=True)
                                            
                                except Exception as e:
                                    logger.warning(f"Error calculating passenger metrics: {str(e)}")
                                    # Still try to add occupancy ratio
                                    derived_vehicle_metrics = pd.DataFrame({
                                        'vehicle_id': distance_df['vehicle_id'],
                                        'timestamp': distance_df['timestamp'],
                                        'metric_name': 'vehicle.occupancy_ratio',
                                        'value': distance_df['vehicle.occupancy_ratio']
                                    })
                                    df = pd.concat([df, derived_vehicle_metrics], ignore_index=True)
                            else:
                                # Only add occupancy ratio if passengers served is not available
                                derived_vehicle_metrics = pd.DataFrame({
                                    'vehicle_id': distance_df['vehicle_id'],
                                    'timestamp': distance_df['timestamp'],
                                    'metric_name': 'vehicle.occupancy_ratio',
                                    'value': distance_df['vehicle.occupancy_ratio']
                                })
                                df = pd.concat([df, derived_vehicle_metrics], ignore_index=True)
                    except Exception as e:
                        logger.warning(f"Error calculating vehicle metrics: {str(e)}")
                else:
                    logger.warning("Missing required distance metrics for vehicle performance calculations")
            except Exception as e:
                logger.warning(f"Error processing vehicle metrics: {str(e)}")
        
        return df

    def _create_visualizations_for_metric(self,
                                        df: pd.DataFrame,
                                        metric_name: str,
                                        subdir: str,
                                        group_by: Optional[str] = None) -> None:
        """Create visualizations for a metric based on its definition."""
        if df.empty:
            return
            
        # Get metric definition directly from registry
        definition = metric_registry.get(metric_name)
        if not definition:
            logger.warning(f"No definition found for metric {metric_name}")
            return
            
        metric_data = df[df['metric_name'] == metric_name]
        if metric_data.empty:
            return
            
        # Time series plot if enabled
        if definition.visualizations.get('time_series', True):
            fig = self.plotter.create_time_series(
                metric_data,
                f'{definition.description} Over Time',
                'Time',
                f'{definition.description} ({definition.unit})',
                group_by=group_by
            )
            self.save_figure(fig, f"{metric_name.replace('.', '_')}_time", subdir)
            
        # Distribution plot if enabled
        if definition.visualizations.get('distribution', True):
            fig = self.plotter.create_distribution_plot(
                metric_data,
                f'{definition.description} Distribution',
                f'Value ({definition.unit})',
                'Count',
                group_by=group_by
            )
            self.save_figure(fig, f"{metric_name.replace('.', '_')}_dist", subdir)
            
        # Box plot if distribution is enabled and we have a group_by
        if definition.visualizations.get('distribution', True) and group_by:
            fig = self.plotter.create_box_plot(
                metric_data,
                f'{definition.description} by {group_by}',
                group_by,
                f'Value ({definition.unit})',
                group_by=group_by
            )
            self.save_figure(fig, f"{metric_name.replace('.', '_')}_box", subdir)
            
        # Log summary statistics if aggregations are defined
        if definition.aggregations:
            stats = {
                'mean': float(metric_data['value'].mean()),
                'median': float(metric_data['value'].median()),
                'std': float(metric_data['value'].std()),
                'min': float(metric_data['value'].min()),
                'max': float(metric_data['value'].max()),
                'count': int(len(metric_data))
            }
            
            for stat_name, stat_value in stats.items():
                if stat_name in definition.aggregations:
                    mlflow.log_metric(f"{metric_name}.{stat_name}", stat_value)

    def analyze_vehicle_performance(self, metrics_df: pd.DataFrame) -> None:
        """Analyze vehicle performance metrics."""
        logger.info("Analyzing vehicle performance")
        
        # Filter vehicle-related metrics
        vehicle_metrics = metrics_df[metrics_df['metric_name'].str.startswith('vehicle.')]
        
        if vehicle_metrics.empty:
            logger.warning("No vehicle metrics available")
            return
            
        # Create visualizations for base metrics only
        base_metric_names = [
            'vehicle.utilization',
            'vehicle.occupied_distance',
            'vehicle.empty_distance',
            'vehicle.dwell_time',
            'vehicle.stops_served',
            'vehicle.passengers_served'
        ]
        
        for metric_name in base_metric_names:
            self._create_visualizations_for_metric(
                vehicle_metrics,
                metric_name,
                "vehicle_analysis",
                group_by='vehicle_id'
            )
            
        # Handle derived metrics separately
        self._analyze_derived_vehicle_metrics(vehicle_metrics)

    def _analyze_derived_vehicle_metrics(self, vehicle_metrics: pd.DataFrame) -> None:
        """Create specialized visualizations for derived vehicle metrics."""
        # 1. Vehicle Occupancy Ratio Analysis
        occupancy_data = vehicle_metrics[
            vehicle_metrics['metric_name'] == 'vehicle.occupancy_ratio'
        ]
        
        if not occupancy_data.empty:
            # Fleet-wide occupancy ratio distribution
            fig = px.box(occupancy_data,
                        x='vehicle_id',
                        y='value',
                        title='Fleet Occupancy Ratio Distribution',
                        labels={
                            'value': 'Occupancy Ratio',
                            'vehicle_id': 'Vehicle ID'
                        })
            self.save_figure(fig, "fleet_occupancy_ratio_dist", "vehicle_analysis")
            
            # Fleet-wide summary
            fleet_summary = occupancy_data.groupby('vehicle_id')['value'].agg(['mean', 'std']).reset_index()
            fig = px.bar(fleet_summary,
                        x='vehicle_id',
                        y='mean',
                        error_y='std',
                        title='Fleet Average Occupancy Ratio',
                        labels={
                            'mean': 'Average Occupancy Ratio',
                            'vehicle_id': 'Vehicle ID'
                        })
            self.save_figure(fig, "fleet_occupancy_ratio_summary", "vehicle_analysis")
            
        # 2. Passengers per KM Analysis
        passengers_per_km = vehicle_metrics[
            vehicle_metrics['metric_name'] == 'vehicle.passengers_per_km'
        ]
        
        if not passengers_per_km.empty:
            # Fleet-wide passengers/km distribution
            fig = px.box(passengers_per_km,
                        x='vehicle_id',
                        y='value',
                        title='Passengers per KM Distribution by Vehicle',
                        labels={
                            'value': 'Passengers per KM',
                            'vehicle_id': 'Vehicle ID'
                        })
            self.save_figure(fig, "fleet_passengers_per_km_dist", "vehicle_analysis")
            
            # Fleet efficiency comparison
            fleet_efficiency = passengers_per_km.groupby('vehicle_id')['value'].agg(['mean', 'std']).reset_index()
            fig = px.bar(fleet_efficiency,
                        x='vehicle_id',
                        y='mean',
                        error_y='std',
                        title='Fleet Passenger Efficiency',
                        labels={
                            'mean': 'Average Passengers per KM',
                            'vehicle_id': 'Vehicle ID'
                        })
            self.save_figure(fig, "fleet_passenger_efficiency", "vehicle_analysis")
            
        # 3. Distance Breakdown (Combined Visualization)
        distance_metrics = vehicle_metrics[
            vehicle_metrics['metric_name'].isin(['vehicle.occupied_distance', 'vehicle.empty_distance'])
        ].pivot_table(
            index='vehicle_id',
            columns='metric_name',
            values='value',
            aggfunc='sum'
        ).reset_index()
        
        if not distance_metrics.empty:
            fig = px.bar(distance_metrics,
                        x='vehicle_id',
                        y=['vehicle.occupied_distance', 'vehicle.empty_distance'],
                        title='Vehicle Distance Breakdown',
                        barmode='stack',
                        labels={
                            'value': 'Distance (meters)',
                            'vehicle_id': 'Vehicle ID',
                            'variable': 'Distance Type'
                        })
            self.save_figure(fig, "vehicle_distance_breakdown", "vehicle_analysis")

    def analyze_passenger_experience(self, metrics_df: pd.DataFrame) -> None:
        """Analyze passenger experience metrics."""
        logger.info("Analyzing passenger experience")
        
        passenger_metrics = metrics_df[metrics_df['metric_name'].str.startswith('passenger.')]
        
        if passenger_metrics.empty:
            logger.warning("No passenger metrics available")
            return
            
        # Create visualizations for base metrics only
        base_metric_names = [
            'passenger.wait_time',
            'passenger.ride_time',
            'passenger.walk_time_to_origin_stop',
            'passenger.walk_time_from_destination_stop'
        ]
        
        for metric_name in base_metric_names:
            self._create_visualizations_for_metric(
                passenger_metrics,
                metric_name,
                "passenger_analysis",
                group_by=None
            )
            
        # Handle derived metrics separately
        self._analyze_derived_passenger_metrics(passenger_metrics)

    def _analyze_derived_passenger_metrics(self, passenger_metrics: pd.DataFrame) -> None:
        """Create specialized visualizations for derived passenger metrics."""
        try:
            # 1. Total Journey Time Analysis
            journey_time = passenger_metrics[
                passenger_metrics['metric_name'] == 'passenger.total_journey_time'
            ]
            
            if not journey_time.empty:
                try:
                    # Ensure we have numeric values
                    journey_time['value'] = pd.to_numeric(journey_time['value'], errors='coerce')
                    journey_time = journey_time.dropna(subset=['value'])
                    
                    if not journey_time.empty:
                        # Journey time distribution
                        fig = px.histogram(journey_time,
                                         x='value',
                                         title='Distribution of Total Journey Times',
                                         labels={'value': 'Journey Time (minutes)'},
                                         nbins=30)
                        self.save_figure(fig, "total_journey_time_dist", "passenger_analysis")
                    else:
                        logger.warning("No valid numeric journey time values found")
                except Exception as e:
                    logger.error(f"Error creating journey time distribution: {str(e)}")
            
            # 2. Journey Components Analysis
            try:
                journey_components = [
                    'passenger.wait_time',
                    'passenger.ride_time',
                    'passenger.walk_time_to_origin_stop',
                    'passenger.walk_time_from_destination_stop'
                ]
                
                # Filter for available components and ensure they exist
                available_components = [
                    comp for comp in journey_components
                    if comp in passenger_metrics['metric_name'].unique()
                ]
                
                if not available_components:
                    logger.warning("No journey components available for analysis")
                    return
                
                # Get component data and ensure numeric values
                components_data = passenger_metrics[
                    passenger_metrics['metric_name'].isin(available_components)
                ].copy()
                
                # Convert to numeric, dropping any non-numeric values
                components_data['value'] = pd.to_numeric(components_data['value'], errors='coerce')
                components_data = components_data.dropna(subset=['value'])
                
                if components_data.empty:
                    logger.warning("No valid numeric values found in journey components")
                    return
                
                # Create pivot table with error handling
                try:
                    components = components_data.pivot_table(
                        index='passenger_id',
                        columns='metric_name',
                        values='value',
                        aggfunc='mean'
                    ).reset_index()
                    
                    if components.empty:
                        logger.warning("No data after pivoting journey components")
                        return
                        
                    # Verify we have numeric columns (excluding passenger_id)
                    numeric_cols = [col for col in components.columns 
                                  if col != 'passenger_id' and 
                                  pd.api.types.is_numeric_dtype(components[col])]
                    
                    if not numeric_cols:
                        logger.warning("No numeric columns found in journey components")
                        return
                    
                    # Create stacked bar chart
                    try:
                        fig = px.bar(components,
                                   x='passenger_id',
                                   y=numeric_cols,
                                   title='Journey Time Components Breakdown',
                                   barmode='stack',
                                   labels={
                                       'value': 'Time (minutes)',
                                       'passenger_id': 'Passenger ID',
                                       'variable': 'Component'
                                   })
                        self.save_figure(fig, "journey_components_breakdown", "passenger_analysis")
                    except Exception as e:
                        logger.error(f"Error creating journey components breakdown: {str(e)}")
                    
                    # Create average composition pie chart
                    try:
                        # Calculate means only for numeric columns
                        avg_composition = components[numeric_cols].mean()
                        
                        if not avg_composition.empty and not avg_composition.isna().all():
                            fig = px.pie(values=avg_composition.values,
                                       names=avg_composition.index,
                                       title='Average Journey Time Composition')
                            self.save_figure(fig, "journey_composition", "passenger_analysis")
                        else:
                            logger.warning("No valid data for journey composition pie chart")
                    except Exception as e:
                        logger.error(f"Error creating journey composition pie chart: {str(e)}")
                        
                except Exception as e:
                    logger.error(f"Error in journey components pivot table creation: {str(e)}")
                    
            except Exception as e:
                logger.error(f"Error in journey components analysis: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error in passenger metrics analysis: {str(e)}")
            # Continue with other analyses if possible

    def analyze_service_efficiency(self, metrics_df: pd.DataFrame) -> None:
        """Analyze service efficiency metrics."""
        logger.info("Analyzing service efficiency")
        
        service_metrics = metrics_df[metrics_df['metric_name'].str.startswith('service.')]
        
        if service_metrics.empty:
            logger.warning("No service metrics available")
            return
            
        # Create visualizations for base metrics only
        base_metric_names = [
            'service.violations',
            'service.capacity_utilization'
        ]
        
        for metric_name in base_metric_names:
            self._create_visualizations_for_metric(
                service_metrics,
                metric_name,
                "service_analysis"
            )
            
        # Handle derived metrics separately
        self._analyze_derived_service_metrics(service_metrics)

    def _analyze_derived_service_metrics(self, service_metrics: pd.DataFrame) -> None:
        """Create specialized visualizations for derived service metrics."""
        # 1. Service Success Rate Analysis
        success_rate = service_metrics[
            service_metrics['metric_name'] == 'service.request_success_rate'
        ]
        
        if not success_rate.empty:
            # Success rate over time
            fig = px.line(success_rate,
                         x='timestamp',
                         y='value',
                         title='Request Success Rate Over Time',
                         labels={
                             'value': 'Success Rate',
                             'timestamp': 'Time'
                         })
            self.save_figure(fig, "request_success_rate", "service_analysis")
            
        # 2. On-time Performance Analysis
        on_time_rate = service_metrics[
            service_metrics['metric_name'] == 'service.on_time_rate'
        ]
        
        if not on_time_rate.empty:
            # On-time rate over time
            fig = px.line(on_time_rate,
                         x='timestamp',
                         y='value',
                         title='On-time Performance Rate',
                         labels={
                             'value': 'On-time Rate',
                             'timestamp': 'Time'
                         })
            self.save_figure(fig, "on_time_rate", "service_analysis")
            
        # 3. Service Violations Analysis
        violations = service_metrics[
            service_metrics['metric_name'] == 'service.violations'
        ]
        
        if not violations.empty:
            # Violations by type
            violations_by_type = violations.pivot_table(
                index='violation_type',
                values='value',
                aggfunc='count'
            ).reset_index()
            
            fig = px.bar(violations_by_type,
                        x='violation_type',
                        y='value',
                        title='Service Violations by Type',
                        labels={
                            'value': 'Count',
                            'violation_type': 'Violation Type'
                        })
            self.save_figure(fig, "violations_breakdown", "service_analysis")
            
            # Violations over time
            violations_timeline = violations.groupby('timestamp')['value'].count().reset_index()
            fig = px.line(violations_timeline,
                         x='timestamp',
                         y='value',
                         title='Service Violations Over Time',
                         labels={
                             'value': 'Number of Violations',
                             'timestamp': 'Time'
                         })
            self.save_figure(fig, "violations_timeline", "service_analysis")

    def analyze_system_performance(self, metrics_df: pd.DataFrame) -> None:
        """Analyze system-wide performance metrics."""
        logger.info("Analyzing system performance")
        
        system_metrics = metrics_df[metrics_df['metric_name'].str.startswith('system.')]
        
        if system_metrics.empty:
            logger.warning("No system metrics available")
            return
            
        # Create visualizations for base metrics only
        base_metric_names = [
            'system.cpu_usage',
            'system.memory_usage'
        ]
        
        for metric_name in base_metric_names:
            self._create_visualizations_for_metric(
                system_metrics,
                metric_name,
                "system_analysis"
            )
            
        # Handle derived metrics separately
        self._analyze_derived_system_metrics(system_metrics)

    def _analyze_derived_system_metrics(self, system_metrics: pd.DataFrame) -> None:
        """Create specialized visualizations for derived system metrics."""
        # 1. System Load Analysis
        load_metrics = system_metrics[
            system_metrics['metric_name'].isin([
                'system.active_requests',
                'system.active_vehicles'
            ])
        ]
        
        if not load_metrics.empty:
            # System load over time
            pivot_load = load_metrics.pivot_table(
                index='timestamp',
                columns='metric_name',
                values='value'
            ).reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pivot_load['timestamp'],
                y=pivot_load['system.active_requests'],
                name='Active Requests',
                mode='lines'
            ))
            fig.add_trace(go.Scatter(
                x=pivot_load['timestamp'],
                y=pivot_load['system.active_vehicles'],
                name='Active Vehicles',
                mode='lines'
            ))
            fig.update_layout(
                title='System Load Over Time',
                xaxis_title='Time',
                yaxis_title='Count',
                hovermode='x unified'
            )
            self.save_figure(fig, "system_load", "system_analysis")
            
            # Calculate and plot request/vehicle ratio
            if 'system.active_vehicles' in pivot_load.columns and 'system.active_requests' in pivot_load.columns:
                pivot_load['request_vehicle_ratio'] = (
                    pivot_load['system.active_requests'] / 
                    pivot_load['system.active_vehicles'].replace(0, 1)  # Avoid division by zero
                )
                
                fig = px.line(pivot_load,
                             x='timestamp',
                             y='request_vehicle_ratio',
                             title='Request to Vehicle Ratio Over Time',
                             labels={
                                 'request_vehicle_ratio': 'Requests per Vehicle',
                                 'timestamp': 'Time'
                             })
                self.save_figure(fig, "request_vehicle_ratio", "system_analysis")

    def cleanup(self) -> None:
        """Clean up all metrics resources and ensure proper MLflow logging."""
        try:
            # Ensure all MLflow operations are complete
            active_run = mlflow.active_run()
            if active_run:
                logger.info(f"Ensuring all artifacts are logged to MLflow run: {active_run.info.run_id}")
                
                # First, save consolidated metrics data through the collector's storage
                archive_paths = self.metrics_collector.storage.save_consolidated_data()
                if archive_paths:
                    parquet_path, json_path = archive_paths
                    # Log the consolidated metrics data to MLflow
                    logger.info(f"Logging consolidated metrics data to MLflow from {parquet_path} and {json_path}")
                    mlflow.log_artifact(str(parquet_path), "archive")
                    mlflow.log_artifact(str(json_path), "archive")
            
            # Clean up the metrics collector (which will clean up storage)
            self.metrics_collector.cleanup()
            
            # Clean up analysis directories
            import shutil
            shutil.rmtree(self.analysis_dir, ignore_errors=True)
            
            logger.info("Metrics cleanup completed successfully")
        except Exception as e:
            logger.error(f"Error during metrics cleanup: {str(e)}\nTraceback: {traceback.format_exc()}") 