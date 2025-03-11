#!/usr/bin/env python3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse
import logging
from pathlib import Path
import traceback
from enum import Enum
import numpy as np
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RejectionReason(str, Enum):
    """Enum for rejection reasons to mirror the original code's RejectionReason class."""
    TIME_WINDOW_CONSTRAINT = "time_window_constraint"
    VEHICLE_ACCESS_TIME_CONSTRAINT = "vehicle_access_time_constraint"
    PASSENGER_WAIT_TIME_CONSTRAINT = "passenger_wait_time_constraint"
    RIDE_TIME_CONSTRAINT = "ride_time_constraint"
    CAPACITY_CONSTRAINT = "capacity_constraint"
    DETOUR_CONSTRAINT = "detour_constraint"
    NO_VEHICLES_AVAILABLE = "no_vehicles_available"
    NO_COMPATIBLE_VEHICLES = "no_compatible_vehicles"
    OUTSIDE_SERVICE_HOURS = "outside_service_hours"
    OUTSIDE_SERVICE_AREA = "outside_service_area"
    EXCESSIVE_WALKING_DISTANCE = "excessive_walking_distance"
    INVALID_BOOKING_TIME = "invalid_booking_time"
    INSUFFICIENT_NOTICE = "insufficient_notice"
    OTHER = "other"

class MetricDefinition:
    """Simple class to define metrics with their properties."""
    def __init__(self, name, description, unit, visualizations=None, aggregations=None):
        self.name = name
        self.description = description
        self.unit = unit
        self.visualizations = visualizations or {"time_series": True, "distribution": True}
        self.aggregations = aggregations or ["mean", "median", "std", "min", "max"]

class MetricRegistry:
    """Registry for metric definitions."""
    def __init__(self):
        self.metrics = {}
        self._initialize_metrics()
    
    def _initialize_metrics(self):
        """Initialize predefined metrics."""
        # Vehicle metrics
        self.add(MetricDefinition("vehicle.utilization", "Vehicle Utilization", "%"))
        self.add(MetricDefinition("vehicle.occupied_distance", "Occupied Distance", "meters"))
        self.add(MetricDefinition("vehicle.empty_distance", "Empty Distance", "meters"))
        self.add(MetricDefinition("vehicle.dwell_time", "Dwell Time", "seconds"))
        self.add(MetricDefinition("vehicle.stops_served", "Stops Served", "count"))
        self.add(MetricDefinition("vehicle.passengers_served", "Passengers Served", "count"))
        self.add(MetricDefinition("vehicle.occupancy_ratio", "Occupancy Ratio", "%"))
        self.add(MetricDefinition("vehicle.passengers_per_km", "Passengers per KM", "count/km"))
        
        # Passenger metrics
        self.add(MetricDefinition("passenger.wait_time", "Wait Time", "seconds"))
        self.add(MetricDefinition("passenger.ride_time", "Ride Time", "seconds"))
        self.add(MetricDefinition("passenger.walk_time_to_origin_stop", "Walk Time to Origin", "seconds"))
        self.add(MetricDefinition("passenger.walk_time_from_destination_stop", "Walk Time from Destination", "seconds"))
        self.add(MetricDefinition("passenger.total_journey_time", "Total Journey Time", "seconds"))
        
        # Service metrics
        self.add(MetricDefinition("service.violations", "Service Violations", "count"))
        self.add(MetricDefinition("service.capacity_utilization", "Capacity Utilization", "%"))
        self.add(MetricDefinition("service.request_success_rate", "Request Success Rate", "%"))
        self.add(MetricDefinition("service.on_time_rate", "On-Time Rate", "%"))
        
        # Request metrics
        self.add(MetricDefinition("request.received", "Requests Received", "count"))
        self.add(MetricDefinition("request.assigned", "Requests Assigned", "count"))
        self.add(MetricDefinition("request.rejected", "Requests Rejected", "count"))
    
    def add(self, metric_definition):
        """Add a metric definition to the registry."""
        self.metrics[metric_definition.name] = metric_definition
    
    def get(self, metric_name):
        """Get a metric definition by name."""
        return self.metrics.get(metric_name)

class MetricPlotter:
    """Creates standardized plots for metrics visualization."""
    
    def create_time_series(self, data, title, x_label, y_label, group_by=None):
        """Create a time series plot."""
        if group_by and group_by in data.columns:
            fig = px.line(
                data,
                x='timestamp',
                y='value',
                color=group_by,
                title=title,
                labels={
                    'timestamp': x_label,
                    'value': y_label,
                    group_by: group_by.capitalize()
                }
            )
        else:
            fig = px.line(
                data,
                x='timestamp',
                y='value',
                title=title,
                labels={
                    'timestamp': x_label,
                    'value': y_label
                }
            )
        return fig
    
    def create_distribution_plot(self, data, title, x_label, y_label, group_by=None):
        """Create a distribution plot (histogram)."""
        if group_by and group_by in data.columns:
            fig = px.histogram(
                data,
                x='value',
                color=group_by,
                title=title,
                labels={
                    'value': x_label,
                    'count': y_label,
                    group_by: group_by.capitalize()
                },
                marginal='box'
            )
        else:
            fig = px.histogram(
                data,
                x='value',
                title=title,
                labels={
                    'value': x_label,
                    'count': y_label
                },
                marginal='box'
            )
        return fig
    
    def create_box_plot(self, data, title, x_label, y_label, group_by=None):
        """Create a box plot for grouped data."""
        if group_by and group_by in data.columns:
            fig = px.box(
                data,
                x=group_by,
                y='value',
                title=title,
                labels={
                    group_by: x_label,
                    'value': y_label
                }
            )
        else:
            # Fallback to regular box plot if group_by is not available
            fig = px.box(
                data,
                y='value',
                title=title,
                labels={
                    'value': y_label
                }
            )
        return fig

class StandaloneMetricsAnalyzer:
    """Standalone metrics analyzer that mimics MetricsManager functionality."""
    
    def __init__(self, metrics_df, output_dir):
        """Initialize the metrics analyzer.
        
        Args:
            metrics_df: DataFrame containing metrics data
            output_dir: Directory to save output plots
        """
        self.metrics_df = metrics_df
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.plotter = MetricPlotter()
        self.metric_registry = MetricRegistry()
        
        # Create subdirectories for different analysis types
        self.vehicle_dir = self.output_dir / "vehicle_analysis"
        self.passenger_dir = self.output_dir / "passenger_analysis"
        self.service_dir = self.output_dir / "service_analysis"
        self.system_dir = self.output_dir / "system_analysis"
        
        for directory in [self.vehicle_dir, self.passenger_dir, self.service_dir, self.system_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def save_figure(self, fig, filename, subdir=None):
        """Save a plotly figure to HTML.
        
        Args:
            fig: The plotly figure to save
            filename: Name of the file without extension
            subdir: Optional subdirectory within output_dir
        """
        save_dir = self.output_dir
        if subdir:
            save_dir = self.output_dir / subdir
            save_dir.mkdir(parents=True, exist_ok=True)
            
        filepath = save_dir / f"{filename}.html"
        fig.write_html(str(filepath))
        logger.info(f"Saved figure to {filepath}")
    
    def add_derived_metrics(self):
        """Add derived metrics to the DataFrame."""
        logger.info("Adding derived metrics to the dataset")
        # Create a copy to avoid modifying the original
        df = self.metrics_df.copy()
        
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
        
        self.metrics_df = df
        return df
    
    def create_visualizations_for_metric(self, df, metric_name, subdir, group_by=None):
        """Create visualizations for a metric based on its definition."""
        if df.empty:
            return
            
        # Get metric definition from registry
        definition = self.metric_registry.get(metric_name)
        if not definition:
            logger.warning(f"No definition found for metric {metric_name}")
            return
            
        metric_data = df[df['metric_name'] == metric_name]
        if metric_data.empty:
            return
            
        # Time series plot if enabled
        if definition.visualizations.get('time_series', True):
            # Convert timestamp to datetime if it's not already
            if 'timestamp' in metric_data.columns:
                try:
                    metric_data['timestamp'] = pd.to_datetime(metric_data['timestamp'])
                except:
                    logger.warning(f"Failed to convert timestamp to datetime for {metric_name}")
            
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
        if definition.visualizations.get('distribution', True) and group_by and group_by in metric_data.columns:
            fig = self.plotter.create_box_plot(
                metric_data,
                f'{definition.description} by {group_by}',
                group_by,
                f'Value ({definition.unit})',
                group_by=group_by
            )
            self.save_figure(fig, f"{metric_name.replace('.', '_')}_box", subdir)
    
    def analyze_vehicle_performance(self):
        """Analyze vehicle performance metrics."""
        logger.info("Analyzing vehicle performance")
        
        # Filter vehicle-related metrics
        vehicle_metrics = self.metrics_df[self.metrics_df['metric_name'].str.startswith('vehicle.')]
        
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
            self.create_visualizations_for_metric(
                vehicle_metrics,
                metric_name,
                "vehicle_analysis",
                group_by='vehicle_id'
            )
            
        # Handle derived metrics separately
        self.analyze_derived_vehicle_metrics(vehicle_metrics)
    
    def analyze_derived_vehicle_metrics(self, vehicle_metrics):
        """Create specialized visualizations for derived vehicle metrics."""
        logger.info("Analyzing derived vehicle metrics")
        
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
        try:
            distance_metrics = vehicle_metrics[
                vehicle_metrics['metric_name'].isin(['vehicle.occupied_distance', 'vehicle.empty_distance'])
            ]
            
            if not distance_metrics.empty:
                # Pivot table to get occupied and empty distance by vehicle
                pivot_distance = distance_metrics.pivot_table(
                    index='vehicle_id',
                    columns='metric_name',
                    values='value',
                    aggfunc='sum',
                    fill_value=0
                ).reset_index()
                
                # Check if any distance metrics exist in the dataframe
                required_columns = ['vehicle.occupied_distance', 'vehicle.empty_distance']
                available_columns = [col for col in required_columns if col in pivot_distance.columns]
                
                if len(available_columns) > 0:
                    fig = px.bar(pivot_distance,
                                x='vehicle_id',
                                y=available_columns,
                                title='Vehicle Distance Breakdown',
                                barmode='stack',
                                labels={
                                    'value': 'Distance (meters)',
                                    'vehicle_id': 'Vehicle ID',
                                    'variable': 'Distance Type'
                                })
                    self.save_figure(fig, "vehicle_distance_breakdown", "vehicle_analysis")
                else:
                    logger.warning("No distance metrics available for visualization")
        except Exception as e:
            logger.warning(f"Could not create vehicle distance breakdown visualization: {str(e)}")
    
    def analyze_passenger_experience(self):
        """Analyze passenger experience metrics."""
        logger.info("Analyzing passenger experience")
        
        passenger_metrics = self.metrics_df[self.metrics_df['metric_name'].str.startswith('passenger.')]
        
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
            self.create_visualizations_for_metric(
                passenger_metrics,
                metric_name,
                "passenger_analysis",
                group_by=None
            )
            
        # Handle derived metrics separately
        self.analyze_derived_passenger_metrics(passenger_metrics)
    
    def analyze_derived_passenger_metrics(self, passenger_metrics):
        """Create specialized visualizations for derived passenger metrics."""
        logger.info("Analyzing derived passenger metrics")
        
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
                                         labels={'value': 'Journey Time (seconds)'},
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
                    # Ensure passenger_id exists
                    if 'passenger_id' not in components_data.columns:
                        logger.warning("No passenger_id column found in journey components data")
                        return
                    
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
                                       'value': 'Time (seconds)',
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
    
    def analyze_service_efficiency(self):
        """Analyze service efficiency metrics."""
        logger.info("Analyzing service efficiency")

        # Analyze request acceptance/rejection ratio
        self.analyze_request_serviced_ratio()
        
        # Analyze request rejections
        self.analyze_request_rejections()
        
        service_metrics = self.metrics_df[self.metrics_df['metric_name'].str.startswith('service.')]
        
        if service_metrics.empty:
            logger.warning("No service metrics available")
            return
            
        # Create visualizations for base metrics only
        base_metric_names = [
            'service.violations',
            'service.capacity_utilization'
        ]
        
        for metric_name in base_metric_names:
            self.create_visualizations_for_metric(
                service_metrics,
                metric_name,
                "service_analysis"
            )
            
        # Handle derived metrics separately
        self.analyze_derived_service_metrics(service_metrics)
    
    def analyze_derived_service_metrics(self, service_metrics):
        """Create specialized visualizations for derived service metrics."""
        logger.info("Analyzing derived service metrics")
        
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
            try:
                # Violations by type if violation_type column exists
                if 'violation_type' in violations.columns:
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
            except Exception as e:
                logger.error(f"Error creating service violations visualizations: {str(e)}")
    
    def analyze_request_rejections(self):
        """Create detailed analysis of request rejections."""
        try:
            logger.info("Analyzing request rejections")
            # Filter rejection metrics
            rejections = self.metrics_df[
                self.metrics_df['metric_name'] == 'request.rejected'
            ]
            
            if rejections.empty:
                logger.warning("No rejection data available for analysis")
                return
            
            # Extract rejection reasons and details
            if 'rejection_reason' not in rejections.columns:
                # Try to extract from details column if it exists
                if 'details' in rejections.columns:
                    try:
                        rejections['rejection_reason'] = rejections['details'].apply(
                            lambda x: x.get('reason', 'unknown') if isinstance(x, dict) else 'unknown'
                        )
                    except:
                        # Create a placeholder rejection reason
                        rejections['rejection_reason'] = 'unknown'
                else:
                    # Create a placeholder rejection reason
                    rejections['rejection_reason'] = 'unknown'
            
            # Ensure rejection_reason is a string
            rejections['rejection_reason'] = rejections['rejection_reason'].astype(str)
            
            # Group rejection reasons by category
            timing_constraints = [
                'time_window_constraint',
                'vehicle_access_time_constraint',
                'passenger_wait_time_constraint',
                'ride_time_constraint'
            ]
            vehicle_constraints = [
                'capacity_constraint',
                'detour_constraint'
            ]
            availability_issues = [
                'no_vehicles_available',
                'no_compatible_vehicles'
            ]
            validation_issues = [
                'outside_service_hours',
                'outside_service_area',
                'excessive_walking_distance',
                'invalid_booking_time',
                'insufficient_notice'
            ]
            
            # 1. Overall Rejection Analysis with Categories
            rejection_counts = rejections['rejection_reason'].value_counts().reset_index()
            rejection_counts.columns = ['reason', 'count']
            
            # Create category mapping
            reason_to_category = {}
            for reason in rejection_counts['reason']:
                if reason in timing_constraints:
                    reason_to_category[reason] = 'Timing Constraints'
                elif reason in vehicle_constraints:
                    reason_to_category[reason] = 'Vehicle Constraints'
                elif reason in availability_issues:
                    reason_to_category[reason] = 'Vehicle Availability'
                elif reason in validation_issues:
                    reason_to_category[reason] = 'Validation Issues'
                else:
                    reason_to_category[reason] = 'Other'
            
            # Add category to the DataFrame
            rejection_counts['category'] = rejection_counts['reason'].map(reason_to_category)
            
            # Create bar chart of rejection reasons
            fig = px.bar(
                rejection_counts,
                x='reason',
                y='count',
                color='category',
                title='Rejection Reasons by Category',
                labels={
                    'reason': 'Rejection Reason',
                    'count': 'Count',
                    'category': 'Category'
                }
            )
            self.save_figure(fig, "rejection_reasons", "service_analysis")
            
            # Create pie chart of rejection categories
            category_counts = rejection_counts.groupby('category')['count'].sum().reset_index()
            fig = px.pie(
                category_counts,
                values='count',
                names='category',
                title='Rejection Categories Distribution'
            )
            self.save_figure(fig, "rejection_categories", "service_analysis")
            
            # Try to create sunburst chart for hierarchical view
            try:
                fig = px.sunburst(
                    rejection_counts,
                    path=['category', 'reason'],
                    values='count',
                    title='Hierarchical View of Rejection Reasons'
                )
                self.save_figure(fig, "rejection_reasons_hierarchy", "service_analysis")
            except Exception as e:
                logger.warning(f"Could not create sunburst chart: {str(e)}")
            
            # 2. Detailed Constraint Violation Analysis
            # This part depends on the structure of the 'details' field in rejections
            # We'll try to extract violation details if they exist
            if 'details' in rejections.columns:
                try:
                    constraint_details = []
                    for _, row in rejections.iterrows():
                        if isinstance(row.get('details'), dict):
                            violations = row['details'].get('violations', {})
                            if isinstance(violations, dict):
                                for violation_type, violation_list in violations.items():
                                    if isinstance(violation_list, list):
                                        for violation in violation_list:
                                            violation_data = {
                                                'timestamp': row['timestamp'],
                                                'violation_type': violation_type,
                                                'vehicle_id': row['details'].get('vehicle_id', 'unknown')
                                            }
                                            if isinstance(violation, dict):
                                                violation_data.update(violation)
                                            constraint_details.append(violation_data)
                    
                    if constraint_details:
                        violations_df = pd.DataFrame(constraint_details)
                        
                        # Convert timestamp to datetime if it's a string
                        if 'timestamp' in violations_df.columns and violations_df['timestamp'].dtype == 'object':
                            try:
                                violations_df['timestamp'] = pd.to_datetime(violations_df['timestamp'])
                            except:
                                pass
                        
                        # Violation magnitude analysis
                        for violation_type in violations_df['violation_type'].unique():
                            type_data = violations_df[violations_df['violation_type'] == violation_type]
                            
                            # Check for common value columns that might exist
                            value_cols = ['wait_time', 'ride_time', 'detour_time', 'access_time']
                            value_col = next((col for col in value_cols if col in type_data.columns), None)
                            
                            if value_col and 'max_allowed' in type_data.columns:
                                try:
                                    fig = px.histogram(
                                        type_data,
                                        x=value_col,
                                        title=f'Distribution of {violation_type.replace("_violations", "")} Values',
                                        labels={value_col: 'Time (minutes)'},
                                        marginal='box'
                                    )
                                    # Add vertical line for max allowed value
                                    fig.add_vline(x=type_data['max_allowed'].iloc[0], 
                                                line_dash="dash", 
                                                line_color="red",
                                                annotation_text="Max Allowed")
                                    
                                    self.save_figure(fig, f"{violation_type}_distribution", "service_analysis")
                                except Exception as e:
                                    logger.warning(f"Could not create histogram for {violation_type}: {str(e)}")
                except Exception as e:
                    logger.warning(f"Could not analyze constraint violations: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error analyzing request rejections: {str(e)}\n{traceback.format_exc()}")

    def analyze_request_density(self):
        """Create visualizations showing request density over time and compare with acceptance/rejection patterns."""
        logger.info("Analyzing request density")
        
        # Filter request-related metrics
        request_metrics = self.metrics_df[
            self.metrics_df['metric_name'].isin(['request.received', 'request.assigned', 'request.rejected'])
        ]
        
        if request_metrics.empty:
            logger.warning("No request metrics available for density analysis")
            return
        
        try:
            # Ensure timestamp is datetime
            request_metrics['timestamp'] = pd.to_datetime(request_metrics['timestamp'])
            
            # Group by timestamp and metric_name, then count occurrences
            request_counts = request_metrics.groupby(['timestamp', 'metric_name']).size().reset_index(name='count')
            
            # Pivot to get columns for each metric type
            pivot_counts = request_counts.pivot_table(
                index='timestamp',
                columns='metric_name',
                values='count',
                fill_value=0
            ).reset_index()
            
            # Ensure all required columns exist
            for col in ['request.received', 'request.assigned', 'request.rejected']:
                if col not in pivot_counts.columns:
                    pivot_counts[col] = 0
            
            # 1. Request Density Plot: Resample to appropriate time bins (e.g., 5-min intervals)
            pivot_counts.set_index('timestamp', inplace=True)
            
            # Determine appropriate bin size based on data timespan
            timespan = (pivot_counts.index.max() - pivot_counts.index.min()).total_seconds() / 60  # in minutes
            
            if timespan > 120:  # If more than 2 hours of data
                bin_size = '15T'  # 15-minute intervals
                bin_label = '15-minute'
            elif timespan > 60:  # If more than 1 hour of data
                bin_size = '10T'  # 10-minute intervals
                bin_label = '10-minute'
            else:
                bin_size = '5T'   # 5-minute intervals
                bin_label = '5-minute'
            
            # Resample with appropriate bin size
            resampled = pivot_counts.resample(bin_size).sum()
            resampled = resampled.reset_index()
            
            # 2. Create combination plot: Request Density vs Acceptance Rate
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add request density (bar chart)
            fig.add_trace(
                go.Bar(
                    x=resampled['timestamp'],
                    y=resampled['request.received'],
                    name='Request Volume',
                    marker_color='lightblue',
                    opacity=0.7
                ),
                secondary_y=False
            )
            
            # Calculate acceptance rate for each bin
            resampled['acceptance_rate'] = resampled['request.assigned'] / resampled['request.received'].replace(0, 1)
            resampled['rejection_rate'] = resampled['request.rejected'] / resampled['request.received'].replace(0, 1)
            
            # Add acceptance rate (line)
            fig.add_trace(
                go.Scatter(
                    x=resampled['timestamp'],
                    y=resampled['acceptance_rate'],
                    name='Acceptance Rate',
                    line=dict(color='green', width=2)
                ),
                secondary_y=True
            )
            
            # Add rejection rate (line)
            fig.add_trace(
                go.Scatter(
                    x=resampled['timestamp'],
                    y=resampled['rejection_rate'],
                    name='Rejection Rate',
                    line=dict(color='red', width=2)
                ),
                secondary_y=True
            )
            
            # Update layout
            fig.update_layout(
                title_text=f'Request Density vs. Acceptance/Rejection Rate ({bin_label} intervals)',
                hovermode="x unified"
            )
            
            # Set y-axes titles
            fig.update_yaxes(title_text="Number of Requests", secondary_y=False)
            fig.update_yaxes(title_text="Acceptance/Rejection Rate", secondary_y=True)
            
            # Set x-axis title
            fig.update_xaxes(title_text="Time")
            
            self.save_figure(fig, "request_density_vs_acceptance_rate", "service_analysis")
            
            # 3. Create heatmap showing request density by hour of day and day of week (if data spans multiple days)
            try:
                # Add hour and day columns
                pivot_df = pivot_counts.reset_index()
                pivot_df['hour'] = pivot_df['timestamp'].dt.hour
                pivot_df['day_of_week'] = pivot_df['timestamp'].dt.day_name()
                
                # Check if we have data spanning multiple days
                unique_days = pivot_df['day_of_week'].nunique()
                
                if unique_days > 1:
                    # Create hour of day x day of week heatmap
                    hourly_requests = pivot_df.groupby(['day_of_week', 'hour'])['request.received'].sum().reset_index()
                    
                    # Pivot for heatmap format
                    heatmap_data = hourly_requests.pivot(
                        index='day_of_week', 
                        columns='hour', 
                        values='request.received'
                    )
                    
                    # Order days correctly
                    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    heatmap_data = heatmap_data.reindex(day_order)
                    
                    # Create heatmap
                    fig = px.imshow(
                        heatmap_data,
                        labels=dict(x="Hour of Day", y="Day of Week", color="Request Count"),
                        x=heatmap_data.columns,
                        y=heatmap_data.index,
                        title="Request Density by Hour and Day",
                        color_continuous_scale="blues"
                    )
                    
                    self.save_figure(fig, "request_density_heatmap", "service_analysis")
                else:
                    # If only one day, create hourly distribution
                    hourly_requests = pivot_df.groupby('hour')['request.received'].sum().reset_index()
                    
                    fig = px.bar(
                        hourly_requests,
                        x='hour',
                        y='request.received',
                        title='Request Distribution by Hour of Day',
                        labels={
                            'hour': 'Hour of Day',
                            'request.received': 'Number of Requests'
                        }
                    )
                    
                    # Set x-axis to show all hours
                    fig.update_xaxes(tickvals=list(range(24)))
                    
                    self.save_figure(fig, "hourly_request_distribution", "service_analysis")
            
            except Exception as e:
                logger.warning(f"Could not create time-based request density visualizations: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error analyzing request density: {str(e)}\n{traceback.format_exc()}")

    def analyze_request_spatial_patterns(self):
        """Create visualizations for spatial patterns of trip requests."""
        logger.info("Analyzing request spatial patterns")
        
        # Filter request data
        request_data = self.metrics_df[
            self.metrics_df['metric_name'] == 'request.received'
        ]
        
        if request_data.empty:
            logger.warning("No request data available for spatial analysis")
            return
        
        try:
            # Extract location data from the details column
            requests_with_locations = []
            
            for _, row in request_data.iterrows():
                try:
                    # Handle both string and dict request formats
                    if isinstance(row['request'], str):
                        request_dict = json.loads(row['request'])
                    elif isinstance(row['request'], dict):
                        request_dict = row['request']
                    else:
                        continue
                        
                    # Extract request details
                    req_id = request_dict.get('id')
                    status = request_dict.get('status')
                    
                    # Extract origin and destination if available
                    origin = request_dict.get('origin')
                    destination = request_dict.get('destination')
                    
                    if origin and destination and isinstance(origin, dict) and isinstance(destination, dict):
                        if all(k in origin for k in ['lat', 'lon']) and all(k in destination for k in ['lat', 'lon']):
                            requests_with_locations.append({
                                'request_id': req_id,
                                'status': status,
                                'origin_lat': float(origin['lat']),
                                'origin_lon': float(origin['lon']),
                                'dest_lat': float(destination['lat']),
                                'dest_lon': float(destination['lon']),
                                'timestamp': row.get('timestamp')
                            })
                except Exception as e:
                    logger.warning(f"Error processing request: {str(e)}")
                    continue
            
            if not requests_with_locations:
                logger.warning("No request location data could be extracted")
                return
            
            # Create DataFrame with location data
            locations_df = pd.DataFrame(requests_with_locations)
            
            # 1. Create scatter map of origins and destinations
            fig = go.Figure()
            
            # Add origins
            fig.add_trace(go.Scattermapbox(
                lat=locations_df['origin_lat'],
                lon=locations_df['origin_lon'],
                mode='markers',
                marker=dict(
                    size=8,
                    color='blue',
                    opacity=0.7
                ),
                name='Origins',
                hoverinfo='text',
                text=locations_df['request_id']
            ))
            
            # Add destinations
            fig.add_trace(go.Scattermapbox(
                lat=locations_df['dest_lat'],
                lon=locations_df['dest_lon'],
                mode='markers',
                marker=dict(
                    size=8,
                    color='red',
                    opacity=0.7
                ),
                name='Destinations',
                hoverinfo='text',
                text=locations_df['request_id']
            ))
            
            # Calculate center of map
            center_lat = locations_df[['origin_lat', 'dest_lat']].values.flatten().mean()
            center_lon = locations_df[['origin_lon', 'dest_lon']].values.flatten().mean()
            
            # Update layout
            fig.update_layout(
                title='Request Origins and Destinations',
                mapbox=dict(
                    style='open-street-map',
                    center=dict(lat=center_lat, lon=center_lon),
                    zoom=12
                ),
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='right',
                    x=1
                ),
                margin=dict(l=0, r=0, t=30, b=0)
            )
            
            self.save_figure(fig, "request_origins_destinations_map", "spatial_analysis")
            
            # 2. Create a heatmap of request origins
            fig = go.Figure()
            
            fig.add_trace(go.Densitymapbox(
                lat=locations_df['origin_lat'],
                lon=locations_df['origin_lon'],
                z=np.ones(len(locations_df)),  # Equal weight for each point
                radius=15,
                colorscale='Blues',
                name='Origin Density'
            ))
            
            fig.update_layout(
                title='Request Origin Density',
                mapbox=dict(
                    style='open-street-map',
                    center=dict(lat=center_lat, lon=center_lon),
                    zoom=12
                ),
                margin=dict(l=0, r=0, t=30, b=0)
            )
            
            self.save_figure(fig, "request_origins_heatmap", "spatial_analysis")
            
            # 3. Create a heatmap of request destinations
            fig = go.Figure()
            
            fig.add_trace(go.Densitymapbox(
                lat=locations_df['dest_lat'],
                lon=locations_df['dest_lon'],
                z=np.ones(len(locations_df)),  # Equal weight for each point
                radius=15,
                colorscale='Reds',
                name='Destination Density'
            ))
            
            fig.update_layout(
                title='Request Destination Density',
                mapbox=dict(
                    style='open-street-map',
                    center=dict(lat=center_lat, lon=center_lon),
                    zoom=12
                ),
                margin=dict(l=0, r=0, t=30, b=0)
            )
            
            self.save_figure(fig, "request_destinations_heatmap", "spatial_analysis")
            
            # 4. Create a visualization of request flows (lines connecting origins to destinations)
            fig = go.Figure()
            
            # Sample the data if there are too many points (to avoid visual clutter)
            max_lines = 100
            sample_df = locations_df if len(locations_df) <= max_lines else locations_df.sample(max_lines)
            
            # Create a scatter plot for origins
            fig.add_trace(go.Scattermapbox(
                lat=sample_df['origin_lat'],
                lon=sample_df['origin_lon'],
                mode='markers',
                marker=dict(
                    size=6,
                    color='blue',
                    opacity=0.7
                ),
                name='Origins',
                hoverinfo='text',
                text=sample_df['request_id']
            ))
            
            # Create a scatter plot for destinations
            fig.add_trace(go.Scattermapbox(
                lat=sample_df['dest_lat'],
                lon=sample_df['dest_lon'],
                mode='markers',
                marker=dict(
                    size=6,
                    color='red',
                    opacity=0.7
                ),
                name='Destinations',
                hoverinfo='text',
                text=sample_df['request_id']
            ))
            
            # Add lines connecting origins to destinations
            for idx, row in sample_df.iterrows():
                fig.add_trace(go.Scattermapbox(
                    lat=[row['origin_lat'], row['dest_lat']],
                    lon=[row['origin_lon'], row['dest_lon']],
                    mode='lines',
                    line=dict(width=1, color='gray'),
                    opacity=0.4,
                    showlegend=False,
                    hoverinfo='none'
                ))
            
            # Update layout
            fig.update_layout(
                title='Request Flow Patterns (Origin → Destination)',
                mapbox=dict(
                    style='open-street-map',
                    center=dict(lat=center_lat, lon=center_lon),
                    zoom=12
                ),
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='right',
                    x=1
                ),
                margin=dict(l=0, r=0, t=30, b=0)
            )
            
            self.save_figure(fig, "request_flows", "spatial_analysis")
            
            # 5. Create a hexbin map of trip distances
            # Calculate trip distances (Haversine formula)
            def haversine(lat1, lon1, lat2, lon2):
                R = 6371  # Earth radius in kilometers
                
                # Convert to radians
                lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
                
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                
                a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
                c = 2 * np.arcsin(np.sqrt(a))
                
                return R * c
            
            locations_df['trip_distance_km'] = locations_df.apply(
                lambda row: haversine(
                    row['origin_lat'], row['origin_lon'], 
                    row['dest_lat'], row['dest_lon']
                ), 
                axis=1
            )
            
            # Create histogram of trip distances
            fig = px.histogram(
                locations_df,
                x='trip_distance_km',
                title='Distribution of Trip Distances',
                labels={'trip_distance_km': 'Trip Distance (km)'},
                nbins=30
            )
            
            self.save_figure(fig, "trip_distance_distribution", "spatial_analysis")
            
            # Make sure spatial_analysis directory exists
            spatial_dir = self.output_dir / "spatial_analysis"
            spatial_dir.mkdir(parents=True, exist_ok=True)
            
        except Exception as e:
            logger.error(f"Error analyzing request spatial patterns: {str(e)}\n{traceback.format_exc()}")
    
    def analyze_request_serviced_ratio(self):
        """Create visualizations showing the request accepted/rejected ratio."""
        logger.info("Analyzing request acceptance ratio")
        
        # Filter request-related metrics
        request_metrics = self.metrics_df[
            self.metrics_df['metric_name'].isin(['request.received', 'request.assigned', 'request.rejected'])
        ]
        
        if request_metrics.empty:
            logger.warning("No request metrics available for acceptance ratio analysis")
            return
        
        try:
            # Ensure timestamp is datetime
            try:
                request_metrics['timestamp'] = pd.to_datetime(request_metrics['timestamp'])
            except:
                pass
            
            # Method 1: Resample by time windows (5-minute intervals)
            # First create pivot table with counts by metric type
            request_counts = request_metrics.groupby(['timestamp', 'metric_name']).size().reset_index(name='count')
            
            # Pivot to get columns for each metric type
            pivot_counts = request_counts.pivot_table(
                index='timestamp',
                columns='metric_name',
                values='count',
                fill_value=0
            ).reset_index()
            
            # Ensure all required columns exist
            for col in ['request.received', 'request.assigned', 'request.rejected']:
                if col not in pivot_counts.columns:
                    pivot_counts[col] = 0
            
            # Resample to 5-minute intervals (if we have enough data)
            if len(pivot_counts) > 10:
                pivot_counts.set_index('timestamp', inplace=True)
                resampled = pivot_counts.resample('5T').sum()
                resampled['acceptance_ratio'] = resampled['request.assigned'] / resampled['request.received'].replace(0, 1)
                resampled['rejection_ratio'] = resampled['request.rejected'] / resampled['request.received'].replace(0, 1)
                resampled = resampled.reset_index()
                
                # Time series of resampled ratios
                fig = px.line(
                    resampled,
                    x='timestamp',
                    y=['acceptance_ratio', 'rejection_ratio'],
                    title='Request Acceptance and Rejection Ratios (5-minute intervals)',
                    labels={
                        'timestamp': 'Time',
                        'value': 'Ratio',
                        'variable': 'Metric'
                    }
                )
                self.save_figure(fig, "request_acceptance_rejection_ratio", "service_analysis")
            else:
                # If not enough data for resampling, calculate ratios directly
                pivot_counts['acceptance_ratio'] = pivot_counts['request.assigned'] / pivot_counts['request.received'].replace(0, 1)
                pivot_counts['rejection_ratio'] = pivot_counts['request.rejected'] / pivot_counts['request.received'].replace(0, 1)
                
                fig = px.line(
                    pivot_counts,
                    x='timestamp',
                    y=['acceptance_ratio', 'rejection_ratio'],
                    title='Request Acceptance and Rejection Ratios Over Time',
                    labels={
                        'timestamp': 'Time',
                        'value': 'Ratio',
                        'variable': 'Metric'
                    }
                )
                self.save_figure(fig, "request_acceptance_rejection_ratio", "service_analysis")
            
            # Method 2: Pie chart for overall acceptance vs rejection
            # Calculate total counts for the entire simulation
            total_assigned = pivot_counts['request.assigned'].sum()
            total_rejected = pivot_counts['request.rejected'].sum()
            total_received = pivot_counts['request.received'].sum()
            
            # Create pie chart data
            if total_received > 0:
                acceptance_pct = (total_assigned / total_received) * 100
                rejection_pct = (total_rejected / total_received) * 100
            else:
                acceptance_pct = 0
                rejection_pct = 0
                
            pie_data = pd.DataFrame({
                'Status': ['Accepted', 'Rejected'],
                'Count': [total_assigned, total_rejected],
                'Percentage': [acceptance_pct, rejection_pct]
            })
            
            # Create pie chart with percentages in labels
            fig = px.pie(
                pie_data,
                values='Count',
                names='Status',
                title=f'Overall Request Acceptance vs Rejection<br><sup>Total Requests: {total_received}</sup>',
                hover_data=['Percentage'],
                labels={'Percentage': 'Percentage (%)'}
            )
            
            # Format the percentage in the hover text
            fig.update_traces(
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{customdata:.1f}%'
            )
            
            # Add percentage to labels in the pie slices
            fig.update_traces(
                texttemplate='%{label}<br>%{percent:.1%}',
                textposition='inside'
            )
            
            self.save_figure(fig, "overall_acceptance_rejection_pie", "service_analysis")
            
            # Method 3: Stacked area chart showing accepted vs rejected requests
            fig = px.area(
                pivot_counts,
                x='timestamp',
                y=['request.assigned', 'request.rejected'],
                title='Accepted vs Rejected Requests Over Time',
                labels={
                    'timestamp': 'Time',
                    'value': 'Count',
                    'variable': 'Request Status'
                }
            )
            self.save_figure(fig, "accepted_rejected_requests", "service_analysis")
            
        except Exception as e:
            logger.error(f"Error analyzing request acceptance ratio: {str(e)}\n{traceback.format_exc()}")
    
    def analyze_system_performance(self):
        """Analyze system-wide performance metrics."""
        logger.info("Analyzing system performance")
        
        system_metrics = self.metrics_df[self.metrics_df['metric_name'].str.startswith('system.')]
        
        if system_metrics.empty:
            logger.warning("No system metrics available")
            return
            
        # Create visualizations for base metrics only
        base_metric_names = [
            'system.cpu_usage',
            'system.memory_usage'
        ]
        
        for metric_name in base_metric_names:
            self.create_visualizations_for_metric(
                system_metrics,
                metric_name,
                "system_analysis"
            )
            
        # Handle derived metrics separately
        self.analyze_derived_system_metrics(system_metrics)
    
    def analyze_derived_system_metrics(self, system_metrics):
        """Create specialized visualizations for derived system metrics."""
        logger.info("Analyzing derived system metrics")
        
        # 1. System Load Analysis
        load_metrics = system_metrics[
            system_metrics['metric_name'].isin([
                'system.active_requests',
                'system.active_vehicles'
            ])
        ]
        
        if not load_metrics.empty:
            try:
                # Ensure timestamp is datetime
                if 'timestamp' in load_metrics.columns:
                    try:
                        load_metrics['timestamp'] = pd.to_datetime(load_metrics['timestamp'])
                    except:
                        pass
                
                # System load over time
                pivot_load = load_metrics.pivot_table(
                    index='timestamp',
                    columns='metric_name',
                    values='value'
                ).reset_index()
                
                # Check if we have both metrics
                required_cols = ['system.active_requests', 'system.active_vehicles']
                if all(col in pivot_load.columns for col in required_cols):
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
                else:
                    # Create individual plots for each metric if both aren't available
                    for metric in required_cols:
                        if metric in pivot_load.columns:
                            fig = px.line(pivot_load,
                                        x='timestamp',
                                        y=metric,
                                        title=f'{metric} Over Time',
                                        labels={
                                            metric: 'Count',
                                            'timestamp': 'Time'
                                        })
                            metric_name = metric.split('.')[-1]
                            self.save_figure(fig, f"{metric_name}_timeline", "system_analysis")
            except Exception as e:
                logger.error(f"Error creating system load visualizations: {str(e)}")
    
    def run_all_analysis(self):
        """Run all analysis functions."""
        logger.info("Starting comprehensive metrics analysis")
        
        if self.metrics_df.empty:
            logger.warning("No metrics data available for analysis")
            return
        
        # Add derived metrics
        self.add_derived_metrics()
        
        # Generate all analyses
        self.analyze_vehicle_performance()
        self.analyze_passenger_experience()
        self.analyze_service_efficiency()
        self.analyze_system_performance()

        self.analyze_request_density()
        self.analyze_request_spatial_patterns()
        
        logger.info("Completed comprehensive metrics analysis")

def load_data(parquet_path):
    """Load metrics data from Parquet file and save as CSV."""
    try:
        logger.info(f"Loading data from {parquet_path}")
        df = pd.read_parquet(parquet_path)
        
        # Check if the required columns exist
        required_columns = ['metric_name', 'value']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return pd.DataFrame()
        
        # Convert value column to numeric if possible
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        # Handle JSON columns if they exist
        for col in ['details', 'metadata']:
            if col in df.columns and df[col].dtype == 'object':
                try:
                    import json
                    # Try to parse JSON strings
                    df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) and x.strip() else x)
                except:
                    logger.warning(f"Could not parse JSON in {col} column")
        
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def main():
    """Main function to run the metrics analysis."""
    parser = argparse.ArgumentParser(description='Analyze metrics data from a CSV file')
    parser.add_argument('parquet_file', help='Path to the Parquet file containing metrics data')
    parser.add_argument('--output-dir', '-o', default='./metrics_analysis', 
                        help='Directory to save analysis output (default: ./metrics_analysis)')
    
    args = parser.parse_args()
    
    # Load data
    metrics_df = load_data(args.parquet_file)
    
    if metrics_df.empty:
        logger.error("No valid data loaded. Exiting.")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create and run analyzer
    analyzer = StandaloneMetricsAnalyzer(metrics_df, output_dir)
    analyzer.run_all_analysis()
    
    logger.info(f"Analysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()