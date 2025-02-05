'''Under Construction'''

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional
from drt_sim.core.logging_config import setup_logger

@dataclass
class VisualizationConfig:
    """Configuration for visualization settings"""
    color_scheme: Dict[str, str] = field(default_factory=lambda: {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e',
        'success': '#2ca02c',
        'warning': '#d62728',
        'neutral': '#7f7f7f'
    })
    map_style: str = 'cartodbpositron'
    default_figure_size: Tuple[int, int] = (1200, 800)
    animation_frame_duration: int = 500
    export_format: str = 'html'
    include_plotly_mode_bar: bool = True

class SimulationVisualizer:
    """Enhanced visualization system for simulation results"""
    
    def __init__(self, 
                 output_dir: Optional[Path] = None,
                 config: Optional[VisualizationConfig] = None):
        self.output_dir = output_dir or Path("visualization_output")
        self.config = config or VisualizationConfig()
        self.logger = setup_logger(self.__class__.__name__)
        
        # Create output directories
        self.static_dir = self.output_dir / "static"
        self.interactive_dir = self.output_dir / "interactive"
        self.animation_dir = self.output_dir / "animations"
        self.map_dir = self.output_dir / "maps"
        
        for dir_path in [self.static_dir, self.interactive_dir, 
                        self.animation_dir, self.map_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def create_performance_dashboard(self, metrics_df: pd.DataFrame) -> None:
        """Create comprehensive performance dashboard"""
        try:
            # Create main dashboard layout
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    'Service Level Over Time',
                    'Vehicle Utilization',
                    'Wait Time Distribution',
                    'Request Status',
                    'Distance Composition',
                    'Fleet Activity'
                ),
                specs=[
                    [{"type": "scatter"}, {"type": "scatter"}],
                    [{"type": "violin"}, {"type": "pie"}],
                    [{"type": "bar"}, {"type": "scatter"}]
                ],
                vertical_spacing=0.12,
                horizontal_spacing=0.1
            )

            # Service Level Plot
            fig.add_trace(
                go.Scatter(
                    x=metrics_df['timestamp'],
                    y=metrics_df['service_level'],
                    name='Service Level',
                    line=dict(color=self.config.color_scheme['primary']),
                    hovertemplate='%{y:.1f}%<br>%{x}'
                ),
                row=1, col=1
            )

            # Vehicle Utilization
            fig.add_trace(
                go.Scatter(
                    x=metrics_df['timestamp'],
                    y=metrics_df['vehicle_utilization'],
                    name='Utilization',
                    line=dict(color=self.config.color_scheme['secondary']),
                    hovertemplate='%{y:.1f}%<br>%{x}'
                ),
                row=1, col=2
            )

            # Wait Time Distribution
            fig.add_trace(
                go.Violin(
                    y=metrics_df['average_wait_time'],
                    name='Wait Time',
                    box_visible=True,
                    meanline_visible=True,
                    fillcolor=self.config.color_scheme['primary'],
                    line_color=self.config.color_scheme['neutral']
                ),
                row=2, col=1
            )

            # Request Status Pie Chart
            completed = metrics_df['completed_requests'].iloc[-1]
            rejected = metrics_df['rejected_requests'].iloc[-1]
            in_progress = (metrics_df['total_requests'].iloc[-1] - 
                         completed - rejected)
            
            fig.add_trace(
                go.Pie(
                    labels=['Completed', 'Rejected', 'In Progress'],
                    values=[completed, rejected, in_progress],
                    marker_colors=[
                        self.config.color_scheme['success'],
                        self.config.color_scheme['warning'],
                        self.config.color_scheme['neutral']
                    ]
                ),
                row=2, col=2
            )

            # Distance Composition
            distance_data = [
                metrics_df['loaded_distance'].sum(),
                metrics_df['empty_distance'].sum(),
                metrics_df['rebalancing_distance'].sum()
            ]
            
            fig.add_trace(
                go.Bar(
                    x=['Loaded', 'Empty', 'Rebalancing'],
                    y=distance_data,
                    marker_color=[
                        self.config.color_scheme['success'],
                        self.config.color_scheme['warning'],
                        self.config.color_scheme['neutral']
                    ]
                ),
                row=3, col=1
            )

            # Fleet Activity
            fig.add_trace(
                go.Scatter(
                    x=metrics_df['timestamp'],
                    y=metrics_df['active_vehicles'],
                    name='Active Vehicles',
                    fill='tozeroy',
                    fillcolor=f"rgba{tuple(list(int(self.config.color_scheme['primary'].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.3])}"
                ),
                row=3, col=2
            )

            # Update layout
            fig.update_layout(
                height=self.config.default_figure_size[1],
                width=self.config.default_figure_size[0],
                title_text="System Performance Dashboard",
                showlegend=True,
                template='plotly_white',
                hovermode='x unified'
            )

            # Save dashboard
            fig.write_html(
                self.interactive_dir / "performance_dashboard.html",
                include_plotlyjs=True,
                full_html=True
            )

        except Exception as e:
            self.logger.error(f"Failed to create performance dashboard: {str(e)}", 
                            exc_info=True)

    def create_map_visualization(self,
                               vehicle_positions: List[Dict],
                               requests: List[Dict],
                               service_area: Tuple[Tuple[float, float], 
                                                 Tuple[float, float]],
                               timestamp: datetime) -> None:
        """Create interactive map visualization"""
        try:
            # Calculate map center
            center_lat = (service_area[0][0] + service_area[1][0]) / 2
            center_lon = (service_area[0][1] + service_area[1][1]) / 2

            # Create base map
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=13,
                tiles=self.config.map_style
            )

            # Create colormap for vehicle status
            status_colors = {
                'idle': self.config.color_scheme['neutral'],
                'enroute_to_pickup': self.config.color_scheme['primary'],
                'picking_up': self.config.color_scheme['secondary'],
                'enroute_to_dropoff': self.config.color_scheme['success'],
                'dropping_off': self.config.color_scheme['warning']
            }

            # Add vehicles to map
            for vehicle in vehicle_positions:
                color = status_colors.get(vehicle['status'], 
                                        self.config.color_scheme['neutral'])
                
                # Create popup content
                popup_content = f"""
                <div style='font-family: Arial, sans-serif;'>
                    <b>Vehicle {vehicle['id']}</b><br>
                    Status: {vehicle['status']}<br>
                    Occupancy: {vehicle['occupancy']}<br>
                    {f"Speed: {vehicle['speed']:.1f} km/h" if 'speed' in vehicle else ''}
                </div>
                """
                
                folium.CircleMarker(
                    location=[vehicle['location'][0], vehicle['location'][1]],
                    radius=8,
                    color=color,
                    fill=True,
                    popup=folium.Popup(popup_content, max_width=200)
                ).add_to(m)

            # Add requests to map
            for request in requests:
                if request['status'] == 'completed':
                    # Create pickup marker
                    folium.CircleMarker(
                        location=[request['pickup'][0], request['pickup'][1]],
                        radius=6,
                        color=self.config.color_scheme['success'],
                        fill=True,
                        popup=f"Pickup: Request {request['id']}"
                    ).add_to(m)
                    
                    # Create dropoff marker
                    folium.CircleMarker(
                        location=[request['dropoff'][0], request['dropoff'][1]],
                        radius=6,
                        color=self.config.color_scheme['warning'],
                        fill=True,
                        popup=f"Dropoff: Request {request['id']}"
                    ).add_to(m)
                    
                    # Draw route line
                    folium.PolyLine(
                        locations=[request['pickup'], request['dropoff']],
                        color=self.config.color_scheme['neutral'],
                        weight=2,
                        opacity=0.8
                    ).add_to(m)

            # Add heat map of request density
            if requests:
                pickup_points = [[r['pickup'][0], r['pickup'][1]] 
                               for r in requests]
                dropoff_points = [[r['dropoff'][0], r['dropoff'][1]] 
                                for r in requests]
                
                folium.plugins.HeatMap(
                    pickup_points,
                    radius=15,
                    blur=10,
                    max_zoom=1
                ).add_to(m)

            # Save map
            output_file = (self.map_dir / 
                          f"map_{timestamp.strftime('%Y%m%d_%H%M%S')}.html")
            m.save(str(output_file))

        except Exception as e:
            self.logger.error(f"Failed to create map visualization: {str(e)}", 
                            exc_info=True)

    def create_kpi_cards(self, metrics: Dict[str, Any]) -> None:
        """Create interactive KPI cards"""
        try:
            fig = go.Figure()

            # Service Level KPI
            fig.add_trace(go.Indicator(
                mode="number+delta",
                value=metrics['service_metrics']['service_level'],
                title="Service Level",
                delta={'reference': 95, 'relative': True},
                domain={'row': 0, 'column': 0}
            ))

            # Average Wait Time KPI
            fig.add_trace(go.Indicator(
                mode="number+delta",
                value=metrics['service_metrics']['average_wait_time'],
                title="Avg Wait Time (min)",
                delta={'reference': 10, 'relative': True},
                domain={'row': 0, 'column': 1}
            ))

            # Vehicle Utilization KPI
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=metrics['operational_metrics']['vehicle_utilization'],
                title="Vehicle Utilization",
                gauge={'axis': {'range': [0, 100]}},
                domain={'row': 1, 'column': 0}
            ))

            # Request Completion Rate KPI
            completion_rate = (metrics['service_metrics']['completed_requests'] / 
                             max(metrics['service_metrics']['total_requests'], 1) * 100)
            
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=completion_rate,
                title="Request Completion Rate",
                gauge={'axis': {'range': [0, 100]}},
                domain={'row': 1, 'column': 1}
            ))

            fig.update_layout(
                grid={'rows': 2, 'columns': 2, 'pattern': "independent"},
                height=600,
                title_text="Key Performance Indicators"
            )

            # Save KPI dashboard
            fig.write_html(self.interactive_dir / "kpi_dashboard.html")

        except Exception as e:
            self.logger.error(f"Failed to create KPI cards: {str(e)}", 
                            exc_info=True)

    def create_temporal_analysis(self, metrics_df: pd.DataFrame) -> None:
        """Create temporal analysis plots"""
        try:
            # Create subplots for temporal analysis
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Hourly Patterns',
                    'Daily Patterns',
                    'Weekly Patterns',
                    'Trend Analysis'
                )
            )

            # Hourly patterns
            hourly_data = metrics_df.groupby(
                metrics_df['timestamp'].dt.hour
            )['service_level'].mean()
            
            fig.add_trace(
                go.Scatter(
                    x=hourly_data.index,
                    y=hourly_data.values,
                    mode='lines+markers',
                    name='Hourly Pattern'
                ),
                row=1, col=1
            )

            # Daily patterns
            daily_data = metrics_df.groupby(
                metrics_df['timestamp'].dt.dayofweek
            )['service_level'].mean()
            
            fig.add_trace(
                go.Bar(
                    x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                    y=daily_data.values,
                    name='Daily Pattern'
                ),
                row=1, col=2
            )

            # Weekly trend
            weekly_data = metrics_df.resample('W', on='timestamp').mean()
            
            fig.add_trace(
                go.Scatter(
                    x=weekly_data.index,
                    y=weekly_data['service_level'],
                    mode='lines+markers',
                    name='Weekly Trend'
                ),
                row=2, col=1
            )

            # Moving average trend
            fig.add_trace(
                go.Scatter(
                    x=metrics_df['timestamp'],
                    y=metrics_df['service_level'].rolling(
                        window='1D'
                    ).mean(),
                    name='Trend (24h MA)'
                ),
                row=2, col=2
            )

            fig.update_layout(
                height=800,
                title_text="Temporal Analysis",
                showlegend=True
            )

            # Save temporal analysis
            fig.write_html(self.interactive_dir / "temporal_analysis.html")

        except Exception as e:
            self.logger.error(f"Failed to create temporal analysis: {str(e)}", 
                            exc_info=True)

    def create_parameter_sensitivity_plot(self,
                                       sensitivity_results: Dict[str, Any]) -> None:
        """Create parameter sensitivity visualization"""
        try:
            # Create figure with secondary y-axis
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            for param_name, metrics in sensitivity_results.items():
                # Plot service level on primary y-axis
                fig.add_trace(
                    go.Scatter(
                        x=[v['parameter_value'] for v in metrics['service_level']],
                        y=[v['metric_value'] for v in metrics['service_level']],
                        name=f'Service Level - {param_name}',
                        mode='lines+markers'
                    )
                )

                # Plot wait time on secondary y-axis
                fig.add_trace(
                    go.Scatter(
                        x=[v['parameter_value'] for v in metrics['average_wait_time']],
                        y=[v['metric_value'] for v in metrics['average_wait_time']],
                        name=f'Wait Time - {param_name}',
                        mode='lines+markers'
                    ),
                    secondary_y=True
                )

            fig.update_layout(
                title=f"Parameter Sensitivity Analysis",
                xaxis_title="Parameter Value",
                height=600,
                width=800,
                showlegend=True
            )
            
            fig.update_yaxes(title_text="Service Level (%)", secondary_y=False)
            fig.update_yaxes(title_text="Average Wait Time (min)", secondary_y=True)

            # Save sensitivity plot
            fig.write_html(self.interactive_dir / "parameter_sensitivity.html")

        except Exception as e:
            self.logger.error(f"Failed to create parameter sensitivity plot: {str(e)}", 
                            exc_info=True)

    def create_fleet_analysis_dashboard(self, metrics_df: pd.DataFrame) -> None:
        """Create comprehensive fleet analysis dashboard"""
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Vehicle Utilization Distribution',
                    'Fleet Activity Timeline',
                    'Distance Composition',
                    'Occupancy Patterns'
                )
            )

            # Vehicle utilization distribution
            fig.add_trace(
                go.Histogram(
                    x=metrics_df['vehicle_utilization'],
                    nbinsx=20,
                    name='Utilization Distribution',
                    marker_color=self.config.color_scheme['primary']
                ),
                row=1, col=1
            )

            # Fleet activity timeline
            fig.add_trace(
                go.Scatter(
                    x=metrics_df['timestamp'],
                    y=metrics_df['active_vehicles'],
                    name='Active Vehicles',
                    fill='tozeroy',
                    line=dict(color=self.config.color_scheme['secondary'])
                ),
                row=1, col=2
            )

            # Distance composition
            distance_types = ['loaded_distance', 'empty_distance', 'rebalancing_distance']
            distance_values = [metrics_df[col].sum() for col in distance_types]
            
            fig.add_trace(
                go.Pie(
                    labels=['Loaded', 'Empty', 'Rebalancing'],
                    values=distance_values,
                    marker_colors=[
                        self.config.color_scheme['success'],
                        self.config.color_scheme['warning'],
                        self.config.color_scheme['neutral']
                    ]
                ),
                row=2, col=1
            )

            # Occupancy patterns
            fig.add_trace(
                go.Scatter(
                    x=metrics_df['timestamp'],
                    y=metrics_df['average_occupancy'],
                    name='Average Occupancy',
                    mode='lines',
                    line=dict(color=self.config.color_scheme['primary'])
                ),
                row=2, col=2
            )

            fig.update_layout(
                height=800,
                showlegend=True,
                title_text="Fleet Analysis Dashboard"
            )

            # Save fleet analysis dashboard
            fig.write_html(self.interactive_dir / "fleet_analysis_dashboard.html")

        except Exception as e:
            self.logger.error(f"Failed to create fleet analysis dashboard: {str(e)}", 
                            exc_info=True)

    def create_request_patterns_visualization(self, 
                                           metrics_df: pd.DataFrame,
                                           include_heatmap: bool = True) -> None:
        """Create visualization of request patterns"""
        try:
            if include_heatmap:
                # Create hourly by day heatmap
                hourly_data = metrics_df.pivot_table(
                    values='total_requests',
                    index=metrics_df['timestamp'].dt.hour,
                    columns=metrics_df['timestamp'].dt.dayofweek,
                    aggfunc='mean'
                )

                fig = go.Figure(data=go.Heatmap(
                    z=hourly_data.values,
                    x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                    y=list(range(24)),
                    colorscale='Viridis',
                    colorbar_title='Requests'
                ))

                fig.update_layout(
                    title='Request Patterns Heatmap',
                    xaxis_title='Day of Week',
                    yaxis_title='Hour of Day',
                    height=500
                )

                fig.write_html(self.interactive_dir / "request_heatmap.html")

            # Create request status timeline
            fig = go.Figure()

            for status in ['completed_requests', 'rejected_requests']:
                fig.add_trace(go.Scatter(
                    x=metrics_df['timestamp'],
                    y=metrics_df[status],
                    name=status.replace('_', ' ').title(),
                    stackgroup='one'
                ))

            fig.update_layout(
                title='Request Status Timeline',
                xaxis_title='Time',
                yaxis_title='Number of Requests',
                height=400
            )

            fig.write_html(self.interactive_dir / "request_timeline.html")

        except Exception as e:
            self.logger.error(f"Failed to create request patterns visualization: {str(e)}", 
                            exc_info=True)

    def create_comparative_analysis(self,
                                  baseline_df: pd.DataFrame,
                                  comparison_df: pd.DataFrame,
                                  metrics_to_compare: List[str]) -> None:
        """Create comparative analysis visualization"""
        try:
            fig = make_subplots(
                rows=len(metrics_to_compare),
                cols=1,
                subplot_titles=[m.replace('_', ' ').title() for m in metrics_to_compare]
            )

            for i, metric in enumerate(metrics_to_compare, 1):
                # Baseline trace
                fig.add_trace(
                    go.Scatter(
                        x=baseline_df['timestamp'],
                        y=baseline_df[metric],
                        name=f'Baseline - {metric}',
                        line=dict(color=self.config.color_scheme['primary'])
                    ),
                    row=i, col=1
                )

                # Comparison trace
                fig.add_trace(
                    go.Scatter(
                        x=comparison_df['timestamp'],
                        y=comparison_df[metric],
                        name=f'Comparison - {metric}',
                        line=dict(dash='dash', color=self.config.color_scheme['secondary'])
                    ),
                    row=i, col=1
                )

            fig.update_layout(
                height=300 * len(metrics_to_compare),
                title_text="Comparative Analysis",
                showlegend=True
            )

            fig.write_html(self.interactive_dir / "comparative_analysis.html")

        except Exception as e:
            self.logger.error(f"Failed to create comparative analysis: {str(e)}", 
                            exc_info=True)

    def export_all_visualizations(self, metrics_df: pd.DataFrame, 
                                analysis_results: Dict[str, Any]) -> None:
        """Export all visualizations for the experiment"""
        try:
            self.create_performance_dashboard(metrics_df)
            self.create_temporal_analysis(metrics_df)
            self.create_fleet_analysis_dashboard(metrics_df)
            self.create_request_patterns_visualization(metrics_df)
            self.create_kpi_cards(analysis_results)
            
            if 'parameter_sensitivity' in analysis_results:
                self.create_parameter_sensitivity_plot(
                    analysis_results['parameter_sensitivity']
                )

            self.logger.info("Successfully exported all visualizations")

        except Exception as e:
            self.logger.error(f"Failed to export all visualizations: {str(e)}", 
                            exc_info=True)