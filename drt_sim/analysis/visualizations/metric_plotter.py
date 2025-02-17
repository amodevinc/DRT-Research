import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging
import traceback

logger = logging.getLogger(__name__)

class MetricPlotter:
    """Generates interactive visualizations for metrics data."""
    
    def __init__(self):
        # Set default styling
        self.color_palette = px.colors.qualitative.Set3
        self.template = "plotly_white"
        
    def create_time_series(self,
                          df: pd.DataFrame,
                          title: str,
                          x_label: str,
                          y_label: str,
                          group_by: Optional[str] = None,
                          rolling_window: Optional[str] = None) -> go.Figure:
        """Create an interactive time series plot."""
        fig = go.Figure()
        
        if group_by and group_by in df.columns:
            # Group data and plot each group
            for i, (name, group) in enumerate(df.groupby(group_by)):
                if rolling_window:
                    # Apply rolling window
                    group = group.set_index('timestamp').sort_index()
                    group['rolling_mean'] = group['value'].rolling(window=rolling_window).mean()
                    y_values = group['rolling_mean']
                    name = f"{name} (Rolling Avg)"
                else:
                    y_values = group['value']
                    
                fig.add_trace(go.Scatter(
                    x=group['timestamp'],
                    y=y_values,
                    name=str(name),
                    line=dict(color=self.color_palette[i % len(self.color_palette)])
                ))
        else:
            # Plot single time series
            if rolling_window:
                df = df.set_index('timestamp').sort_index()
                df['rolling_mean'] = df['value'].rolling(window=rolling_window).mean()
                y_values = df['rolling_mean']
                name = "Rolling Average"
            else:
                y_values = df['value']
                name = "Value"
                
            fig.add_trace(go.Scatter(
                x=df['timestamp'] if 'timestamp' in df else df.index,
                y=y_values,
                name=name,
                line=dict(color=self.color_palette[0])
            ))
                
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            template=self.template,
            showlegend=True if group_by else False
        )
        
        return fig
        
    def create_distribution_plot(self,
                               df: pd.DataFrame,
                               title: str,
                               x_label: str,
                               y_label: str,
                               group_by: Optional[str] = None) -> go.Figure:
        """Create a distribution plot (histogram + kde)."""
        if group_by and group_by in df.columns:
            fig = go.Figure()
            for i, (name, group) in enumerate(df.groupby(group_by)):
                fig.add_trace(go.Histogram(
                    x=group['value'],
                    name=str(name),
                    nbinsx=30,
                    opacity=0.7
                ))
            fig.update_layout(barmode='overlay')
        else:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df['value'],
                nbinsx=30
            ))
            
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            template=self.template,
            showlegend=True if group_by else False
        )
        
        return fig
        
    def create_box_plot(self,
                       df: pd.DataFrame,
                       title: str,
                       x_label: str,
                       y_label: str,
                       group_by: Optional[str] = None) -> go.Figure:
        """Create a box plot showing distribution statistics."""
        fig = go.Figure()
        
        if group_by and group_by in df.columns:
            fig.add_trace(go.Box(
                x=df[group_by],
                y=df['value']
            ))
        else:
            fig.add_trace(go.Box(
                y=df['value']
            ))
            
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            template=self.template
        )
        
        return fig
        
    def create_multi_series(self,
                          df: pd.DataFrame,
                          title: str,
                          x_label: str,
                          series_names: Dict[str, str],
                          use_secondary_y: bool = True) -> go.Figure:
        """Create a plot with multiple time series, optionally using secondary y-axis."""
        if use_secondary_y:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
        else:
            fig = go.Figure()
            
        # Pivot data if needed
        if 'metric_name' in df.columns:
            pivot_df = df.pivot(
                index='timestamp',
                columns='metric_name',
                values='value'
            ).reset_index()
        else:
            pivot_df = df
            
        for i, (metric_name, display_name) in enumerate(series_names.items()):
            if use_secondary_y and i == 1:
                secondary_y = True
            else:
                secondary_y = False
                
            fig.add_trace(
                go.Scatter(
                    x=pivot_df['timestamp'],
                    y=pivot_df[metric_name],
                    name=display_name,
                    line=dict(color=self.color_palette[i])
                ),
                secondary_y=secondary_y
            )
            
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            template=self.template
        )
        
        if use_secondary_y:
            # Update y-axis titles based on series names
            fig.update_yaxes(title_text=list(series_names.values())[0], secondary_y=False)
            if len(series_names) > 1:
                fig.update_yaxes(title_text=list(series_names.values())[1], secondary_y=True)
                
        return fig 