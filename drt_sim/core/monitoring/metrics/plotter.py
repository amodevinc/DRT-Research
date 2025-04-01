import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Optional, Dict, Any

class MetricPlotter:
    """A utility class for creating standardized metric visualizations."""
    
    def __init__(self):
        """Initialize the MetricPlotter with default settings."""
        self.default_layout = {
            'template': 'plotly_white',
            'font': {'size': 12},
            'showlegend': True,
            'legend': {'orientation': 'h', 'y': -0.2}
        }
    
    def create_time_series(self, 
                          df: pd.DataFrame,
                          x_col: str,
                          y_col: str,
                          title: str,
                          group_col: Optional[str] = None,
                          layout_updates: Optional[Dict[str, Any]] = None) -> go.Figure:
        """Create a time series plot from the given dataframe.
        
        Args:
            df: DataFrame containing the data to plot
            x_col: Column name for x-axis (typically timestamp)
            y_col: Column name for y-axis (metric value)
            title: Plot title
            group_col: Optional column to group by (for multiple lines)
            layout_updates: Optional dictionary of layout updates
            
        Returns:
            Plotly figure object
        """
        if group_col:
            fig = px.line(df, x=x_col, y=y_col, color=group_col, title=title)
        else:
            fig = px.line(df, x=x_col, y=y_col, title=title)
            
        layout = self.default_layout.copy()
        if layout_updates:
            layout.update(layout_updates)
            
        fig.update_layout(**layout)
        return fig
    
    def create_box_plot(self,
                       df: pd.DataFrame,
                       y_col: str,
                       title: str,
                       group_col: Optional[str] = None,
                       layout_updates: Optional[Dict[str, Any]] = None) -> go.Figure:
        """Create a box plot from the given dataframe.
        
        Args:
            df: DataFrame containing the data to plot
            y_col: Column name for y-axis (metric value)
            title: Plot title
            group_col: Optional column to group by
            layout_updates: Optional dictionary of layout updates
            
        Returns:
            Plotly figure object
        """
        if group_col:
            fig = px.box(df, y=y_col, color=group_col, title=title)
        else:
            fig = px.box(df, y=y_col, title=title)
            
        layout = self.default_layout.copy()
        if layout_updates:
            layout.update(layout_updates)
            
        fig.update_layout(**layout)
        return fig
    
    def create_histogram(self,
                        df: pd.DataFrame,
                        x_col: str,
                        title: str,
                        nbins: int = 30,
                        layout_updates: Optional[Dict[str, Any]] = None) -> go.Figure:
        """Create a histogram from the given dataframe.
        
        Args:
            df: DataFrame containing the data to plot
            x_col: Column name for x-axis (metric value)
            title: Plot title
            nbins: Number of bins for the histogram
            layout_updates: Optional dictionary of layout updates
            
        Returns:
            Plotly figure object
        """
        fig = px.histogram(df, x=x_col, title=title, nbins=nbins)
        
        layout = self.default_layout.copy()
        if layout_updates:
            layout.update(layout_updates)
            
        fig.update_layout(**layout)
        return fig
    
    def create_pie_chart(self,
                        labels: list,
                        values: list,
                        title: str,
                        layout_updates: Optional[Dict[str, Any]] = None) -> go.Figure:
        """Create a pie chart from the given data.
        
        Args:
            labels: List of labels for each segment
            values: List of values for each segment
            title: Plot title
            layout_updates: Optional dictionary of layout updates
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
        
        layout = self.default_layout.copy()
        if layout_updates:
            layout.update(layout_updates)
            
        fig.update_layout(**layout)
        return fig 