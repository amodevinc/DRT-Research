import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
from jinja2 import Environment, FileSystemLoader

from drt_sim.core.monitoring.types.metrics import MetricDefinition
from drt_sim.analysis.visualizations.metric_plotter import MetricPlotter
from drt_sim.analysis.utils.statistical_analysis import StatisticalAnalyzer

class ReportGenerator:
    """Generates comprehensive analysis reports from metrics data."""
    
    def __init__(self, 
                 output_dir: Union[str, Path],
                 template_dir: Optional[Union[str, Path]] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.plotter = MetricPlotter(self.output_dir / "figures")
        self.analyzer = StatisticalAnalyzer()
        
        # Set up template environment
        if template_dir is None:
            template_dir = Path(__file__).parent / "templates"
        self.env = Environment(loader=FileSystemLoader(str(template_dir)))
        
    def generate_metric_report(self,
                             df: pd.DataFrame,
                             metric_def: MetricDefinition,
                             group_by: Optional[str] = None) -> str:
        """Generate a detailed report for a single metric."""
        report_data = {
            'metric_name': metric_def.name,
            'description': metric_def.description,
            'unit': metric_def.unit,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'sections': []
        }
        
        # Basic statistics
        stats_section = self._generate_basic_stats(df, metric_def)
        report_data['sections'].append(stats_section)
        
        # Time series analysis
        time_series_section = self._generate_time_series_analysis(df, metric_def)
        report_data['sections'].append(time_series_section)
        
        # Distribution analysis
        dist_section = self._generate_distribution_analysis(df, metric_def)
        report_data['sections'].append(dist_section)
        
        # Group analysis if specified
        if group_by:
            group_section = self._generate_group_analysis(df, metric_def, group_by)
            report_data['sections'].append(group_section)
        
        # Generate HTML report
        template = self.env.get_template('metric_report.html')
        html_content = template.render(**report_data)
        
        # Save report
        report_path = self.output_dir / f"{metric_def.name}_report.html"
        with open(report_path, 'w') as f:
            f.write(html_content)
            
        return str(report_path)
        
    def generate_comparison_report(self,
                                 dfs: Dict[str, pd.DataFrame],
                                 metric_def: MetricDefinition) -> str:
        """Generate a report comparing multiple datasets for the same metric."""
        report_data = {
            'metric_name': metric_def.name,
            'description': metric_def.description,
            'unit': metric_def.unit,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'sections': []
        }
        
        # Basic comparison statistics
        stats_section = {
            'title': 'Basic Statistics Comparison',
            'content': [],
            'figures': []
        }
        
        comparison_stats = {}
        for name, df in dfs.items():
            stats = df['value'].describe()
            comparison_stats[name] = stats.to_dict()
            
        stats_section['content'].append({
            'type': 'table',
            'data': pd.DataFrame(comparison_stats).to_html()
        })
        
        # Statistical tests
        all_values = [df['value'].values for df in dfs.values()]
        if len(dfs) == 2:
            t_test = stats.ttest_ind(all_values[0], all_values[1])
            stats_section['content'].append({
                'type': 'text',
                'data': f"T-test results: statistic={t_test.statistic:.4f}, p-value={t_test.pvalue:.4f}"
            })
            
        report_data['sections'].append(stats_section)
        
        # Time series comparison
        time_series_section = {
            'title': 'Time Series Comparison',
            'content': [],
            'figures': []
        }
        
        # Create combined time series plot
        fig = go.Figure()
        for name, df in dfs.items():
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['value'],
                name=name
            ))
            
        fig.update_layout(
            title=f"{metric_def.name} Comparison",
            xaxis_title="Time",
            yaxis_title=f"Value ({metric_def.unit})"
        )
        
        # Save figure
        fig_path = f"comparison_{metric_def.name}_timeseries.html"
        self.plotter.save_figure(fig, fig_path)
        time_series_section['figures'].append(fig_path)
        
        report_data['sections'].append(time_series_section)
        
        # Distribution comparison
        dist_section = {
            'title': 'Distribution Comparison',
            'content': [],
            'figures': []
        }
        
        # Create distribution comparison plot
        fig = go.Figure()
        for name, df in dfs.items():
            fig.add_trace(go.Histogram(
                x=df['value'],
                name=name,
                opacity=0.7,
                nbinsx=30
            ))
            
        fig.update_layout(
            title=f"{metric_def.name} Distribution Comparison",
            xaxis_title=f"Value ({metric_def.unit})",
            yaxis_title="Frequency",
            barmode='overlay'
        )
        
        # Save figure
        fig_path = f"comparison_{metric_def.name}_distribution.html"
        self.plotter.save_figure(fig, fig_path)
        dist_section['figures'].append(fig_path)
        
        report_data['sections'].append(dist_section)
        
        # Generate HTML report
        template = self.env.get_template('comparison_report.html')
        html_content = template.render(**report_data)
        
        # Save report
        report_path = self.output_dir / f"{metric_def.name}_comparison_report.html"
        with open(report_path, 'w') as f:
            f.write(html_content)
            
        return str(report_path)
        
    def generate_summary_report(self,
                              df: pd.DataFrame,
                              metric_defs: List[MetricDefinition],
                              title: Optional[str] = 'Simulation Metrics Summary Report') -> str:
        """Generate a summary report covering multiple metrics.
        
        Args:
            df: DataFrame containing metrics data
            metric_defs: List of metric definitions to include in report
            title: Optional custom title for the report
        """
        report_data = {
            'title': title,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'sections': []
        }
        
        # Overview section
        # Ensure timestamps are datetime objects before comparison
        timestamps = pd.to_datetime(df['timestamp'])
        overview_section = {
            'title': 'Overview',
            'content': [
                {
                    'type': 'text',
                    'data': f"Analysis period: {timestamps.min()} to {timestamps.max()}"
                },
                {
                    'type': 'text',
                    'data': f"Total metrics collected: {len(df)}"
                }
            ]
        }
        report_data['sections'].append(overview_section)
        
        # Metrics summary
        for metric_def in metric_defs:
            metric_df = df[df['metric_name'] == metric_def.name]
            if not metric_df.empty:
                metric_section = {
                    'title': metric_def.name,
                    'content': [],
                    'figures': []
                }
                
                # Basic statistics
                stats = metric_df['value'].describe()
                metric_section['content'].append({
                    'type': 'table',
                    'data': stats.to_frame().to_html()
                })
                
                # Time series plot
                fig = self.plotter.create_time_series(metric_df, metric_def)
                fig_path = f"summary_{metric_def.name}_timeseries.html"
                self.plotter.save_figure(fig, fig_path)
                metric_section['figures'].append(fig_path)
                
                report_data['sections'].append(metric_section)
        
        # Correlation analysis
        corr_section = {
            'title': 'Metric Correlations',
            'content': [],
            'figures': []
        }
        
        # Create correlation matrix
        fig = self.plotter.create_correlation_matrix(df, [m.name for m in metric_defs])
        fig_path = "summary_correlation_matrix.html"
        self.plotter.save_figure(fig, fig_path)
        corr_section['figures'].append(fig_path)
        
        report_data['sections'].append(corr_section)
        
        # Generate HTML report
        template = self.env.get_template('summary_report.html')
        html_content = template.render(**report_data)
        
        # Save report
        report_path = self.output_dir / "summary_report.html"
        with open(report_path, 'w') as f:
            f.write(html_content)
            
        return str(report_path)
        
    def _generate_basic_stats(self,
                            df: pd.DataFrame,
                            metric_def: MetricDefinition) -> Dict:
        """Generate basic statistics section."""
        section = {
            'title': 'Basic Statistics',
            'content': [],
            'figures': []
        }
        
        # Compute statistics
        stats = df['value'].describe()
        section['content'].append({
            'type': 'table',
            'data': stats.to_frame().to_html()
        })
        
        # Additional statistics
        additional_stats = {
            'skewness': stats.skew(),
            'kurtosis': stats.kurtosis(),
            'missing_values': df['value'].isnull().sum()
        }
        
        section['content'].append({
            'type': 'text',
            'data': f"Additional Statistics:\n" + \
                   f"Skewness: {additional_stats['skewness']:.4f}\n" + \
                   f"Kurtosis: {additional_stats['kurtosis']:.4f}\n" + \
                   f"Missing Values: {additional_stats['missing_values']}"
        })
        
        return section
        
    def _generate_time_series_analysis(self,
                                     df: pd.DataFrame,
                                     metric_def: MetricDefinition) -> Dict:
        """Generate time series analysis section."""
        section = {
            'title': 'Time Series Analysis',
            'content': [],
            'figures': []
        }
        
        # Create time series plot
        fig = self.plotter.create_time_series(df, metric_def)
        fig_path = f"{metric_def.name}_timeseries.html"
        self.plotter.save_figure(fig, fig_path)
        section['figures'].append(fig_path)
        
        # Trend analysis
        # trend_result = self.analyzer.analyze_trend(df)
        # section['content'].append({
        #     'type': 'text',
        #     'data': f"Trend Analysis:\n{trend_result.interpretation}"
        # })
        
        return section
        
    def _generate_distribution_analysis(self,
                                      df: pd.DataFrame,
                                      metric_def: MetricDefinition) -> Dict:
        """Generate distribution analysis section."""
        section = {
            'title': 'Distribution Analysis',
            'content': [],
            'figures': []
        }
        
        # Create distribution plot
        fig = self.plotter.create_distribution_plot(df, metric_def)
        fig_path = f"{metric_def.name}_distribution.html"
        self.plotter.save_figure(fig, fig_path)
        section['figures'].append(fig_path)
        
        # Normality test
        # normality_result = self.analyzer.test_normality(df['value'].values)
        # section['content'].append({
        #     'type': 'text',
        #     'data': f"Normality Test:\n{normality_result.interpretation}"
        # })
        
        # Outlier analysis
        # outliers = self.analyzer.detect_outliers(df['value'].values)
        # n_outliers = np.sum(outliers)
        # section['content'].append({
        #     'type': 'text',
        #     'data': f"Outlier Analysis:\nDetected {n_outliers} outliers " + \
        #            f"({n_outliers/len(df)*100:.2f}% of data)"
        # })
        
        return section
        
    def _generate_group_analysis(self,
                               df: pd.DataFrame,
                               metric_def: MetricDefinition,
                               group_by: str) -> Dict:
        """Generate group analysis section."""
        section = {
            'title': f'Group Analysis by {group_by}',
            'content': [],
            'figures': []
        }
        
        # Create box plot
        fig = self.plotter.create_box_plot(df, metric_def, group_by)
        fig_path = f"{metric_def.name}_boxplot.html"
        self.plotter.save_figure(fig, fig_path)
        section['figures'].append(fig_path)
        
        # Statistical comparison
        # comparison_results = self.analyzer.compare_groups(df, group_by)
        # for result in comparison_results:
        #     section['content'].append({
        #         'type': 'text',
        #         'data': f"{result.test_name}:\n{result.interpretation}"
        #     })
            
        return section 