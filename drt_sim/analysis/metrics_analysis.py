from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
from dataclasses import dataclass
from enum import Enum
from drt_sim.core.logging_config import setup_logger

class AnalysisType(str, Enum):
    """Types of analysis that can be performed"""
    SERVICE_QUALITY = "service_quality"
    OPERATIONAL_EFFICIENCY = "operational_efficiency"
    FLEET_UTILIZATION = "fleet_utilization"
    TEMPORAL_PATTERNS = "temporal_patterns"
    PARAMETER_SENSITIVITY = "parameter_sensitivity"

@dataclass
class AnalysisConfig:
    """Configuration for analysis parameters"""
    peak_hours: List[tuple] = (7, 9), (16, 19)
    time_windows: List[str] = "1H", "4H", "1D"
    percentiles: List[float] = 50.0, 90.0, 95.0
    smoothing_window: str = "30min"
    min_data_points: int = 10

class MetricsAnalyzer:
    """Enhanced system for analyzing simulation metrics"""
    
    def __init__(self, output_dir: Optional[Path] = None, config: Optional[AnalysisConfig] = None):
        self.output_dir = output_dir or Path("analysis_output")
        self.config = config or AnalysisConfig()
        self.logger = setup_logger(self.__class__.__name__)
        
        # Create output directories
        self.reports_dir = self.output_dir / "reports"
        self.plots_dir = self.output_dir / "plots"
        self.data_dir = self.output_dir / "processed_data"
        
        for dir_path in [self.reports_dir, self.plots_dir, self.data_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def analyze_experiment(self, 
                         metrics_df: pd.DataFrame, 
                         experiment_config: Optional[Dict] = None) -> Dict[str, Any]:
        """Perform comprehensive analysis of experiment results"""
        try:
            self.logger.info("Starting experiment analysis")
            
            # Preprocess data
            processed_df = self._preprocess_data(metrics_df)
            
            # Perform various analyses
            analysis_results = {
                'service_quality': self._analyze_service_quality(processed_df),
                'operational_efficiency': self._analyze_operational_efficiency(processed_df),
                'fleet_utilization': self._analyze_fleet_utilization(processed_df),
                'statistical_summary': self._generate_statistical_summary(processed_df)
            }
            
            # Add parameter sensitivity analysis if experiment config provided
            if experiment_config:
                analysis_results['parameter_sensitivity'] = (
                    self._analyze_parameter_sensitivity(processed_df, experiment_config)
                )
            
            # Save analysis results
            self._save_analysis_results(analysis_results)
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}", exc_info=True)
            raise

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess metrics data for analysis"""
        processed_df = df.copy()
        
        # Flatten nested dictionaries into columns
        # Handle timing data
        timing_df = pd.json_normalize(processed_df['timing'].apply(eval))
        processed_df = pd.concat([
            processed_df.drop('timing', axis=1),
            timing_df.add_prefix('timing_')
        ], axis=1)
        
        # Handle resources data
        resources_df = pd.json_normalize(processed_df['resources'].apply(eval))
        processed_df = pd.concat([
            processed_df.drop('resources', axis=1),
            resources_df.add_prefix('resources_')
        ], axis=1)
        
        # Handle requests data
        requests_df = pd.json_normalize(processed_df['requests'].apply(eval))
        processed_df = pd.concat([
            processed_df.drop('requests', axis=1),
            requests_df.add_prefix('requests_')
        ], axis=1)
        
        # Handle performance data
        performance_df = pd.json_normalize(processed_df['performance'].apply(eval))
        processed_df = pd.concat([
            processed_df.drop('performance', axis=1),
            performance_df.add_prefix('performance_')
        ], axis=1)
        
        # Handle system data
        system_df = pd.json_normalize(processed_df['system'].apply(eval))
        processed_df = pd.concat([
            processed_df.drop('system', axis=1),
            system_df.add_prefix('system_')
        ], axis=1)
        
        # Handle parameters data
        params_df = pd.json_normalize(processed_df['parameters'].apply(eval))
        processed_df = pd.concat([
            processed_df.drop('parameters', axis=1),
            params_df.add_prefix('params_')
        ], axis=1)
        
        # Calculate derived metrics using the new column names
        processed_df['request_satisfaction_rate'] = (
            processed_df['requests_completed'] /
            processed_df['requests_total'].replace(0, 1)  # Avoid division by zero
        ) * 100
        
        # Handle missing values
        processed_df = processed_df.fillna(method='ffill').fillna(method='bfill')
        
        # Add simulation duration in hours
        processed_df['simulation_duration_hours'] = processed_df['timing_simulation_time'] / 3600
        
        # Add efficiency metrics
        processed_df['events_per_second'] = processed_df['timing_events_processed'] / processed_df['timing_execution_time']
        processed_df['cpu_memory_ratio'] = (
            processed_df['resources_average_cpu_usage'] / 
            processed_df['resources_peak_memory_usage']
        )
        
        return processed_df

    def _analyze_service_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze service quality metrics"""
        return {
            'average_wait_time': df['performance_average_wait_time'].mean(),
            'average_ride_time': df['performance_average_ride_time'].mean(),
            'average_detour_ratio': df['performance_average_detour_ratio'].mean(),
            'request_satisfaction_rate': df['request_satisfaction_rate'].mean()
        }

    def _analyze_operational_efficiency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze operational efficiency metrics"""
        return {
            'average_execution_time': df['timing_execution_time'].mean(),
            'average_simulation_speed': df['timing_simulation_speed'].mean(),
            'average_cpu_usage': df['resources_average_cpu_usage'].mean(),
            'average_memory_usage': df['resources_peak_memory_usage'].mean(),
            'events_per_second': df['events_per_second'].mean()
        }

    def _analyze_fleet_utilization(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze fleet utilization metrics"""
        return {
            'average_vehicle_utilization': df['performance_vehicle_utilization'].mean(),
            'reoptimizations_per_hour': df['system_reoptimizations'].mean() / df['simulation_duration_hours'].mean()
        }

    def _analyze_parameter_sensitivity(self, 
                                    df: pd.DataFrame,
                                    experiment_config: Dict) -> Dict[str, Any]:
        """Analyze sensitivity to parameter variations"""
        try:
            sensitivity_results = {}
            
            # Extract parameter ranges from experiment config
            for param_range in experiment_config.get('parameter_ranges', []):
                param_path = param_range['parameter_path']
                param_values = param_range['values']
                
                # Calculate key metrics for each parameter value
                metric_variations = {}
                for metric in ['service_level', 'vehicle_utilization', 'average_wait_time']:
                    variations = []
                    for value in param_values:
                        metric_mean = float(
                            df[df[param_path] == value][metric].mean()
                        )
                        variations.append({
                            'parameter_value': value,
                            'metric_value': metric_mean
                        })
                    metric_variations[metric] = variations
                
                sensitivity_results[param_path] = metric_variations
            
            return sensitivity_results
            
        except Exception as e:
            self.logger.error(f"Parameter sensitivity analysis failed: {str(e)}", exc_info=True)
            return {}

    def _generate_statistical_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate statistical summary of key metrics"""
        try:
            metrics_to_summarize = [
                'service_level', 'average_wait_time', 'vehicle_utilization',
                'average_occupancy', 'request_satisfaction_rate'
            ]
            
            summary = {}
            for metric in metrics_to_summarize:
                summary[metric] = {
                    'mean': float(df[metric].mean()),
                    'std': float(df[metric].std()),
                    'min': float(df[metric].min()),
                    'max': float(df[metric].max()),
                    'percentiles': {
                        f'p{p}': float(df[metric].quantile(p/100))
                        for p in self.config.percentiles
                    }
                }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Statistical summary generation failed: {str(e)}", exc_info=True)
            return {}

    def _analyze_time_windows(self, 
                            df: pd.DataFrame, 
                            metric: str,
                            windows: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze metric patterns across different time windows"""
        windows = windows or self.config.time_windows
        analysis = {}
        
        for window in windows:
            window_data = df.resample(window, on='timestamp')[metric].mean()
            analysis[window] = self._calculate_pattern_metrics(window_data)
            
        return analysis

    def _calculate_pattern_metrics(self, series: pd.Series) -> Dict[str, float]:
        """Calculate pattern metrics for a time series"""
        return {
            'mean': float(series.mean()),
            'std': float(series.std()),
            'trend': float(self._calculate_trend(series)),
            'seasonality': float(self._calculate_seasonality(series))
        }

    def _calculate_trend(self, series: pd.Series) -> float:
        """Calculate linear trend in time series"""
        try:
            x = np.arange(len(series))
            z = np.polyfit(x, series, 1)
            return float(z[0])  # Return slope
        except:
            return 0.0

    def _calculate_seasonality(self, series: pd.Series) -> float:
        """Calculate seasonality strength in time series"""
        try:
            # Simple seasonality strength calculation
            if len(series) < 24:  # Need at least 24 points for daily seasonality
                return 0.0
                
            # Calculate daily seasonality
            daily_pattern = series.groupby(series.index.hour).mean()
            seasonality_strength = daily_pattern.std() / series.std()
            return float(seasonality_strength)
        except:
            return 0.0

    def _save_analysis_results(self, results: Dict[str, Any]) -> None:
        """Save analysis results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_file = self.reports_dir / f"analysis_report_{timestamp}.json"
        
        try:
            with open(analysis_file, 'w') as f:
                json.dump(results, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save analysis results: {str(e)}", exc_info=True)