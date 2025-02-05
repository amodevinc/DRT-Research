import csv
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import yaml
from drt_sim.core.monitoring.types.metrics import MetricName
from drt_sim.core.logging_config import setup_logger

logger = setup_logger(__name__)


# -----------------------------------------------
# Define the MetricDefinition and MetricRegistry
# -----------------------------------------------
class MetricDefinition:
    def __init__(self, name: str, description: str, metric_type: str, unit: str,
                 required_context: List[str], aggregations: List[str]):
        self.name = name
        self.description = description
        self.metric_type = metric_type  # "event", "snapshot", "aggregate"
        self.unit = unit
        self.required_context = required_context
        self.aggregations = aggregations

    def __repr__(self):
        return f"MetricDefinition(name={self.name})"


class MetricRegistry:
    def __init__(self):
        self.definitions: Dict[str, MetricDefinition] = {}

    def register(self, definition: MetricDefinition):
        self.definitions[definition.name] = definition

    def get(self, name: str) -> Optional[MetricDefinition]:
        return self.definitions.get(name)

    def load_from_yaml(self, yaml_path: str):
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            for metric_data in data.get('metrics', []):
                definition = MetricDefinition(
                    name=metric_data['name'],
                    description=metric_data['description'],
                    metric_type=metric_data['metric_type'],
                    unit=metric_data['unit'],
                    required_context=metric_data.get('required_context', []),
                    aggregations=metric_data.get('aggregations', [])
                )
                self.register(definition)
        logger.info(f"Loaded {len(self.definitions)} metric definitions from YAML.")


# Create a global registry and load definitions from the YAML file.
metric_registry = MetricRegistry()
current_dir = Path(__file__).parent
metric_registry.load_from_yaml(str(current_dir / "metrics.yaml"))


# -------------------------------
# Updated MetricsCollector class
# -------------------------------
class MetricsCollector:
    """
    A centralized metrics collector for Demand Responsive Transit (DRT) research.

    This collector supports hierarchical logging (Study -> Experiment -> Scenario -> Replication), 
    flexible aggregation, and CSV export for robust analysis.
    """

    def __init__(self, default_context: Optional[Dict[str, Any]] = None):
        """
        Initialize the metrics collector with an optional default context.
        """
        self.metrics: List[Dict[str, Any]] = []  # Store raw metric records
        self.default_context = default_context or {}  # Default hierarchical IDs
        self.study_id = self.default_context.get('study_id')
        self.experiment_id = self.default_context.get('experiment_id')
        self.scenario_id = self.default_context.get('scenario_id')
        self.replication_id = self.default_context.get('replication_id')

        # Get paths from context
        self.study_paths = self.default_context.get('study_paths')
        self.experiment_paths = self.default_context.get('experiment_paths')
        self.scenario_paths = self.default_context.get('scenario_paths')
        self.replication_paths = self.default_context.get('replication_paths')

        logger.info(f"Metrics Collector initialized with default context: {self.default_context}")

    def log(self, metric_name: MetricName, value: float, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log an individual metric with hierarchical context.
        """
        if not isinstance(metric_name, MetricName):
            raise ValueError(f"Invalid metric name: {metric_name}")

        context = context or {}

        # Retrieve the metric definition from our registry.
        definition = metric_registry.get(metric_name.value)
        if not definition:
            raise ValueError(f"No metric definition found for '{metric_name.value}'.")

        # Check that all required context keys are present.
        missing_keys = [key for key in definition.required_context if key not in context]
        if missing_keys:
            raise ValueError(f"Missing context keys for metric '{metric_name.value}': {missing_keys}")

        record = {
            "timestamp": context.get("current_time", datetime.utcnow().isoformat()),
            "metric": metric_name.value,
            "description": definition.description,
            "unit": definition.unit,
            "value": value,
            "context": context,
            "study_id": self.study_id,
            "experiment_id": self.experiment_id,
            "scenario_id": self.scenario_id,
            "replication_id": self.replication_id
        }

        self.metrics.append(record)

    def aggregate(self, group_by: List[str]) -> Dict[str, Any]:
        """
        Aggregate metrics based on hierarchical levels.
        """
        results = {}
        for metric in self.metrics:
            group_key = tuple([metric.get(key, 'unknown') for key in group_by])
            m_name = metric['metric']

            if group_key not in results:
                results[group_key] = {}
            if m_name not in results[group_key]:
                results[group_key][m_name] = []
            results[group_key][m_name].append(metric['value'])

        aggregated_results = {}
        for group_key, metrics in results.items():
            group_dict = dict(zip(group_by, group_key))
            aggregated_results[json_serialize(group_dict)] = {
                m_name: {
                    'count': len(values),
                    'mean': mean(values),
                    'min': min(values),
                    'max': max(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0
                }
                for m_name, values in metrics.items()
            }
        return aggregated_results

    def flush_to_csv(self, output_file: str) -> None:
        """
        Export raw metric logs to a CSV file.
        """
        if not self.metrics:
            return

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        headers = sorted(list({key for metric in self.metrics for key in metric.keys()}))

        with open_csv(output_file, 'w') as csvfile:
            write_csv_header(csvfile, headers)
            for metric in self.metrics:
                row = [metric.get(header, '') for header in headers]
                write_csv_row(csvfile, row)

    def flush_hierarchical(self) -> None:
        """
        Flush metrics to CSV files at each hierarchical level.
        """
        if self.replication_paths and self.replication_id:
            replication_metrics_path = Path(self.replication_paths.metrics) / "raw_metrics.csv"
            self.flush_to_csv(str(replication_metrics_path))

        if self.scenario_paths and self.scenario_id:
            scenario_metrics = self._collect_from_replications()
            scenario_metrics_path = Path(self.scenario_paths.metrics) / "raw_metrics.csv"
            self._write_raw_metrics(scenario_metrics, scenario_metrics_path)

        if self.experiment_paths and self.experiment_id:
            experiment_metrics = self._collect_from_scenarios()
            experiment_metrics_path = Path(self.experiment_paths.metrics) / "raw_metrics.csv"
            self._write_raw_metrics(experiment_metrics, experiment_metrics_path)

        if self.study_paths and self.study_id:
            study_metrics = self._collect_from_experiments()
            study_metrics_path = Path(self.study_paths.metrics) / "raw_metrics.csv"
            self._write_raw_metrics(study_metrics, study_metrics_path)

    def _collect_from_replications(self) -> List[Dict[str, Any]]:
        if not self.scenario_paths:
            return []
        replications_path = Path(self.scenario_paths.root) / "replications"
        metrics_data = []
        for rep_dir in replications_path.glob("rep_*"):
            metrics_file = rep_dir / "metrics" / "raw_metrics.csv"
            if metrics_file.exists():
                metrics_data.extend(self._read_metrics_file(metrics_file))
        return metrics_data + self.metrics

    def _collect_from_scenarios(self) -> List[Dict[str, Any]]:
        if not self.experiment_paths:
            return []
        scenarios_path = Path(self.experiment_paths.root) / "scenarios"
        metrics_data = []
        for scenario_dir in scenarios_path.glob("*"):
            metrics_file = scenario_dir / "metrics" / "raw_metrics.csv"
            if metrics_file.exists():
                metrics_data.extend(self._read_metrics_file(metrics_file))
        return metrics_data + self.metrics

    def _collect_from_experiments(self) -> List[Dict[str, Any]]:
        if not self.study_paths:
            return []
        experiments_path = Path(self.study_paths.root) / "experiments"
        metrics_data = []
        for exp_dir in experiments_path.glob("*"):
            metrics_file = exp_dir / "metrics" / "raw_metrics.csv"
            if metrics_file.exists():
                metrics_data.extend(self._read_metrics_file(metrics_file))
        return metrics_data + self.metrics

    def _read_metrics_file(self, file_path: Path) -> List[Dict[str, Any]]:
        if not file_path.exists():
            return []
        metrics_data = []
        with open(file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                metrics_data.append(row)
        return metrics_data

    def _write_raw_metrics(self, metrics: List[Dict[str, Any]], output_path: Path) -> None:
        if not metrics:
            return
        output_path.parent.mkdir(parents=True, exist_ok=True)
        headers = sorted(list({key for metric in metrics for key in metric.keys()}))
        with open_csv(output_path, 'w') as csvfile:
            write_csv_header(csvfile, headers)
            for metric in metrics:
                row = [metric.get(header, '') for header in headers]
                write_csv_row(csvfile, row)

    def clear_metrics(self) -> None:
        """Clear all logged metrics from memory."""
        self.metrics = []


def open_csv(file_path: str, mode: str):
    """Open a CSV file."""
    return open(file_path, mode, newline='')


def write_csv_header(csvfile, headers: List[str]) -> None:
    writer = csv.writer(csvfile)
    writer.writerow(headers)


def write_csv_row(csvfile, row: List[Any]) -> None:
    writer = csv.writer(csvfile)
    writer.writerow(row)


def json_serialize(data: Dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True)


def mean(values: List[float]) -> float:
    return statistics.mean(values) if values else 0.0
