from typing import Dict, Optional, List
import yaml
import logging
from pathlib import Path
from drt_sim.core.monitoring.types.metrics import MetricDefinition

logger = logging.getLogger(__name__)

class MetricRegistry:
    """Registry for metric definitions with type safety."""
    
    def __init__(self):
        self.definitions: Dict[str, MetricDefinition] = {}

    def register(self, 
                name: str, 
                description: str, 
                metric_type: str,
                unit: str, 
                required_context: Optional[List[str]] = None,
                aggregations: Optional[List[str]] = None,
                visualizations: Optional[Dict[str, bool]] = None):
        """Register a metric definition."""
        self.definitions[name] = MetricDefinition(
            name=name,
            description=description,
            metric_type=metric_type,
            unit=unit,
            required_context=required_context or [],
            aggregations=aggregations or ['mean', 'min', 'max', 'std', 'count'],
            visualizations=visualizations
        )

    def get(self, name: str) -> Optional[MetricDefinition]:
        """Get metric definition."""
        return self.definitions.get(name)

    def load_from_yaml(self, yaml_path: str):
        """Load metric definitions from YAML."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            
            # Load metric definitions
            for metric_data in data.get('metrics', []):
                self.register(
                    name=metric_data['name'],
                    description=metric_data['description'],
                    metric_type=metric_data['metric_type'],
                    unit=metric_data.get('unit', ''),
                    required_context=metric_data.get('required_context', []),
                    aggregations=metric_data.get('aggregations', None),
                    visualizations=metric_data.get('visualizations', None)
                )
                
        logger.info(f"Loaded {len(self.definitions)} metric definitions from YAML.")

# Initialize the global registry
metric_registry = MetricRegistry()
current_dir = Path(__file__).parent
metric_registry.load_from_yaml(str(current_dir / "metrics.yaml"))