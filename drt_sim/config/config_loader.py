# drt_sim/config/config_loader.py
import yaml
from pathlib import Path
from typing import Dict, Any, Union
import json
from datetime import datetime, timedelta
from .parameters import ScenarioParameters, SimulationParameters, VehicleParameters, DemandParameters, StopParameters, AlgorithmParameters
class ConfigLoader:
    """Loads and validates configuration files"""
    
    def __init__(self, config_dir: Union[str, Path]):
        self.config_dir = Path(config_dir)
        self._cache: Dict[str, ScenarioParameters] = {}
        
    def load_scenario(self, scenario_name: str) -> ScenarioParameters:
        """Load a scenario configuration by name"""
        if scenario_name in self._cache:
            return self._cache[scenario_name]
            
        config_file = self.config_dir / f"{scenario_name}.yaml"
        if not config_file.exists():
            raise ValueError(f"Scenario configuration not found: {scenario_name}")
            
        with config_file.open() as f:
            config_data = yaml.safe_load(f)
            
        scenario = self._parse_scenario(config_data)
        self._cache[scenario_name] = scenario
        return scenario
    
    def _parse_scenario(self, config_data: Dict) -> ScenarioParameters:
        """Parse and validate scenario configuration"""
        # Parse datetime and timedelta values
        for key, value in config_data.get('simulation', {}).items():
            if key.endswith('_time'):
                if isinstance(value, str):
                    config_data['simulation'][key] = datetime.fromisoformat(value)
            elif key.endswith('_period'):
                if isinstance(value, (int, float)):
                    config_data['simulation'][key] = timedelta(minutes=value)
        
        # Create parameter objects
        simulation_params = SimulationParameters(**config_data.get('simulation', {}))
        vehicle_params = VehicleParameters(**config_data.get('vehicle', {}))
        demand_params = DemandParameters(**config_data.get('demand', {}))
        stop_params = StopParameters(**config_data.get('stop', {}))
        algorithm_params = AlgorithmParameters(**config_data.get('algorithm', {}))
        
        return ScenarioParameters(
            name=config_data['name'],
            description=config_data.get('description', ''),
            simulation=simulation_params,
            vehicle=vehicle_params,
            demand=demand_params,
            stop=stop_params,
            algorithm=algorithm_params,
            network_file=Path(config_data['network_file']),
            output_directory=Path(config_data.get('output_directory', 'output'))
        )
    
    def save_scenario(self, scenario: ScenarioParameters, filename: str) -> None:
        """Save a scenario configuration to file"""
        config_file = self.config_dir / filename
        
        # Convert scenario to dictionary
        config_data = {
            'name': scenario.name,
            'description': scenario.description,
            'simulation': scenario.simulation.dict(),
            'vehicle': scenario.vehicle.dict(),
            'demand': scenario.demand.dict(),
            'stop': scenario.stop.dict(),
            'algorithm': scenario.algorithm.dict(),
            'network_file': str(scenario.network_file),
            'output_directory': str(scenario.output_directory)
        }
        
        # Save to YAML
        with config_file.open('w') as f:
            yaml.dump(config_data, f, default_flow_style=False)