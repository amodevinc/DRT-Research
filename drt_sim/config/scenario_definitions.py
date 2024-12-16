# drt_sim/config/scenario_definitions.py
from dataclasses import dataclass
from typing import Dict, Any, List, Union
from pathlib import Path
from .parameters import ScenarioParameters
import json

@dataclass
class ScenarioDefinition:
    """Defines a complete simulation scenario"""
    parameters: ScenarioParameters
    metadata: Dict[str, Any]
    description: str
    tags: List[str]
    
    def to_dict(self) -> Dict:
        """Convert scenario definition to dictionary"""
        return {
            'parameters': self.parameters.dict(),
            'metadata': self.metadata,
            'description': self.description,
            'tags': self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ScenarioDefinition':
        """Create scenario definition from dictionary"""
        return cls(
            parameters=ScenarioParameters(**data['parameters']),
            metadata=data['metadata'],
            description=data['description'],
            tags=data['tags']
        )

class ScenarioManager:
    """Manages scenario definitions and variations"""
    
    def __init__(self, scenarios_dir: Union[str, Path]):
        self.scenarios_dir = Path(scenarios_dir)
        self.scenarios_dir.mkdir(parents=True, exist_ok=True)
        self._scenarios: Dict[str, ScenarioDefinition] = {}
        self._load_scenarios()
    
    def _load_scenarios(self) -> None:
        """Load all scenario definitions from directory"""
        for scenario_file in self.scenarios_dir.glob('*.json'):
            with scenario_file.open() as f:
                data = json.load(f)
                self._scenarios[scenario_file.stem] = ScenarioDefinition.from_dict(data)
    
    def get_scenario(self, name: str) -> ScenarioDefinition:
        """Get a scenario by name"""
        if name not in self._scenarios:
            raise ValueError(f"Scenario not found: {name}")
        return self._scenarios[name]
    
    def create_variation(self, 
                        base_scenario: str, 
                        changes: Dict[str, Any]) -> ScenarioDefinition:
        """Create a variation of an existing scenario"""
        base = self.get_scenario(base_scenario)
        new_params = base.parameters.copy()
        
        # Apply changes to parameters
        for path, value in changes.items():
            self._apply_parameter_change(new_params, path, value)
        
        return ScenarioDefinition(
            parameters=new_params,
            metadata={**base.metadata, 'parent': base_scenario},
            description=f"Variation of {base_scenario}",
            tags=[*base.tags, 'variation']
        )
    
    def _apply_parameter_change(self, 
                              params: ScenarioParameters, 
                              path: str, 
                              value: Any) -> None:
        """Apply a change to nested parameters"""
        parts = path.split('.')
        target = params
        
        for part in parts[:-1]:
            target = getattr(target, part)
        
        setattr(target, parts[-1], value)
    
    def save_scenario(self, name: str, scenario: ScenarioDefinition) -> None:
        """Save a scenario definition"""
        scenario_file = self.scenarios_dir / f"{name}.json"
        with scenario_file.open('w') as f:
            json.dump(scenario.to_dict(), f, indent=2)
        self._scenarios[name] = scenario