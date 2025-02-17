from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

@dataclass
class SimulationPaths:
    """Advanced path management for DRT simulation research platform"""
    root: Path
    
    def __post_init__(self):
        """Initialize derived paths after root is set"""
        # Top level structure
        self.studies = self.root / "studies"
        self.visualizations = self.root / "visualizations"
        self.analysis = self.root / "analysis"
        
    def get_study_paths(self, study_name: str, timestamp: Optional[str] = None) -> "StudyPaths":
        """Get paths for a specific study"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return StudyPaths(self.studies / f"{study_name}_{timestamp}")
    
    def ensure_base_structure(self) -> None:
        """Create base directory structure"""
        for path in [self.studies, self.visualizations, self.analysis]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Create visualization directories
        (self.visualizations / "plots").mkdir(exist_ok=True)
        (self.visualizations / "animations").mkdir(exist_ok=True)
        (self.visualizations / "dashboards").mkdir(exist_ok=True)
        
        # Create analysis directories
        (self.analysis / "comparative").mkdir(exist_ok=True)
        (self.analysis / "statistical").mkdir(exist_ok=True)
        (self.analysis / "reports").mkdir(exist_ok=True)

@dataclass
class StudyPaths:
    """Manages paths for a specific study"""
    root: Path
    
    def __post_init__(self):
        # Study structure
        self.config = self.root / "config"
        self.logs = self.root / "logs"
        self.mlruns = self.root / "mlruns"
        self.experiments = self.root / "experiments"
        self.metrics = self.root / "metrics"
        self.artifacts = self.root / "artifacts"
        
        # Study artifact subdirectories
        self.plots = self.artifacts / "plots"
        self.animations = self.artifacts / "animations"
        self.reports = self.artifacts / "reports"
        
    def get_experiment_paths(self, experiment_name: str) -> "ExperimentPaths":
        """Get paths for a specific experiment"""
        return ExperimentPaths(self.experiments / experiment_name)
        
    def ensure_study_structure(self) -> None:
        """Create study directory structure"""
        # Create main directories
        for path in [self.config, self.logs, self.mlruns, self.experiments, self.metrics, self.artifacts, self.plots, self.animations, 
                    self.reports]:
            path.mkdir(parents=True, exist_ok=True)
            
@dataclass
class ExperimentPaths:
    """Manages paths for a specific experiment"""
    root: Path
    
    def __post_init__(self):
        # Experiment structure
        self.config = self.root / "config"
        self.logs = self.root / "logs"
        self.scenarios = self.root / "scenarios"
        self.metrics = self.root / "metrics"
        self.artifacts = self.root / "artifacts"
        
    def get_scenario_paths(self, scenario_name: str) -> "ScenarioPaths":
        """Get paths for a specific scenario"""
        return ScenarioPaths(self.scenarios / scenario_name)
        
    def ensure_experiment_structure(self) -> None:
        """Create experiment directory structure"""
        # Create main directories
        for path in [self.config, self.logs, self.scenarios, self.metrics, self.artifacts]:
            path.mkdir(parents=True, exist_ok=True)
            

@dataclass
class ScenarioPaths:
    """Manages paths for a specific scenario"""
    root: Path
    
    def __post_init__(self):
        # Scenario structure
        self.config = self.root / "config"
        self.logs = self.root / "logs"
        self.replications = self.root / "replications"
        self.metrics = self.root / "metrics"
        self.states = self.root / "states"
        self.snapshots = self.root / "snapshots"
        self.artifacts = self.root / "artifacts"
        
    def get_replication_paths(self, replication: int) -> "ReplicationPaths":
        """Get paths for a specific replication"""
        return ReplicationPaths(self.replications / f"rep_{replication}")
        
    def ensure_scenario_structure(self) -> None:
        """Create scenario directory structure"""
        for path in [self.config, self.logs, self.replications, self.metrics,
                    self.states, self.snapshots, self.artifacts]:
            path.mkdir(parents=True, exist_ok=True)

@dataclass
class ReplicationPaths:
    """Manages paths for a specific replication"""
    root: Path
    
    def __post_init__(self):
        # Replication structure
        self.events = self.root / "events"
        self.metrics = self.root / "metrics"
        self.states = self.root / "states"
        self.logs = self.root / "logs"
        self.results = self.root / "results"
        self.artifacts = self.root / "artifacts"
        
    def ensure_replication_structure(self) -> None:
        """Create replication directory structure"""
        for path in [self.events, self.metrics, self.states, self.logs, self.results, self.artifacts]:
            path.mkdir(parents=True, exist_ok=True)

# Example usage
def create_simulation_environment(base_dir: str = "drt_sim_output") -> SimulationPaths:
    """Create the complete simulation environment"""
    base_path = Path(base_dir)
    base_path.mkdir(exist_ok=True)  # Ensure base directory exists
    sim_paths = SimulationPaths(base_path)
    sim_paths.ensure_base_structure()
    return sim_paths