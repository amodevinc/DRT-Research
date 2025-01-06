from typing import Dict, List, Any, Optional
from enum import Enum
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from pathlib import Path

class ParameterType(Enum):
    """Types of parameters that can be swept"""
    INTEGER = "integer"
    FLOAT = "float"
    TIMEDELTA = "timedelta"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"

class ParameterRange(BaseModel):
    """Defines a range for a parameter sweep"""
    parameter_path: str  # Dot-notation path to parameter (e.g., 'vehicle.fleet_size')
    parameter_type: ParameterType
    values: List[Any]  # List of values to sweep
    description: Optional[str] = None

class ExperimentType(Enum):
    """Types of experiments supported"""
    SINGLE_RUN = "single_run"  # Single scenario run
    PARAMETER_SWEEP = "parameter_sweep"  # Sweep over parameter ranges
    MULTI_PARAMETER = "multi_parameter"  # Multiple parameter combinations

class ExperimentStatus(str, Enum):
    """Status states for experiment tracking"""
    INITIALIZED = "initialized"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ANALYZING = "analyzing"

class ExperimentProgress(BaseModel):
    """Progress tracking for experiments"""
    total_runs: int = 0
    completed_runs: int = 0
    failed_runs: int = 0
    current_phase: str = "initializing"



class ExperimentConfig(BaseModel):
    """Configuration for an experiment"""
    experiment_id: str
    experiment_type: ExperimentType
    description: str
    base_scenario: str  # Name of base scenario configuration
    parameter_ranges: Optional[List[ParameterRange]] = None
    replications: int = Field(default=1, ge=1)  # Number of replications per parameter set
    output_directory: Path
    random_seed_base: Optional[int] = None  # Base random seed
    max_parallel_runs: int = Field(default=1, ge=1)
    
    # Analysis configuration
    save_state_history: bool = False
    metrics_to_collect: List[str] = Field(default_factory=list)
    aggregation_period: Optional[str] = None
    
    # Validation rules
    def model_post_init(self, *args, **kwargs):
        if self.experiment_type != ExperimentType.SINGLE_RUN and not self.parameter_ranges:
            raise ValueError("Parameter ranges required for parameter sweep experiments")
        
class ExperimentMetadata(BaseModel):
    """Metadata about an experiment run"""
    experiment_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: ExperimentStatus = ExperimentStatus.INITIALIZED
    progress: ExperimentProgress
    configuration: ExperimentConfig
    git_commit: Optional[str] = None  # For reproducibility
    error: Optional[str] = None
    analysis_results: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True
            