from pathlib import Path
from typing import Dict, Any, Optional
import logging
import yaml
import pickle
from datetime import datetime, timedelta
import mlflow

from drt_sim.core.simulation.orchestrator import SimulationOrchestrator
from drt_sim.core.monitoring.metrics_collector import MetricsCollector
from drt_sim.config.config import (
    ScenarioConfig,
    SimulationConfig,
)
from drt_sim.models.simulation import SimulationStatus
import json
from enum import Enum
logger = logging.getLogger(__name__)

class ScenarioRunner:
    """
    Runner for executing individual simulation scenarios.
    Handles core simulation execution, metrics collection, and state management.
    """
    
    def __init__(
        self,
        cfg: ScenarioConfig,
        sim_cfg: SimulationConfig,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize the scenario runner.
        
        Args:
            cfg: Scenario configuration
            sim_cfg: Simulation configuration
            output_dir: Optional output directory path
        """
        self.cfg = cfg if isinstance(cfg, ScenarioConfig) else ScenarioConfig(**cfg)
        self.sim_cfg = sim_cfg
        self.output_dir = Path(output_dir) if output_dir else Path(f"scenarios/{self.cfg.name}")
        
        # Initialize components as None
        self.orchestrator: Optional[SimulationOrchestrator] = None
        self.metrics_collector: Optional[MetricsCollector] = None
        self.current_replication: int = 0
        self.state_history: Dict[int, Any] = {}
        
    def setup(self, replication: int) -> None:
        """
        Set up scenario for execution.
        
        Args:
            replication: Replication number for this run
        """
        self.current_replication = replication
        self._setup_directories()
        self._initialize_components(replication)
        logger.info(f"Scenario {self.cfg.name} setup completed for replication {replication}")
    
    def _setup_directories(self) -> Dict[str, Path]:
        """Create necessary directories for scenario outputs"""
        directories = {
            "base": self.output_dir,
            "results": self.output_dir / "results",
            "logs": self.output_dir / "logs",
            "metrics": self.output_dir / "metrics",
            "states": self.output_dir / "states",
            "replication": self.output_dir / f"replication_{self.current_replication}"
        }
        
        for path in directories.values():
            path.mkdir(parents=True, exist_ok=True)
            
        return directories
        
    def _initialize_components(self, replication: int) -> None:
        """Initialize simulation components for this replication"""
        logger.info(f"Initializing replication {replication}")
        
        # Initialize orchestrator with full configuration context
        self.orchestrator = SimulationOrchestrator(
            cfg=self.cfg,
            sim_cfg=self.sim_cfg
        )
        
        # Initialize metrics collection if configured
        # if self.cfg.metrics:
        #     self.metrics_collector = MetricsCollector(
        #         metrics_config=self.cfg.metrics
        #     )
            
        # Initialize state tracking if enabled
        if self.sim_cfg.save_state:
            self.state_history.clear()
    
    def run(self, replication: int) -> Dict[str, Any]:
        """
        Execute the scenario for a specific replication.
        
        Args:
            replication: Replication number to run
            
        Returns:
            Dict containing scenario results
        """
        try:
            # Set up for this replication
            self.setup(replication)
            
            # Start MLflow run for this replication
            with mlflow.start_run(run_name=f"{self.cfg.name}_rep_{replication}", nested=True):
                self._log_scenario_params(replication)
                results = self._execute_simulation()
                
                # Process and save results
                processed_results = self._process_results(results)
                self._save_results(processed_results)
                self._log_metrics(processed_results)
                
                return processed_results
                
        except Exception as e:
            import traceback
            logger.error(f"Scenario execution failed: {str(e)}\n{traceback.format_exc()}")
            self._save_error_state()
            raise
        finally:
            self.cleanup()
            
    def _execute_simulation(self) -> Dict[str, Any]:
        """Execute the core simulation loop"""
        step_results = []
        last_metrics_collection = datetime.now() - timedelta(seconds=self.cfg.metrics.collect_interval)
        last_state_save = datetime.now() - timedelta(seconds=self.sim_cfg.save_interval)
        
        logger.info(f"Starting simulation for scenario {self.cfg.name}")
        
        try:
            # Initialize simulation
            self.orchestrator.initialize()
            initial_state = self.orchestrator.get_state()
            
            if self.sim_cfg.save_state:
                self.state_history[0] = initial_state
            
            # Main simulation loop
            while self.orchestrator.context.status not in [SimulationStatus.COMPLETED, SimulationStatus.FAILED, SimulationStatus.STOPPED]:
                current_time = self.orchestrator.context.current_time
                
                # Execute simulation step
                step_result = self.orchestrator.step()
                step_results.append(step_result)
                
                # Collect metrics if configured
                if (self.metrics_collector and 
                    self.cfg.metrics and 
                    (current_time - last_metrics_collection).total_seconds() >= self.cfg.metrics.collect_interval):
                    self.metrics_collector.collect(self.orchestrator.get_state())
                    last_metrics_collection = current_time
                
                # Save state if enabled
                if (self.sim_cfg.save_state and 
                    (current_time - last_state_save).total_seconds() >= self.sim_cfg.save_interval):
                    self.state_history[current_time] = self.orchestrator.get_state()
                    last_state_save = current_time
                
                # Check if we've reached the end time
                if current_time >= self.orchestrator.context.end_time:
                    self.orchestrator.context.status = SimulationStatus.COMPLETED
                    break
                
                # Log progress periodically
                if current_time.second % self.sim_cfg.time_step == 0:
                    logger.debug(f"Simulation time: {current_time}")
                    
            # Collect final metrics if configured
            if self.metrics_collector and self.cfg.metrics:
                self.metrics_collector.collect(self.orchestrator.get_state())
                
            return {
                "step_results": step_results,
                "final_state": self.orchestrator.get_state(),
                "simulation_time": self.orchestrator.context.current_time
            }
            
        except Exception as e:
            self.orchestrator.context.status = SimulationStatus.FAILED
            logger.error("Error during simulation execution", exc_info=True)
            raise
            
    def _process_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw simulation results"""
        processed = {
            "scenario_name": self.cfg.name,
            "description": self.cfg.description,
            "replication": self.current_replication,
            "timestamp": datetime.now().isoformat(),
            "simulation_info": {
                "duration": results["simulation_time"],
                "num_steps": len(results["step_results"]),
                "start_time": self.sim_cfg.start_time,
                "end_time": self.sim_cfg.end_time
            },
            "configuration": {
                "vehicle": self.cfg.vehicle.__dict__,
                "demand": self.cfg.demand.__dict__,
                "algorithm": self.cfg.algorithm.__dict__,
                "network": self.cfg.network.__dict__
            }
        }
        
        # Add simulation-specific results
        processed["step_results"] = results["step_results"]
        
        return processed
    
    def _log_scenario_params(self, replication: int) -> None:
        """Log scenario parameters to MLflow"""
        # Log basic scenario information
        mlflow.log_params({
            "scenario_name": self.cfg.name,
            "replication": replication,
            "fleet_size": self.cfg.vehicle.fleet_size,
            "vehicle_capacity": self.cfg.vehicle.capacity,
            "dispatch_strategy": self.cfg.algorithm.dispatch_strategy,
            "matching_algorithm": self.cfg.algorithm.matching_algorithm,
            "demand_generator": self.cfg.demand.generator_type
        })
    
    def _log_metrics(self, results: Dict[str, Any]) -> None:
        """Log metrics to MLflow"""
        # Prepare the simulation duration
        simulation_duration = results["simulation_info"]["duration"]
        if isinstance(simulation_duration, datetime):
            # Convert datetime to a numeric value (e.g., total seconds since the start of simulation)
            simulation_duration = simulation_duration.timestamp()  # Or calculate the difference if it's not a timestamp

        # Log general metrics
        mlflow.log_metrics({
            "simulation_duration": simulation_duration,
            "num_steps": results["simulation_info"]["num_steps"]
        })
        
        # Log collected metrics
        if "metrics" in results:
            for key, value in results["metrics"].items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)

    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save processed results"""
        results_dir = self.output_dir / "results"
        results_path = results_dir / f"scenario_results_rep_{self.current_replication}.yaml"
        
        # Convert any dataclass objects in results to dictionaries
        serializable_results = self._make_serializable(results)
        
        with open(results_path, 'w') as f:
            yaml.safe_dump(serializable_results, f, default_flow_style=False)
            
        # Save state history if enabled
        if self.sim_cfg.save_state and self.state_history:
            states_path = self.output_dir / "states" / f"state_history_rep_{self.current_replication}.pkl"
            with open(states_path, 'wb') as f:
                pickle.dump(self.state_history, f)
                
        logger.info(f"Saved scenario results to {results_path}")

    def _make_serializable(self, obj: Any) -> Any:
        """Helper method to make objects YAML serializable"""
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(i) for i in obj]
        elif isinstance(obj, (datetime, timedelta)):
            return str(obj)
        elif isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, Path):
            return str(obj)
        return obj
    
    def _save_error_state(self) -> None:
        """Save state information in case of error"""
        try:
            error_dir = self.output_dir / "errors"
            error_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            error_path = error_dir / f"error_state_rep_{self.current_replication}_{timestamp}.json"
            
            if self.orchestrator:
                error_state = {
                    'current_state': self.orchestrator.get_state().to_dict(),
                    'state_history': self.state_history,
                    'last_step_time': self.orchestrator.context.current_time.isoformat(),
                    'scenario_config': self.cfg.__dict__,
                    'simulation_config': self.sim_cfg.__dict__
                }
                
                with open(error_path, 'w') as f:
                    json.dump(error_state, f, indent=2, default=str)
                    
                logger.info(f"Saved error state to {error_path}")
        except Exception as e:
            logger.error(f"Failed to save error state: {str(e)}")
    
    def cleanup(self) -> None:
        """Clean up scenario resources"""
        try:
            if self.orchestrator:
                self.orchestrator.cleanup()
            if self.metrics_collector:
                self.metrics_collector.cleanup()
            self.state_history.clear()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")