from pathlib import Path
from typing import Dict, Any, Optional
import logging
import pickle
from datetime import datetime, timedelta
import mlflow
import json
import pandas as pd

from drt_sim.core.simulation.orchestrator import SimulationOrchestrator
from drt_sim.config.config import ScenarioConfig, SimulationConfig
from drt_sim.models.simulation import SimulationStatus
from drt_sim.core.paths import ScenarioPaths, ReplicationPaths
from drt_sim.core.logging_config import setup_logger
from drt_sim.models.base import SimulationEncoder
from drt_sim.core.monitoring.metrics_collector import MetricsCollector
logger = setup_logger(__name__)

class ScenarioRunner:
    """Runner for executing individual simulation scenarios"""
    
    def __init__(
        self,
        cfg: ScenarioConfig,
        sim_cfg: SimulationConfig,
        paths: ScenarioPaths,
        experiment_metadata: Dict[str, Any]
    ):
        """
        Initialize the scenario runner.
        
        Args:
            cfg: Scenario configuration
            sim_cfg: Simulation configuration
            paths: Scenario paths manager
            experiment_metadata: Metadata from parent experiment
        """
        self.cfg = cfg if isinstance(cfg, ScenarioConfig) else ScenarioConfig(**cfg)
        self.sim_cfg = sim_cfg
        self.paths = paths
        self.experiment_metadata = experiment_metadata
        
        # Generate a unique scenario ID
        self.scenario_id = f"{self.cfg.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Store scenario metadata
        self.scenario_metadata = {
            **self.experiment_metadata,
            'scenario_id': self.scenario_id,
            'scenario_name': self.cfg.name,
            'scenario_type': self.cfg.type if hasattr(self.cfg, 'type') else None,
            'fleet_size': self.cfg.vehicle.fleet_size,
            'vehicle_capacity': self.cfg.vehicle.capacity,
            'scenario_paths': self.paths
        }
        
        # Initialize components as None
        self.orchestrator: Optional[SimulationOrchestrator] = None
        self.current_replication: int = 0
        self.state_history: Dict[int, Any] = {}
        
    def setup(self, rep_paths: ReplicationPaths) -> None:
        """
        Set up scenario for execution.
        
        Args:
            rep_paths: Paths for specific replication
        """
        logger.info(f"Setting up scenario {self.cfg.name} for replication {self.current_replication}")
        
        # Set up logging for this replication
        self._setup_logging(rep_paths)
        
        # Initialize components
        self._initialize_components(rep_paths)
        
        # Save configuration
        self._save_config(rep_paths)
        
        logger.info(f"Scenario setup completed for replication {self.current_replication}")
    
    def _setup_logging(self, rep_paths: ReplicationPaths) -> None:
        """Configure logging for this replication"""
        log_file = rep_paths.logs / "replication.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)
        
    def _initialize_components(self, rep_paths: ReplicationPaths) -> None:
        """Initialize simulation components for this replication"""
        logger.info(f"Initializing components for replication {self.current_replication}")
        
        # Create metrics collector for this replication with full context
        replication_id = f"{self.scenario_id}_rep_{self.current_replication}"
        metrics_collector = MetricsCollector(default_context={
            **self.scenario_metadata,
            'replication_id': replication_id,
            'replication_number': self.current_replication,
            'replication_paths': rep_paths
        })
        
        # Initialize orchestrator with full configuration context
        self.orchestrator = SimulationOrchestrator(
            cfg=self.cfg,
            sim_cfg=self.sim_cfg,
            output_dir=rep_paths.root,
            metrics_collector=metrics_collector
        )
            
        # Initialize state tracking if enabled
        if self.sim_cfg.save_state:
            self.state_history.clear()
    
    def _save_config(self, rep_paths: ReplicationPaths) -> None:
        """Save configuration for this replication"""
        config = {
            'scenario_config': self.cfg.to_dict(),
            'simulation_config': self.sim_cfg.to_dict(),
            'replication': self.current_replication,
            'timestamp': datetime.now().isoformat()
        }
        
        config_file = rep_paths.root / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
    
    async def run(self, replication: int) -> Dict[str, Any]:
        """
        Execute the scenario for a specific replication.
        
        Args:
            replication: Replication number to run
            
        Returns:
            Dict containing scenario results
        """
        self.current_replication = replication
        rep_paths = self.paths.get_replication_paths(replication)
        rep_paths.ensure_replication_structure()
        
        try:
            # Set up for this replication
            self.setup(rep_paths)
            
            # Start MLflow run for this replication
            with mlflow.start_run(run_name=f"{self.cfg.name}_rep_{replication}", nested=True):
                self._log_scenario_params(replication)
                results = await self._execute_simulation(rep_paths)
                self._log_metrics(results)
                return results
                
        except Exception as e:
            logger.error(f"Scenario execution failed: {str(e)}")
            await self._save_error_state(rep_paths)
            raise
            
        finally:
            self.cleanup()
            
    async def _execute_simulation(self, rep_paths: ReplicationPaths) -> Dict[str, Any]:
        """Execute the core simulation loop"""
        step_results = []
        last_state_save = datetime.now()
        
        logger.info(f"Starting simulation for scenario {self.cfg.name}")
        
        try:
            # Initialize simulation
            self.orchestrator.initialize()
            initial_state = self.orchestrator.get_state()
            
            if self.sim_cfg.save_state:
                self.state_history[0] = initial_state
                self._save_state(initial_state, 0, rep_paths)
            
            # Main simulation loop
            while self.orchestrator.context.status not in [
                SimulationStatus.COMPLETED,
                SimulationStatus.FAILED,
                SimulationStatus.STOPPED
            ]:
                current_time = self.orchestrator.context.current_time
                
                # Execute simulation step
                step_result = await self.orchestrator.step()
                step_results.append(step_result)
                
                
                # Save state if enabled
                if (self.sim_cfg.save_state and 
                    (current_time - last_state_save).total_seconds() >= 
                    self.sim_cfg.save_interval):
                    state = self.orchestrator.get_state()
                    self.state_history[int(current_time.timestamp())] = state
                    self._save_state(state, int(current_time.timestamp()), rep_paths)
                    last_state_save = current_time
                
                # Check if we've reached the end time
                if current_time >= self.orchestrator.context.end_time:
                    self.orchestrator.context.status = SimulationStatus.COMPLETED
                    break
                
                # Log progress periodically
                if current_time.second % self.sim_cfg.time_step == 0:
                    logger.debug(f"Simulation time: {current_time}")
                    
            # Collect final metrics and state
            final_state = self.orchestrator.get_state()
            
            if self.sim_cfg.save_state:
                self._save_state(final_state, int(current_time.timestamp()), rep_paths)
            
            # Prepare results
            results = {
                "step_results": step_results,
                "final_state": json.dumps(final_state, cls=SimulationEncoder),
                "simulation_time": self.orchestrator.context.current_time,
                "status": self.orchestrator.context.status.value
            }
            # Save results
            # self._save_results(results, rep_paths)
            # Save event history
            self.orchestrator.save_event_history(rep_paths.events / "event_history.json")
            return results
        except Exception as e:
            self.orchestrator.context.status = SimulationStatus.FAILED
            logger.error("Error during simulation execution", exc_info=True)
            raise
    
    def _save_state(self, state: Any, timestamp: int, rep_paths: ReplicationPaths) -> None:
        """
        Save simulation state as JSON.
        
        Args:
            state: Current simulation state
            timestamp: Current simulation timestamp
            rep_paths: Replication paths manager
        """
        state_file = rep_paths.states / f"state_{timestamp}.json"
        
        # Convert state to serializable format
        serializable_state = self._make_serializable(state)
        
        # Save as formatted JSON
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_state, f, indent=2, default=str)

    def _log_scenario_params(self, replication: int) -> None:
        """Log scenario parameters to MLflow"""
        params = {
            "scenario_name": self.cfg.name,
            "replication": replication,
            "fleet_size": self.cfg.vehicle.fleet_size,
            "vehicle_capacity": self.cfg.vehicle.capacity,
            "matching": self.cfg.matching,
            "demand_generator": self.cfg.demand.generator_type,
            "start_time": self.sim_cfg.start_time,
            "end_time": self.sim_cfg.end_time,
            "time_step": self.sim_cfg.time_step
        }
        mlflow.log_params(params)
    
    def _log_metrics(self, results: Dict[str, Any]) -> None:
        """Log metrics to MLflow"""
        if isinstance(results.get("metrics"), dict):
            for metric_type, metrics in results["metrics"].items():
                if isinstance(metrics, dict):
                    for name, value in metrics.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(f"{metric_type}_{name}", value)
    
    def _save_results(self, results: Dict[str, Any], rep_paths: ReplicationPaths) -> None:
        """Save simulation results"""
        # Save main results
        results_file = rep_paths.results / "simulation_results.json"
        with open(results_file, 'w') as f:
            json.dump(self._make_serializable(results), f, indent=2, default=str)
        
        # Save summary
        summary = {
            "status": results["status"],
            "simulation_time": results["simulation_time"].isoformat(),
            "num_steps": len(results["step_results"]),
            "final_metrics": results.get("metrics", {})
        }
        
        summary_file = rep_paths.results / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Saved simulation results to {results_file}")
    
    async def _save_error_state(self, rep_paths: ReplicationPaths) -> None:
        """Save state information in case of error"""
        try:
            error_dir = rep_paths.root / "errors"
            error_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            error_file = error_dir / f"error_state_{timestamp}.json"
            
            if self.orchestrator:
                error_state = {
                    'current_state': self.orchestrator.get_state(),
                    'last_step_time': self.orchestrator.context.current_time.isoformat(),
                    'scenario_config': self.cfg.to_dict(),
                    'simulation_config': self.sim_cfg.to_dict(),
                    'error_time': datetime.now().isoformat()
                }
                
                with open(error_file, 'w') as f:
                    json.dump(self._make_serializable(error_state), f, indent=2, default=str)
                    
                logger.info(f"Saved error state to {error_file}")
        except Exception as e:
            logger.error(f"Failed to save error state: {str(e)}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Helper method to make objects JSON serializable"""
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(i) for i in obj]
        elif isinstance(obj, (datetime, timedelta)):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        return obj
    
    def cleanup(self) -> None:
        """Clean up scenario resources"""
        try:
            if self.orchestrator:
                self.orchestrator.cleanup()
            self.state_history.clear()
            
            # Remove file handlers to avoid duplicate logging
            for handler in logger.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    logger.removeHandler(handler)
                    handler.close()
                    
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")