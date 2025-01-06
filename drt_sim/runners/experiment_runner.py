from pathlib import Path
from typing import Dict, Any, Union
import logging
from datetime import datetime
import mlflow
import yaml
import ray

from .scenario_runner import ScenarioRunner
from drt_sim.config.config import (
    ExperimentConfig,
    ScenarioConfig,
    SimulationConfig,
)

logger = logging.getLogger(__name__)

@ray.remote
class RayScenarioRunner:
    """Ray actor wrapper for ScenarioRunner"""
    def __init__(self, scenario_cfg: ScenarioConfig, output_dir: str):
        self.runner = ScenarioRunner(
            cfg=scenario_cfg,
            output_dir=Path(output_dir)
        )
        
    def run(self, replication: int) -> Dict[str, Any]:
        return self.runner.run(replication)
        
    def cleanup(self):
        self.runner.cleanup()

class ExperimentRunner:
    """
    Runner for managing scenarios within an experiment.
    Handles both distributed and sequential execution of scenarios.
    """
    
    def __init__(self, cfg: Union[ExperimentConfig, Dict[str, Any]], sim_cfg: SimulationConfig):
        """
        Initialize the experiment runner.
        
        Args:
            cfg: Experiment configuration
        """
        # Ensure proper configuration type
        self.cfg = cfg if isinstance(cfg, ExperimentConfig) else ExperimentConfig(**cfg)
        self.sim_cfg = sim_cfg
        self.scenario_runners: Dict[str, Union[ScenarioRunner, ray.actor.ActorHandle]] = {}
        self._setup_paths()
        self._validate_config()
        self.setup()
        
    def _setup_paths(self) -> None:
        """Set up experiment directory structure"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = Path(f"experiments/{self.cfg.name}/{timestamp}")
        
        self.paths = {
            "base": base_dir,
            "results": base_dir / "results",
            "logs": base_dir / "logs",
            "configs": base_dir / "configs",
            "metrics": base_dir / "metrics",
            "scenarios": base_dir / "scenarios"
        }
        
        # Create all directories
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)

    def _validate_config(self) -> None:
        """Validate experiment configuration"""
        if not self.cfg.name:
            raise ValueError("Experiment name is required")
        if not self.cfg.scenarios:
            raise ValueError("At least one scenario is required")

    def setup(self) -> None:
        """Set up experiment environment and initialize scenarios"""
        logger.info(f"Setting up experiment: {self.cfg.name}")
        
        self._setup_mlflow()
        self._initialize_scenarios()
        self._save_experiment_config()
        
        logger.info(f"Experiment setup completed for: {self.cfg.name}")

    def _setup_mlflow(self) -> None:
        """Configure MLflow tracking for this experiment"""
        mlflow.set_experiment(self.cfg.name)
        
        # Log experiment metadata
        mlflow.set_tags({
            "experiment_name": self.cfg.name,
            "description": self.cfg.description,
            "variant": self.cfg.variant,
            "tags": ", ".join(self.cfg.tags)
        })

    def _initialize_scenarios(self) -> None:
        """Initialize scenario runners"""
        logger.info(f"Initializing {len(self.cfg.scenarios)} scenarios")
        
        for scenario_name, scenario_cfg in self.cfg.scenarios.items():
            # Ensure proper ScenarioConfig type
            if not isinstance(scenario_cfg, ScenarioConfig):
                scenario_cfg = ScenarioConfig(**scenario_cfg)
                
            # Apply experiment metrics if scenario doesn't have metrics
            if not scenario_cfg.metrics and self.cfg.metrics:
                scenario_cfg.metrics = self.cfg.metrics
            
            scenario_dir = self.paths["scenarios"] / scenario_name
            
            # Use study's execution config instead of experiment's
            is_distributed = False  # Default to sequential execution
            if hasattr(self.cfg, 'execution'):
                is_distributed = getattr(self.cfg.execution, 'distributed', False)
            
            if is_distributed:
                # Initialize Ray if not already done
                if not ray.is_initialized():
                    ray.init(
                        num_cpus=getattr(self.cfg.execution, 'max_parallel', 4),
                        ignore_reinit_error=True
                    )
                # Create distributed scenario runner
                self.scenario_runners[scenario_name] = RayScenarioRunner.remote(
                    scenario_cfg=scenario_cfg,
                    output_dir=str(scenario_dir)
                )
            else:
                # Create local scenario runner
                self.scenario_runners[scenario_name] = ScenarioRunner(
                    cfg=scenario_cfg,
                    sim_cfg=self.sim_cfg,
                    output_dir=scenario_dir
                )

    def run(self) -> Dict[str, Any]:
        """Execute all scenarios in the experiment"""
        logger.info(f"Running experiment: {self.cfg.name}")
        active_run = None
        
        try:
            # Start MLflow run with nested=True since this is part of a study
            active_run = mlflow.start_run(
                run_name=self.cfg.name,
                nested=True  # This is crucial for proper nesting under the study run
            )
            
            self._log_experiment_params()
            
            # Execute scenarios based on study's execution configuration
            is_distributed = False
            if hasattr(self.cfg, 'execution'):
                is_distributed = getattr(self.cfg.execution, 'distributed', False)
            
            if is_distributed:
                results = self._run_distributed()
            else:
                results = self._run_sequential()
                
            # Process and save results
            processed_results = self._process_results(results)
            self._save_results(processed_results)
            self._log_experiment_metrics(processed_results)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Experiment execution failed: {str(e)}")
            raise
            
        finally:
            # Ensure MLflow run is ended
            if active_run:
                mlflow.end_run()

    def _run_sequential(self) -> Dict[str, Any]:
        """Run scenarios sequentially"""
        results = {}
        
        logger.info(f"Running {len(self.scenario_runners)} scenarios sequentially")
        for scenario_name, runner in self.scenario_runners.items():
            scenario_results = []
            
            # Note: Using simulation.replications from cfg if it exists, otherwise default to 1
            num_replications = (
                self.cfg.simulation.replications 
                if hasattr(self.cfg, 'simulation') and hasattr(self.cfg.simulation, 'replications')
                else 1
            )

            logger.info(f"Running {num_replications} replications for scenario {scenario_name}")
            for rep in range(num_replications):
                try:
                    logger.info(f"Running scenario {scenario_name} replication {rep}")
                    rep_results = runner.run(rep)
                    scenario_results.append({
                        "replication": rep,
                        "status": "completed",
                        "results": rep_results
                    })
                except Exception as e:
                    logger.error(f"Scenario {scenario_name} replication {rep} failed: {str(e)}")
                    scenario_results.append({
                        "replication": rep,
                        "status": "failed",
                        "error": str(e)
                    })
            
            results[scenario_name] = scenario_results
        
        return results

    def _run_distributed(self) -> Dict[str, Any]:
        logger.info("Running scenarios in parallel using Ray")
        """Run scenarios in parallel using Ray"""
        results = {}
        pending_tasks = []
        
        # Get number of replications
        num_replications = (
            self.cfg.simulation.replications 
            if hasattr(self.cfg, 'simulation') and hasattr(self.cfg.simulation, 'replications')
            else 1
        )
        
        # Launch all scenarios and replications
        for scenario_name, runner in self.scenario_runners.items():
            scenario_tasks = []
            for rep in range(num_replications):
                task = runner.run.remote(rep)
                scenario_tasks.append((scenario_name, rep, task))
            pending_tasks.extend(scenario_tasks)
        
        # Process results as they complete
        while pending_tasks:
            done_ids, pending_ids = ray.wait(
                [task for _, _, task in pending_tasks],
                num_returns=1,
                timeout=self.cfg.execution.timeout if hasattr(self.cfg.execution, 'timeout') else None
            )
            
            # Find and process completed task
            for scenario_name, rep, task in pending_tasks[:]:
                if task in done_ids:
                    try:
                        rep_results = ray.get(task)
                        results.setdefault(scenario_name, []).append({
                            "replication": rep,
                            "status": "completed",
                            "results": rep_results
                        })
                    except Exception as e:
                        logger.error(f"Scenario {scenario_name} replication {rep} failed: {str(e)}")
                        results.setdefault(scenario_name, []).append({
                            "replication": rep,
                            "status": "failed",
                            "error": str(e)
                        })
                        
                        if hasattr(self.cfg.execution, 'continue_on_error') and not self.cfg.execution.continue_on_error:
                            # Cancel remaining tasks
                            for _, _, remaining_task in pending_tasks:
                                ray.cancel(remaining_task)
                            raise
                    
                    pending_tasks.remove((scenario_name, rep, task))
                    break
        
        return results

    def _process_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Process and aggregate results from all scenarios"""
        processed = {
            "experiment_name": self.cfg.name,
            "description": self.cfg.description,
            "timestamp": datetime.now().isoformat(),
            "scenarios": results,
            "summary": {
                "total_scenarios": len(self.cfg.scenarios),
                "completed_scenarios": sum(
                    1 for scenario in results.values()
                    for rep in scenario
                    if rep["status"] == "completed"
                ),
                "failed_scenarios": sum(
                    1 for scenario in results.values()
                    for rep in scenario
                    if rep["status"] == "failed"
                )
            }
        }
        
        return processed

    def _log_experiment_params(self) -> None:
        """Log experiment parameters to MLflow"""
        params = {
            "experiment_name": self.cfg.name,
            "description": self.cfg.description,
            "num_scenarios": len(self.cfg.scenarios),
            "random_seed": self.cfg.random_seed
        }
        
        if hasattr(self.cfg, 'simulation') and hasattr(self.cfg.simulation, 'replications'):
            params["replications"] = self.cfg.simulation.replications
            
        mlflow.log_params(params)

    def _log_experiment_metrics(self, results: Dict[str, Any]) -> None:
        """Log experiment metrics to MLflow"""
        metrics = {
            "completed_scenarios": results["summary"]["completed_scenarios"],
            "failed_scenarios": results["summary"]["failed_scenarios"],
            "completion_rate": (results["summary"]["completed_scenarios"] / 
                              (results["summary"]["total_scenarios"] * 
                               (self.cfg.simulation.replications if hasattr(self.cfg, 'simulation') 
                                and hasattr(self.cfg.simulation, 'replications') else 1)))
        }
        mlflow.log_metrics(metrics)

    def _save_experiment_config(self) -> None:
        """Save experiment configuration"""
        config_path = self.paths["configs"] / "experiment_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(self.cfg.__dict__, f, default_flow_style=False)
        logger.info(f"Saved experiment configuration to {config_path}")

    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save experiment results"""
        results_path = self.paths["results"] / "experiment_results.yaml"
        
        # Create backup if file exists
        if results_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = results_path.with_name(f"experiment_results_{timestamp}.yaml")
            results_path.rename(backup_path)
        
        # Save new results
        with open(results_path, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
        logger.info(f"Saved experiment results to {results_path}")

    def cleanup(self) -> None:
        """Clean up experiment resources"""
        logger.info("Cleaning up experiment resources")
        try:
            for runner in self.scenario_runners.values():
                try:
                    runner.cleanup()
                except Exception as e:
                    logger.error(f"Error during cleanup: {str(e)}")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")