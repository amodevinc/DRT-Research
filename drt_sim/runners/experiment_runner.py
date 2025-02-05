from pathlib import Path
from typing import Dict, Any, Union, List, Optional
import logging
from datetime import datetime
import mlflow
import yaml
import ray
import json
import pandas as pd
import asyncio
import random
import numpy as np
from .scenario_runner import ScenarioRunner
from ..config.config import (
    ExperimentConfig,
    ScenarioConfig,
    SimulationConfig,
)
from drt_sim.core.paths import ExperimentPaths
from drt_sim.core.logging_config import setup_logger

logger = setup_logger(__name__)

@ray.remote
class RayScenarioRunner:
    """Ray actor wrapper for ScenarioRunner"""
    def __init__(self, scenario_cfg: ScenarioConfig, sim_cfg: SimulationConfig, paths: Path, experiment_metadata: Dict[str, Any]):
        self.runner = ScenarioRunner(
            cfg=scenario_cfg,
            sim_cfg=sim_cfg,
            paths=paths,
            experiment_metadata=experiment_metadata
        )
        
    async def run(self, replication: int) -> Dict[str, Any]:
        return await self.runner.run(replication)
        
    def cleanup(self):
        self.runner.cleanup()

class ExperimentRunner:
    """Runner for managing scenarios within an experiment"""
    
    def __init__(
        self,
        cfg: Union[ExperimentConfig, Dict[str, Any]],
        sim_cfg: SimulationConfig,
        paths: ExperimentPaths,
        study_metadata: Dict[str, Any]
    ):
        """
        Initialize the experiment runner.
        
        Args:
            cfg: Experiment configuration
            sim_cfg: Simulation configuration
            paths: Experiment paths manager
            study_metadata: Metadata from parent study
        """
        self.cfg = cfg if isinstance(cfg, ExperimentConfig) else ExperimentConfig(**cfg)
        self.sim_cfg = sim_cfg
        self.paths = paths
        self.study_metadata = study_metadata
        
        # Generate a unique experiment ID
        self.experiment_id = f"{self.cfg.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Store experiment metadata
        self.experiment_metadata = {
            **self.study_metadata,
            'experiment_id': self.experiment_id,
            'experiment_name': self.cfg.name,
            'experiment_variant': self.cfg.variant,
            'experiment_tags': ','.join(self.cfg.tags) if hasattr(self.cfg, 'tags') else '',
            'experiment_paths': self.paths
        }
        
        self.scenario_runners: Dict[str, Union[ScenarioRunner, ray.actor.ActorHandle]] = {}
        
        self._validate_config()
        self.setup()
        
    def _validate_config(self) -> None:
        """Validate experiment configuration"""
        if not self.cfg.name:
            raise ValueError("Experiment name is required")
        if not self.cfg.scenarios:
            raise ValueError("At least one scenario is required")
            
    def setup(self) -> None:
        """Set up experiment environment and initialize scenarios"""
        logger.info(f"Setting up experiment: {self.cfg.name}")
        
        # Ensure directory structure exists
        self.paths.ensure_experiment_structure()
        
        # Set up logging to file
        self._setup_logging()
        
        # Set up MLflow tracking for this experiment
        self._setup_mlflow()
        
        # Initialize scenarios
        self._initialize_scenarios()
        
        # Save experiment configuration
        self._save_experiment_config()
        
        logger.info(f"Experiment setup completed for: {self.cfg.name}")
        
    def _setup_logging(self) -> None:
        """Configure logging to file"""
        log_file = self.paths.logs / "experiment.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)
        
    def _setup_mlflow(self) -> None:
        """Configure MLflow tracking for this experiment"""
        # Set experiment-specific tags
        mlflow.set_tags({
            "experiment_name": self.cfg.name,
            "description": self.cfg.description,
            "variant": self.cfg.variant,
            "tags": ", ".join(self.cfg.tags) if hasattr(self.cfg, 'tags') else ""
        })
        
    def _initialize_scenarios(self) -> None:
        """Initialize scenario runners"""
        logger.info(f"Initializing {len(self.cfg.scenarios)} scenarios")
        
        for scenario_name, scenario_cfg in self.cfg.scenarios.items():
            # Get paths for this scenario
            scenario_paths = self.paths.get_scenario_paths(scenario_name)
            scenario_paths.ensure_scenario_structure()
            
            # Ensure proper ScenarioConfig type
            if not isinstance(scenario_cfg, ScenarioConfig):
                scenario_cfg = ScenarioConfig(**scenario_cfg)
                
            # Apply experiment metrics if scenario doesn't have metrics
            if not scenario_cfg.metrics and self.cfg.metrics:
                scenario_cfg.metrics = self.cfg.metrics
            
            # Determine if using distributed execution
            is_distributed = (hasattr(self.cfg, 'execution') and 
                           getattr(self.cfg.execution, 'distributed', False))
            
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
                    sim_cfg=self.sim_cfg,
                    paths=scenario_paths,
                    experiment_metadata=self.experiment_metadata
                )
            else:
                # Create local scenario runner
                self.scenario_runners[scenario_name] = ScenarioRunner(
                    cfg=scenario_cfg,
                    sim_cfg=self.sim_cfg,
                    paths=scenario_paths,
                    experiment_metadata=self.experiment_metadata
                )
                
            logger.info(f"Initialized scenario runner: {scenario_name}")
            
    async def run(self) -> Dict[str, Any]:
        """Execute all scenarios in the experiment"""
        logger.info(f"Running experiment: {self.cfg.name}")
        start_time = datetime.now()
        
        try:
            with mlflow.start_run(run_name=self.cfg.name, nested=True):
                self._log_experiment_params()
                
                # Determine execution mode
                is_distributed = (hasattr(self.cfg, 'execution') and 
                               getattr(self.cfg.execution, 'distributed', False))
                
                if is_distributed:
                    results = await self._run_distributed()
                else:
                    results = await self._run_sequential()
                    
                # Process and save results
                processed_results = self._process_results(results, start_time)
                # self._save_results(processed_results)
                self._log_experiment_metrics(processed_results)
                
                return processed_results
                
        except Exception as e:
            logger.error(f"Experiment execution failed: {str(e)}")
            raise
            
    async def _run_sequential(self) -> Dict[str, Any]:
        """Run scenarios sequentially"""
        results = {}
        
        for scenario_name, runner in self.scenario_runners.items():
            scenario_results = []
            num_replications = runner.cfg.replications

            logger.info(f"Running {num_replications} replications for scenario {scenario_name}")
            
            base_seed = self.sim_cfg.random_seed
            for rep in range(num_replications):
                new_seed = base_seed + rep
                logger.info("Starting replication %d for scenario '%s' with seed %d.",
                         rep + 1, runner.cfg.name, new_seed)
                # Initialize random generators with the new seed
                random.seed(new_seed)
                np.random.seed(new_seed)
                try:
                    logger.info(f"Running scenario {scenario_name} replication {rep}")
                    rep_results = await runner.run(rep)
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
                    raise
            
            results[scenario_name] = scenario_results
            
            # Save intermediate results
            # self._save_scenario_results(scenario_name, scenario_results)
        
        return results
        
    async def _run_distributed(self) -> Dict[str, Any]:
        """Run scenarios in parallel using Ray"""
        results = {}
        tasks = []
    
        
        # Create tasks for all scenarios and replications
        for scenario_name, runner in self.scenario_runners.items():
            num_replications = runner.cfg.replications
            for rep in range(num_replications):
                if isinstance(runner, ray.actor.ActorHandle):
                    task = runner.run.remote(rep)
                else:
                    task = asyncio.create_task(runner.run(rep))
                tasks.append((scenario_name, rep, task))
        
        # Process tasks as they complete
        pending_tasks = tasks.copy()
        while pending_tasks:
            done_tasks = []
            for scenario_name, rep, task in pending_tasks:
                try:
                    if isinstance(task, ray.actor.ObjectRef):
                        result = await asyncio.get_event_loop().run_in_executor(
                            None, ray.get, task
                        )
                    else:
                        result = await task
                        
                    results.setdefault(scenario_name, []).append({
                        "replication": rep,
                        "status": "completed",
                        "results": result
                    })
                    done_tasks.append((scenario_name, rep, task))
                    
                    # Save intermediate results
                    self._save_scenario_results(scenario_name, results[scenario_name])
                    
                except Exception as e:
                    logger.error(f"Scenario {scenario_name} replication {rep} failed: {str(e)}")
                    results.setdefault(scenario_name, []).append({
                        "replication": rep,
                        "status": "failed",
                        "error": str(e)
                    })
                    done_tasks.append((scenario_name, rep, task))
                    
                    if not getattr(self.cfg.execution, 'continue_on_error', True):
                        # Cancel remaining tasks
                        for _, _, remaining_task in pending_tasks:
                            if isinstance(remaining_task, asyncio.Task):
                                remaining_task.cancel()
                        return results
            
            # Remove completed tasks
            for done_task in done_tasks:
                pending_tasks.remove(done_task)
        
        return results
        
    def _process_results(self, results: Dict[str, Any], start_time: datetime) -> Dict[str, Any]:
        """Process and aggregate results from all scenarios"""
        end_time = datetime.now()
        
        processed = {
            "experiment_name": self.cfg.name,
            "description": self.cfg.description,
            "variant": self.cfg.variant,
            "timestamp": datetime.now().isoformat(),
            "execution_time": str(end_time - start_time),
            "scenarios": results,
            "summary": {
                "total_scenarios": len(self.cfg.scenarios),
                "total_replications": sum(len(reps) for reps in results.values()),
                "completed_replications": sum(
                    sum(1 for rep in reps if rep["status"] == "completed")
                    for reps in results.values()
                ),
                "failed_replications": sum(
                    sum(1 for rep in reps if rep["status"] == "failed")
                    for reps in results.values()
                )
            }
        }
        
        # Add metric summaries if available
        metric_summaries = self._calculate_metric_summaries(results)
        if metric_summaries:
            processed["metric_summaries"] = metric_summaries
        
        return processed
        
    def _calculate_metric_summaries(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics for metrics across replications"""
        metric_summaries = {}
        
        for scenario_name, replications in results.items():
            scenario_metrics = []
            
            for rep in replications:
                if rep["status"] == "completed" and "metrics" in rep["results"]:
                    scenario_metrics.append(rep["results"]["metrics"])
            
            if scenario_metrics:
                # Convert to DataFrame for easy statistical calculations
                df = pd.DataFrame(scenario_metrics)
                
                metric_summaries[scenario_name] = {
                    "mean": df.mean().to_dict(),
                    "std": df.std().to_dict(),
                    "min": df.min().to_dict(),
                    "max": df.max().to_dict()
                }
        
        return metric_summaries
        
    def _log_experiment_params(self) -> None:
        """Log experiment parameters to MLflow"""
        params = {
            "experiment_name": self.cfg.name,
            "description": self.cfg.description,
            "num_scenarios": len(self.cfg.scenarios),
            "variant": self.cfg.variant,
        }
            
        mlflow.log_params(params)
        
    def _log_experiment_metrics(self, results: Dict[str, Any]) -> None:
        """Log experiment metrics to MLflow"""
        metrics = {
            "completed_replications": results["summary"]["completed_replications"],
            "failed_replications": results["summary"]["failed_replications"],
            "completion_rate": (results["summary"]["completed_replications"] / 
                              results["summary"]["total_replications"])
        }
        
        # Log metric summaries if available
        if "metric_summaries" in results:
            for scenario, summary in results["metric_summaries"].items():
                for metric, stats in summary["mean"].items():
                    metrics[f"{scenario}_{metric}_mean"] = stats
                    
        mlflow.log_metrics(metrics)
        
    def _save_experiment_config(self) -> None:
        """Save experiment configuration"""
        config_path = self.paths.config / "experiment_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(self.cfg.to_dict(), f, default_flow_style=False)
            
        logger.info(f"Saved experiment configuration to {config_path}")
        
    def _save_scenario_results(self, scenario_name: str, results: List[Dict[str, Any]]) -> None:
        """Save intermediate scenario results"""
        scenario_dir = self.paths.results / scenario_name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = scenario_dir / "scenario_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save experiment results"""
        # Save main results
        results_path = self.paths.results / "experiment_results.yaml"
        
        # Create backup if file exists
        if results_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.paths.results / f"experiment_results_{timestamp}.yaml"
            results_path.rename(backup_path)
        
        # Save new results
        with open(results_path, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
        
        # Save metric summaries to CSV for easy analysis
        if "metric_summaries" in results:
            metrics_dir = self.paths.metrics / "summaries"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            
            for scenario, summary in results["metric_summaries"].items():
                # Save mean metrics
                mean_df = pd.DataFrame(summary["mean"], index=[0])
                mean_df.to_csv(metrics_dir / f"{scenario}_mean_metrics.csv", index=False)
                
                # Save all statistics
                stats_df = pd.concat([
                    pd.DataFrame(summary[stat], index=[0])
                    for stat in ["mean", "std", "min", "max"]
                ], keys=["mean", "std", "min", "max"])
                stats_df.to_csv(metrics_dir / f"{scenario}_metric_stats.csv")
        
        # Save temporal metrics if available
        temporal_dir = self.paths.metrics / "temporal"
        temporal_dir.mkdir(parents=True, exist_ok=True)
        
        for scenario_name, scenario_results in results["scenarios"].items():
            temporal_data = []
            for rep in scenario_results:
                if rep["status"] == "completed" and "temporal_metrics" in rep["results"]:
                    rep_data = rep["results"]["temporal_metrics"]
                    rep_data["replication"] = rep["replication"]
                    temporal_data.extend(rep_data)
            
            if temporal_data:
                pd.DataFrame(temporal_data).to_csv(
                    temporal_dir / f"{scenario_name}_temporal.csv",
                    index=False
                )
        
        logger.info(f"Saved experiment results to {results_path}")
        
    def get_scenario_results(self, scenario_name: str) -> Optional[Dict[str, Any]]:
        """
        Get results for a specific scenario.
        
        Args:
            scenario_name: Name of the scenario
            
        Returns:
            Optional[Dict[str, Any]]: Scenario results if available
        """
        results_file = self.paths.results / scenario_name / "scenario_results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                return json.load(f)
        return None
        
    def get_metric_summary(self, scenario_name: str) -> Optional[pd.DataFrame]:
        """
        Get metric summary statistics for a scenario.
        
        Args:
            scenario_name: Name of the scenario
            
        Returns:
            Optional[pd.DataFrame]: DataFrame with metric statistics if available
        """
        stats_file = self.paths.metrics / "summaries" / f"{scenario_name}_metric_stats.csv"
        if stats_file.exists():
            return pd.read_csv(stats_file)
        return None
        
    def get_temporal_metrics(self, scenario_name: str) -> Optional[pd.DataFrame]:
        """
        Get temporal metrics for a scenario.
        
        Args:
            scenario_name: Name of the scenario
            
        Returns:
            Optional[pd.DataFrame]: DataFrame with temporal metrics if available
        """
        temporal_file = self.paths.metrics / "temporal" / f"{scenario_name}_temporal.csv"
        if temporal_file.exists():
            return pd.read_csv(temporal_file)
        return None
        
    def export_experiment_data(self, output_dir: Path) -> None:
        """
        Export experiment data to a specified directory.
        
        Args:
            output_dir: Directory to export data to
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export configuration
        config_dir = output_dir / "config"
        config_dir.mkdir(exist_ok=True)
        with open(config_dir / "experiment_config.yaml", 'w') as f:
            yaml.dump(self.cfg.to_dict(), f)
        
        # Export results
        results_dir = output_dir / "results"
        results_dir.mkdir(exist_ok=True)
        
        results_file = self.paths.results / "experiment_results.yaml"
        if results_file.exists():
            import shutil
            shutil.copy2(results_file, results_dir / "experiment_results.yaml")
        
        # Export metrics
        metrics_dir = output_dir / "metrics"
        if self.paths.metrics.exists():
            shutil.copytree(self.paths.metrics, metrics_dir, dirs_exist_ok=True)
        logger.info(f"Exported experiment data to {output_dir}")
        
    def cleanup(self) -> None:
        """Clean up experiment resources"""
        logger.info(f"Cleaning up experiment: {self.cfg.name}")
        
        try:
            # Clean up scenario runners
            for runner in self.scenario_runners.values():
                try:
                    if isinstance(runner, ray.actor.ActorHandle):
                        ray.get(runner.cleanup.remote())
                    else:
                        runner.cleanup()
                except Exception as e:
                    logger.error(f"Error during scenario cleanup: {str(e)}")
            
            # Remove file handlers
            for handler in logger.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    logger.removeHandler(handler)
                    handler.close()
            
            logger.info("Experiment cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during experiment cleanup: {str(e)}")
            raise