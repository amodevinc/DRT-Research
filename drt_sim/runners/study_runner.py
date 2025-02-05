from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import yaml
import mlflow
from datetime import datetime, timedelta
import json
import asyncio
import ray
from itertools import product

from .experiment_runner import ExperimentRunner
from drt_sim.config.config import (
    StudyConfig,
    StudyType,
    ExperimentConfig,
)
from drt_sim.core.paths import create_simulation_environment
from drt_sim.core.logging_config import setup_logger

logger = setup_logger(__name__)

class StudyRunner:
    """Runner for managing simulation studies with multiple experiments"""
    
    def __init__(self, cfg: StudyConfig):
        """
        Initialize the study runner.
        
        Args:
            cfg: Study configuration defining the study parameters and structure
        """
        self.cfg = cfg
        self.experiment_runners: Dict[str, ExperimentRunner] = {}
        
        # Initialize path management
        self.sim_paths = create_simulation_environment()
        self.study_paths = self.sim_paths.get_study_paths(self.cfg.metadata.name)
        
        # Generate a unique study ID
        self.study_id = f"{self.cfg.metadata.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Store study metadata
        self.study_metadata = {
            'study_id': self.study_id,
            'study_name': self.cfg.metadata.name,
            'study_type': self.cfg.type.value,
            'study_version': self.cfg.metadata.version,
            'study_paths': self.study_paths
        }
        
        # Validate configuration
        self._validate_config()
        
    def _validate_config(self) -> None:
        """Validate study configuration"""
        if not self.cfg.metadata.name:
            raise ValueError("Study name is required in metadata")
            
        if self.cfg.type == StudyType.PARAMETER_SWEEP:
            if not self.cfg.parameter_sweep or not self.cfg.parameter_sweep.enabled:
                raise ValueError("Parameter sweep configuration is required for parameter sweep studies")
                
        elif not self.cfg.experiments:
            raise ValueError("At least one experiment configuration is required for non-parameter sweep studies")
            
    def setup(self) -> None:
        """Set up study environment and initialize experiments"""
        logger.info(f"Setting up study: {self.cfg.metadata.name}")
        
        # Create directory structure
        self.study_paths.ensure_study_structure()
        
        # Set up logging to file
        self._setup_logging()
        
        # Set up MLflow tracking
        self._setup_tracking()
        
        # Initialize experiments
        self._initialize_experiments()
        
        # Save study configuration
        self._save_study_config()
        
        logger.info(f"Study setup completed for: {self.cfg.metadata.name}")
        
    def _setup_logging(self) -> None:
        """Configure logging to file"""
        log_file = self.study_paths.logs / "study.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)
        
    def _setup_tracking(self) -> None:
        """Configure MLflow tracking"""
        mlflow.set_tracking_uri(str(self.study_paths.mlruns))
        mlflow.set_experiment(self.cfg.metadata.name)
        
        # Log study metadata
        mlflow.set_tags({
            "study_type": self.cfg.type.value,
            "study_version": self.cfg.metadata.version,
            "authors": ", ".join(self.cfg.metadata.authors),
            "tags": ", ".join(self.cfg.metadata.tags),
            "timestamp": datetime.now().isoformat()
        })
        
    def _initialize_experiments(self) -> None:
        """Initialize experiments based on study type"""
        if self.cfg.type == StudyType.PARAMETER_SWEEP:
            self._initialize_parameter_sweep()
        else:
            self._initialize_standard_experiments()
            
    def _initialize_parameter_sweep(self) -> None:
        """Initialize experiments for parameter sweep study"""
        if not self.cfg.parameter_sweep or not self.cfg.parameter_sweep.enabled:
            raise ValueError("Parameter sweep configuration is missing or disabled")
            
        logger.info("Initializing parameter sweep experiments")
        param_configs = self._generate_parameter_combinations()
        logger.info(f"Param Configs: {param_configs}")
        logger.info(f"Generated {len(param_configs)} parameter combinations")
        
        for i, param_config in enumerate(param_configs):
            exp_name = f"sweep_{i}"
            exp_paths = self.study_paths.get_experiment_paths(exp_name)
            exp_paths.ensure_experiment_structure()
            
            exp_config = self._create_sweep_experiment_config(param_config, exp_name)
            
            self.experiment_runners[exp_name] = ExperimentRunner(
                cfg=exp_config,
                sim_cfg=self.cfg.simulation,
                paths=exp_paths,
                study_metadata=self.study_metadata
            )
            
    def _generate_parameter_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations for sweep"""
        sweep_config = self.cfg.parameter_sweep
        if not sweep_config or not sweep_config.parameters:
            raise ValueError("Parameter sweep configuration is invalid")
            
        param_names = list(sweep_config.parameters.keys())
        param_values = [sweep_config.parameters[name] for name in param_names]
        
        combinations = []
        for values in product(*param_values):
            combinations.append(dict(zip(param_names, values)))
            
        return combinations
        
    def _create_sweep_experiment_config(self, param_config: Dict[str, Any], name: str) -> ExperimentConfig:
        """Create experiment configuration for a parameter combination"""
        # Start with base configuration
        config = self.cfg.base_config.copy() if self.cfg.base_config else {}
        
        # Update with sweep parameters
        for param_path, value in param_config.items():
            current = config
            *parts, last = param_path.split('.')
            
            for part in parts:
                current = current.setdefault(part, {})
            current[last] = value
        
        logger.info(f'Base Config for sweep: {config}')
            
        scenario_config = config.copy()
            
        return ExperimentConfig(
            name=name,
            description=f"Parameter sweep configuration: {param_config}",
            scenarios={"default": scenario_config},
            metrics=self.cfg.metrics if hasattr(self.cfg, 'metrics') else None,
            variant=f"sweep_{name}",
            tags=[f"{k}={v}" for k, v in param_config.items()]
        )
        
    def _initialize_standard_experiments(self) -> None:
        """Initialize standard (non-sweep) experiments"""
        logger.info("Initializing standard experiments")
        
        for exp_name, exp_cfg in self.cfg.experiments.items():
            logger.info(f"Initializing experiment: {exp_name}")
            
            # Get paths for this experiment
            exp_paths = self.study_paths.get_experiment_paths(exp_name)
            exp_paths.ensure_experiment_structure()
            
            # Ensure experiment config is properly typed
            if not isinstance(exp_cfg, ExperimentConfig):
                exp_cfg = ExperimentConfig(**exp_cfg)
            
            # Apply study metrics if experiment doesn't have metrics defined
            if not exp_cfg.metrics and hasattr(self.cfg, 'metrics'):
                exp_cfg.metrics = self.cfg.metrics
                
            self.experiment_runners[exp_name] = ExperimentRunner(
                cfg=exp_cfg,
                sim_cfg=self.cfg.simulation,
                paths=exp_paths,
                study_metadata=self.study_metadata
            )
            
    async def run(self) -> Dict[str, Any]:
        """Execute all experiments in the study"""
        results = {}
        active_run = None
        
        try:
            # End any existing runs
            try:
                mlflow.end_run()
            except Exception:
                pass
                
            active_run = mlflow.start_run(run_name=self.cfg.metadata.name)
            self._log_study_params()

            # Determine execution mode
            if (hasattr(self.cfg, 'execution') and 
                self.cfg.execution.distributed and 
                self.cfg.settings.get('parallel_experiments', False)):
                results = await self._run_parallel()
            else:
                results = await self._run_sequential()

            # Process and save results
            processed_results = self._process_results(results)
            self._log_study_metrics(processed_results)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Study execution failed: {str(e)}")
            raise
            
        finally:
            if active_run:
                mlflow.end_run()
                
    async def _run_sequential(self) -> Dict[str, Any]:
        """Run experiments sequentially"""
        results = {}
        
        for exp_name, runner in self.experiment_runners.items():
            try:
                logger.info(f"Running experiment: {exp_name}")
                results[exp_name] = await runner.run()
            except Exception as e:
                logger.error(f"Experiment {exp_name} failed: {str(e)}")
                results[exp_name] = {"status": "failed", "error": str(e)}
                
                if not self.cfg.settings.get('continue_on_error', False):
                    raise
                    
        return results
        
    async def _run_parallel(self) -> Dict[str, Any]:
        """Run experiments in parallel using Ray"""
        if not ray.is_initialized():
            ray.init(
                num_cpus=self.cfg.execution.max_parallel,
                ignore_reinit_error=True
            )
        
        results = {}
        tasks = []
        
        for exp_name, runner in self.experiment_runners.items():
            task = asyncio.create_task(runner.run())
            tasks.append((exp_name, task))
        
        for exp_name, task in tasks:
            try:
                results[exp_name] = await task
            except Exception as e:
                logger.error(f"Experiment {exp_name} failed: {str(e)}")
                results[exp_name] = {"status": "failed", "error": str(e)}
                
                if not self.cfg.settings.get('continue_on_error', False):
                    # Cancel remaining tasks
                    for _, remaining_task in tasks:
                        remaining_task.cancel()
                    break
                    
        return results
        
    def _process_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Process and aggregate results from all experiments"""
        processed = {
            "study_name": self.cfg.metadata.name,
            "study_type": self.cfg.type.value,
            "timestamp": datetime.now().isoformat(),
            "experiments": results,
            "summary": {
                "total_experiments": len(self.experiment_runners),
                "completed_experiments": sum(
                    1 for exp in results.values()
                    if isinstance(exp, dict) and exp.get("status") != "failed"
                ),
                "failed_experiments": sum(
                    1 for exp in results.values()
                    if isinstance(exp, dict) and exp.get("status") == "failed"
                )
            }
        }
        
        return processed
        
    def _log_study_params(self) -> None:
        """Log study parameters to MLflow"""
        params = {
            "study_name": self.cfg.metadata.name,
            "study_type": self.cfg.type.value,
            "num_experiments": len(self.experiment_runners),
            "version": self.cfg.metadata.version,
            "random_seed": self.cfg.random_seed if hasattr(self.cfg, 'random_seed') else None
        }
        
        if self.cfg.type == StudyType.PARAMETER_SWEEP:
            params.update({
                "sweep_parameters": str(list(self.cfg.parameter_sweep.parameters.keys())),
                "num_combinations": len(self._generate_parameter_combinations())
            })
            
        mlflow.log_params(params)
        
    def _log_study_metrics(self, results: Dict[str, Any]) -> None:
        """Log study-level metrics to MLflow"""
        metrics = {
            "completed_experiments": results["summary"]["completed_experiments"],
            "failed_experiments": results["summary"]["failed_experiments"],
            "completion_rate": (results["summary"]["completed_experiments"] / 
                              results["summary"]["total_experiments"])
        }
        mlflow.log_metrics(metrics)
        
    def _save_study_config(self) -> None:
        """Save the study configuration"""
        config_path = self.study_paths.config / "study_config.yaml"
        self.cfg.save(config_path)
        logger.info(f"Saved study configuration to {config_path}")
        
    def _extract_experiment_summary(self, exp_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key information for experiment summary"""
        if not isinstance(exp_results, dict):
            return {"status": "invalid_results"}
        return {
            "status": exp_results.get("status", "unknown"),
            "num_scenarios": len(exp_results.get("scenarios", {})),
            "error": exp_results.get("error"),
            "execution_time": exp_results.get("execution_time"),
            "metrics": exp_results.get("metrics", {})
        }
        
    def _make_serializable(self, obj: Any) -> Any:
        """Helper method to make objects serializable"""
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
        """Clean up study resources"""
        logger.info("Cleaning up study resources")
        
        try:
            # Clean up experiment runners
            for runner in self.experiment_runners.values():
                try:
                    runner.cleanup()
                except Exception as e:
                    logger.error(f"Error during experiment cleanup: {str(e)}")
            
            # Clean up Ray if it was initialized
            if (hasattr(self.cfg, 'execution') and 
                self.cfg.execution.distributed and 
                ray.is_initialized()):
                ray.shutdown()
            
            # Clean up MLflow
            try:
                mlflow.end_run()
            except Exception:
                pass
            
            # Remove file handlers
            for handler in logger.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    logger.removeHandler(handler)
                    handler.close()
                    
            logger.info("Study cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during study cleanup: {str(e)}")
            raise
            
    def export_study_artifacts(self, output_dir: Path) -> None:
        """
        Export study artifacts to a specified directory.
        
        Args:
            output_dir: Directory to export artifacts to
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export study configuration
        config_dir = output_dir / "config"
        config_dir.mkdir(exist_ok=True)
        self.cfg.save(config_dir / "study_config.yaml")
        
        # Export MLflow artifacts
        mlflow_dir = output_dir / "mlflow"
        mlflow_dir.mkdir(exist_ok=True)
        # This would require implementing MLflow export functionality
        
        logger.info(f"Exported study artifacts to {output_dir}")