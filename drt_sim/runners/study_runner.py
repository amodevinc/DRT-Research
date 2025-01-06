from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import yaml
import mlflow
from datetime import datetime

from .experiment_runner import ExperimentRunner
from ..config.config import (
    StudyConfig,
    StudyType,
    ExperimentConfig,
    StudyMetadata,
    ParameterSweepConfig
)

logger = logging.getLogger(__name__)

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
        
        # Set up MLflow tracking
        self._setup_tracking()
        
        # Initialize experiments
        self._initialize_experiments()
        
        # Save study configuration
        self._save_study_config()
        
        logger.info(f"Study setup completed for: {self.cfg.metadata.name}")

    def _setup_tracking(self) -> None:
        """Configure MLflow tracking"""
        # Use MLflow config if provided, otherwise use default paths
        tracking_uri = (self.cfg.paths.get_study_dir(self.cfg.metadata.name) / "mlflow")

        logger.info(f"Setting up MLflow tracking with URI: {tracking_uri}")
        
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(self.cfg.metadata.name)
        
        # Log study metadata
        mlflow.set_tags({
            "study_type": self.cfg.type.value,
            "study_version": self.cfg.metadata.version,
            "authors": ", ".join(self.cfg.metadata.authors),
            "tags": ", ".join(self.cfg.metadata.tags)
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
        
        for i, param_config in enumerate(param_configs):
            exp_name = f"sweep_{i}"
            exp_config = self._create_sweep_experiment_config(param_config, exp_name)
            self.experiment_runners[exp_name] = ExperimentRunner(exp_config, self.cfg.simulation)
            
        logger.info(f"Created {len(param_configs)} parameter sweep experiments")

    def _generate_parameter_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations for sweep"""
        from itertools import product
        
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
            
        # Create ExperimentConfig
        return ExperimentConfig(
            name=name,
            description=f"Parameter sweep configuration: {param_config}",
            scenarios={"default": config},
            metrics=self.cfg.metrics if hasattr(self.cfg, 'metrics') else None,
            random_seed=42
        )

    def _initialize_standard_experiments(self) -> None:
        """Initialize standard (non-sweep) experiments"""
        logger.info("Initializing standard experiments")
        
        for exp_name, exp_cfg in self.cfg.experiments.items():
            logger.info(f"Initializing experiment: {exp_name}")
            
            # Ensure experiment config is properly typed
            if not isinstance(exp_cfg, ExperimentConfig):
                exp_cfg = ExperimentConfig(**exp_cfg)
            
            # Apply study metrics if experiment doesn't have metrics defined
            if not exp_cfg.metrics and hasattr(self.cfg, 'metrics'):
                exp_cfg.metrics = self.cfg.metrics
                
            self.experiment_runners[exp_name] = ExperimentRunner(exp_cfg, self.cfg.simulation)

    def run(self) -> Dict[str, Any]:
        """Execute all experiments in the study"""
        results = {}
        active_run = None
        
        try:
            # End any existing runs
            try:
                mlflow.end_run()
            except Exception:
                pass
                
            # Start new run
            active_run = mlflow.start_run(run_name=self.cfg.metadata.name)
            self._log_study_params()

            # Execute experiments based on execution configuration
            if (hasattr(self.cfg, 'execution') and 
                hasattr(self.cfg.execution, 'distributed') and 
                self.cfg.execution.distributed and 
                self.cfg.settings.get('parallel_experiments', False)):
                results = self._run_parallel()
            else:
                results = self._run_sequential()

            # Process and save results
            self._save_results(results)
            self._log_study_metrics(results)

            return results
        except Exception as e:
            logger.error(f"Study execution failed: {str(e)}")
            raise
        finally:
            # Ensure MLflow run is ended
            if active_run:
                mlflow.end_run()

    def _run_sequential(self) -> Dict[str, Any]:
        """Run experiments sequentially"""
        results = {}
        
        for exp_name, runner in self.experiment_runners.items():
            try:
                logger.info(f"Running experiment: {exp_name}")
                results[exp_name] = runner.run()
            except Exception as e:
                logger.error(f"Experiment {exp_name} failed: {str(e)}")
                results[exp_name] = {"status": "failed", "error": str(e)}
                
                if not self.cfg.settings.get('continue_on_error', False):
                    raise
                    
        return results

    def _run_parallel(self) -> Dict[str, Any]:
        """Run experiments in parallel using Ray"""
        import ray
        
        if not ray.is_initialized():
            ray.init(
                num_cpus=self.cfg.execution.max_parallel,
                ignore_reinit_error=True
            )
        
        # Convert experiment runners to Ray actors
        futures = []
        for exp_name, runner in self.experiment_runners.items():
            future = runner.run.remote()
            futures.append((exp_name, future))
        
        # Collect results
        results = {}
        for exp_name, future in futures:
            try:
                results[exp_name] = ray.get(future)
            except Exception as e:
                logger.error(f"Experiment {exp_name} failed: {str(e)}")
                results[exp_name] = {"status": "failed", "error": str(e)}
                
                if not self.cfg.settings.get('continue_on_error', False):
                    break
                    
        return results

    def _log_study_params(self) -> None:
        """Log study parameters to MLflow"""
        mlflow.log_params({
            "study_name": self.cfg.metadata.name,
            "study_type": self.cfg.type.value,
            "num_experiments": len(self.experiment_runners),
            "version": self.cfg.metadata.version
        })

    def _log_study_metrics(self, results: Dict[str, Any]) -> None:
        """Log study-level metrics to MLflow"""
        metrics = {
            "completed_experiments": sum(1 for r in results.values() 
                                      if isinstance(r, dict) and r.get("status") != "failed"),
            "failed_experiments": sum(1 for r in results.values() 
                                   if isinstance(r, dict) and r.get("status") == "failed")
        }
        mlflow.log_metrics(metrics)

    def _save_study_config(self) -> None:
        """Save the study configuration"""
        config_path = self.cfg.paths.get_study_dir(self.cfg.metadata.name) / "configs" / "study_config.yaml"
        self.cfg.save(config_path)
        logger.info(f"Saved study configuration to {config_path}")

    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save study results"""
        study_dir = self.cfg.paths.get_study_dir(self.cfg.metadata.name)
        results_path = study_dir / "results" / "study_results.yaml"
        
        # Create backup if file exists and backup is enabled
        if results_path.exists() and self.cfg.settings.get('backup_existing', True):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = results_path.with_name(f"study_results_{timestamp}.yaml")
            results_path.rename(backup_path)
        
        # Serialize results using the `to_dict` method for dataclasses
        def serialize(obj: Any) -> Any:
            if hasattr(obj, "to_dict"):
                return obj.to_dict()
            elif isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [serialize(v) for v in obj]
            elif isinstance(obj, tuple):
                return tuple(serialize(v) for v in obj)
            return obj

        serialized_results = serialize(results)

        # Save results
        with open(results_path, 'w') as f:
            yaml.safe_dump(serialized_results, f, default_flow_style=False)
            
        logger.info(f"Saved study results to {results_path}")


    def cleanup(self) -> None:
        """Clean up study resources"""
        logger.info("Cleaning up study resources")
        for runner in self.experiment_runners.values():
            try:
                runner.cleanup()
            except Exception as e:
                logger.error(f"Error during cleanup: {str(e)}")