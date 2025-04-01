"""
Study execution for simulation runs.
"""
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any
import yaml
import glob
import click
from datetime import datetime

from drt_sim.config.config import StudyConfig
from drt_sim.core.logging_config import configure_logging, cleanup_logging
from scripts.simulation.mlflow_manager import MLflowManager
from drt_sim.runners.parameter_set_runner import ParameterSetRunner
from scripts.simulation.metrics import compute_study_summary_metrics

logger = logging.getLogger(__name__)

def find_study_config(study_name: str) -> Path:
    """Find the study configuration file by name."""
    config_dir = Path(__file__).parent.parent.parent / "studies" / "configs"
    study_files = glob.glob(str(config_dir / "*.yaml"))
    for file_path in study_files:
        try:
            with open(file_path, 'r') as f:
                config = yaml.safe_load(f)
            config_name = (
                config.get('name') or
                config.get('metadata', {}).get('name') or
                Path(file_path).stem
            )
            if config_name == study_name:
                return Path(file_path)
        except Exception as e:
            logger.warning(f"Error reading config file {file_path}: {str(e)}")
            continue
    raise click.ClickException(f"No study configuration found with name: {study_name}")

class StudyRunner:
    def __init__(
        self,
        study_name: str,
        output_dir: str,
        parameter_sets: List[str],
        max_parallel: int,
        parallel: bool
    ):
        self.study_name = study_name
        self.output_dir = output_dir
        self.parameter_sets = parameter_sets
        self.max_parallel = max_parallel
        self.parallel = parallel
        self.study_config = None
        self.mlflow_manager = None
        self.output_path = None

    async def run(self) -> None:
        """Run the study with all parameter sets."""
        try:
            await self._setup()
            await self._execute()
        except Exception as e:
            logger.error("Error running study", exc_info=True)
            raise click.ClickException(str(e))
        finally:
            cleanup_logging()

    async def _setup(self) -> None:
        """Set up the study environment."""
        # Set up base logging configuration
        output_path = Path(self.output_dir)
        base_log_dir = output_path / self.study_name / "logs"
        configure_logging(base_log_dir=base_log_dir, log_level=logging.DEBUG)
        logger.info(f"Configured logging to directory: {base_log_dir}")
        
        # Find and load study configuration
        config_file = find_study_config(self.study_name)
        self.study_config = StudyConfig.load(config_file)

        # Set up output directory
        self.output_path = Path(self.output_dir) / self.study_config.name / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Set up MLflow
        self.mlflow_manager = MLflowManager(self.study_config)
        self.mlflow_manager.setup()

        # Save and log study configuration
        config_output = self.output_path / "study_config.yaml"
        with open(config_output, "w") as f:
            yaml.dump(self.study_config.to_dict(), f)
        self.mlflow_manager.log_artifact(str(config_output), "config")

    async def _execute(self) -> None:
        """Execute the study runs."""
        # Determine which parameter sets to run
        param_sets_to_run = (
            list(self.parameter_sets) if self.parameter_sets
            else list(self.study_config.parameter_sets.keys())
        )
        logger.info(f"Starting study '{self.study_config.name}' with {len(param_sets_to_run)} parameter sets")

        # Start the parent (study) run
        with self.mlflow_manager.start_study_run(
            find_study_config(self.study_name),
            self.output_path
        ) as parent_run:
            # Run simulations either sequentially or in parallel
            results = []
            if self.parallel:
                tasks = [
                    self._run_parameter_set(param_set)
                    for param_set in param_sets_to_run
                ]
                results = await asyncio.gather(*tasks)
            else:
                for param_set in param_sets_to_run:
                    try:
                        result = await self._run_parameter_set(param_set)
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Error running parameter set {param_set}: {str(e)}", exc_info=True)
                        if not self.study_config.execution.continue_on_error:
                            raise

            # Log summary metrics for the study
            summary_metrics = compute_study_summary_metrics(results)
            if summary_metrics:
                self.mlflow_manager.log_metrics(summary_metrics)

            logger.info(f"Study completed. Results saved to {self.output_path}")

    async def _run_parameter_set(self, parameter_set_name: str) -> Dict[str, Any]:
        """Run a single parameter set."""
        runner = ParameterSetRunner(
            parameter_set_name=parameter_set_name,
            study_config=self.study_config,
            output_dir=self.output_path,
            mlflow_manager=self.mlflow_manager,
            is_parallel=self.parallel
        )
        return await runner.run() 