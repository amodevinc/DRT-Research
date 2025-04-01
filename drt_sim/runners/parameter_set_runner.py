"""
Parameter set execution for simulation runs.
"""
import logging
from pathlib import Path
from typing import Dict, Any, List

from drt_sim.runners.simulation_runner import SimulationRunner
from scripts.simulation.mlflow_manager import MLflowManager
from scripts.simulation.metrics import compute_parameter_set_metrics

logger = logging.getLogger(__name__)

class ParameterSetRunner:
    def __init__(
        self,
        parameter_set_name: str,
        study_config: Any,
        output_dir: Path,
        mlflow_manager: MLflowManager,
        is_parallel: bool = False
    ):
        self.parameter_set_name = parameter_set_name
        self.study_config = study_config
        self.output_dir = output_dir
        self.mlflow_manager = mlflow_manager
        self.is_parallel = is_parallel
        self.parameter_set = study_config.get_parameter_set(parameter_set_name)

    async def run(self) -> Dict[str, Any]:
        """Run all replications for a parameter set."""
        logger.info(f"Starting parameter set: {self.parameter_set_name}")

        with self.mlflow_manager.start_parameter_set_run(
            self.parameter_set_name,
            self.is_parallel
        ) as parameter_set_run:
            try:
                results = await self._run_replications()
                metrics = compute_parameter_set_metrics(results)
                self.mlflow_manager.log_metrics(metrics)
                
                return {
                    "parameter_set": self.parameter_set_name,
                    "results": results
                }
            except Exception as e:
                logger.error(f"Error in parameter set {self.parameter_set_name}: {str(e)}", exc_info=True)
                raise

    async def _run_replications(self) -> List[Dict[str, Any]]:
        """Run all replications for this parameter set."""
        results = []
        runner = SimulationRunner(
            parameter_set=self.parameter_set,
            sim_cfg=self.study_config.simulation,
            output_dir=self.output_dir / self.parameter_set_name,
            run_name=self.parameter_set_name,
            parent_run_id=self.mlflow_manager.parent_run_id,
            tags={
                "study": self.study_config.name,
                "parameter_set": self.parameter_set_name
            },
            experiment_name=self.study_config.mlflow.experiment_name,
            is_parallel=self.is_parallel
        )

        for rep in range(self.parameter_set.replications):
            try:
                logger.info(
                    f"Starting replication {rep + 1}/{self.parameter_set.replications} "
                    f"for parameter set {self.parameter_set_name}"
                )
                result = await runner.run_replication(rep + 1)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in replication {rep + 1}: {str(e)}", exc_info=True)
                if not self.study_config.execution.continue_on_error:
                    raise

        return results 