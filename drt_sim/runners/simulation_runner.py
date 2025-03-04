from pathlib import Path
from typing import Dict, Any, Optional
import logging
import mlflow
import json
from drt_sim.core.simulation.orchestrator import SimulationOrchestrator
from drt_sim.config.config import ParameterSet, SimulationConfig
from drt_sim.models.state import SimulationStatus
from drt_sim.models.base import SimulationEncoder
from drt_sim.core.monitoring.metrics.collector import MetricsCollector
from drt_sim.core.monitoring.metrics.manager import MetricsManager
from drt_sim.core.logging_config import add_replication_file_handler
import traceback
from datetime import timedelta

logger = logging.getLogger(__name__)

class SimulationRunner:
    """Unified runner for executing simulations with MLflow tracking."""

    def __init__(
        self,
        parameter_set: ParameterSet,
        sim_cfg: SimulationConfig,
        output_dir: Path,
        run_name: Optional[str] = None,
        parent_run_id: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        mlruns_dir: Optional[str] = None,
        experiment_name: Optional[str] = None,
        is_parallel: bool = False
    ):
        """
        Initialize the simulation runner.
        """
        self.parameter_set = parameter_set
        self.sim_cfg = sim_cfg
        self.output_dir = Path(output_dir)
        self.run_name = run_name or parameter_set.name
        self.parent_run_id = parent_run_id
        self.tags = tags or {}
        self.experiment_name = experiment_name or self.run_name
        self.is_parallel = is_parallel

        self.tags.update({
            "parallel_execution": str(is_parallel).lower(),
            "parameter_set": parameter_set.name,
            "description": parameter_set.description
        })

        # Create output directories.
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Note: We don't create logs directory here as it's managed by configure_logging
        (self.output_dir / "results").mkdir(exist_ok=True)
        (self.output_dir / "artifacts").mkdir(exist_ok=True)

        # Initialize components.
        self.orchestrator: Optional[SimulationOrchestrator] = None
        self.metrics_collector: Optional[MetricsCollector] = None
        self.active_run: Optional[mlflow.ActiveRun] = None
        self._run_id: Optional[str] = None
        self._experiment_id: Optional[str] = None

        # Get experiment ID from MLflow.
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment {self.experiment_name} not found. Please create it first.")
        self._experiment_id = experiment.experiment_id

        # Add extra parameter set tags.
        self.tags.update({
            "parameter_set": parameter_set.name,
            "description": parameter_set.description,
            **(dict(zip(parameter_set.tags, parameter_set.tags)))
        })

    def _initialize_components(self, replication: int) -> None:
        """Initialize simulation components."""
        replication_id = f"rep_{replication}"
        
        self.metrics_collector = MetricsCollector(
            base_path=str(self.output_dir),
            replication_id=replication_id
        )
        
        self.orchestrator = SimulationOrchestrator(
            cfg=self.parameter_set,
            sim_cfg=self.sim_cfg,
            output_dir=self.output_dir,
            metrics_collector=self.metrics_collector,
            replication_id=replication_id
        )

    async def run_replication(self, replication: int) -> Dict[str, Any]:
        """
        Run a single replication of the simulation.
        """
        replication_handlers = []
        try:
            # Always start a new nested MLflow run for this replication.
            with mlflow.start_run(
                run_name=f"{self.run_name}_rep_{replication}",
                experiment_id=self._experiment_id,
                nested=True,
                tags={
                    **self.tags,
                    "replication": str(replication),
                    "type": "replication",
                    "run_type": "replication"
                }
            ) as replication_run:
                replication_id = f"rep_{replication}"
                # Use the parameter set specific logs directory
                logs_dir = self.output_dir / "logs"
                replication_handlers = add_replication_file_handler(logs_dir, replication_id)
                logger.info(f"Started logging for replication {replication} in {logs_dir}/replications/{replication_id}")
                
                # Force the parent_run_id so that the run is shown as a child of the parameter set.
                mlflow.set_tag("mlflow.parentRunId", self.parent_run_id)
                self.active_run = replication_run
                self._run_id = replication_run.info.run_id

                logger.info(f"Starting replication {replication} for parameter set {self.run_name}")

                # Log configurations.
                mlflow.log_params({
                    "parameter_set": self.parameter_set.name,
                    "replication": replication,
                    **self._flatten_config(self.parameter_set)
                })
                mlflow.log_dict(self.sim_cfg.to_dict(), "simulation_config.yaml")

                # Initialize and run the simulation.
                self._initialize_components(replication)
                await self.orchestrator.initialize()
                results = await self._run_simulation_loop()

                # Initialize metrics manager and analyze results
                metrics_manager = MetricsManager(
                    metrics_collector=self.metrics_collector,
                    output_dir=self.output_dir,
                    replication_id=replication_id
                )

                # Analyze all metrics
                metrics_manager.generate_all_analysis()

                # Log results.
                self._log_results(results)
                results["status"] = "completed"

                # Clean up analysis resources
                metrics_manager.cleanup()
                return results

        except Exception as e:
            logger.error(f"Error in replication {replication}: {traceback.format_exc()}")
            if self.active_run:
                try:
                    mlflow.set_tag("status", "failed")
                    mlflow.set_tag("error", str(e))
                except Exception as mlflow_error:
                    logger.error(f"Error setting failure tags: {str(mlflow_error)}")
            raise

        finally:
            # Clean up all handlers
            for handler in replication_handlers:
                logging.getLogger().removeHandler(handler)
                handler.close()
            self.cleanup()

    async def _run_simulation_loop(self) -> Dict[str, Any]:
        """Execute the main simulation loop."""
        step_results = []
        try:
            while self.orchestrator.context.status not in [
                SimulationStatus.COMPLETED,
                SimulationStatus.FAILED,
                SimulationStatus.STOPPED
            ]:
                step_result = await self.orchestrator.step()
                step_result = step_result.to_dict()
                step_results.append(step_result)

                if self.orchestrator.context.current_time.second % self.sim_cfg.time_step == 0:
                    logger.debug(f"Simulation time: {self.orchestrator.context.current_time}")

                if self.orchestrator.context.current_time >= self.orchestrator.context.end_time:
                    self.orchestrator.context.status = SimulationStatus.COMPLETED
                    break

            results = {
                "step_results": step_results,
                "final_state": self.orchestrator.get_state().to_dict(),
                "simulation_time": self.orchestrator.context.current_time.isoformat(),
                "status": self.orchestrator.context.status.value
            }
            return results

        except Exception as e:
            self.orchestrator.context.status = SimulationStatus.FAILED
            logger.error("Error during simulation execution", exc_info=True)
            raise

    def _log_results(self, results: Dict[str, Any]) -> None:
        """Log simulation results to MLflow."""
        try:
            if isinstance(results.get("metrics"), dict):
                metrics = {}
                for metric_type, metric_values in results["metrics"].items():
                    if isinstance(metric_values, dict):
                        for name, value in metric_values.items():
                            if isinstance(value, (int, float)):
                                metrics[f"{metric_type}.{name}"] = value
                if metrics:
                    mlflow.log_metrics(metrics)

            # Get event history from the event manager and log it
            if self.orchestrator and self.orchestrator.event_manager:
                event_history = self.orchestrator.event_manager.get_serializable_history()
                event_history_path = self.output_dir / "artifacts" / f"{self.run_name}_events.json"
                with open(event_history_path, 'w') as f:
                    json.dump(event_history, f, indent=2, cls=SimulationEncoder)
                mlflow.log_artifact(str(event_history_path), f"replications/{self.run_name}/events")

            mlflow.log_artifacts(str(self.output_dir / "logs"), f"replications/{self.run_name}/logs")
            if "final_state" in results:
                mlflow.log_dict(results["final_state"], f"replications/{self.run_name}/final_state.json")
            mlflow.set_tag("simulation_status", results.get("status", "unknown"))

        except Exception as e:
            logger.error(f"Error logging results to MLflow: {str(e)}")
            raise

    def cleanup(self) -> None:
        """Clean up simulation resources."""
        try:
            if self.orchestrator:
                self.orchestrator.cleanup()
                self.orchestrator = None
                
            logger.info("Simulation cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            # Don't re-raise the exception to allow for graceful cleanup

    def _flatten_config(self, config: Any, prefix: str = "") -> Dict[str, Any]:
        """Flatten nested configuration for MLflow logging."""
        params = {}
        if hasattr(config, "__dict__"):
            for key, value in config.__dict__.items():
                if key.startswith("_"):
                    continue
                param_name = f"{prefix}{key}" if prefix else key
                if hasattr(value, "__dict__"):
                    params.update(self._flatten_config(value, f"{param_name}."))
                elif isinstance(value, (int, float, str, bool)):
                    params[param_name] = value
        return params
