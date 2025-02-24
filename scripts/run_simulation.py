#!/usr/bin/env python3
import asyncio
import click
from pathlib import Path
import mlflow
import logging
from datetime import datetime, timedelta
import yaml
import json
from typing import List, Dict, Any
import glob

from drt_sim.config.config import StudyConfig
from drt_sim.runners.simulation_runner import SimulationRunner
from drt_sim.core.logging_config import configure_logging, cleanup_logging
from drt_sim.models.base import SimulationEncoder

logger = logging.getLogger(__name__)

#
# ─── PARAMETER SET RUNNERS ──────────────────────────────────────────────────────
#

async def run_parameter_set(
    parameter_set_name: str,
    study_config: StudyConfig,
    output_dir: Path,
    parent_run_id: str,
    is_parallel: bool = False
) -> Dict[str, Any]:
    """
    Run all replications for a parameter set. Always create a new MLflow run
    (with nested=True) for this parameter set and then set its parent run id
    manually so that the UI will show the study → parameter_set → replication hierarchy.
    """
    logger.info(f"Starting parameter set: {parameter_set_name}")
    parameter_set = study_config.get_parameter_set(parameter_set_name)

    # Always start a new nested run for the parameter set.
    experiment = mlflow.get_experiment_by_name(study_config.mlflow.experiment_name)
    with mlflow.start_run(
        run_name=parameter_set_name,
        experiment_id=experiment.experiment_id,
        tags={
            "parameter_set": parameter_set_name,
            "type": "parameter_set",
            "study": study_config.name,
            "parallel_execution": str(is_parallel)
        },
        nested=True
    ) as parameter_set_run:
        # Force the parent run id so that the run is nested under the study.
        mlflow.set_tag("mlflow.parentRunId", parent_run_id)
        
        try:
            result = await _run_parameter_set_internal(
                parameter_set, study_config, output_dir,
                parameter_set_run.info.run_id, parameter_set_name, is_parallel
            )
            return result
        except Exception as e:
            logger.error(f"Error in parameter set {parameter_set_name}: {str(e)}", exc_info=True)
            raise


async def _run_parameter_set_internal(
    parameter_set: Any,
    study_config: StudyConfig,
    output_dir: Path,
    parent_run_id: str,
    parameter_set_name: str,
    is_parallel: bool
) -> Dict[str, Any]:
    """Internal function to run the replications for a parameter set."""
    results = []

    # Create the simulation runner (each replication run will be nested further).
    runner = SimulationRunner(
        parameter_set=parameter_set,
        sim_cfg=study_config.simulation,
        output_dir=output_dir / parameter_set_name,
        run_name=parameter_set_name,
        parent_run_id=parent_run_id,
        tags={
            "study": study_config.name,
            "parameter_set": parameter_set_name
        },
        experiment_name=study_config.mlflow.experiment_name,
        is_parallel=is_parallel
    )

    # Run replications sequentially for this parameter set.
    for rep in range(parameter_set.replications):
        try:
            logger.info(
                f"Starting replication {rep + 1}/{parameter_set.replications} for parameter set {parameter_set_name}"
            )
            result = await runner.run_replication(rep + 1)
            results.append(result)
        except Exception as e:
            logger.error(f"Error in replication {rep + 1}: {str(e)}", exc_info=True)
            if not study_config.execution.continue_on_error:
                raise

    # Log aggregated metrics for the parameter set.
    param_set_metrics = _compute_parameter_set_metrics(results)
    if param_set_metrics:
        mlflow.log_metrics(param_set_metrics)

    return {
        "parameter_set": parameter_set_name,
        "results": results
    }


def _compute_parameter_set_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute aggregate metrics for a parameter set."""
    try:
        metrics = {}
        successful_runs = 0
        failed_runs = 0

        for result in results:
            if result.get("status") == "FAILED":
                failed_runs += 1
            else:
                successful_runs += 1
                if isinstance(result.get("metrics"), dict):
                    for metric_type, metric_values in result["metrics"].items():
                        if isinstance(metric_values, dict):
                            for name, value in metric_values.items():
                                if isinstance(value, (int, float)):
                                    metric_key = f"mean_{metric_type}.{name}"
                                    metrics.setdefault(metric_key, []).append(value)

        # Compute means.
        summary = {
            "successful_replications": successful_runs,
            "failed_replications": failed_runs,
            "replication_success_rate": successful_runs / len(results) if results else 0
        }
        for metric_name, values in metrics.items():
            if values:
                summary[metric_name] = sum(values) / len(values)
        return summary

    except Exception as e:
        logging.getLogger(__name__).error(f"Error computing parameter set metrics: {str(e)}")
        return {}


def run_parameter_set_sync(
    parameter_set_name: str,
    study_config: StudyConfig,
    output_dir: Path,
    parent_run_id: str,
    is_parallel: bool = False
) -> Dict[str, Any]:
    """
    Synchronous wrapper for run_parameter_set so that we can call it from a separate thread.
    """
    return asyncio.run(
        run_parameter_set(parameter_set_name, study_config, output_dir, parent_run_id, is_parallel)
    )


#
# ─── MAIN ASYNC FUNCTION ───────────────────────────────────────────────────────
#

async def main_async(
    study_name: str,
    output_dir: str,
    parameter_sets: List[str],
    max_parallel: int,
    parallel: bool
):
    """Async implementation of the main function."""
    try:
        # Set up base logging configuration once at startup
        output_path = Path(output_dir)
        base_log_dir = output_path / study_name / "logs"
        configure_logging(base_log_dir=base_log_dir, log_level=logging.DEBUG)
        logger.info(f"Configured logging to directory: {base_log_dir}")
        
        # Find and load study configuration.
        config_file = await find_study_config(study_name)
        study_config = StudyConfig.load(config_file)

        # Set up output directory.
        output_path = Path(output_dir) / study_config.name / datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path.mkdir(parents=True, exist_ok=True)

        # Configure MLflow tracking.
        if study_config.mlflow.tracking_uri.startswith("sqlite:"):
            db_path = study_config.mlflow.tracking_uri.replace("sqlite:///", "")
            db_dir = Path(db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)
            if not Path(db_path).exists():
                import sqlite3
                conn = sqlite3.connect(db_path)
                conn.close()
            mlflow.set_tracking_uri(study_config.mlflow.tracking_uri)
            if study_config.mlflow.artifact_location:
                artifact_root = Path(study_config.mlflow.artifact_location).absolute()
            else:
                artifact_root = db_dir / "artifacts"
        else:
            artifact_root = Path.cwd() / "mlruns"
            mlflow.set_tracking_uri(f"file://{artifact_root.absolute()}")

        artifact_root.mkdir(parents=True, exist_ok=True)

        # End any existing runs.
        if mlflow.active_run():
            mlflow.end_run()

        # Get or create the experiment.
        experiment_name = study_config.mlflow.experiment_name
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.info(f"Creating new experiment: {experiment_name}")
            experiment_id = mlflow.create_experiment(
                name=experiment_name,
                artifact_location=f"file://{(artifact_root / experiment_name).absolute()}",
                tags=study_config.mlflow.tags
            )
        else:
            experiment_id = experiment.experiment_id
            logger.info(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")
            client = mlflow.tracking.MlflowClient()
            for key, value in study_config.mlflow.tags.items():
                client.set_experiment_tag(experiment_id, key, value)

        mlflow.set_experiment(experiment_name)
        logger.info("MLflow Configuration:")
        logger.info(f"  Tracking URI: {mlflow.get_tracking_uri()}")
        logger.info(f"  Artifact Location: {experiment.artifact_location}")
        logger.info(f"  Experiment ID: {experiment_id}")
        logger.info(f"  Experiment Name: {experiment_name}")

        # Start the parent (study) run.
        with mlflow.start_run(
            run_name=study_config.name,
            experiment_id=experiment_id,
            tags={
                "study_name": study_config.name,
                "config_file": str(config_file),
                "type": "study",
                **study_config.mlflow.tags
            }
        ) as parent_run:
            # Log study parameters.
            mlflow.log_params({
                "config_file": str(config_file),
                "output_dir": str(output_path),
                "start_time": datetime.now().isoformat(),
                "parameter_sets": ",".join(parameter_sets) if parameter_sets else "all",
                "max_parallel": str(max_parallel or "auto"),
                "mlflow_tracking_uri": mlflow.get_tracking_uri(),
                "mlflow_artifact_location": str(artifact_root)
            })

            # Save and log study configuration.
            config_output = output_path / "study_config.yaml"
            with open(config_output, "w") as f:
                yaml.dump(study_config.to_dict(), f)
            mlflow.log_artifact(str(config_output), "config")

            # Determine which parameter sets to run.
            param_sets_to_run = (
                list(parameter_sets) if parameter_sets
                else list(study_config.parameter_sets.keys())
            )
            logger.info(f"Starting study '{study_config.name}' with {len(param_sets_to_run)} parameter sets")

            # Run simulations either sequentially or in parallel.
            results = []
            if parallel:
                # In parallel mode, run each parameter set in its own thread to isolate MLflow context.
                tasks = [
                    asyncio.to_thread(
                        run_parameter_set_sync,
                        param_set,
                        study_config,
                        output_path,
                        parent_run.info.run_id,
                        True
                    )
                    for param_set in param_sets_to_run
                ]
                results = await asyncio.gather(*tasks)
            else:
                # Sequentially run each parameter set.
                for param_set in param_sets_to_run:
                    try:
                        result = await run_parameter_set(
                            param_set,
                            study_config,
                            output_path,
                            parent_run.info.run_id,
                            False
                        )
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Error running parameter set {param_set}: {str(e)}", exc_info=True)
                        if not study_config.execution.continue_on_error:
                            raise
            # Log summary metrics for the study.
            summary_metrics = _compute_study_summary_metrics(results)
            if summary_metrics:
                mlflow.log_metrics(summary_metrics)

            logger.info(f"Study completed. Results saved to {output_path}")

    except Exception as e:
        logger.error("Error running study", exc_info=True)
        raise click.ClickException(str(e))
    finally:
        cleanup_logging()


async def find_study_config(study_name: str) -> Path:
    """Find the study configuration file by name."""
    config_dir = Path(__file__).parent.parent / "studies" / "configs"
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


def _compute_study_summary_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute summary metrics for the entire study."""
    try:
        metrics = {}
        successful_runs = 0
        failed_runs = 0

        for result in results:
            if isinstance(result, dict):
                if result.get("status") == "failed":
                    failed_runs += 1
                else:
                    successful_runs += 1

                if "results" in result and isinstance(result["results"], list):
                    for run_result in result["results"]:
                        if isinstance(run_result.get("metrics"), dict):
                            for metric_type, metric_values in run_result["metrics"].items():
                                if isinstance(metric_values, dict):
                                    for name, value in metric_values.items():
                                        if isinstance(value, (int, float)):
                                            metric_key = f"mean_{metric_type}.{name}"
                                            metrics.setdefault(metric_key, []).append(value)

        summary = {
            "successful_runs": successful_runs,
            "failed_runs": failed_runs,
            "success_rate": successful_runs / (successful_runs + failed_runs) if (successful_runs + failed_runs) > 0 else 0
        }
        for metric_name, values in metrics.items():
            if values:
                summary[metric_name] = sum(values) / len(values)
        return summary

    except Exception as e:
        logger.error(f"Error computing summary metrics: {str(e)}")
        return {}


@click.command()
@click.argument('study_name', type=str)
@click.option('--output-dir', type=click.Path(), default='studies/results',
              help='Directory for simulation outputs')
@click.option('--parameter-sets', '-p', multiple=True,
              help='Specific parameter sets to run (default: all)')
@click.option('--max-parallel', type=int, default=None,
              help='Maximum number of parallel simulations')
@click.option('--parallel', is_flag=True, default=False,
              help='Run parameter sets in parallel')
def main(study_name: str, output_dir: str, parameter_sets: List[str], max_parallel: int, parallel: bool):
    """Run simulations based on study configuration."""
    asyncio.run(main_async(study_name, output_dir, parameter_sets, max_parallel, parallel))


if __name__ == "__main__":
    main()
