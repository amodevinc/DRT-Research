"""
MLflow configuration and management for simulation runs.
"""
import mlflow
import logging
from pathlib import Path
from typing import Dict, Any
import yaml
from datetime import datetime
import sqlite3

logger = logging.getLogger(__name__)

class MLflowManager:
    def __init__(self, study_config: Any):
        self.study_config = study_config
        self.experiment_id = None
        self.parent_run_id = None
        self.artifact_root = None

    def setup(self) -> None:
        """Configure MLflow tracking and create/get experiment."""
        self._setup_tracking_uri()
        self._setup_experiment()

    def _setup_tracking_uri(self) -> None:
        """Set up MLflow tracking URI and artifact location."""
        if self.study_config.mlflow.tracking_uri.startswith("sqlite:"):
            self._setup_sqlite_tracking()
        else:
            self.artifact_root = Path.cwd() / "mlruns"
            mlflow.set_tracking_uri(f"file://{self.artifact_root.absolute()}")

        self.artifact_root.mkdir(parents=True, exist_ok=True)

    def _setup_sqlite_tracking(self) -> None:
        """Set up SQLite-based MLflow tracking."""
        db_path = self.study_config.mlflow.tracking_uri.replace("sqlite:///", "")
        db_dir = Path(db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        
        if not Path(db_path).exists():
            conn = sqlite3.connect(db_path)
            conn.close()
            
        mlflow.set_tracking_uri(self.study_config.mlflow.tracking_uri)
        self.artifact_root = (
            Path(self.study_config.mlflow.artifact_location).absolute()
            if self.study_config.mlflow.artifact_location
            else db_dir / "artifacts"
        )

    def _setup_experiment(self) -> None:
        """Create or get experiment and set up tags."""
        experiment_name = self.study_config.mlflow.experiment_name
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            logger.info(f"Creating new experiment: {experiment_name}")
            self.experiment_id = mlflow.create_experiment(
                name=experiment_name,
                artifact_location=f"file://{(self.artifact_root / experiment_name).absolute()}",
                tags=self.study_config.mlflow.tags
            )
        else:
            self.experiment_id = experiment.experiment_id
            logger.info(f"Using existing experiment: {experiment_name} (ID: {self.experiment_id})")
            client = mlflow.tracking.MlflowClient()
            for key, value in self.study_config.mlflow.tags.items():
                client.set_experiment_tag(self.experiment_id, key, value)

        mlflow.set_experiment(experiment_name)

    def _create_run(self, run_name: str, tags: Dict[str, str], nested: bool = False) -> mlflow.ActiveRun:
        """Create a new MLflow run with the given parameters."""
        if not nested and mlflow.active_run():
            mlflow.end_run()

        return mlflow.start_run(
            run_name=run_name,
            experiment_id=self.experiment_id,
            tags=tags,
            nested=nested
        )

    def start_study_run(self, config_file: Path, output_path: Path) -> mlflow.ActiveRun:
        """Start a new MLflow run for the study."""
        run = self._create_run(
            run_name=self.study_config.name,
            tags={
                "study_name": self.study_config.name,
                "config_file": str(config_file),
                "type": "study",
                **self.study_config.mlflow.tags
            }
        )
        self.parent_run_id = run.info.run_id

        mlflow.log_params({
            "config_file": str(config_file),
            "output_dir": str(output_path),
            "start_time": datetime.now().isoformat(),
            "mlflow_tracking_uri": mlflow.get_tracking_uri(),
            "mlflow_artifact_location": str(self.artifact_root)
        })

        return run

    def start_parameter_set_run(self, parameter_set_name: str, is_parallel: bool) -> mlflow.ActiveRun:
        """Start a new MLflow run for a parameter set."""
        run = self._create_run(
            run_name=parameter_set_name,
            tags={
                "parameter_set": parameter_set_name,
                "type": "parameter_set",
                "study": self.study_config.name,
                "parallel_execution": str(is_parallel)
            },
            nested=True
        )
        mlflow.set_tag("mlflow.parentRunId", self.parent_run_id)
        return run

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log metrics to MLflow."""
        if metrics:
            mlflow.log_metrics(metrics)

    def log_artifact(self, local_path: str, artifact_path: str) -> None:
        """Log an artifact to MLflow."""
        mlflow.log_artifact(local_path, artifact_path) 