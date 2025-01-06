import os
import sys
import logging
import yaml
import argparse
from pathlib import Path
from drt_sim.runners.study_runner import StudyRunner
from drt_sim.config.config import StudyConfig, StudyType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_study_config(study_name: str) -> StudyConfig:
    """Load study configuration from YAML file"""
    config_dir = Path("drt_sim/studies/configs")
    config_file = config_dir / f"{study_name}.yaml"
    
    if not config_file.exists():
        raise FileNotFoundError(
            f"Study configuration file not found: {config_file}\n"
            f"Available studies: {list_available_studies()}"
        )
    
    return StudyConfig.load(config_file)

def list_available_studies() -> list:
    """List all available study configurations."""
    study_dir = Path("drt_sim/studies/configs")
    if not study_dir.exists():
        return []
    return [f.stem for f in study_dir.glob("*.yaml")]

def validate_study_config(config: StudyConfig) -> None:
    """Validate the study configuration."""
    if not config.metadata.name:
        raise ValueError("Study name is required in metadata")
    if config.type == StudyType.PARAMETER_SWEEP and not config.parameter_sweep:
        raise ValueError("Parameter sweep configuration is required for parameter sweep studies")
    if not config.experiments and config.type != StudyType.PARAMETER_SWEEP:
        raise ValueError("At least one experiment configuration is required for non-parameter sweep studies")

def main():
    parser = argparse.ArgumentParser(
        description='''
DRT Simulation Study Runner

This script runs simulation studies based on YAML configuration files. 
Studies are stored in drt_sim/studies/configs/ directory as YAML files.
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('study_name', nargs='?', help='Name of the study config to use (without .yaml extension)')
    parser.add_argument('--list', '-l', action='store_true', help='List all available study configurations')
    
    args = parser.parse_args()

    if args.list:
        studies = list_available_studies()
        if not studies:
            print("No studies found in drt_sim/studies/configs/")
            return
        print("\nAvailable studies:")
        for study in sorted(studies):
            print(f"  - {study}")
        print("\nUse: python run_study.py <study_name> to run a specific study")
        return

    if not args.study_name:
        parser.error("study_name is required when not using --list")

    try:
        # Load and validate study configuration
        study_config = load_study_config(args.study_name)
        validate_study_config(study_config)
        
        logger.info(f"Running study: {study_config.metadata.name}")
        logger.info(f"Study type: {study_config.type.value}")
        logger.info(f"Description: {study_config.metadata.description}")
        
        # Create and run study
        runner = StudyRunner(study_config)
        try:
            runner.setup()
            results = runner.run()
            logger.info("Study completed successfully")
            return results
        finally:
            runner.cleanup()
    except Exception as e:
        logger.error("Study failed with error", exc_info=True)
        raise

if __name__ == "__main__":
    main()