#!/usr/bin/env python3

import os
import sys
import logging
import yaml
import argparse
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from drt_sim.runners.study_runner import StudyRunner
from drt_sim.config.config import StudyConfig, StudyType
from drt_sim.core.paths import create_simulation_environment, SimulationPaths
from drt_sim.core.logging_config import setup_logger

logger = setup_logger(__name__)

def load_study_config(study_name: str) -> StudyConfig:
    """
    Load study configuration from YAML file.
    
    Args:
        study_name: Name of the study configuration to load
        
    Returns:
        StudyConfig object containing the study configuration
        
    Raises:
        FileNotFoundError: If the study configuration file is not found
    """
    config_file = Path("studies/configs") / f"{study_name}.yaml"
    
    if not config_file.exists():
        raise FileNotFoundError(
            f"Study configuration file not found: {config_file}\n"
            f"Available studies: {list_available_studies()}"
        )
        
    try:
        logger.info(f"Loading study configuration from {config_file}")
        return StudyConfig.load(config_file)
    except Exception as e:
        logger.error(f"Failed to load study configuration: {str(e)}")
        raise

def list_available_studies() -> list:
    """
    List all available study configurations.
    
    Returns:
        List of available study names (without .yaml extension)
    """
    configs = Path("studies/configs")
    
    if not configs.exists():
        return []
        
    return sorted([f.stem for f in configs.glob("*.yaml")])

def setup_logging(sim_paths: SimulationPaths, study_name: Optional[str] = None) -> None:
    """
    Configure logging for the study runner.
    
    Args:
        sim_paths: SimulationPaths object
        study_name: Optional name of the study to include in log filename
    """
    # Create log directory
    log_dir = sim_paths.root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_prefix = f"{study_name}_{timestamp}" if study_name else f"study_run_{timestamp}"
    log_file = log_dir / f"{file_prefix}.log"
    
    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all logs
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # File handler for DEBUG logs
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)  # Log everything to the file
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler for INFO logs and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Log only INFO and above to console
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Log system information
    logger = logging.getLogger(__name__)
    logger.info("Starting DRT Simulation Platform")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Log file: {log_file}")


def validate_study_config(config: StudyConfig) -> None:
    """
    Validate the study configuration.
    
    Args:
        config: Study configuration to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    if not config.metadata.name:
        raise ValueError("Study name is required in metadata")
        
    if config.type == StudyType.PARAMETER_SWEEP and not config.parameter_sweep:
        raise ValueError("Parameter sweep configuration is required for parameter sweep studies")
        
    if not config.experiments and config.type != StudyType.PARAMETER_SWEEP:
        raise ValueError("At least one experiment configuration is required for non-parameter sweep studies")
        
    logger.info("Study configuration validation successful")

async def run_study(study_config: StudyConfig) -> Dict[str, Any]:
    """
    Run a study asynchronously.
    
    Args:
        study_config: Configuration for the study to run
        
    Returns:
        Dict containing study results
        
    Raises:
        Exception: If study execution fails
    """
    logger.info(f"Running study: {study_config.metadata.name}")
    logger.info(f"Study type: {study_config.type.value}")
    logger.info(f"Description: {study_config.metadata.description}")
    
    # Create and run study
    runner = StudyRunner(study_config)
    try:
        runner.setup()
        results = await runner.run()
        logger.info("Study completed successfully")
        return results
    except Exception as e:
        logger.error("Study execution failed", exc_info=True)
        raise
    finally:
        runner.cleanup()

async def async_main() -> Optional[Dict[str, Any]]:
    """
    Async main function handling command line arguments and study execution.
    
    Returns:
        Optional[Dict[str, Any]]: Study results if successful, None otherwise
    """
    parser = argparse.ArgumentParser(
        description='''
        DRT Simulation Study Runner
        
        This script runs simulation studies based on YAML configuration files.
        Studies are stored in the studies/configs directory as YAML files.
        Results, logs, and artifacts are organized in the studies directory.
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('study_name', nargs='?', 
                       help='Name of the study config to use (without .yaml extension)')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List all available study configurations')
    parser.add_argument('--clean', '-c', action='store_true',
                       help='Clean previous study results before running')
    
    args = parser.parse_args()
    
    # Initialize simulation environment
    sim_paths = create_simulation_environment()
    
    try:
        if args.list:
            studies = list_available_studies()
            if not studies:
                print("\nNo studies found in studies/configs/")
                return None
                
            print("\nAvailable studies:")
            for study in studies:
                config_path = Path("studies/configs") / f"{study}.yaml"
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                    description = config.get('metadata', {}).get('description', 'No description')
                print(f" - {study}: {description}")
            print("\nUse: python run_study.py <study_name> to run a specific study")
            return None
            
        if not args.study_name:
            parser.error("study_name is required when not using --list")
            
        # Load and validate study configuration
        study_config = load_study_config(args.study_name)
        setup_logging(sim_paths, study_config.metadata.name)
        logger.info(f"Study Config Loaded: {study_config}")
        validate_study_config(study_config)
        
        # Clean previous results if requested
        if args.clean:
            study_dir = sim_paths.studies / f"{study_config.metadata.name}"
            if study_dir.exists():
                import shutil
                logger.info(f"Cleaning previous study results: {study_dir}")
                shutil.rmtree(study_dir)
        
        # Run study asynchronously
        results = await run_study(study_config)
        return results
        
    except Exception as e:
        logger.error("Fatal error in main execution", exc_info=True)
        raise

def main() -> Optional[Dict[str, Any]]:
    """
    Synchronous entry point that runs the async main.
    
    Returns:
        Optional[Dict[str, Any]]: Study results if successful, None otherwise
    """
    try:
        # Get or create event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If no loop exists, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run the async main
        results = loop.run_until_complete(async_main())
        
        # Close the loop
        loop.close()
        
        return results
        
    except KeyboardInterrupt:
        logger.info("Study execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error("Fatal error in main execution", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()