import os
import yaml
import logging
from drt_sim.core.study import Study
from drt_sim.core.study_runner import StudyRunner

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_scenarios(
    config_file: str,
    output_dir: str,
    scenarios: list = None
):
    """
    Run demand forecast error scenarios
    
    Args:
        config_file: Path to study configuration file
        output_dir: Directory to save scenario results
        scenarios: List of scenario names to run (None for all scenarios)
    """
    # Load study configuration
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create study
    study = Study.from_config(config)
    
    # Get parameter sets
    parameter_sets = study.config.parameter_sets
    
    # Filter scenarios if specified
    if scenarios:
        parameter_sets = {
            name: params
            for name, params in parameter_sets.items()
            if name in scenarios
        }
    
    # Create study runner
    runner = StudyRunner(
        study=study,
        output_dir=output_dir
    )
    
    # Run scenarios
    runner.run_parameter_sets(parameter_sets)
    
    logger.info(f"Completed running {len(parameter_sets)} scenarios")

if __name__ == '__main__':
    # Configuration
    STUDY_CONFIG = 'config/study_config.yaml'
    OUTPUT_DIR = 'studies/forecast_scenarios'
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Run scenarios
    run_scenarios(
        config_file=STUDY_CONFIG,
        output_dir=OUTPUT_DIR
    ) 