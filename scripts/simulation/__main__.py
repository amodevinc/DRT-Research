"""
Main entry point for simulation runs.
"""
import asyncio
import click
from typing import List

from drt_sim.runners.study_runner import StudyRunner

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
    runner = StudyRunner(
        study_name=study_name,
        output_dir=output_dir,
        parameter_sets=parameter_sets,
        max_parallel=max_parallel,
        parallel=parallel
    )
    asyncio.run(runner.run())

if __name__ == "__main__":
    main() 