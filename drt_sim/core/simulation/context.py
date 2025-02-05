# core/simulation/context.py
from datetime import datetime, timedelta
import logging
from typing import Optional

from drt_sim.models.simulation import SimulationStatus
from drt_sim.core.events.manager import EventManager
from drt_sim.core.monitoring.metrics_collector import MetricsCollector

logger = logging.getLogger(__name__)

class SimulationContext:
    """
    Maintains simulation context including time, status, and configuration.
    Acts as a central point for accessing simulation state and services.
    """
    
    def __init__(
        self,
        start_time: datetime,
        end_time: datetime,
        time_step: timedelta,
        warm_up_duration: timedelta,
        event_manager: EventManager,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        """
        Initialize simulation context.
        
        Args:
            start_time: Simulation start time
            end_time: Simulation end time
            time_step: Time step for simulation advancement
            warm_up_duration: Duration of warm-up period
            event_manager: Event manager instance
            metrics_collector: Optional metrics collector instance
        """
        self.start_time = start_time
        self.end_time = end_time
        self.current_time = start_time
        self.time_step = time_step
        self.warm_up_duration = warm_up_duration
        self.event_manager = event_manager
        self.metrics_collector = metrics_collector
        self.status = SimulationStatus.WARMING_UP
        
        # Derived times
        self.warm_up_end_time = start_time + warm_up_duration
        
        # Validation
        self._validate_times()
        
    def _validate_times(self) -> None:
        """Validate time configurations"""
        if self.start_time >= self.end_time:
            raise ValueError("Start time must be before end time")
        
        if self.time_step.total_seconds() <= 0:
            raise ValueError("Time step must be positive")
        
        if self.warm_up_duration.total_seconds() < 0:
            raise ValueError("Warm-up duration cannot be negative")
        
        if self.warm_up_end_time > self.end_time:
            raise ValueError("Warm-up period extends beyond simulation end time")
    
    def advance_time(self) -> None:
        """Advance simulation time by one time step."""
        self.current_time += self.time_step
        
        # Check if warm-up period is complete
        if (self.current_time - self.start_time >= self.warm_up_duration and 
            self.status == SimulationStatus.WARMING_UP):
            self.status = SimulationStatus.RUNNING
            logger.info("Warm-up period completed")
        
        # Check for simulation completion
        if self.current_time >= self.end_time:
            self.status = SimulationStatus.COMPLETED
            logger.info("Simulation completed")
    
    @property
    def is_warmed_up(self) -> bool:
        """Check if warm-up period is completed"""
        return self.current_time >= self.warm_up_end_time
    
    @property
    def is_completed(self) -> bool:
        """Check if simulation is completed"""
        return self.current_time >= self.end_time or self.status == SimulationStatus.COMPLETED
    
    @property
    def simulation_time(self) -> timedelta:
        """Get current simulation time as duration from start"""
        return self.current_time - self.start_time
    
    @property
    def progress(self) -> float:
        """Get simulation progress as percentage"""
        total_duration = (self.end_time - self.start_time).total_seconds()
        elapsed = (self.current_time - self.start_time).total_seconds()
        return (elapsed / total_duration) * 100 if total_duration > 0 else 0

    def get_elapsed_time(self) -> timedelta:
        """Get elapsed simulation time."""
        return self.current_time - self.start_time
        
    def get_remaining_time(self) -> timedelta:
        """Get remaining simulation time."""
        return self.end_time - self.current_time