# core/simulation/context.py
from datetime import datetime, timedelta
from typing import Optional
import logging

from ...models.simulation import SimulationStatus

logger = logging.getLogger(__name__)

class SimulationContext:
    """
    Manages the simulation context including time progression and simulation status.
    Provides centralized access to simulation state information.
    """
    
    def __init__(
        self,
        start_time: datetime,
        end_time: datetime,
        time_step: timedelta,
        warm_up_duration: timedelta
    ):
        """
        Initialize simulation context.
        
        Args:
            start_time: Simulation start time
            end_time: Simulation end time
            time_step: Time step for simulation progression
            warm_up_duration: Warm-up period duration
        """
        self.start_time = start_time
        self.end_time = end_time
        self.time_step = time_step
        self.warm_up_duration = warm_up_duration
        
        self.current_time: datetime = start_time
        self.status: SimulationStatus = SimulationStatus.INITIALIZED
        
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
        """Advance simulation time by one time step"""
        self.current_time += self.time_step
        
        # Check for warm-up completion
        if (self.status == SimulationStatus.WARMING_UP and 
            self.current_time >= self.warm_up_end_time):
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