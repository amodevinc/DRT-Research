# drt_sim/core/state/manager.py
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from pathlib import Path
import json


from drt_sim.core.state.base import StateWorker
from drt_sim.core.state.workers import (
    VehicleStateWorker,
    RequestStateWorker,
    RouteStateWorker,
    PassengerStateWorker,
    StopStateWorker
)
from drt_sim.models.simulation import SimulationState, SimulationStatus
from drt_sim.config.config import ScenarioConfig, SimulationConfig
from drt_sim.core.logging_config import setup_logger

class StateManager:
    """Coordinates multiple state workers and manages system-wide state"""
    
    def __init__(self, config: ScenarioConfig, sim_cfg: SimulationConfig):
        self.config = config
        self.logger = setup_logger(self.__class__.__name__)
        
        # Initialize state workers
        self.stop_worker = StopStateWorker()
        self.vehicle_worker = VehicleStateWorker()
        self.request_worker = RequestStateWorker()
        self.route_worker = RouteStateWorker()
        self.passenger_worker = PassengerStateWorker()
        
        self.workers: List[StateWorker] = [
            self.vehicle_worker,
            self.request_worker,
            self.route_worker,
            self.passenger_worker,
            self.stop_worker
        ]
        
        self.metrics: Dict[str, float] = {}
        
        # Initialize base state immediately
        self.state = SimulationState(
            current_time=datetime.fromisoformat(sim_cfg.start_time),
            status=SimulationStatus.INITIALIZED,
            metrics={},
            vehicles={},
            requests={},
            routes={}
        )
        
        # Initialize workers with their configs
        self.initialize_workers()
        
    def initialize_workers(self) -> None:
        """Initialize all state workers with their respective config sections"""
        try:
            self.vehicle_worker.initialize(self.config.vehicle)
            self.request_worker.initialize()
            self.route_worker.initialize()
            self.stop_worker.initialize()
            self.passenger_worker.initialize()
            self.logger.info("State manager initialized successfully")
        except Exception as e:
            self.logger.error(f"State initialization failed: {str(e)}")
            raise
            
    def take_snapshot(self, timestamp: datetime) -> None:
        """Take snapshots from all workers"""
        try:
            for worker in self.workers:
                worker.take_snapshot(timestamp)
        except Exception as e:
            self.logger.error(f"Failed to take snapshot: {str(e)}")
            raise
            
    def get_metrics(self) -> Dict[str, Any]:
        """Collect metrics from all workers"""
        metrics = {}
        for worker in self.workers:
            metrics.update(worker.get_metrics())
        return metrics
        
    def set_state(self, state: SimulationState) -> None:
        """Set the current state"""
        if not state:
            raise ValueError("Cannot set None state")
            
        try:
            self.state = state
            # Update workers with relevant state portions
            self.vehicle_worker.update_state(state.vehicles)
            self.request_worker.update_state(state.requests)
            self.route_worker.update_state(state.routes)
            self.logger.debug(f"State updated successfully: {state.status}")
        except Exception as e:
            self.logger.error(f"Failed to set state: {str(e)}")
            raise
            
    def get_state(self) -> SimulationState:
        """Get current state from all workers and return as dictionary"""
        try:
            current_state = self.get_current_state()
            return current_state
        except Exception as e:
            self.logger.error(f"Failed to get state: {str(e)}")
            raise
            
    def get_current_state(self) -> SimulationState:
        """Get current state from all workers"""
        if not self.state:
            raise RuntimeError("State not initialized")
            
        return SimulationState(
            current_time=self.state.current_time,
            status=self.state.status,
            metrics=self.get_metrics(),
            vehicles=self.vehicle_worker.get_state(),
            requests=self.request_worker.get_state(),
            routes=self.route_worker.get_state()
        )
        
    def begin_transaction(self) -> None:
        """Begin state transaction across all workers"""
        for worker in self.workers:
            worker.begin_transaction()
            
    def commit_transaction(self) -> None:
        """Commit current transaction across all workers"""
        for worker in self.workers:
            worker.commit_transaction()
            
    def rollback_transaction(self) -> None:
        """Rollback current transaction across all workers"""
        for worker in self.workers:
            worker.rollback_transaction()
            
    def save_state(self, filepath: Path) -> None:
        """Save current state to file"""
        try:
            state = self.get_current_state()
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(state.to_dict(), f, indent=2)
            self.logger.info(f"State saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save state: {str(e)}")
            raise
            
    def load_state(self, filepath: Path) -> None:
        """Load state from file"""
        try:
            with open(filepath, 'r') as f:
                state_data = json.load(f)
                
            # Validate state structure
            required_keys = {'current_time', 'status', 'metrics', 'vehicles', 
                           'requests', 'routes'}
            if not all(key in state_data for key in required_keys):
                raise ValueError("Invalid state file structure")
                
            # Create new state and restore workers
            state = SimulationState.from_dict(state_data)
            self.set_state(state)
            self.logger.info(f"State loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to load state: {str(e)}")
            raise
            
    def cleanup(self) -> None:
        """Clean up all workers"""
        for worker in self.workers:
            try:
                worker.cleanup()
            except Exception as e:
                self.logger.error(f"Error cleaning up worker: {str(e)}")