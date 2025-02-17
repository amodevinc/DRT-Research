# drt_sim/core/state/manager.py
from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path
import json


from drt_sim.core.state.base import StateWorker
from drt_sim.core.state.workers import (
    VehicleStateWorker,
    RequestStateWorker,
    RouteStateWorker,
    PassengerStateWorker,
    StopStateWorker,
    StopAssignmentStateWorker,
    AssignmentStateWorker
)
from drt_sim.models.state import (
    SimulationState, 
    SimulationStatus
)
from drt_sim.config.config import ParameterSet, SimulationConfig
import logging
logger = logging.getLogger(__name__)
class StateManager:
    """Coordinates multiple state workers and manages system-wide state"""
    
    def __init__(self, config: ParameterSet, sim_cfg: SimulationConfig):
        self.config = config
        
        # Initialize state workers
        self.stop_worker = StopStateWorker()
        self.vehicle_worker = VehicleStateWorker()
        self.request_worker = RequestStateWorker()
        self.route_worker = RouteStateWorker()
        self.passenger_worker = PassengerStateWorker()
        self.stop_assignment_worker = StopAssignmentStateWorker()
        self.assignment_worker = AssignmentStateWorker()
        self.workers: List[StateWorker] = [
            self.vehicle_worker,
            self.request_worker,
            self.route_worker,
            self.passenger_worker,
            self.stop_worker,
            self.stop_assignment_worker,
            self.assignment_worker
        ]
        
        self.metrics: Dict[str, float] = {}
        
        # Initialize base state immediately
        self.state = SimulationState(
            current_time=datetime.fromisoformat(sim_cfg.start_time),
            status=SimulationStatus.INITIALIZED,
            vehicles={},
            requests={},
            routes={},
            passengers={},
            stops={},
            stop_assignments={},
            assignments={}
        )
        
        # Initialize workers with their configs
        self.initialize_workers()
        
    def initialize_workers(self) -> None:
        """Initialize all state workers with their respective config sections"""
        try:
            self.vehicle_worker.initialize(self.config.vehicle)
            self.request_worker.initialize()
            self.route_worker.initialize()
            self.stop_worker.initialize(
                {
                    "depot_locations": self.config.vehicle.depot_locations
                }
            )
            self.passenger_worker.initialize()
            self.stop_assignment_worker.initialize()
            self.assignment_worker.initialize()
            logger.info("State manager initialized successfully")
        except Exception as e:
            logger.error(f"State initialization failed: {str(e)}")
            raise
            
    def take_snapshot(self, timestamp: datetime) -> None:
        """Take snapshots from all workers"""
        try:
            for worker in self.workers:
                worker.take_snapshot(timestamp)
        except Exception as e:
            logger.error(f"Failed to take snapshot: {str(e)}")
            raise
        
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
            self.stop_worker.update_state(state.stops)
            self.stop_assignment_worker.update_state(state.stop_assignments)
            self.assignment_worker.update_state(state.assignments)
            logger.debug(f"State updated successfully: {state.status}")
        except Exception as e:
            logger.error(f"Failed to set state: {str(e)}")
            raise
            
    def get_state(self) -> SimulationState:
        """Get current state from all workers and return as dictionary"""
        try:
            current_state = self.get_current_state()
            return current_state
        except Exception as e:
            logger.error(f"Failed to get state: {str(e)}")
            raise
            
    def get_current_state(self) -> SimulationState:
        """Get current state from all workers"""
        if not self.state:
            raise RuntimeError("State not initialized")
            
        # Get component states
        vehicle_state = self.vehicle_worker.get_state()
        request_state = self.request_worker.get_state()
        passenger_state = self.passenger_worker.get_state()
        route_state = self.route_worker.get_state()
        stop_state = self.stop_worker.get_state()
        stop_assignment_state = self.stop_assignment_worker.get_state()
        assignment_state = self.assignment_worker.get_state()
        return SimulationState(
            current_time=self.state.current_time,
            status=self.state.status,
            vehicles=vehicle_state,
            requests=request_state,
            passengers=passenger_state,
            routes=route_state,
            stops=stop_state,
            stop_assignments=stop_assignment_state,
            assignments=assignment_state
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
            logger.info(f"State saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save state: {str(e)}")
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
            logger.info(f"State loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load state: {str(e)}")
            raise
            
    def cleanup(self) -> None:
        """Clean up all workers"""
        for worker in self.workers:
            try:
                worker.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up worker: {str(e)}")