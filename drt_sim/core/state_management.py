# drt_sim/core/state_management.py
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
import copy

from ..models.vehicle import Vehicle, VehicleState
from ..models.passenger import Passenger
from ..models.request import Request
from ..models.route import Route
from ..models.stop import Stop
from ..config.parameters import ScenarioParameters, VehicleParameters, StopParameters

@dataclass
class SystemSnapshot:
    """Represents a complete snapshot of the system state"""
    timestamp: datetime
    vehicles: Dict[str, VehicleState]
    passengers: Dict[str, Passenger]
    requests: Dict[str, Request]
    routes: Dict[str, Route]
    stops: Dict[str, Stop]
    metrics: Dict[str, float]

class StateManager:
    """Manages the state of all simulation entities"""
    
    def __init__(self, scenario: ScenarioParameters):
        self.scenario = scenario
        self.logger = logging.getLogger(__name__)
        
        # Core state containers
        self.vehicles: Dict[str, Vehicle] = {}
        self.vehicle_states: Dict[str, VehicleState] = {}
        self.passengers: Dict[str, Passenger] = {}
        self.requests: Dict[str, Request] = {}
        self.routes: Dict[str, Route] = {}
        self.stops: Dict[str, Stop] = {}
        
        # Historical state tracking
        self._state_history: List[SystemSnapshot] = []
        self._snapshot_interval = scenario.simulation.snapshot_interval
        self._last_snapshot_time: Optional[datetime] = None
        
        # Metrics tracking
        self.current_metrics: Dict[str, float] = {}
        
    def initialize(self, scenario: ScenarioParameters) -> None:
        """Initialize simulation state from configuration"""
        self.logger.info("Initializing simulation state")
        
        try:
            # Initialize vehicles from config
            self._initialize_vehicles(scenario.vehicle)
            
            # Initialize stops from config
            self._initialize_stops(scenario.stop)
            
            # Take initial snapshot
            self.take_snapshot(scenario.simulation.start_time)
            
            self.logger.info("State initialization completed successfully")
            
        except Exception as e:
            self.logger.error(f"State initialization failed: {str(e)}")
            raise
            
    def _initialize_vehicles(self, vehicle_config: VehicleParameters) -> None:
        """Initialize vehicle fleet from configuration"""
        for i in range(vehicle_config.fleet_size):
            vehicle = Vehicle(
                id=f"vehicle_{i}",
                type=vehicle_config.vehicle_type,
                capacity=vehicle_config.vehicle_capacity,
                depot_location=vehicle_config.depot_locations[
                    i % len(vehicle_config.depot_locations)
                ],
                current_state=VehicleState(
                    vehicle_id=f"vehicle_{i}",
                    location=vehicle_config.depot_locations[
                        i % len(vehicle_config.depot_locations)
                    ],
                    status="idle"
                )
            )
            self.vehicles[vehicle.id] = vehicle
            self.vehicle_states[vehicle.id] = vehicle.current_state
            
    def _initialize_stops(self, stop_config: StopParameters) -> None:
        """Initialize system stops from configuration"""
        for stop_data in stop_config.initial_stops:
            stop = Stop(
                id=stop_data['id'],
                location=stop_data['location'],
                type=stop_data['type'],
                capacity=stop_data['capacity']
            )
            self.stops[stop.id] = stop
            
    def update_vehicle_state(self, 
                           vehicle_id: str, 
                           updates: Dict[str, Any]) -> None:
        """Update the state of a specific vehicle"""
        if vehicle_id not in self.vehicle_states:
            raise ValueError(f"Vehicle {vehicle_id} not found")
            
        vehicle_state = self.vehicle_states[vehicle_id]
        for key, value in updates.items():
            setattr(vehicle_state, key, value)
        vehicle_state.updated_at = datetime.now()
            
    def add_request(self, request: Request) -> None:
        """Add a new request to the system"""
        self.requests[request.id] = request
        self.logger.debug(f"Added request {request.id}")
        
    def update_request(self, 
                      request_id: str, 
                      updates: Dict[str, Any]) -> None:
        """Update an existing request"""
        if request_id not in self.requests:
            raise ValueError(f"Request {request_id} not found")
            
        request = self.requests[request_id]
        for key, value in updates.items():
            setattr(request, key, value)
        request.updated_at = datetime.now()
            
    def add_passenger(self, passenger: Passenger) -> None:
        """Add a new passenger to the system"""
        self.passengers[passenger.id] = passenger
        self.logger.debug(f"Added passenger {passenger.id}")
        
    def update_route(self, route: Route) -> None:
        """Update or add a route"""
        self.routes[route.id] = route
        self.logger.debug(f"Updated route {route.id}")
        
    def take_snapshot(self, timestamp: datetime) -> None:
        """Take a snapshot of the current system state"""
        snapshot = SystemSnapshot(
            timestamp=timestamp,
            vehicles=copy.deepcopy(self.vehicle_states),
            passengers=copy.deepcopy(self.passengers),
            requests=copy.deepcopy(self.requests),
            routes=copy.deepcopy(self.routes),
            stops=copy.deepcopy(self.stops),
            metrics=copy.deepcopy(self.current_metrics)
        )
        
        self._state_history.append(snapshot)
        self._last_snapshot_time = timestamp
        
        # Cleanup old snapshots if needed
        self._cleanup_old_snapshots()
        
    def get_state_at_time(self, 
                         timestamp: datetime) -> Optional[SystemSnapshot]:
        """Get system state at a specific time"""
        # Find the closest snapshot before the requested time
        for snapshot in reversed(self._state_history):
            if snapshot.timestamp <= timestamp:
                return snapshot
        return None
        
    def _cleanup_old_snapshots(self) -> None:
        """Clean up old snapshots based on retention policy"""
        retention_period = self.scenario.simulation.snapshot_retention_period
        if retention_period and len(self._state_history) > 0:
            cutoff_time = self._state_history[-1].timestamp - retention_period
            self._state_history = [
                snapshot for snapshot in self._state_history
                if snapshot.timestamp > cutoff_time
            ]
            
    def update_metrics(self, metrics_updates: Dict[str, float]) -> None:
        """Update system metrics"""
        self.current_metrics.update(metrics_updates)
        
    def get_metrics(self) -> Dict[str, float]:
        """Get current system metrics"""
        return self.current_metrics.copy()
        
    def export_state(self, filepath: Path) -> None:
        """Export current state to file"""
        state_data = {
            'timestamp': datetime.now().isoformat(),
            'vehicles': {k: asdict(v) for k, v in self.vehicle_states.items()},
            'passengers': {k: asdict(v) for k, v in self.passengers.items()},
            'requests': {k: asdict(v) for k, v in self.requests.items()},
            'routes': {k: asdict(v) for k, v in self.routes.items()},
            'stops': {k: asdict(v) for k, v in self.stops.items()},
            'metrics': self.current_metrics
        }
        
        with open(filepath, 'w') as f:
            json.dump(state_data, f, indent=2)
            
    def import_state(self, filepath: Path) -> None:
        """Import state from file"""
        with open(filepath, 'r') as f:
            state_data = json.load(f)
            
        # Clear current state
        self.vehicles.clear()
        self.vehicle_states.clear()
        self.passengers.clear()
        self.requests.clear()
        self.routes.clear()
        self.stops.clear()
        
        # Import new state
        for v_id, v_data in state_data['vehicles'].items():
            self.vehicle_states[v_id] = VehicleState(**v_data)
            
        for p_id, p_data in state_data['passengers'].items():
            self.passengers[p_id] = Passenger(**p_data)
            
        for r_id, r_data in state_data['requests'].items():
            self.requests[r_id] = Request(**r_data)
            
        for route_id, route_data in state_data['routes'].items():
            self.routes[route_id] = Route(**route_data)
            
        for stop_id, stop_data in state_data['stops'].items():
            self.stops[stop_id] = Stop(**stop_data)
            
        self.current_metrics = state_data['metrics']