from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass
import logging

from drt_sim.core.state.base import StateWorker, StateContainer
from drt_sim.models.vehicle import Vehicle, VehicleStatus, VehicleType, VehicleState
from drt_sim.config.config import VehicleConfig
from drt_sim.models.location import Location

logger = logging.getLogger(__name__)

@dataclass
class VehicleMetrics:
    """Metrics tracked for vehicles"""
    total_distance: float = 0.0
    total_time: float = 0.0
    idle_time: float = 0.0
    occupancy_rate: float = 0.0
    service_count: int = 0

class VehicleStateWorker(StateWorker):
    """Manages state for vehicle fleet"""
    
    def __init__(self):
        self.vehicles = StateContainer[Vehicle]()
        self.metrics = VehicleMetrics()
        self.initialized = False
        self.logger = logging.getLogger(__name__)
        
    def initialize(self, config: Optional[VehicleConfig] = None) -> None:
        """Initialize vehicle state worker with configuration"""
        if not config:
            raise ValueError("Vehicle configuration required")
        self.config = config
        self.initialized = True
        # Initialize fleet when worker is initialized
        self.initialize_fleet()
        self.logger.info("Vehicle state worker initialized")
        
    def initialize_fleet(self) -> None:
        """Initialize the vehicle fleet based on configuration"""
        if not self.initialized:
            raise RuntimeError("Worker must be initialized before creating fleet")
            
        try:
            for idx in range(self.config.fleet_size):
                # Get depot location for this vehicle
                depot_location = Location(
                    lat=self.config.depot_locations[idx % len(self.config.depot_locations)][0],
                    lon=self.config.depot_locations[idx % len(self.config.depot_locations)][1]
                )
                
                vehicle = Vehicle(
                    id=f"vehicle_{idx}",
                    current_state=VehicleState(
                        status=VehicleStatus.IDLE,
                        current_location=depot_location
                    ),
                    capacity=self.config.capacity,
                    type=VehicleType.STANDARD,
                    depot_location=depot_location,
                    registration=f"REG-{idx}",
                    manufacturer="Default",
                    model="Standard",
                    year=2024,
                    fuel_efficiency=10.0,
                    maintenance_schedule={},
                    features=[],
                    accessibility_options=[],
                    max_range_km=500.0
                )
                self.vehicles.add(vehicle.id, vehicle)
                
            self.logger.info(f"Initialized fleet with {self.config.fleet_size} vehicles")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize fleet: {str(e)}")
            raise
            
    def add_vehicle(self, vehicle: Vehicle) -> None:
        """Add a new vehicle to the fleet"""
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
            
        try:
            self.vehicles.add(vehicle.id, vehicle)
            self.logger.debug(f"Added vehicle {vehicle.id} to fleet")
        except Exception as e:
            self.logger.error(f"Failed to add vehicle {vehicle.id}: {str(e)}")
            raise
            
    def remove_vehicle(self, vehicle_id: str) -> None:
        """Remove a vehicle from the fleet"""
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
            
        try:
            self.vehicles.remove(vehicle_id)
            self.logger.debug(f"Removed vehicle {vehicle_id} from fleet")
        except Exception as e:
            self.logger.error(f"Failed to remove vehicle {vehicle_id}: {str(e)}")
            raise
            
    def get_vehicle(self, vehicle_id: str) -> Optional[Vehicle]:
        """Get vehicle by ID"""
        return self.vehicles.get(vehicle_id)
        
    def get_available_vehicles(self) -> List[Vehicle]:
        """Get list of available (idle) vehicles"""
        return [
            vehicle for vehicle in self.vehicles.items.values()
            if vehicle.current_state.status == VehicleStatus.IDLE
        ]
        
    def update_vehicle_status(self, vehicle_id: str, status: VehicleStatus) -> None:
        """Update vehicle status"""
        vehicle = self.get_vehicle(vehicle_id)
        if not vehicle:
            raise ValueError(f"Vehicle {vehicle_id} not found")
            
        try:
            vehicle.current_state.status = status
            self.logger.debug(f"Updated vehicle {vehicle_id} status to {status}")
        except Exception as e:
            self.logger.error(f"Failed to update vehicle {vehicle_id} status: {str(e)}")
            raise
            
    def update_vehicle_location(self, vehicle_id: str, location: Location) -> None:
        """Update vehicle location"""
        vehicle = self.get_vehicle(vehicle_id)
        if not vehicle:
            raise ValueError(f"Vehicle {vehicle_id} not found")
            
        try:
            vehicle.current_state.current_location = location
            self.logger.debug(f"Updated vehicle {vehicle_id} location")
        except Exception as e:
            self.logger.error(f"Failed to update vehicle {vehicle_id} location: {str(e)}")
            raise
    
    def take_snapshot(self, timestamp: datetime) -> None:
        """Take snapshot of vehicle states"""
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
        self.vehicles.take_snapshot(timestamp)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current vehicle metrics"""
        return {
            "vehicle_total_distance": self.metrics.total_distance,
            "vehicle_total_time": self.metrics.total_time,
            "vehicle_idle_time": self.metrics.idle_time,
            "vehicle_occupancy_rate": self.metrics.occupancy_rate,
            "vehicle_service_count": self.metrics.service_count
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get current vehicle states"""
        return {
            vehicle_id: vehicle.to_dict() 
            for vehicle_id, vehicle in self.vehicles.items.items()
        }
    
    def update_state(self, state: Dict[str, Any]) -> None:
        """Update vehicle states"""
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
            
        for vehicle_id, vehicle_data in state.items():
            if vehicle_id in self.vehicles.items:
                self.vehicles.update(vehicle_id, vehicle_data)
            else:
                vehicle = Vehicle.from_dict(vehicle_data)
                self.vehicles.add(vehicle_id, vehicle)
    
    def restore_state(self, saved_state: Dict[str, Any]) -> None:
        """Restore vehicle states from saved state"""
        self.vehicles = StateContainer[Vehicle]()
        for vehicle_id, vehicle_data in saved_state.items():
            vehicle = Vehicle.from_dict(vehicle_data)
            self.vehicles.add(vehicle_id, vehicle)
    
    def begin_transaction(self) -> None:
        """Begin state transaction"""
        self.vehicles.begin_transaction()
    
    def commit_transaction(self) -> None:
        """Commit current transaction"""
        self.vehicles.commit_transaction()
    
    def rollback_transaction(self) -> None:
        """Rollback current transaction"""
        self.vehicles.rollback_transaction()
    
    def cleanup(self) -> None:
        """Clean up resources"""
        self.vehicles.clear_history()
        self.initialized = False