from typing import Dict, Any, Optional, List
from datetime import datetime
from drt_sim.core.state.base import StateWorker, StateContainer
from drt_sim.models.vehicle import Vehicle, VehicleStatus, VehicleType, VehicleState
from drt_sim.config.config import VehicleConfig
from drt_sim.models.location import Location
from collections import defaultdict
from drt_sim.models.state import VehicleSystemState
import logging
logger = logging.getLogger(__name__)

class VehicleStateWorker(StateWorker):
    """Manages state for vehicle fleet"""
    
    def __init__(self):
        self.vehicles = StateContainer[Vehicle]()
        self.initialized = False
        
    def initialize(self, config: Optional[VehicleConfig] = None) -> None:
        """Initialize vehicle state worker with configuration"""
        if not config:
            raise ValueError("Vehicle configuration required")
        self.config = config
        self.initialized = True
        # Initialize fleet when worker is initialized
        self.initialize_fleet()
        logger.info("Vehicle state worker initialized")
        
    def initialize_fleet(self) -> None:
        """Initialize the vehicle fleet based on configuration and distribute across depot locations"""
        if not self.initialized:
            raise RuntimeError("Worker must be initialized before creating fleet")
            
        try:
            # Calculate the number of vehicles per depot
            num_depots = len(self.config.depot_locations)
            vehicles_per_depot = self.config.fleet_size // num_depots
            remaining_vehicles = self.config.fleet_size % num_depots
            
            vehicle_id = 0
            for depot_idx, depot in enumerate(self.config.depot_locations):
                # Determine the number of vehicles for this depot
                num_vehicles = vehicles_per_depot + (1 if depot_idx < remaining_vehicles else 0)
                
                for _ in range(num_vehicles):
                    depot_location = Location(lat=depot[1], lon=depot[0])
                    
                    vehicle = Vehicle(
                        id=f"vehicle_{vehicle_id}",
                        current_state=VehicleState(
                            status=VehicleStatus.IDLE,
                            current_location=depot_location
                        ),
                        config=self.config,
                        capacity=self.config.capacity,
                        type=VehicleType.STANDARD,
                        registration=f"REG-{vehicle_id}",
                        manufacturer="Default_Manufacturer",
                        model="Default_Model",
                        year=2024,
                        fuel_efficiency=10.0,
                        maintenance_schedule={},
                        features=[],
                        accessibility_options=[],
                        max_range_km=500.0
                    )
                    self.vehicles.add(vehicle.id, vehicle)
                    vehicle_id += 1
            logger.info(f"Initialized fleet with {self.config.fleet_size} vehicles across {num_depots} depots")
            
        except Exception as e:
            logger.error(f"Failed to initialize fleet: {str(e)}")
            raise
            
    def add_vehicle(self, vehicle: Vehicle) -> None:
        """Add a new vehicle to the fleet"""
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
            
        try:
            self.vehicles.add(vehicle.id, vehicle)
            logger.debug(f"Added vehicle {vehicle.id} to fleet")
        except Exception as e:
            logger.error(f"Failed to add vehicle {vehicle.id}: {str(e)}")
            raise
            
    def remove_vehicle(self, vehicle_id: str) -> None:
        """Remove a vehicle from the fleet"""
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
            
        try:
            vehicle = self.get_vehicle(vehicle_id)
            if vehicle:
                self.vehicles.remove(vehicle_id)
            logger.debug(f"Removed vehicle {vehicle_id} from fleet")
            self.vehicles.remove(vehicle_id)
            logger.debug(f"Removed vehicle {vehicle_id} from fleet")
        except Exception as e:
            logger.error(f"Failed to remove vehicle {vehicle_id}: {str(e)}")
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
        
    def update_vehicle_status(self, vehicle_id: str, status: VehicleStatus, current_time: datetime) -> None:
        """Update vehicle status and track in-service time for utilization metric"""
        vehicle = self.get_vehicle(vehicle_id)
        if not vehicle:
            raise ValueError(f"Vehicle {vehicle_id} not found")

        try:
            old_status = vehicle.current_state.status

            # Initialize tracking attributes if not already present
            if not hasattr(vehicle.current_state, 'cumulative_occupied_time'):
                vehicle.current_state.cumulative_occupied_time = 0.0
            if not hasattr(vehicle.current_state, 'in_service_start_time'):
                vehicle.current_state.in_service_start_time = None

            # If transitioning into IN_SERVICE, record the start time
            if status == VehicleStatus.IN_SERVICE and old_status != VehicleStatus.IN_SERVICE:
                vehicle.current_state.in_service_start_time = current_time
            # If transitioning out of IN_SERVICE, update cumulative occupied time
            elif old_status == VehicleStatus.IN_SERVICE and status != VehicleStatus.IN_SERVICE:
                if vehicle.current_state.in_service_start_time is not None:
                    time_delta = (current_time - vehicle.current_state.in_service_start_time).total_seconds()
                    vehicle.current_state.cumulative_occupied_time += time_delta
                    vehicle.current_state.in_service_start_time = None

            vehicle.current_state.status = status
            logger.debug(f"Updated vehicle {vehicle_id} status to {status}")
        except Exception as e:
            logger.error(f"Failed to update vehicle {vehicle_id} status: {str(e)}")
            raise

    def update_vehicle_active_route_id(self, vehicle_id: str, route_id: str) -> None:
        """Update vehicle active route"""
        vehicle = self.get_vehicle(vehicle_id)
        if not vehicle:
            raise ValueError(f"Vehicle {vehicle_id} not found")
        vehicle.current_state.update_active_route_id(route_id)
        return True
    def update_vehicle_location(self, vehicle_id: str, location: Location, distance_traveled: Optional[float] = None) -> None:
        """Update vehicle location and record distance traveled"""
        vehicle = self.get_vehicle(vehicle_id)
        if not vehicle:
            raise ValueError(f"Vehicle {vehicle_id} not found")
            
        try:
            vehicle.current_state.current_location = location
                
            logger.debug(f"Updated vehicle {vehicle_id} location")
        except Exception as e:
            logger.error(f"Failed to update vehicle {vehicle_id} location: {str(e)}")
            raise

    def increment_vehicle_occupancy(self, vehicle_id: str) -> None:
        """Increment vehicle occupancy"""
        vehicle = self.get_vehicle(vehicle_id)
        if not vehicle:
            raise ValueError(f"Vehicle {vehicle_id} not found")
            
        try:
            vehicle.current_state.current_occupancy += 1
            logger.debug(f"Updated vehicle {vehicle_id} occupancy to {vehicle.current_state.current_occupancy}")
        except Exception as e:
            logger.error(f"Failed to update vehicle {vehicle_id} occupancy: {str(e)}")
            raise
    
    def decrement_vehicle_occupancy(self, vehicle_id: str) -> None:
        """Decrement vehicle occupancy"""
        vehicle = self.get_vehicle(vehicle_id)
        if not vehicle:
            raise ValueError(f"Vehicle {vehicle_id} not found")
            
        try:
            vehicle.current_state.current_occupancy -= 1
            logger.debug(f"Updated vehicle {vehicle_id} occupancy to {vehicle.current_state.current_occupancy}")
        except Exception as e:
            logger.error(f"Failed to update vehicle {vehicle_id} occupancy: {str(e)}")
            raise
    
    def take_snapshot(self, timestamp: datetime) -> None:
        """Take snapshot of vehicle states"""
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
        self.vehicles.take_snapshot(timestamp)
    
    def get_state(self) -> VehicleSystemState:
        """Get current state of the vehicle system"""
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
            
        try:
            # Get vehicles by status
            vehicles_by_status = defaultdict(list)
            for vid, vehicle in self.vehicles.items.items():
                vehicles_by_status[vehicle.current_state.status].append(vid)
            
            
            return VehicleSystemState(
                vehicles={vid: v.current_state for vid, v in self.vehicles.items.items()},
                vehicles_by_status=vehicles_by_status
            )
            
        except Exception as e:
            logger.error(f"Failed to get vehicle system state: {str(e)}")
            raise

    def update_state(self, state: VehicleSystemState) -> None:
        """Update vehicle system state"""
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
            
        try:
            # Update vehicle states
            for vehicle_id, vehicle_state in state.vehicles.items():
                vehicle = self.get_vehicle(vehicle_id)
                if vehicle:
                    vehicle.current_state = vehicle_state
                else:
                    logger.warning(f"Vehicle {vehicle_id} not found during state update")
            logger.info("Vehicle system state updated successfully")
            
        except Exception as e:
            logger.error(f"Failed to update vehicle system state: {str(e)}")
            raise

    def restore_state(self, saved_state: Dict[str, Any]) -> None:
        """Restore vehicle states from a saved VehicleSystemState"""
        try:
            # First create a VehicleSystemState object to validate the data
            vehicle_system_state = VehicleSystemState.from_dict(saved_state)
            
            # Reset current state
            self.vehicles = StateContainer[Vehicle]()
            
            # Restore vehicles using update_state
            self.update_state(vehicle_system_state)
            
            logger.debug("Restored vehicle system state")

        except Exception as e:
            logger.error(f"Error restoring vehicle system state: {str(e)}")
            raise
    
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
        logger.info("Vehicle state worker cleaned up")