# drt_sim/models/passenger.py
from dataclasses import dataclass
from typing import Optional, List
from enum import Enum
from datetime import datetime
from .base import ModelBase
from .location import Location
from .vehicle import VehicleType

class PassengerStatus(Enum):
    WALKING_TO_PICKUP = "walking_to_pickup"
    ARRIVED_AT_PICKUP = "arrived_at_pickup"
    WAITING_FOR_VEHICLE = "waiting_for_vehicle"
    PICKUP_STARTED = "pickup_started"
    PICKUP_COMPLETED = "pickup_completed"
    IN_VEHICLE = "in_vehicle"
    DETOUR_STARTED = "detour_started"
    DETOUR_ENDED = "detour_ended"
    RESUMED_VEHICLE_TRIP = "resumed_vehicle_trip"
    DROPOFF_STARTED = "dropoff_started"
    DROPOFF_COMPLETED = "dropoff_completed"
    WALKING_TO_DESTINATION = "walking_to_destination"
    ARRIVED_AT_DESTINATION = "arrived_at_destination"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

@dataclass
class PassengerPreferences:
    max_wait_time: int  # minutes
    max_detour_time: int  # minutes
    preferred_vehicle_type: Optional[VehicleType]
    accessibility_needs: List[str]
    baggage: float  # in cubic meters
    sharing_preference: bool  # whether willing to share rides

@dataclass
class Passenger(ModelBase):
    id: str
    request_id: str
    pickup_location: Location
    dropoff_location: Location
    requested_pickup_time: datetime
    preferences: PassengerPreferences
    status: PassengerStatus
    assigned_vehicle_id: Optional[str] = None

@dataclass
class PassengerState:
    """Represents the state of a passenger during their journey"""
    id: str
    request_id: str
    status: PassengerStatus
    current_location: Location
    origin: Location
    destination: Location
    pickup_point: Optional[Location] = None
    dropoff_point: Optional[Location] = None
    creation_time: datetime = None
    assigned_vehicle: Optional[str] = None
    
    # Timing metrics
    walking_to_pickup_start: Optional[datetime] = None
    walking_to_pickup_end: Optional[datetime] = None
    waiting_start: Optional[datetime] = None
    waiting_end: Optional[datetime] = None
    boarding_start: Optional[datetime] = None
    boarding_end: Optional[datetime] = None
    in_vehicle_start: Optional[datetime] = None
    in_vehicle_end: Optional[datetime] = None
    walking_to_destination_start: Optional[datetime] = None
    walking_to_destination_end: Optional[datetime] = None
    completion_time: Optional[datetime] = None
    
    # Journey metrics
    access_walking_distance: Optional[float] = None
    egress_walking_distance: Optional[float] = None
    total_wait_time: Optional[float] = None
    total_in_vehicle_time: Optional[float] = None
    total_journey_time: Optional[float] = None
    route_deviation_ratio: Optional[float] = None
    service_level_violations: List[str] = None