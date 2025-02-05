# drt_sim/models/passenger.py
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime
from .location import Location
from .stop import Stop

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
class PassengerState:
    """Tracks the complete dynamic state of a passenger"""
    id: str
    request_id: str
    status: PassengerStatus
    current_location: Location

    # Stop-related state
    assigned_origin_stop: Optional[Stop] = None
    assigned_destination_stop: Optional[Stop] = None

    # Vehicle assignment
    assigned_vehicle_id: Optional[str] = None

    # Timing state
    estimated_pickup_time: Optional[datetime] = None
    actual_pickup_time: Optional[datetime] = None
    estimated_dropoff_time: Optional[datetime] = None
    actual_dropoff_time: Optional[datetime] = None
    
    # Journey progress
    walking_to_pickup_start_time: Optional[datetime] = None
    walking_to_pickup_end_time: Optional[datetime] = None
    waiting_start_time: Optional[datetime] = None
    waiting_end_time: Optional[datetime] = None
    in_vehicle_start_time: Optional[datetime] = None
    in_vehicle_end_time: Optional[datetime] = None
    walking_to_destination_start_time: Optional[datetime] = None
    walking_to_destination_end_time: Optional[datetime] = None
    boarding_start_time: Optional[datetime] = None
    boarding_end_time: Optional[datetime] = None
    alighting_start_time: Optional[datetime] = None
    alighting_end_time: Optional[datetime] = None

    #Journey times
    wait_time: Optional[float] = None
    walk_time_to_origin_stop: Optional[float] = None
    in_vehicle_time: Optional[float] = None
    walk_time_from_destination_stop: Optional[float] = None
    total_journey_time: Optional[float] = None

    # Cancellation
    cancellation_reason: Optional[str] = None
    cancellation_time: Optional[datetime] = None

    # Service Level Violations
    service_violations: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert PassengerState to dictionary"""
        return {
            'id': self.id,
            'request_id': self.request_id,
            'status': self.status.value,
            'current_location': self.current_location.to_dict(),
            'assigned_origin_stop': self.assigned_origin_stop.to_dict() if self.assigned_origin_stop else None,
            'assigned_destination_stop': self.assigned_destination_stop.to_dict() if self.assigned_destination_stop else None,
            'assigned_vehicle_id': self.assigned_vehicle_id,
            'estimated_pickup_time': self.estimated_pickup_time.isoformat() if self.estimated_pickup_time else None,
            'actual_pickup_time': self.actual_pickup_time.isoformat() if self.actual_pickup_time else None,
            'estimated_dropoff_time': self.estimated_dropoff_time.isoformat() if self.estimated_dropoff_time else None,
            'actual_dropoff_time': self.actual_dropoff_time.isoformat() if self.actual_dropoff_time else None,
            'walking_to_pickup_start_time': self.walking_to_pickup_start_time.isoformat() if self.walking_to_pickup_start_time else None,
            'walking_to_pickup_end_time': self.walking_to_pickup_end_time.isoformat() if self.walking_to_pickup_end_time else None,
            'waiting_start_time': self.waiting_start_time.isoformat() if self.waiting_start_time else None,
            'waiting_end_time': self.waiting_end_time.isoformat() if self.waiting_end_time else None,
            'in_vehicle_start_time': self.in_vehicle_start_time.isoformat() if self.in_vehicle_start_time else None,
            'in_vehicle_end_time': self.in_vehicle_end_time.isoformat() if self.in_vehicle_end_time else None,
            'walking_to_destination_start_time': self.walking_to_destination_start_time.isoformat() if self.walking_to_destination_start_time else None,
            'walking_to_destination_end_time': self.walking_to_destination_end_time.isoformat() if self.walking_to_destination_end_time else None,
            'boarding_start_time': self.boarding_start_time.isoformat() if self.boarding_start_time else None,
            'boarding_end_time': self.boarding_end_time.isoformat() if self.boarding_end_time else None,
            'alighting_start_time': self.alighting_start_time.isoformat() if self.alighting_start_time else None,
            'alighting_end_time': self.alighting_end_time.isoformat() if self.alighting_end_time else None,
            'cancellation_reason': self.cancellation_reason,
            'cancellation_time': self.cancellation_time.isoformat() if self.cancellation_time else None,
            'service_violations': self.service_violations
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PassengerState':
        """Create PassengerState from dictionary"""
        return cls(
            id=data['id'],
            request_id=data['request_id'],
            status=PassengerStatus(data['status']),
            current_location=Location.from_dict(data['current_location']),
            assigned_origin_stop=Stop.from_dict(data['assigned_origin_stop']) if data['assigned_origin_stop'] else None,
            assigned_destination_stop=Stop.from_dict(data['assigned_destination_stop']) if data['assigned_destination_stop'] else None,
            assigned_vehicle_id=data['assigned_vehicle_id'],
            estimated_pickup_time=datetime.fromisoformat(data['estimated_pickup_time']) if data['estimated_pickup_time'] else None,
            actual_pickup_time=datetime.fromisoformat(data['actual_pickup_time']) if data['actual_pickup_time'] else None,
            estimated_dropoff_time=datetime.fromisoformat(data['estimated_dropoff_time']) if data['estimated_dropoff_time'] else None,
            actual_dropoff_time=datetime.fromisoformat(data['actual_dropoff_time']) if data['actual_dropoff_time'] else None,
            walking_to_pickup_start_time=datetime.fromisoformat(data['walking_to_pickup_start_time']) if data['walking_to_pickup_start_time'] else None,
            walking_to_pickup_end_time=datetime.fromisoformat(data['walking_to_pickup_end_time']) if data['walking_to_pickup_end_time'] else None,
            waiting_start_time=datetime.fromisoformat(data['waiting_start_time']) if data['waiting_start_time'] else None,
            waiting_end_time=datetime.fromisoformat(data['waiting_end_time']) if data['waiting_end_time'] else None,
            in_vehicle_start_time=datetime.fromisoformat(data['in_vehicle_start_time']) if data['in_vehicle_start_time'] else None,
            in_vehicle_end_time=datetime.fromisoformat(data['in_vehicle_end_time']) if data['in_vehicle_end_time'] else None,
            walking_to_destination_start_time=datetime.fromisoformat(data['walking_to_destination_start_time']) if data['walking_to_destination_start_time'] else None,
            walking_to_destination_end_time=datetime.fromisoformat(data['walking_to_destination_end_time']) if data['walking_to_destination_end_time'] else None,
            boarding_start_time=datetime.fromisoformat(data['boarding_start_time']) if data['boarding_start_time'] else None,
            boarding_end_time=datetime.fromisoformat(data['boarding_end_time']) if data['boarding_end_time'] else None,
            alighting_start_time=datetime.fromisoformat(data['alighting_start_time']) if data['alighting_start_time'] else None,
            alighting_end_time=datetime.fromisoformat(data['alighting_end_time']) if data['alighting_end_time'] else None,
            cancellation_reason=data['cancellation_reason'],
            cancellation_time=datetime.fromisoformat(data['cancellation_time']) if data['cancellation_time'] else None,
            service_violations=data['service_violations']
        )
