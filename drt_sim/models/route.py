from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum

from drt_sim.models.base import ModelBase
from drt_sim.models.stop import Stop
from drt_sim.models.location import Location
class RouteStatus(Enum):
    """Status of a vehicle route"""
    CREATED = "created"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    PLANNED = "planned"


@dataclass
class RouteStop(ModelBase):
    """Represents a stop within a route context with additional route-specific information"""
    stop: Stop
    sequence: int
    current_load: int = 0
    pickup_passengers: List[str] = field(default_factory=list)  # List of request IDs to pick up
    dropoff_passengers: List[str] = field(default_factory=list)  # List of request IDs to drop off
    service_time: Optional[float] = 0
    earliest_passenger_arrival_time: Optional[datetime] = None
    latest_passenger_arrival_time: Optional[datetime] = None
    planned_arrival_time: Optional[datetime] = None
    actual_arrival_time: Optional[datetime] = None
    planned_departure_time: Optional[datetime] = None
    actual_departure_time: Optional[datetime] = None
    completed: bool = False
    # Fields for boarding state tracking
    boarded_request_ids: List[str] = field(default_factory=list)  # List of request IDs that have boarded
    wait_start_time: Optional[datetime] = None
    wait_timeout_event_id: Optional[str] = None

    def __post_init__(self):
        super().__init__()

    def __str__(self) -> str:
        """Provides a concise string representation of the route stop"""
        status = "✓" if self.completed else "⌛"
        pickup_ids = [pid[:8] for pid in self.pickup_passengers]
        dropoff_ids = [pid[:8] for pid in self.dropoff_passengers]
        pickup_str = f"+[{','.join(pickup_ids)}]" if pickup_ids else ""
        dropoff_str = f"-[{','.join(dropoff_ids)}]" if dropoff_ids else ""
        ops_str = f"{pickup_str}{dropoff_str}" if (pickup_str or dropoff_str) else "no-ops"
        
        # Format planned times if they exist
        arr_time = f"|arr={self.planned_arrival_time.strftime('%H:%M:%S')}" if self.planned_arrival_time else ""
        dep_time = f"|dep={self.planned_departure_time.strftime('%H:%M:%S')}" if self.planned_departure_time else ""
        
        return f"Stop[{self.id[:8]}|seq={self.sequence}|load={self.current_load}|{ops_str}{arr_time}{dep_time}|{status}]"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'stop': self.stop.to_dict(),
            'pickup_passengers': self.pickup_passengers,
            'dropoff_passengers': self.dropoff_passengers,
            'sequence': self.sequence,
            'service_time': self.service_time,
            'current_load': self.current_load,
            'earliest_passenger_arrival_time': self.earliest_passenger_arrival_time.isoformat() if self.earliest_passenger_arrival_time else None,
            'latest_passenger_arrival_time': self.latest_passenger_arrival_time.isoformat() if self.latest_passenger_arrival_time else None,
            'planned_arrival_time': self.planned_arrival_time.isoformat() if self.planned_arrival_time else None,
            'actual_arrival_time': self.actual_arrival_time.isoformat() if self.actual_arrival_time else None,
            'planned_departure_time': self.planned_departure_time.isoformat() if self.planned_departure_time else None,
            'actual_departure_time': self.actual_departure_time.isoformat() if self.actual_departure_time else None,
            'completed': self.completed,
            'boarded_request_ids': self.boarded_request_ids,
            'wait_start_time': self.wait_start_time.isoformat() if self.wait_start_time else None,
            'wait_timeout_event_id': self.wait_timeout_event_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RouteStop':
        route_stop = cls(
            stop=Stop.from_dict(data['stop']),
            pickup_passengers=data['pickup_passengers'],
            dropoff_passengers=data['dropoff_passengers'],
            sequence=data['sequence'],
            service_time=data['service_time'],
            current_load=data['current_load'],
            earliest_passenger_arrival_time=datetime.fromisoformat(data['earliest_passenger_arrival_time']) if data['earliest_passenger_arrival_time'] else None,
            latest_passenger_arrival_time=datetime.fromisoformat(data['latest_passenger_arrival_time']) if data['latest_passenger_arrival_time'] else None,
            planned_arrival_time=datetime.fromisoformat(data['planned_arrival_time']) if data['planned_arrival_time'] else None,
            actual_arrival_time=datetime.fromisoformat(data['actual_arrival_time']) if data['actual_arrival_time'] else None,
            planned_departure_time=datetime.fromisoformat(data['planned_departure_time']) if data['planned_departure_time'] else None,
            actual_departure_time=datetime.fromisoformat(data['actual_departure_time']) if data['actual_departure_time'] else None,
            completed=data['completed'],
            boarded_request_ids=data['boarded_request_ids'],
            wait_start_time=datetime.fromisoformat(data['wait_start_time']) if data['wait_start_time'] else None,
            wait_timeout_event_id=data['wait_timeout_event_id']
        )
        route_stop.id = data['id']
        route_stop.created_at = datetime.fromisoformat(data['created_at'])
        route_stop.updated_at = datetime.fromisoformat(data['updated_at'])
        return route_stop
@dataclass
class RouteSegment(ModelBase):
    """Represents a segment of a route between two stops or locations"""
    origin: Optional[RouteStop]  # Made optional to handle vehicle's current position
    destination: Optional[RouteStop]  # Made optional to handle return-to-base
    estimated_duration: float  # in seconds
    estimated_distance: float  # in meters
    completed: bool = False
    actual_duration: Optional[float] = None
    actual_distance: Optional[float] = None
    origin_location: Optional[Location] = None  # Added to handle non-stop origins
    destination_location: Optional[Location] = None  # Added to handle non-stop destinations
    waypoints: List[Dict[str, Any]] = field(default_factory=list)  # List of waypoints along the path
    current_waypoint_index: int = 0  # Index of current waypoint in path
    last_waypoint_update: Optional[datetime] = None  # Timestamp of last waypoint update
    movement_start_time: Optional[datetime] = None  # Timestamp of when the segment movement started

    def __post_init__(self):
        super().__init__()

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'origin': self.origin.to_dict() if self.origin else None,
            'destination': self.destination.to_dict() if self.destination else None,
            'estimated_duration': self.estimated_duration,
            'estimated_distance': self.estimated_distance,
            'completed': self.completed,
            'actual_duration': self.actual_duration,
            'actual_distance': self.actual_distance,
            'origin_location': self.origin_location.to_dict() if self.origin_location else None,
            'destination_location': self.destination_location.to_dict() if self.destination_location else None,
            'waypoints': self.waypoints,
            'current_waypoint_index': self.current_waypoint_index,
            'last_waypoint_update': self.last_waypoint_update.isoformat() if self.last_waypoint_update else None,
            'movement_start_time': self.movement_start_time.isoformat() if self.movement_start_time else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RouteSegment':
        segment = cls(
            origin=RouteStop.from_dict(data['origin']) if data.get('origin') else None,
            destination=RouteStop.from_dict(data['destination']) if data.get('destination') else None,
            estimated_duration=data['estimated_duration'],
            estimated_distance=data['estimated_distance'],
            completed=data['completed'],
            actual_duration=data['actual_duration'],
            actual_distance=data['actual_distance'],
            origin_location=Location.from_dict(data['origin_location']) if data.get('origin_location') else None,
            destination_location=Location.from_dict(data['destination_location']) if data.get('destination_location') else None,
            waypoints=data.get('waypoints', []),
            current_waypoint_index=data.get('current_waypoint_index', 0),
            last_waypoint_update=datetime.fromisoformat(data['last_waypoint_update']) if data.get('last_waypoint_update') else None,
            movement_start_time=datetime.fromisoformat(data['movement_start_time']) if data.get('movement_start_time') else None
        )
        segment.id = data['id']
        segment.created_at = datetime.fromisoformat(data['created_at'])
        segment.updated_at = datetime.fromisoformat(data['updated_at'])
        return segment

    def __str__(self) -> str:
        """Provides a concise string representation of the route segment"""
        status = "✓" if self.completed else "⌛"
        origin_id = self.origin.id[:8] if self.origin else (self.origin_location.id[:8] if self.origin_location else "N/A")
        dest_id = self.destination.id[:8] if self.destination else (self.destination_location.id[:8] if self.destination_location else "N/A")
        dist = f"{self.actual_distance:.1f}m" if self.actual_distance else f"{self.estimated_distance:.1f}m"
        dur = f"{self.actual_duration:.1f}s" if self.actual_duration else f"{self.estimated_duration:.1f}s"
        waypoint_info = f"|wp={self.current_waypoint_index}/{len(self.waypoints)}" if self.waypoints else ""
        return f"Segment[{self.id[:8]}|{origin_id}→{dest_id}|{dist}|{dur}{waypoint_info}|{status}]"

    def get_current_waypoint(self) -> Optional[Dict[str, Any]]:
        """Get the current waypoint in the path"""
        if self.waypoints and 0 <= self.current_waypoint_index < len(self.waypoints):
            return self.waypoints[self.current_waypoint_index]
        return None

    def get_next_waypoint(self) -> Optional[Dict[str, Any]]:
        """Get the next waypoint in the path"""
        if self.waypoints and self.current_waypoint_index + 1 < len(self.waypoints):
            return self.waypoints[self.current_waypoint_index + 1]
        return None

    def advance_waypoint(self) -> bool:
        """
        Advance to the next waypoint in the path.
        Returns True if successfully advanced, False if at end of path.
        """
        if self.current_waypoint_index + 1 < len(self.waypoints):
            self.current_waypoint_index += 1
            self.last_waypoint_update = datetime.now()
            return True
        return False

    def get_progress(self) -> float:
        """Get progress through the segment as a percentage"""
        if not self.waypoints:
            return 100.0 if self.completed else 0.0
        return (self.current_waypoint_index / (len(self.waypoints) - 1)) * 100 if len(self.waypoints) > 1 else 100.0

class DeviationType(Enum):
    """Types of route deviations"""
    TIME = "time"           # Temporal deviation (early/late)
    DISTANCE = "distance"   # Spatial deviation from planned route
    SPEED = "speed"        # Speed deviation from expected
    STOP_DURATION = "stop_duration"  # Deviation in stop duration
    OTHER = "other"        # Other types of deviations

@dataclass
class RouteDeviation(ModelBase):
    """Represents a deviation from the planned route"""
    type: DeviationType
    value: float  # The magnitude of the deviation
    time: datetime  # When the deviation occurred
    segment_id: Optional[str] = None  # Reference to the route segment if applicable
    description: Optional[str] = None  # Optional description of the deviation

    def __post_init__(self):
        super().__init__()

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'type': self.type.value,
            'value': self.value,
            'time': self.time.isoformat(),
            'segment_id': self.segment_id,
            'description': self.description
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RouteDeviation':
        deviation = cls(
            type=DeviationType(data['type']),
            value=data['value'],
            time=datetime.fromisoformat(data['time']),
            segment_id=data.get('segment_id'),
            description=data.get('description')
        )
        deviation.id = data['id']
        deviation.created_at = datetime.fromisoformat(data['created_at'])
        deviation.updated_at = datetime.fromisoformat(data['updated_at'])
        return deviation

    def __str__(self) -> str:
        """Provides a concise string representation of the route deviation"""
        segment = f"|seg={self.segment_id[:8]}" if self.segment_id else ""
        return f"Deviation[{self.id[:8]}|{self.type.value}|{self.value:.2f}{segment}|{self.time.strftime('%H:%M:%S')}]"

@dataclass
class Route(ModelBase):
    """Represents a complete vehicle route"""
    vehicle_id: str
    stops: List[RouteStop]
    status: RouteStatus
    current_segment_index: int = 0
    scheduled_start_time: Optional[datetime] = None
    scheduled_end_time: Optional[datetime] = None
    actual_start_time: Optional[datetime] = None
    actual_end_time: Optional[datetime] = None
    deviations: List[RouteDeviation] = field(default_factory=list)
    segments: Optional[List[RouteSegment]] = None  # Make segments optional
    total_distance: Optional[float] = None  # Make optional since it depends on segments
    total_duration: Optional[float] = None  # Make optional since it depends on segments

    def __str__(self) -> str:
        """Provides a concise string representation of the route"""
        #map pickup passengers to the stop id
        pickup_passengers_map = {passenger: stop_id for stop_id, stop in enumerate(self.stops) for passenger in stop.pickup_passengers}
        #map dropoff passengers to the stop id
        dropoff_passengers_map = {passenger: stop_id for stop_id, stop in enumerate(self.stops) for passenger in stop.dropoff_passengers}
        n_completed = sum(1 for stop in self.stops if stop.completed)
        n_stops = len(self.stops)
        #prepare the __str__ output
        str_output = f"Route[{self.id[:8]}|v={self.vehicle_id[:8]}|{self.status.value}|stops={n_completed}/{n_stops}|pickup={pickup_passengers_map}|dropoff={dropoff_passengers_map}]"
        return str_output
    
    @property
    def current_passengers(self) -> List[str]:
        """Get the current passengers in the route"""
        return [passenger for stop in self.stops for passenger in stop.pickup_passengers]

    def __post_init__(self):
        super().__init__()

    def get_current_segment(self) -> Optional[RouteSegment]:
        """Get the current active segment"""
        if self.segments and 0 <= self.current_segment_index < len(self.segments):
            return self.segments[self.current_segment_index]
        return None
    
    def recalc_current_segment_index(self) -> int:
        """
        Recalculates and updates the current_segment_index based on the first uncompleted segment.
        Returns the new current_segment_index.
        """
        for i, segment in enumerate(self.segments):
            if not segment.completed:
                self.current_segment_index = i
                return i
        # If all segments are completed, set it to the last segment
        self.current_segment_index = len(self.segments) - 1
        return self.current_segment_index
    
    def get_next_stop(self) -> Optional[RouteStop]:
        """Get the next stop in the route"""
        current_segment = self.get_current_segment()
        if current_segment:
            return current_segment.destination
        return None

    def is_completed(self) -> bool:
        """Check if route is completed"""
        return all(segment.completed for segment in self.segments)

    def add_deviation(
        self,
        deviation_type: DeviationType,
        value: float,
        time: datetime,
        segment_id: Optional[str] = None,
        description: Optional[str] = None
    ) -> RouteDeviation:
        """Add a new deviation to the route"""
        deviation = RouteDeviation(
            type=deviation_type,
            value=value,
            time=time,
            segment_id=segment_id,
            description=description
        )
        self.deviations.append(deviation)
        return deviation

    def validate_passenger_consistency(self) -> tuple[bool, Optional[str]]:
        """
        Validates that passenger pickups and dropoffs are consistent throughout the route.
        
        Checks:
        1. Each passenger is picked up exactly once
        2. Each passenger is dropped off exactly once
        3. Pickup occurs before dropoff for each passenger
        4. No passenger is dropped off before being picked up
        
        Returns:
            tuple[bool, Optional[str]]: A tuple containing:
                - Boolean indicating if the route is valid (True) or has inconsistencies (False)
                - Error message string if invalid, None if valid
        """
        try:
            # Track pickup and dropoff occurrences for each passenger
            pickups = {}
            dropoffs = {}
            
            # Track currently onboard passengers
            onboard = set()
            
            # Iterate through all stops sequentially
            for stop in self.stops:
                # Check pickups at this stop
                for request_id in stop.pickup_passengers:
                    # Check if passenger was already picked up
                    if request_id in onboard:
                        return False, f"Passenger with request id {request_id} was picked up multiple times"
                    
                    # Record pickup
                    pickups[request_id] = pickups.get(request_id, 0) + 1
                    onboard.add(request_id)
                
                # Check dropoffs at this stop
                for request_id in stop.dropoff_passengers:
                    # Check if passenger was not picked up yet
                    if request_id not in onboard:
                        return False, f"Passenger with request id {request_id} was dropped off before being picked up"
                    
                    # Record dropoff
                    dropoffs[request_id] = dropoffs.get(request_id, 0) + 1
                    onboard.remove(request_id)
            
            # After processing all stops, verify consistency
            all_passengers = set(pickups.keys()) | set(dropoffs.keys())
            
            for request_id in all_passengers:
                pickup_count = pickups.get(request_id, 0)
                dropoff_count = dropoffs.get(request_id, 0)
                
                if pickup_count != 1:
                    return False, f"Passenger with request id {request_id} was picked up {pickup_count} times (should be exactly once)"
                
                if dropoff_count != 1:
                    return False, f"Passenger with request id {request_id} was dropped off {dropoff_count} times (should be exactly once)"
            
            # Check if any passengers are still onboard
            if onboard:
                return False, f"Passengers with request ids {onboard} were picked up but never dropped off"
            
            return True, None
            
        except Exception as e:
            import traceback
            return False, f"Error validating passenger consistency: {str(e)}\nTraceback: {traceback.format_exc()}"
    
    def validate_capacity(self, vehicle_capacity: int) -> tuple[bool, Optional[str]]:
        """
        Validates if the route can be completed within vehicle capacity constraints.
        
        Args:
            vehicle_capacity: Maximum number of passengers the vehicle can carry
            
        Returns:
            tuple[bool, Optional[str]]: A tuple containing:
                - Boolean indicating if the route is valid (True) or would exceed capacity (False)
                - Error message string if invalid, None if valid
        """
        if not self.stops:
            return True, None
            
        # Simulate route traversal to check capacity
        current_load = 0
        
        # Process stops in sequence
        for i, stop in enumerate(self.stops):
            # Apply pickups and dropoffs
            pickup_count = len(stop.pickup_passengers)
            dropoff_count = len(stop.dropoff_passengers)
            
            # Calculate new load
            new_load = current_load + pickup_count - dropoff_count
            
            # Validate capacity constraints
            if new_load > vehicle_capacity:
                return False, f"Capacity exceeded at stop {i} (ID: {stop.stop.id}): {new_load} passengers vs capacity of {vehicle_capacity}"
            
            if new_load < 0:
                return False, f"Invalid negative load at stop {i} (ID: {stop.stop.id}): {new_load} passengers (current: {current_load}, pickup: {pickup_count}, dropoff: {dropoff_count})"
            
            # Update current load for next stop
            current_load = new_load
        
        # Ensure all passengers are dropped off by the end
        if current_load != 0:
            return False, f"Route ends with {current_load} passengers still onboard"
        
        return True, None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'vehicle_id': self.vehicle_id,
            'status': self.status.value,
            'stops': [stop.to_dict() for stop in self.stops],
            'segments': [segment.to_dict() for segment in self.segments],
            'current_segment_index': self.current_segment_index,
            'scheduled_start_time': self.scheduled_start_time,
            'scheduled_end_time': self.scheduled_end_time,
            'actual_start_time': self.actual_start_time,
            'actual_end_time': self.actual_end_time,
            'deviations': [d.to_dict() for d in self.deviations]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Route':
        route = cls(
            vehicle_id=data['vehicle_id'],
            stops=[RouteStop.from_dict(s) for s in data['stops']],
            status=RouteStatus(data['status']),
            current_segment_index=data['current_segment_index'],
            scheduled_start_time=datetime.fromisoformat(data['scheduled_start_time']) if data.get('scheduled_start_time') else None,
            scheduled_end_time=datetime.fromisoformat(data['scheduled_end_time']) if data.get('scheduled_end_time') else None,
            actual_start_time=datetime.fromisoformat(data['actual_start_time']) if data.get('actual_start_time') else None,
            actual_end_time=datetime.fromisoformat(data['actual_end_time']) if data.get('actual_end_time') else None,
            deviations=[RouteDeviation.from_dict(d) for d in data.get('deviations', [])]
        )
        route.id = data['id']
        route.created_at = datetime.fromisoformat(data['created_at'])
        route.updated_at = datetime.fromisoformat(data['updated_at'])
        return route

