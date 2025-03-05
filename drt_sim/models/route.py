from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set, ClassVar
from enum import Enum
import logging

from drt_sim.models.base import ModelBase
from drt_sim.models.stop import Stop
from drt_sim.models.location import Location

logger = logging.getLogger(__name__)

class RouteStatus(Enum):
    """Status of a vehicle route"""
    CREATED = "created"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    PLANNED = "planned"
    MODIFIED = "modified"  # Status to track modifications

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
    # Fields for tracking modifications
    initial_sequence: Optional[int] = None  # Original sequence before any modifications
    modification_count: int = 0  # Number of times this stop has been modified
    # Movement fields (replacing segment information)
    origin_location: Optional[Location] = None  # For first stop or after modifications
    estimated_duration_to_stop: float = 0.0  # Estimated travel time to this stop in seconds
    estimated_distance_to_stop: float = 0.0  # Estimated distance to this stop in meters
    actual_duration_to_stop: Optional[float] = None  # Actual travel time to this stop in seconds
    actual_distance_to_stop: Optional[float] = None  # Actual travel distance to this stop in meters
    movement_start_time: Optional[datetime] = None  # When vehicle started moving toward this stop
    in_progress: bool = False  # Whether vehicle is currently moving toward this stop
    progress_percentage: float = 0.0  # Progress percentage toward this stop (0-100)
    vehicle_current_location: Optional[Location] = None  # Current vehicle location during movement
    arrived_pickup_request_ids: Set[str] = field(default_factory=set)
    completed_dropoff_request_ids: Set[str] = field(default_factory=set)
    vehicle_is_present: bool = False
    last_vehicle_arrival_time: Optional[datetime] = None
    # Fields to track route changes while at stop
    pending_route_change: bool = False
    new_route_id: Optional[str] = None
    route_change_time: Optional[datetime] = None
    
    def __post_init__(self):
        super().__init__()
        if self.initial_sequence is None:
            self.initial_sequence = self.sequence

    def __str__(self) -> str:
        """Provides a concise string representation of the route stop"""
        status = "✓" if self.completed else "⌛"
        if self.in_progress:
            status = f"→{self.progress_percentage:.1f}%"
            
        pickup_ids = [pid[:8] for pid in self.pickup_passengers]
        dropoff_ids = [pid[:8] for pid in self.dropoff_passengers]
        pickup_str = f"+[{','.join(pickup_ids)}]" if pickup_ids else ""
        dropoff_str = f"-[{','.join(dropoff_ids)}]" if dropoff_ids else ""
        ops_str = f"{pickup_str}{dropoff_str}" if (pickup_str or dropoff_str) else "no-ops"
        
        # Format planned times if they exist
        arr_time = f"|arr={self.planned_arrival_time.strftime('%H:%M:%S')}" if self.planned_arrival_time else ""
        dep_time = f"|dep={self.planned_departure_time.strftime('%H:%M:%S')}" if self.planned_departure_time else ""
        
        # Add travel information
        travel_info = ""
        if not self.completed and not self.in_progress:
            dist = f"{self.estimated_distance_to_stop:.1f}m"
            dur = f"{self.estimated_duration_to_stop:.1f}s"
            travel_info = f"|travel={dist},{dur}"
        
        return f"RouteStop[{self.id[:8]}|ActualStopId={self.stop.id}|seq={self.sequence}|load={self.current_load}|{ops_str}{arr_time}{dep_time}{travel_info}|{status}]"

    def to_dict(self) -> Dict[str, Any]:
        """Convert RouteStop to dictionary representation"""
        base_dict = {
            'id': self.id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
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
            'wait_timeout_event_id': self.wait_timeout_event_id,
            'initial_sequence': self.initial_sequence,
            'modification_count': self.modification_count,
            'origin_location': self.origin_location.to_dict() if self.origin_location else None,
            'estimated_duration_to_stop': self.estimated_duration_to_stop,
            'estimated_distance_to_stop': self.estimated_distance_to_stop,
            'actual_duration_to_stop': self.actual_duration_to_stop,
            'actual_distance_to_stop': self.actual_distance_to_stop,
            'movement_start_time': self.movement_start_time.isoformat() if self.movement_start_time else None,
            'in_progress': self.in_progress,
            'progress_percentage': self.progress_percentage,
            'vehicle_current_location': self.vehicle_current_location.to_dict() if self.vehicle_current_location else None
        }
        return base_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RouteStop':
        """Create RouteStop from dictionary representation"""
        # Ensure required data exists
        if not data.get('stop'):
            raise ValueError("Missing required 'stop' data for RouteStop")
            
        route_stop = cls(
            stop=Stop.from_dict(data['stop']),
            pickup_passengers=data.get('pickup_passengers', []),
            dropoff_passengers=data.get('dropoff_passengers', []),
            sequence=data.get('sequence', 0),
            service_time=data.get('service_time', 0),
            current_load=data.get('current_load', 0),
            earliest_passenger_arrival_time=datetime.fromisoformat(data['earliest_passenger_arrival_time']) if data.get('earliest_passenger_arrival_time') else None,
            latest_passenger_arrival_time=datetime.fromisoformat(data['latest_passenger_arrival_time']) if data.get('latest_passenger_arrival_time') else None,
            planned_arrival_time=datetime.fromisoformat(data['planned_arrival_time']) if data.get('planned_arrival_time') else None,
            actual_arrival_time=datetime.fromisoformat(data['actual_arrival_time']) if data.get('actual_arrival_time') else None,
            planned_departure_time=datetime.fromisoformat(data['planned_departure_time']) if data.get('planned_departure_time') else None,
            actual_departure_time=datetime.fromisoformat(data['actual_departure_time']) if data.get('actual_departure_time') else None,
            completed=data.get('completed', False),
            boarded_request_ids=data.get('boarded_request_ids', []),
            wait_start_time=datetime.fromisoformat(data['wait_start_time']) if data.get('wait_start_time') else None,
            wait_timeout_event_id=data.get('wait_timeout_event_id'),
            initial_sequence=data.get('initial_sequence'),
            modification_count=data.get('modification_count', 0),
            estimated_duration_to_stop=data.get('estimated_duration_to_stop', 0.0),
            estimated_distance_to_stop=data.get('estimated_distance_to_stop', 0.0),
            actual_duration_to_stop=data.get('actual_duration_to_stop'),
            actual_distance_to_stop=data.get('actual_distance_to_stop'),
            in_progress=data.get('in_progress', False),
            progress_percentage=data.get('progress_percentage', 0.0)
        )
        
        # Handle location objects
        if data.get('origin_location'):
            route_stop.origin_location = Location.from_dict(data['origin_location'])
            
        if data.get('vehicle_current_location'):
            route_stop.vehicle_current_location = Location.from_dict(data['vehicle_current_location'])
        
        # Handle movement_start_time
        if data.get('movement_start_time'):
            route_stop.movement_start_time = datetime.fromisoformat(data['movement_start_time'])
        
        # Set the ID and timestamps if they exist
        if 'id' in data:
            route_stop.id = data['id']
        if 'created_at' in data and data['created_at']:
            route_stop.created_at = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data and data['updated_at']:
            route_stop.updated_at = datetime.fromisoformat(data['updated_at'])
            
        return route_stop
    
    def register_passenger_arrival(self, request_id: str) -> bool:
        """Register passenger arrival, return True if boarding should be attempted"""
        self.arrived_pickup_request_ids.add(request_id)
        return self.vehicle_is_present
        
    def register_vehicle_arrival(self, current_time: datetime) -> Set[str]:
        """Register vehicle arrival, return passenger IDs ready for boarding"""
        self.vehicle_is_present = True
        self.last_vehicle_arrival_time = current_time
        return self.arrived_pickup_request_ids - set(self.boarded_request_ids)
        
    def register_boarding(self, request_id: str) -> None:
        """Register passenger boarding"""
        if request_id not in self.boarded_request_ids:
            self.boarded_request_ids.append(request_id)
            
    def register_dropoff(self, request_id: str) -> None:
        """Register passenger dropoff"""
        self.completed_dropoff_request_ids.add(request_id)
        
    def is_pickup_complete(self) -> bool:
        """Check if all pickups are complete"""
        return set(self.boarded_request_ids).issuperset(self.pickup_passengers)
        
    def is_dropoff_complete(self) -> bool:
        """Check if all dropoffs are complete"""
        return self.completed_dropoff_request_ids.issuperset(self.dropoff_passengers)
        
    def is_operation_complete(self) -> bool:
        """Check if all operations are complete"""
        return self.is_pickup_complete() and self.is_dropoff_complete()
        
    def start_wait_timer(self, event_id: str, current_time: datetime) -> None:
        """Start wait timer"""
        self.wait_start_time = current_time
        self.wait_timeout_event_id = event_id
        
    def cancel_wait_timer(self) -> Optional[str]:
        """Cancel wait timer, return event ID if one was active"""
        event_id = self.wait_timeout_event_id
        self.wait_start_time = None
        self.wait_timeout_event_id = None
        return event_id
        
    def has_passenger_operation(self) -> bool:
        """Check if this stop has any passenger operations (pickup or dropoff)"""
        return bool(self.pickup_passengers or self.dropoff_passengers)
        
    def add_pickup(self, request_id: str) -> None:
        """Add a passenger pickup to this stop"""
        if request_id not in self.pickup_passengers:
            self.pickup_passengers.append(request_id)
            self.modification_count += 1
            
    def add_dropoff(self, request_id: str) -> None:
        """Add a passenger dropoff to this stop"""
        if request_id not in self.dropoff_passengers:
            self.dropoff_passengers.append(request_id)
            self.modification_count += 1

    def has_pending_operations(self) -> bool:
        """Check if the stop has any pending operations (not yet completed)"""
        if self.completed:
            return False
        return self.has_passenger_operation()
    
    def mark_in_progress(self, current_location: Optional[Location] = None) -> None:
        """Mark this stop as in progress (vehicle is currently moving toward it)"""
        self.in_progress = True
        
        if current_location:
            self.vehicle_current_location = current_location
            
        # Initialize movement start time if not set
        if not self.movement_start_time:
            self.movement_start_time = datetime.now()
    
    def update_progress(self, percentage: float, current_location: Optional[Location] = None) -> None:
        """Update progress toward this stop"""
        self.progress_percentage = max(0.0, min(100.0, percentage))
        
        if current_location:
            self.vehicle_current_location = current_location

    def mark_completed(self, actual_arrival_time: Optional[datetime] = None, 
                      actual_departure_time: Optional[datetime] = None,
                      actual_duration: Optional[float] = None, 
                      actual_distance: Optional[float] = None) -> None:
        """Mark this stop as completed with actual metrics"""
        self.completed = True
        self.in_progress = False
        self.progress_percentage = 100.0
        
        if actual_arrival_time:
            self.actual_arrival_time = actual_arrival_time
            
        if actual_departure_time:
            self.actual_departure_time = actual_departure_time
            
        if actual_duration is not None:
            self.actual_duration_to_stop = actual_duration
            
        if actual_distance is not None:
            self.actual_distance_to_stop = actual_distance
        
        # If actual values not provided, use estimates
        if self.actual_duration_to_stop is None:
            self.actual_duration_to_stop = self.estimated_duration_to_stop
            
        if self.actual_distance_to_stop is None:
            self.actual_distance_to_stop = self.estimated_distance_to_stop

    def register_route_change(self, new_route_id: str, change_time: datetime) -> None:
        """Register a route change while at stop"""
        self.pending_route_change = True
        self.new_route_id = new_route_id
        self.route_change_time = change_time
        
    def clear_route_change(self) -> None:
        """Clear pending route change"""
        self.pending_route_change = False
        self.new_route_id = None
        self.route_change_time = None
        
    def has_pending_route_change(self) -> bool:
        """Check if there is a pending route change"""
        return self.pending_route_change and self.new_route_id is not None

@dataclass
class Route(ModelBase):
    """Represents a complete vehicle route, simplified without segments"""
    vehicle_id: str
    stops: List[RouteStop]
    status: RouteStatus
    current_stop_index: int = 0  # Index of the current active stop
    scheduled_start_time: Optional[datetime] = None
    scheduled_end_time: Optional[datetime] = None
    actual_start_time: Optional[datetime] = None
    actual_end_time: Optional[datetime] = None
    total_distance: Optional[float] = None
    total_duration: Optional[float] = None
    # Fields for enhanced route tracking
    active_stop_id: Optional[str] = None  # ID of the stop currently being approached/serviced
    last_modification_time: Optional[datetime] = None
    modification_count: int = 0
    last_validation_time: Optional[datetime] = None
    is_consistent: bool = True  # Flag indicating if route is in a consistent state
    version: int = 1  # Track route version for dispatch coordination
    _version: ClassVar[str] = "1.0"

    def __post_init__(self):
        super().__init__()
        # Initialize total distance and duration if not set
        if self.total_distance is None:
            self.recalculate_total_distance()
            
        if self.total_duration is None:
            self.recalculate_total_duration()

    def __str__(self) -> str:
        """Provides a concise string representation of the route"""
        n_completed = sum(1 for stop in self.stops if stop.completed)
        n_stops = len(self.stops)
        
        # Get list of current passengers
        current_passengers = self.current_passengers
        passengers_str = f"passengers={len(current_passengers)}" if current_passengers else ""
        
        # Get current active stop info
        active_stop = self.get_active_stop()
        if active_stop and active_stop.in_progress:
            active_stop_info = f"curr_stop={active_stop.id[:8]}@{active_stop.progress_percentage:.1f}%"
        else:
            active_stop_info = f"curr_stop={active_stop.id[:8] if active_stop else 'None'}"
        
        return f"Route[{self.id[:8]}|v={self.vehicle_id[:8]}|{self.status.value}|stops={n_completed}/{n_stops}|{active_stop_info}|{passengers_str}|mods={self.modification_count}]"
    
    @property
    def current_passengers(self) -> List[str]:
        """Get the current passengers in the route"""
        current_passengers = set()
        
        # Track pickups and dropoffs in order
        for stop in self.stops:
            if stop.completed:
                # For completed stops, add passengers who were picked up
                for passenger_id in stop.pickup_passengers:
                    if passenger_id in stop.boarded_request_ids:
                        current_passengers.add(passenger_id)
                        
                # Remove passengers who were dropped off
                for passenger_id in stop.dropoff_passengers:
                    if passenger_id in current_passengers:
                        current_passengers.remove(passenger_id)
            else:
                # For future stops, we use the calculated current_load
                break
                
        return list(current_passengers)

    def get_active_stop(self) -> Optional[RouteStop]:
        """Get the active stop that is currently being approached or serviced"""
        if self.active_stop_id:
            for stop in self.stops:
                if stop.id == self.active_stop_id:
                    return stop
                    
        # Fall back to current stop index if no active stop ID
        return self.get_current_stop()

    def get_current_stop(self) -> Optional[RouteStop]:
        """Get the current stop based on the current_stop_index"""
        if not self.stops:
            return None
            
        if 0 <= self.current_stop_index < len(self.stops):
            return self.stops[self.current_stop_index]
            
        return None
    
    def get_next_stop(self) -> Optional[RouteStop]:
        """Get the next stop in the route after the current active stop"""
        active_stop = self.get_active_stop()
        if not active_stop:
            # Find first uncompleted stop if no active stop
            for stop in self.stops:
                if not stop.completed:
                    return stop
            return None
        
        # Find the next uncompleted stop after the active one
        for i, stop in enumerate(self.stops):
            if stop.id == active_stop.id:
                # Look for the next uncompleted stop
                for j in range(i + 1, len(self.stops)):
                    if not self.stops[j].completed:
                        return self.stops[j]
                break
        
        # If we couldn't find a next stop, return None
        return None
    
    def recalc_current_stop_index(self) -> int:
        """
        Recalculates and updates the current_stop_index based on stop status.
        Returns the new current_stop_index.
        """
        if not self.stops:
            self.current_stop_index = 0
            return 0
        
        # First check if we have an active stop
        for i, stop in enumerate(self.stops):
            if stop.in_progress:
                self.current_stop_index = i
                self.active_stop_id = stop.id
                return i
                
        # Otherwise find the first uncompleted stop
        for i, stop in enumerate(self.stops):
            if not stop.completed:
                self.current_stop_index = i
                return i
                
        # If all stops are completed, set to the last stop
        self.current_stop_index = len(self.stops) - 1
        return self.current_stop_index

    def is_completed(self) -> bool:
        """Check if route is completed (all stops are completed)"""
        return all(stop.completed for stop in self.stops)
        
    def set_active_stop(self, stop_id: str, current_location: Optional[Location] = None) -> bool:
        """
        Set a stop as the active stop being approached
        
        Args:
            stop_id: ID of stop to mark as active
            current_location: Current vehicle location
            
        Returns:
            True if successful, False if stop not found
        """
        for i, stop in enumerate(self.stops):
            # First clear active state on all stops
            stop.in_progress = False
            
            # Then set the specified stop as active
            if stop.id == stop_id:
                stop.mark_in_progress(current_location)
                self.current_stop_index = i
                self.active_stop_id = stop_id
                return True
                
        return False
        
    def mark_stop_completed(self, stop_id: str, 
                          actual_arrival_time: Optional[datetime] = None,
                          actual_departure_time: Optional[datetime] = None,
                          actual_duration: Optional[float] = None, 
                          actual_distance: Optional[float] = None) -> bool:
        """
        Mark a stop as completed with actual metrics
        
        Args:
            stop_id: ID of the stop to mark as completed
            actual_arrival_time: Actual arrival time at the stop
            actual_departure_time: Actual departure time from the stop
            actual_duration: Actual duration of travel to the stop
            actual_distance: Actual distance traveled to the stop
            
        Returns:
            True if successful, False if stop not found
        """
        for i, stop in enumerate(self.stops):
            if stop.id == stop_id:
                stop.mark_completed(
                    actual_arrival_time=actual_arrival_time,
                    actual_departure_time=actual_departure_time,
                    actual_duration=actual_duration,
                    actual_distance=actual_distance
                )
                
                # Clear active stop if this was the active one
                if self.active_stop_id == stop_id:
                    self.active_stop_id = None
                
                # Recalculate current stop index to point to the next uncompleted stop
                self.recalc_current_stop_index()
                
                # Update total metrics
                self.recalculate_total_distance()
                self.recalculate_total_duration()
                
                return True
                
        # If we get here, we couldn't find the stop
        return False
    
    def recalculate_total_distance(self) -> float:
        """Recalculate total distance based on stops"""
        total = 0.0
        for stop in self.stops:
            if stop.completed and stop.actual_distance_to_stop is not None:
                total += stop.actual_distance_to_stop
            else:
                total += stop.estimated_distance_to_stop
                
        self.total_distance = total
        return total
        
    def recalculate_total_duration(self) -> float:
        """Recalculate total duration based on stops and service times"""
        # Sum up stop movement durations
        movement_duration = 0.0
        for stop in self.stops:
            if stop.completed and stop.actual_duration_to_stop is not None:
                movement_duration += stop.actual_duration_to_stop
            else:
                movement_duration += stop.estimated_duration_to_stop
        
        # Add stop service times
        service_duration = 0.0
        for stop in self.stops:
            if stop.has_pending_operations() or stop.completed:
                service_duration += stop.service_time
        
        self.total_duration = movement_duration + service_duration
        return self.total_duration

    def validate_passenger_consistency(self) -> tuple[bool, Optional[str]]:
        """
        Validates that passenger pickups and dropoffs are consistent throughout the route.
        Accounts for completed stops and pending operations.
        
        Returns:
            tuple[bool, Optional[str]]: Validity status and error message if invalid
        """
        try:
            # Separate validation for completed and future operations
            # First check consistency of completed operations
            completed_pickups = {}
            completed_dropoffs = {}
            onboard_from_completed = set()
            
            # Then check planned operations
            planned_pickups = {}
            planned_dropoffs = {}
            
            # Track what's already onboard from completed stops
            for stop in self.stops:
                if stop.completed:
                    # For completed stops, use actual boarding information
                    for request_id in stop.pickup_passengers:
                        if request_id in stop.boarded_request_ids:
                            completed_pickups[request_id] = completed_pickups.get(request_id, 0) + 1
                            onboard_from_completed.add(request_id)
                    
                    # For completed stops, check dropoffs
                    for request_id in stop.dropoff_passengers:
                        completed_dropoffs[request_id] = completed_dropoffs.get(request_id, 0) + 1
                        if request_id in onboard_from_completed:
                            onboard_from_completed.remove(request_id)
                else:
                    # For future stops
                    for request_id in stop.pickup_passengers:
                        planned_pickups[request_id] = planned_pickups.get(request_id, 0) + 1
                    
                    for request_id in stop.dropoff_passengers:
                        planned_dropoffs[request_id] = planned_dropoffs.get(request_id, 0) + 1
            
            # Check for errors in completed operations
            for request_id in completed_pickups:
                if completed_pickups[request_id] > 1:
                    return False, f"Passenger with request id {request_id} was picked up multiple times in completed stops"
            
            for request_id in completed_dropoffs:
                if completed_dropoffs[request_id] > 1:
                    return False, f"Passenger with request id {request_id} was dropped off multiple times in completed stops"
                
                if request_id not in completed_pickups and request_id not in onboard_from_completed:
                    return False, f"Passenger with request id {request_id} was dropped off without being picked up"
            
            # Validate planned operations
            # Each passenger should be picked up exactly once
            for request_id in planned_pickups:
                if request_id in completed_pickups:
                    return False, f"Passenger with request id {request_id} is scheduled for pickup but was already picked up"
                
                if planned_pickups[request_id] != 1:
                    return False, f"Passenger with request id {request_id} is scheduled for pickup {planned_pickups[request_id]} times (should be exactly once)"
            
            # Each passenger should be dropped off exactly once
            for request_id in planned_dropoffs:
                if request_id in completed_dropoffs:
                    return False, f"Passenger with request id {request_id} is scheduled for dropoff but was already dropped off"
                
                if planned_dropoffs[request_id] != 1:
                    return False, f"Passenger with request id {request_id} is scheduled for dropoff {planned_dropoffs[request_id]} times (should be exactly once)"
            
            # All passengers should have both pickup and dropoff
            all_passengers = set(list(completed_pickups.keys()) + list(planned_pickups.keys()))
            all_passengers.update(list(completed_dropoffs.keys()) + list(planned_dropoffs.keys()))
            
            for request_id in all_passengers:
                has_pickup = request_id in completed_pickups or request_id in planned_pickups
                has_dropoff = request_id in completed_dropoffs or request_id in planned_dropoffs
                
                if not has_pickup:
                    return False, f"Passenger with request id {request_id} has a dropoff but no pickup"
                
                if not has_dropoff:
                    return False, f"Passenger with request id {request_id} has a pickup but no dropoff"
            
            # Check for sequence violations (pickup must come before dropoff)
            # This is more complex with mixed completed/pending stops
            pickup_seq = {}
            dropoff_seq = {}
            
            for i, stop in enumerate(self.stops):
                for request_id in stop.pickup_passengers:
                    pickup_seq[request_id] = i
                    
                for request_id in stop.dropoff_passengers:
                    dropoff_seq[request_id] = i
            
            for request_id in all_passengers:
                if request_id in pickup_seq and request_id in dropoff_seq:
                    if pickup_seq[request_id] >= dropoff_seq[request_id]:
                        return False, f"Passenger with request id {request_id} has pickup at or after dropoff"
            
            # All checks passed
            return True, None
            
        except Exception as e:
            import traceback
            logger.error(f"Error in passenger consistency check: {str(e)}\n{traceback.format_exc()}")
            return False, f"Error validating passenger consistency: {str(e)}"
    
    def validate_capacity(self, vehicle_capacity: int) -> tuple[bool, Optional[str]]:
        """
        Validates if the route can be completed within vehicle capacity constraints.
        Accounts for completed operations and their impact on capacity.
        
        Returns:
            tuple[bool, Optional[str]]: Validity status and error message if invalid
        """
        if not self.stops:
            return True, None
            
        # Get current onboard passengers from completed stops
        current_load = len(self.current_passengers)
        
        # Process remaining stops in sequence, starting with current load
        for i, stop in enumerate(self.stops):
            if stop.completed:
                # Skip completed stops as they're already accounted for in current_load
                continue
                
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
        
    def find_stop_for_request(self, request_id: str, is_pickup: bool = True) -> Optional[RouteStop]:
        """Find a stop in the route for a specific request ID"""
        for stop in self.stops:
            if is_pickup and request_id in stop.pickup_passengers:
                return stop
            elif not is_pickup and request_id in stop.dropoff_passengers:
                return stop
        return None
        
    def update_for_reroute(self, current_location: Location, 
                         current_stop_id: Optional[str] = None) -> None:
        """
        Update route for a rerouting operation
        
        Args:
            current_location: Current location of the vehicle
            current_stop_id: ID of the stop the vehicle is currently approaching (if known)
        """
        # Find the active stop
        active_stop = None
        active_stop_index = -1
        
        if current_stop_id:
            for i, stop in enumerate(self.stops):
                if stop.id == current_stop_id:
                    active_stop = stop
                    active_stop_index = i
                    break
        
        if active_stop is None and self.active_stop_id:
            for i, stop in enumerate(self.stops):
                if stop.id == self.active_stop_id:
                    active_stop = stop
                    active_stop_index = i
                    break
        
        # If we have an active stop, update its state
        if active_stop:
            # Update the vehicle location on the active stop
            active_stop.vehicle_current_location = current_location
            
            # Mark stops before the active one as completed if they aren't already
            for i, stop in enumerate(self.stops):
                if i < active_stop_index and not stop.completed:
                    logger.warning(f"Marking stop {stop.id} (seq={stop.sequence}) as completed during reroute")
                    stop.mark_completed()
        
        # Update the active stop ID and recalculate current stop index
        if active_stop:
            self.active_stop_id = active_stop.id
            self.current_stop_index = active_stop_index
        else:
            # If we can't find the active stop, just recalculate
            self.recalc_current_stop_index()
    
    def remove_stop(self, stop_id: str) -> bool:
        """
        Remove a stop from the route
        Used during route modifications when rejecting a stop
        
        Args:
            stop_id: ID of the stop to remove
            
        Returns:
            True if successful, False if stop not found or already completed
        """
        # Find the stop to remove
        stop_index = -1
        for i, stop in enumerate(self.stops):
            if stop.id == stop_id:
                stop_index = i
                break
                
        if stop_index == -1:
            logger.warning(f"Stop {stop_id} not found for removal")
            return False
            
        # Cannot remove completed stops
        if self.stops[stop_index].completed:
            logger.warning(f"Cannot remove completed stop {stop_id}")
            return False
            
        # Remove the stop
        del self.stops[stop_index]
        
        # Update stop sequences
        self.update_stop_sequences()
        
        # Recalculate indices and route metrics
        self.recalc_current_stop_index()
        self.recalculate_total_distance()
        self.recalculate_total_duration()
        
        return True
        
    def update_stop_sequences(self) -> None:
        """Update sequence numbers for all stops"""
        for i, stop in enumerate(self.stops):
            stop.sequence = i
    
    def mark_as_modified(self) -> None:
        """Mark the route as having been modified"""
        self.status = RouteStatus.MODIFIED
        self.last_modification_time = datetime.now()
        self.modification_count += 1
        self.version += 1  # Increment version number when route is modified
        logger.info(f"Route {self.id} modified: version incremented to {self.version}")
        
    def validate_route_integrity(self) -> tuple[bool, Optional[str]]:
        """
        Perform comprehensive validation of the route integrity.
        Combines passenger consistency and other structural checks.
        
        Returns:
            tuple[bool, Optional[str]]: Validity status and error message if invalid
        """
        # Check basic stop sequence consistency
        if not self.stops:
            return True, None  # Empty route is technically valid
            
        # Check passenger consistency
        is_valid, error_msg = self.validate_passenger_consistency()
        if not is_valid:
            return False, error_msg
            
        # Check stop sequencing
        for i in range(len(self.stops) - 1):
            if self.stops[i].sequence >= self.stops[i+1].sequence:
                return False, f"Stops are not in sequential order at position {i}"
        
        # Set the route as consistent
        self.is_consistent = True
        self.last_validation_time = datetime.now()
        
        return True, None

    def to_dict(self) -> Dict[str, Any]:
        """Convert Route to dictionary representation"""
        return {
            'id': self.id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'vehicle_id': self.vehicle_id,
            'status': self.status.value,
            'stops': [stop.to_dict() for stop in self.stops],
            'current_stop_index': self.current_stop_index,
            'scheduled_start_time': self.scheduled_start_time.isoformat() if self.scheduled_start_time else None,
            'scheduled_end_time': self.scheduled_end_time.isoformat() if self.scheduled_end_time else None,
            'actual_start_time': self.actual_start_time.isoformat() if self.actual_start_time else None,
            'actual_end_time': self.actual_end_time.isoformat() if self.actual_end_time else None,
            'total_distance': self.total_distance,
            'total_duration': self.total_duration,
            'active_stop_id': self.active_stop_id,
            'last_modification_time': self.last_modification_time.isoformat() if self.last_modification_time else None,
            'modification_count': self.modification_count,
            'last_validation_time': self.last_validation_time.isoformat() if self.last_validation_time else None,
            'is_consistent': self.is_consistent,
            'version': self.version,
            '_version': self._version
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Route':
        """Create Route from dictionary representation"""
        try:
            # Create the route with required fields
            route = cls(
                vehicle_id=data['vehicle_id'],
                stops=[RouteStop.from_dict(s) for s in data.get('stops', [])],
                status=RouteStatus(data['status']),
                current_stop_index=data.get('current_stop_index', 0),
                total_distance=data.get('total_distance'),
                total_duration=data.get('total_duration'),
                active_stop_id=data.get('active_stop_id'),
                modification_count=data.get('modification_count', 0),
                is_consistent=data.get('is_consistent', True),
                version=data.get('version', 1)
            )
            
            # Handle optional datetime fields
            for dt_field in ['scheduled_start_time', 'scheduled_end_time', 'actual_start_time', 
                          'actual_end_time', 'last_modification_time', 'last_validation_time']:
                if data.get(dt_field):
                    setattr(route, dt_field, datetime.fromisoformat(data[dt_field]))
            
            # Set ID and timestamp fields if provided
            if 'id' in data:
                route.id = data['id']
                
            if 'created_at' in data and data['created_at']:
                route.created_at = datetime.fromisoformat(data['created_at'])
                
            if 'updated_at' in data and data['updated_at']:
                route.updated_at = datetime.fromisoformat(data['updated_at'])
            
            return route
            
        except (KeyError, ValueError) as e:
            raise ValueError(f"Error creating Route from dict: {str(e)}")