from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Set
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
    MODIFIED = "modified"  # New status to track modifications

class SegmentExecutionStatus(Enum):
    """Tracks the execution status of a segment"""
    PENDING = "pending"      # Not yet started
    IN_PROGRESS = "in_progress"  # Vehicle is currently on this segment
    COMPLETED = "completed"  # Segment has been completed
    SKIPPED = "skipped"      # Segment was skipped due to rerouting

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
    # New fields for tracking modifications
    initial_sequence: Optional[int] = None  # Original sequence before any modifications
    modification_count: int = 0  # Number of times this stop has been modified

    def __post_init__(self):
        super().__init__()
        if self.initial_sequence is None:
            self.initial_sequence = self.sequence

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
        
        return f"RouteStop[{self.id[:8]}|ActualStopId={self.stop.id}|seq={self.sequence}|load={self.current_load}|{ops_str}{arr_time}{dep_time}|{status}]"

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
            'modification_count': self.modification_count
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
            modification_count=data.get('modification_count', 0)
        )
        
        # Set the ID and timestamps if they exist
        if 'id' in data:
            route_stop.id = data['id']
        if 'created_at' in data and data['created_at']:
            route_stop.created_at = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data and data['updated_at']:
            route_stop.updated_at = datetime.fromisoformat(data['updated_at'])
            
        return route_stop
        
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

@dataclass
class RouteSegment(ModelBase):
    """Represents a segment of a route between two stops or locations"""
    origin: Optional[RouteStop] = None  # Made optional to handle vehicle's current position
    destination: Optional[RouteStop] = None  # Made optional to handle return-to-base
    estimated_duration: float = 0.0  # in seconds
    estimated_distance: float = 0.0  # in meters
    completed: bool = False
    actual_duration: Optional[float] = None
    actual_distance: Optional[float] = None
    origin_location: Optional[Location] = None  # Added to handle non-stop origins
    destination_location: Optional[Location] = None  # Added to handle non-stop destinations
    waypoints: List[Dict[str, Any]] = field(default_factory=list)  # List of waypoints along the path
    current_waypoint_index: int = 0  # Index of current waypoint in path
    last_waypoint_update: Optional[datetime] = None  # Timestamp of last waypoint update
    movement_start_time: Optional[datetime] = None  # Timestamp of when the segment movement started
    # New fields for enhanced segment tracking
    execution_status: SegmentExecutionStatus = SegmentExecutionStatus.PENDING
    progress_percentage: float = 0.0  # Progress from 0 to 100
    vehicle_current_location: Optional[Location] = None  # Current location in segment if in progress
    is_active: bool = False  # Flag to mark active segment being executed

    def __post_init__(self):
        super().__init__()
        self._validate()
        
    def _validate(self) -> None:
        """Validate segment has valid origin and destination"""
        has_origin = bool(self.origin or self.origin_location)
        has_destination = bool(self.destination or self.destination_location)
        
        if not has_origin or not has_destination:
            raise ValueError("RouteSegment must have either origin/destination stops or locations")

    def __str__(self) -> str:
        """Provides a concise string representation of the route segment"""
        status_map = {
            SegmentExecutionStatus.PENDING: "⌛",
            SegmentExecutionStatus.IN_PROGRESS: "→",
            SegmentExecutionStatus.COMPLETED: "✓",
            SegmentExecutionStatus.SKIPPED: "⨯"
        }
        status = status_map.get(self.execution_status, "?")
        
        origin_id = self.origin.id[:8] if self.origin else (self.origin_location.id[:8] if self.origin_location else "N/A")
        dest_id = self.destination.id[:8] if self.destination else (self.destination_location.id[:8] if self.destination_location else "N/A")
        dist = f"{self.actual_distance:.1f}m" if self.actual_distance else f"{self.estimated_distance:.1f}m"
        dur = f"{self.actual_duration:.1f}s" if self.actual_duration else f"{self.estimated_duration:.1f}s"
        waypoint_info = f"|wp={self.current_waypoint_index}/{len(self.waypoints)}" if self.waypoints else ""
        active_str = "ACTIVE" if self.is_active else ""
        
        return f"Segment[{self.id[:8]}|{origin_id}→{dest_id}|{dist}|{dur}{waypoint_info}|{status}|{self.progress_percentage:.1f}%|{active_str}]"

    def to_dict(self) -> Dict[str, Any]:
        """Convert RouteSegment to dictionary representation"""
        base_dict = {
            'id': self.id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
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
            'movement_start_time': self.movement_start_time.isoformat() if self.movement_start_time else None,
            'execution_status': self.execution_status.value,
            'progress_percentage': self.progress_percentage,
            'vehicle_current_location': self.vehicle_current_location.to_dict() if self.vehicle_current_location else None,
            'is_active': self.is_active
        }
        return base_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RouteSegment':
        """Create RouteSegment from dictionary representation"""
        try:
            # Handle origin and destination carefully
            origin = None
            if data.get('origin'):
                origin = RouteStop.from_dict(data['origin'])
                
            destination = None
            if data.get('destination'):
                destination = RouteStop.from_dict(data['destination'])
                
            origin_location = None
            if data.get('origin_location'):
                origin_location = Location.from_dict(data['origin_location'])
                
            destination_location = None
            if data.get('destination_location'):
                destination_location = Location.from_dict(data['destination_location'])
                
            vehicle_current_location = None
            if data.get('vehicle_current_location'):
                vehicle_current_location = Location.from_dict(data['vehicle_current_location'])
            
            # Handle execution status
            execution_status = SegmentExecutionStatus.PENDING
            if data.get('execution_status'):
                execution_status = SegmentExecutionStatus(data['execution_status'])
            
            segment = cls(
                origin=origin,
                destination=destination,
                estimated_duration=data.get('estimated_duration', 0.0),
                estimated_distance=data.get('estimated_distance', 0.0),
                completed=data.get('completed', False),
                actual_duration=data.get('actual_duration'),
                actual_distance=data.get('actual_distance'),
                origin_location=origin_location,
                destination_location=destination_location,
                waypoints=data.get('waypoints', []),
                current_waypoint_index=data.get('current_waypoint_index', 0),
                execution_status=execution_status,
                progress_percentage=data.get('progress_percentage', 0.0),
                vehicle_current_location=vehicle_current_location,
                is_active=data.get('is_active', False)
            )
            
            # Handle timestamps
            if data.get('last_waypoint_update'):
                segment.last_waypoint_update = datetime.fromisoformat(data['last_waypoint_update'])
                
            if data.get('movement_start_time'):
                segment.movement_start_time = datetime.fromisoformat(data['movement_start_time'])
            
            # Set ID and creation timestamps
            if 'id' in data:
                segment.id = data['id']
                
            if 'created_at' in data and data['created_at']:
                segment.created_at = datetime.fromisoformat(data['created_at'])
                
            if 'updated_at' in data and data['updated_at']:
                segment.updated_at = datetime.fromisoformat(data['updated_at'])
                
            return segment
            
        except (ValueError, KeyError) as e:
            raise ValueError(f"Error creating RouteSegment from dict: {str(e)}")

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
        if not self.waypoints:
            return False
            
        if self.current_waypoint_index + 1 < len(self.waypoints):
            self.current_waypoint_index += 1
            self.last_waypoint_update = datetime.now()
            
            # Update progress percentage
            if len(self.waypoints) > 1:
                self.progress_percentage = (self.current_waypoint_index / (len(self.waypoints) - 1)) * 100
            else:
                self.progress_percentage = 100.0
                
            return True
            
        return False

    def get_progress(self) -> float:
        """Get progress through the segment as a percentage"""
        if self.execution_status == SegmentExecutionStatus.COMPLETED:
            return 100.0
            
        if self.execution_status == SegmentExecutionStatus.SKIPPED:
            return 100.0
            
        return self.progress_percentage
        
    def get_origin_location(self) -> Optional[Location]:
        """Get the origin location regardless of whether it's from a stop or direct location"""
        if self.origin and self.origin.stop:
            return self.origin.stop.location
        return self.origin_location
        
    def get_destination_location(self) -> Optional[Location]:
        """Get the destination location regardless of whether it's from a stop or direct location"""
        if self.destination and self.destination.stop:
            return self.destination.stop.location
        return self.destination_location
        
    def mark_completed(self, actual_duration: Optional[float] = None, actual_distance: Optional[float] = None) -> None:
        """Mark this segment as completed with actual metrics"""
        self.completed = True
        self.execution_status = SegmentExecutionStatus.COMPLETED
        self.progress_percentage = 100.0
        self.is_active = False
        
        if actual_duration is not None:
            self.actual_duration = actual_duration
            
        if actual_distance is not None:
            self.actual_distance = actual_distance
        
        # If actual values not provided, use estimates
        if self.actual_duration is None:
            self.actual_duration = self.estimated_duration
            
        if self.actual_distance is None:
            self.actual_distance = self.estimated_distance
            
    def mark_in_progress(self, current_location: Optional[Location] = None) -> None:
        """Mark this segment as in progress"""
        self.execution_status = SegmentExecutionStatus.IN_PROGRESS
        self.is_active = True
        
        if current_location:
            self.vehicle_current_location = current_location
            
    def mark_skipped(self) -> None:
        """Mark this segment as skipped (due to rerouting)"""
        self.execution_status = SegmentExecutionStatus.SKIPPED
        self.completed = True
        self.progress_percentage = 100.0
        self.is_active = False
        
    def update_progress(self, percentage: float, current_location: Optional[Location] = None) -> None:
        """Update progress through the segment"""
        self.progress_percentage = max(0.0, min(100.0, percentage))
        
        if current_location:
            self.vehicle_current_location = current_location
            
        # Update waypoint index based on progress if we have waypoints
        if self.waypoints and len(self.waypoints) > 1:
            target_index = int((self.progress_percentage / 100.0) * (len(self.waypoints) - 1))
            self.current_waypoint_index = max(0, min(len(self.waypoints) - 1, target_index))

@dataclass
class Route(ModelBase):
    """Represents a complete vehicle route"""
    vehicle_id: str
    stops: List[RouteStop]
    status: RouteStatus
    segments: List[RouteSegment] = field(default_factory=list)
    current_segment_index: int = 0
    scheduled_start_time: Optional[datetime] = None
    scheduled_end_time: Optional[datetime] = None
    actual_start_time: Optional[datetime] = None
    actual_end_time: Optional[datetime] = None
    total_distance: Optional[float] = None
    total_duration: Optional[float] = None
    # New fields for enhanced route tracking
    active_segment_id: Optional[str] = None  # ID of the segment currently being executed
    last_modification_time: Optional[datetime] = None
    modification_count: int = 0
    last_validation_time: Optional[datetime] = None
    is_consistent: bool = True  # Flag indicating if route is in a consistent state

    def __post_init__(self):
        super().__init__()
        # Initialize total distance and duration if not set
        if self.total_distance is None and self.segments:
            self.recalculate_total_distance()
            
        if self.total_duration is None and self.segments:
            self.recalculate_total_duration()

    def __str__(self) -> str:
        """Provides a concise string representation of the route"""
        n_completed = sum(1 for stop in self.stops if stop.completed)
        n_stops = len(self.stops)
        n_active_segments = sum(1 for seg in self.segments if seg.is_active)
        active_segment_str = f"active_seg={n_active_segments}" if n_active_segments > 0 else ""
        
        # Get list of current passengers
        current_passengers = self.current_passengers
        passengers_str = f"passengers={len(current_passengers)}" if current_passengers else ""
        
        # Get current active segment info
        active_segment = self.get_active_segment()
        active_segment_info = f"curr_seg={active_segment.id[:8]}@{active_segment.progress_percentage:.1f}%" if active_segment else ""
        
        return f"Route[{self.id[:8]}|v={self.vehicle_id[:8]}|{self.status.value}|stops={n_completed}/{n_stops}|{active_segment_info}|{active_segment_str}|{passengers_str}|mods={self.modification_count}]"
    
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

    def get_active_segment(self) -> Optional[RouteSegment]:
        """Get the active segment that is currently being executed"""
        if self.active_segment_id:
            for segment in self.segments:
                if segment.id == self.active_segment_id:
                    return segment
                    
        # Fall back to current segment index if no active segment ID
        return self.get_current_segment()

    def get_current_segment(self) -> Optional[RouteSegment]:
        """Get the current segment based on the current_segment_index"""
        if not self.segments:
            return None
            
        if 0 <= self.current_segment_index < len(self.segments):
            return self.segments[self.current_segment_index]
            
        return None
    
    def recalc_current_segment_index(self) -> int:
        """
        Recalculates and updates the current_segment_index based on segment execution status.
        Returns the new current_segment_index.
        """
        if not self.segments:
            self.current_segment_index = 0
            return 0
        
        # First check if we have an active segment
        for i, segment in enumerate(self.segments):
            if segment.is_active or segment.execution_status == SegmentExecutionStatus.IN_PROGRESS:
                self.current_segment_index = i
                self.active_segment_id = segment.id
                return i
                
        # Otherwise find the first pending segment
        for i, segment in enumerate(self.segments):
            if segment.execution_status == SegmentExecutionStatus.PENDING:
                self.current_segment_index = i
                return i
                
        # If all segments are completed or skipped, set to the last segment
        self.current_segment_index = len(self.segments) - 1
        return self.current_segment_index
    
    def get_next_stop(self) -> Optional[RouteStop]:
        """Get the next stop in the route"""
        active_segment = self.get_active_segment()
        if active_segment and active_segment.destination:
            return active_segment.destination
            
        # If no current segment or segment has no destination stop,
        # find first uncompleted stop
        for stop in self.stops:
            if not stop.completed:
                return stop
                
        return None

    def is_completed(self) -> bool:
        """Check if route is completed"""
        if not self.segments:
            # Check based on stops if no segments
            return all(stop.completed for stop in self.stops)
            
        # Check if all segments are either completed or skipped
        for segment in self.segments:
            if segment.execution_status not in [SegmentExecutionStatus.COMPLETED, SegmentExecutionStatus.SKIPPED]:
                return False
                
        return True
        
    def set_active_segment(self, segment_id: str, current_location: Optional[Location] = None) -> bool:
        """
        Set a segment as the active segment being executed
        
        Args:
            segment_id: ID of segment to mark as active
            current_location: Current vehicle location
            
        Returns:
            True if successful, False if segment not found
        """
        for i, segment in enumerate(self.segments):
            # First clear active state on all segments
            segment.is_active = False
            
            # Then set the specified segment as active
            if segment.id == segment_id:
                segment.is_active = True
                segment.mark_in_progress(current_location)
                self.current_segment_index = i
                self.active_segment_id = segment_id
                return True
                
        return False
        
    def mark_segment_completed(self, 
                               segment_id: Optional[str] = None,
                               segment_index: Optional[int] = None, 
                               actual_duration: Optional[float] = None, 
                               actual_distance: Optional[float] = None) -> bool:
        """
        Mark a segment as completed.
        
        Args:
            segment_id: ID of segment to mark completed
            segment_index: Index of segment to mark completed
            actual_duration: Actual duration of the segment in seconds
            actual_distance: Actual distance of the segment in meters
            
        Returns:
            True if successfully marked segment as completed, False otherwise
        """
        # First try to find by ID if provided
        if segment_id:
            for i, segment in enumerate(self.segments):
                if segment.id == segment_id:
                    segment.mark_completed(actual_duration, actual_distance)
                    
                    # Clear active segment if this was the active one
                    if self.active_segment_id == segment_id:
                        self.active_segment_id = None
                    
                    # Recalculate current segment index
                    self.recalc_current_segment_index()
                    
                    # Update total metrics
                    self.recalculate_total_distance()
                    self.recalculate_total_duration()
                    
                    return True
        
        # Otherwise try by index if provided            
        if segment_index is not None:
            idx = segment_index
            if 0 <= idx < len(self.segments):
                segment = self.segments[idx]
                segment.mark_completed(actual_duration, actual_distance)
                
                # Clear active segment if this was the active one
                if self.active_segment_id == segment.id:
                    self.active_segment_id = None
                
                # Recalculate current segment index
                self.recalc_current_segment_index()
                
                # Update total metrics
                self.recalculate_total_distance()
                self.recalculate_total_duration()
                
                return True
                
        # If we get here, we couldn't find the segment
        return False
    
    def recalculate_total_distance(self) -> float:
        """Recalculate total distance based on segments"""
        if not self.segments:
            self.total_distance = 0.0
            return 0.0
            
        # Sum up actual distances for completed segments, estimated for others
        total = 0.0
        for segment in self.segments:
            if segment.execution_status == SegmentExecutionStatus.SKIPPED:
                # Skip distance for skipped segments
                continue
                
            if segment.execution_status == SegmentExecutionStatus.COMPLETED and segment.actual_distance is not None:
                total += segment.actual_distance
            else:
                total += segment.estimated_distance
                
        self.total_distance = total
        return total
        
    def recalculate_total_duration(self) -> float:
        """Recalculate total duration based on segments and stop service times"""
        if not self.segments:
            self.total_duration = 0.0
            return 0.0
            
        # Sum up segment durations, skipping segments that were skipped due to rerouting
        segment_duration = 0.0
        for segment in self.segments:
            # Skip duration for skipped segments
            if segment.execution_status == SegmentExecutionStatus.SKIPPED:
                continue
                
            if segment.execution_status == SegmentExecutionStatus.COMPLETED and segment.actual_duration is not None:
                segment_duration += segment.actual_duration
            else:
                segment_duration += segment.estimated_duration
        
        # Add stop service times for stops that haven't been skipped
        service_duration = 0.0
        for stop in self.stops:
            if stop.has_pending_operations() or stop.completed:
                service_duration += stop.service_time
        
        self.total_duration = segment_duration + service_duration
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
        
    def mark_stop_completed(self, stop_id: str, actual_arrival_time: Optional[datetime] = None, 
                        actual_departure_time: Optional[datetime] = None) -> bool:
        """
        Mark a stop as completed with actual timing information
        
        Args:
            stop_id: ID of the stop to mark as completed
            actual_arrival_time: When the vehicle actually arrived
            actual_departure_time: When the vehicle actually departed
            
        Returns:
            True if successful, False if stop not found
        """
        for stop in self.stops:
            if stop.id == stop_id:
                stop.completed = True
                
                if actual_arrival_time:
                    stop.actual_arrival_time = actual_arrival_time
                    
                if actual_departure_time:
                    stop.actual_departure_time = actual_departure_time
                    
                return True
                
        return False
        
    def update_segments_for_reroute(self, active_vehicle_location: Location, 
                                 current_segment_id: Optional[str] = None) -> None:
        """
        Update segment states for a rerouting operation
        
        Args:
            active_vehicle_location: Current location of the vehicle
            current_segment_id: ID of the segment the vehicle is currently on (if known)
        """
        # Find the active segment
        active_segment = None
        active_segment_index = -1
        
        if current_segment_id:
            for i, segment in enumerate(self.segments):
                if segment.id == current_segment_id:
                    active_segment = segment
                    active_segment_index = i
                    break
        
        if active_segment is None and self.active_segment_id:
            for i, segment in enumerate(self.segments):
                if segment.id == self.active_segment_id:
                    active_segment = segment
                    active_segment_index = i
                    break
        
        # If we have an active segment, update its state
        if active_segment:
            # Update the vehicle location on the active segment
            active_segment.vehicle_current_location = active_vehicle_location
            
            # Mark segments before the active one as completed if they aren't already
            for i, segment in enumerate(self.segments):
                if i < active_segment_index and segment.execution_status == SegmentExecutionStatus.PENDING:
                    segment.mark_skipped()
        
        # Update the active segment ID and recalculate current segment index
        if active_segment:
            self.active_segment_id = active_segment.id
            self.current_segment_index = active_segment_index
        else:
            # If we can't find the active segment, just recalculate
            self.recalc_current_segment_index()
    
    def remove_stop_and_related_segments(self, stop_id: str) -> bool:
        """
        Remove a stop and its related segments from the route
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
            
        # Find segments that involve this stop
        segments_to_remove = []
        for i, segment in enumerate(self.segments):
            origin_is_target = segment.origin and segment.origin.id == stop_id
            dest_is_target = segment.destination and segment.destination.id == stop_id
            
            if origin_is_target or dest_is_target:
                segments_to_remove.append(i)
        
        # Remove segments (in reverse order to avoid index issues)
        for i in sorted(segments_to_remove, reverse=True):
            del self.segments[i]
            
        # Remove the stop
        del self.stops[stop_index]
        
        # Update stop sequences
        self.update_stop_sequences()
        
        # Recalculate segment indices and route metrics
        self.recalc_current_segment_index()
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
            
        # Check segment consistency - each segment must connect consecutive stops
        for i, segment in enumerate(self.segments):
            # Skip segments without proper stops
            if not segment.origin or not segment.destination:
                continue
                
            # Find the stops in the route
            origin_idx = -1
            dest_idx = -1
            
            for j, stop in enumerate(self.stops):
                if stop.id == segment.origin.id:
                    origin_idx = j
                if stop.id == segment.destination.id:
                    dest_idx = j
            
            # Check that the stops exist and are in the right order
            if origin_idx == -1 or dest_idx == -1:
                return False, f"Segment {i} connects stops that don't exist in the route"
                
            if dest_idx <= origin_idx:
                return False, f"Segment {i} connects stops in the wrong order (origin index: {origin_idx}, destination index: {dest_idx})"
                
            # Check for gaps in segments
            if i > 0 and dest_idx - origin_idx > 1:
                # There might be stops between the origin and destination
                # that don't have connecting segments
                return False, f"Segment {i} skips intermediate stops"
        
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
            'segments': [segment.to_dict() for segment in self.segments],
            'current_segment_index': self.current_segment_index,
            'scheduled_start_time': self.scheduled_start_time.isoformat() if self.scheduled_start_time else None,
            'scheduled_end_time': self.scheduled_end_time.isoformat() if self.scheduled_end_time else None,
            'actual_start_time': self.actual_start_time.isoformat() if self.actual_start_time else None,
            'actual_end_time': self.actual_end_time.isoformat() if self.actual_end_time else None,
            'total_distance': self.total_distance,
            'total_duration': self.total_duration,
            'active_segment_id': self.active_segment_id,
            'last_modification_time': self.last_modification_time.isoformat() if self.last_modification_time else None,
            'modification_count': self.modification_count,
            'last_validation_time': self.last_validation_time.isoformat() if self.last_validation_time else None,
            'is_consistent': self.is_consistent
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
                current_segment_index=data.get('current_segment_index', 0),
                total_distance=data.get('total_distance'),
                total_duration=data.get('total_duration'),
                active_segment_id=data.get('active_segment_id'),
                modification_count=data.get('modification_count', 0),
                is_consistent=data.get('is_consistent', True)
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
            
            # Add segments if provided
            if 'segments' in data:
                route.segments = [RouteSegment.from_dict(s) for s in data['segments']]
            
            return route
            
        except (KeyError, ValueError) as e:
            raise ValueError(f"Error creating Route from dict: {str(e)}")