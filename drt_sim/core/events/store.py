from typing import Dict, List, Optional, Set, Any, Tuple
from datetime import datetime
import os
import json
from collections import defaultdict
import threading
from enum import Enum
import heapq
from dataclasses import dataclass, asdict, field
import logging
import copy

from drt_sim.models.event import Event, EventType, EventStatus
from drt_sim.models.base import SimulationEncoder
logger = logging.getLogger(__name__)

class EventHistoryStore:
    """
    Enhanced event history storage with advanced indexing and dual-purpose outputs.
    
    This class provides:
    1. Rich debug-oriented history with event lifecycle tracking
    2. Entity-centered visualization data for simulation replay
    3. Efficient querying by various dimensions
    4. Automatic file persistence with configurable formats
    """
    
    def __init__(
        self,
        max_memory_events: int = 10000,
        auto_save_events: bool = True,
        debug_file_path: str = "debug_events.jsonl",
        viz_file_path: str = "visualization_events.json",
        save_interval: int = 1000  # Save after this many new events
    ):
        # Core storage
        self.events_by_id: Dict[str, List[Event]] = defaultdict(list)
        self.events_by_timestamp: List[Event] = []  # Sorted list for time-based lookups
        
        # Indexing for fast lookups
        self.events_by_type: Dict[EventType, List[str]] = defaultdict(list)  # event_type -> [event_ids]
        self.events_by_entity: Dict[str, Dict[str, Set[str]]] = {
            "vehicle": defaultdict(set),    # vehicle_id -> set(event_ids)
            "request": defaultdict(set),    # request_id -> set(event_ids)
            "passenger": defaultdict(set),  # passenger_id -> set(event_ids)
            "route": defaultdict(set),      # route_id -> set(event_ids)
            "stop": defaultdict(set),       # stop_id -> set(event_ids)
        }
        
        # Lifecycle tracking - map original event IDs to all their updates
        self.event_lifecycle: Dict[str, List[str]] = defaultdict(list)
        
        # Visualization-specific storage (entity-centered)
        self.entity_timelines: Dict[str, Dict[str, List[Dict]]] = {
            "vehicle": defaultdict(list),    # vehicle_id -> [state_changes]
            "request": defaultdict(list),    # request_id -> [state_changes]
            "passenger": defaultdict(list),  # passenger_id -> [state_changes]
            "route": defaultdict(list),      # route_id -> [state_changes]
            "stop": defaultdict(list),       # stop_id -> [state_changes]
        }
        
        # File persistence
        self.max_memory_events = max_memory_events
        self.auto_save_events = auto_save_events
        self.debug_file_path = debug_file_path
        self.viz_file_path = viz_file_path
        self.save_interval = save_interval
        self.event_count = 0
        
        # Thread safety
        self.lock = threading.RLock()
    
    def add_event(self, event: Event) -> None:
        """
        Add an event to the history store with comprehensive indexing.
        
        This method:
        1. Updates all indexes
        2. Maintains event lifecycle tracking
        3. Updates visualization timelines
        4. Handles periodic saving if enabled
        """
        with self.lock:
            event_id = event.id
            
            # Store the event itself
            self.events_by_id[event_id].append(event)
            
            # Add to timestamp-ordered list (using binary search would be more efficient)
            # For simplicity, we'll append and sort, but in production you'd use insort
            self.events_by_timestamp.append(event)
            self.events_by_timestamp.sort(key=lambda e: e.timestamp)
            
            # Update type index
            self.events_by_type[event.event_type].append(event_id)
            
            # Update entity indexes
            if event.vehicle_id:
                self.events_by_entity["vehicle"][event.vehicle_id].add(event_id)
            if event.request_id:
                self.events_by_entity["request"][event.request_id].add(event_id)
            if event.passenger_id:
                self.events_by_entity["passenger"][event.passenger_id].add(event_id)
            if event.route_id:
                self.events_by_entity["route"][event.route_id].add(event_id)
            if event.stop_id:
                self.events_by_entity["stop"][event.stop_id].add(event_id)
            
            # Update lifecycle tracking
            # Check if this is a status update for an existing event
            if event.status != EventStatus.PENDING:
                # Look for existing event with same ID but different status
                existing_events = [e for e in self.events_by_id.get(event_id, []) 
                                   if e.status != event.status]
                if existing_events:
                    # This is an update to an existing event
                    self.event_lifecycle[event_id].append(event_id)
            
            # Update visualization timelines
            self._update_visualization_timelines(event)
            
            # Increment event count and potentially save
            self.event_count += 1
            if self.auto_save_events and self.event_count % self.save_interval == 0:
                self.save_to_files()
            
            # Trim memory if needed
            if len(self.events_by_timestamp) > self.max_memory_events:
                self._trim_memory_history()
    
    def _update_visualization_timelines(self, event: Event) -> None:
        """
        Update entity-centered timelines for visualization.
        Skip events with 'processing' status unless they're crucial for visualization.
        """
        # Skip processing events for visualization to reduce data volume
        # We only want completed, failed, or cancelled events (important state transitions)
        if event.status == EventStatus.PROCESSING:
            return
        # Extract relevant state change info based on event type
        viz_data = self._create_visualization_data(event)
        
        if not viz_data:
            return  # Skip events not relevant to visualization
            
        # Add to appropriate timeline(s)
        entity_type, entity_id = self._get_primary_entity(event)
        if entity_type and entity_id:
            self.entity_timelines[entity_type][entity_id].append(viz_data)
    
    def _create_visualization_data(self, event: Event) -> Optional[Dict]:
        """
        Extract visualization-relevant data from an event.
        Only include data that's important for visualization and omit unnecessary details.
        """
        # Base data included in all visualization events
        base_data = {
            "timestamp": event.timestamp.isoformat(),
            "event_type": event.event_type.value,
            "status": event.status.value,
        }
        
        # Add location if available - crucial for map visualization
        if event.location:
            base_data["location"] = event.location
        
        # Add event-type specific data - only include what's necessary for visualization
        if event.event_type == EventType.VEHICLE_POSITION_UPDATE:
            base_data["vehicle_id"] = event.vehicle_id
            # Extract only the specific fields needed for visualization
            base_data["progress"] = event.data["progress_percentage"]
            base_data["location"] = event.data["location"]
        elif event.event_type == EventType.PASSENGER_WALKING_TO_PICKUP:
            base_data["passenger_id"] = event.passenger_id
            base_data["request_id"] = event.request_id
            base_data["origin"] = event.data["origin"]
            base_data["destination"] = event.data["destination"]
            base_data["estimated_walking_time"] = event.data["estimated_walking_time"]
            base_data["estimated_walking_distance"] = event.data["estimated_walking_distance"]
            if "walking_path" in event.data:
                base_data["walking_path"] = event.data["walking_path"]
        elif event.event_type == EventType.PASSENGER_WALKING_TO_DESTINATION:
            base_data["passenger_id"] = event.passenger_id
            base_data["request_id"] = event.request_id
            base_data["origin"] = event.data["origin"]
            base_data["destination"] = event.data["destination"]
            base_data["estimated_walking_time"] = event.data["estimated_walking_time"]
            base_data["estimated_walking_distance"] = event.data["estimated_walking_distance"]
            if "walking_path" in event.data:
                base_data["walking_path"] = event.data["walking_path"]
        elif event.event_type == EventType.PASSENGER_BOARDING_COMPLETED:
            base_data["passenger_id"] = event.passenger_id
            base_data["vehicle_id"] = event.vehicle_id
            base_data["request_id"] = event.request_id
        elif event.event_type == EventType.PASSENGER_ALIGHTING_COMPLETED:
            base_data["passenger_id"] = event.passenger_id
            base_data["vehicle_id"] = event.vehicle_id
            base_data["request_id"] = event.request_id
            base_data["location"] = event.data["location"]
        elif event.event_type == EventType.PASSENGER_ARRIVED_DESTINATION:
            base_data["passenger_id"] = event.passenger_id
            base_data["request_id"] = event.request_id
            base_data["location"] = event.data["location"]
            base_data["walking_time"] = event.data["walking_time"]
        elif event.event_type == EventType.VEHICLE_ARRIVED_STOP:
            base_data["vehicle_id"] = event.vehicle_id
            stop = event.data["route_stop"].stop
            base_data["stop_id"] = stop.id
            loc = stop.location
            base_data["location"] = loc
        elif event.event_type == EventType.REQUEST_REJECTED:
            base_data["request_id"] = event.request_id
            if "reason" in event.data:
                base_data["reason"] = event.data["reason"]
        elif event.event_type == EventType.STOP_ACTIVATED:
            base_data["stop_id"] = event.stop_id
            base_data["location"] = event.data["location"]
            base_data["state"] = "active"
        elif event.event_type == EventType.STOP_DEACTIVATED:
            base_data["stop_id"] = event.stop_id
            base_data["location"] = event.data["location"]
            base_data["state"] = "inactive"
        elif event.event_type == EventType.STOP_CONGESTED:
            base_data["stop_id"] = event.stop_id
            base_data["location"] = event.data["location"]
            base_data["state"] = "congested"
        elif event.event_type == EventType.STOP_CAPACITY_EXCEEDED:
            base_data["stop_id"] = event.stop_id
            base_data["location"] = event.data["location"]
            base_data["state"] = "capacity_exceeded"
                
        # Add more event type handlers as needed
        
        # Special case handling: If this is an event type we don't specifically handle
        # but it has important IDs, include those for entity tracking
        if not any(key in base_data for key in ["vehicle_id", "passenger_id", "request_id", "stop_id", "route_id"]):
            if event.vehicle_id:
                base_data["vehicle_id"] = event.vehicle_id
            if event.passenger_id:
                base_data["passenger_id"] = event.passenger_id
            if event.request_id:
                base_data["request_id"] = event.request_id
            if event.stop_id:
                base_data["stop_id"] = event.stop_id
            if event.route_id:
                base_data["route_id"] = event.route_id
                
        return base_data
    
    def _get_primary_entity(self, event: Event) -> Tuple[Optional[str], Optional[str]]:
        """Determine the primary entity affected by this event"""
        # Priority order for determining primary entity
        if event.vehicle_id:
            return "vehicle", event.vehicle_id
        elif event.passenger_id:
            return "passenger", event.passenger_id
        elif event.request_id:
            return "request", event.request_id
        elif event.route_id:
            return "route", event.route_id
        elif event.stop_id:
            return "stop", event.stop_id
        else:
            return None, None
    
    def _trim_memory_history(self) -> None:
        """Remove oldest events to stay under memory limit"""
        # Calculate how many to remove
        excess = len(self.events_by_timestamp) - self.max_memory_events
        if excess <= 0:
            return
            
        # Remove oldest events
        events_to_remove = self.events_by_timestamp[:excess]
        self.events_by_timestamp = self.events_by_timestamp[excess:]
        
        # Update all indexes
        for event in events_to_remove:
            event_id = event.id
            
            # Remove from ID index
            if event_id in self.events_by_id:
                self.events_by_id[event_id].remove(event)
                if not self.events_by_id[event_id]:
                    del self.events_by_id[event_id]
            
            # Remove from type index
            if event.event_type in self.events_by_type:
                if event_id in self.events_by_type[event.event_type]:
                    self.events_by_type[event.event_type].remove(event_id)
            
            # Remove from entity indexes
            for entity_type, entity_id, index_dict in [
                ("vehicle", event.vehicle_id, self.events_by_entity["vehicle"]),
                ("request", event.request_id, self.events_by_entity["request"]),
                ("passenger", event.passenger_id, self.events_by_entity["passenger"]),
                ("route", event.route_id, self.events_by_entity["route"]),
                ("stop", event.stop_id, self.events_by_entity["stop"])
            ]:
                if entity_id and entity_id in index_dict:
                    index_dict[entity_id].discard(event_id)
                    if not index_dict[entity_id]:
                        del index_dict[entity_id]
    
    def get_event_by_id(self, event_id: str) -> List[Event]:
        """Get all versions of an event by its ID"""
        with self.lock:
            return self.events_by_id.get(event_id, [])
    
    def get_events_by_type(self, event_type: EventType) -> List[Event]:
        """Get all events of a specific type"""
        with self.lock:
            event_ids = self.events_by_type.get(event_type, [])
            return [event for event_id in event_ids 
                    for event in self.events_by_id.get(event_id, [])]
    
    def get_events_by_entity(
        self, 
        entity_type: str, 
        entity_id: str
    ) -> List[Event]:
        """Get all events associated with a specific entity"""
        with self.lock:
            if entity_type not in self.events_by_entity:
                return []
                
            event_ids = self.events_by_entity[entity_type].get(entity_id, set())
            return [event for event_id in event_ids 
                    for event in self.events_by_id.get(event_id, [])]
    
    def get_events_in_timerange(
        self, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[Event]:
        """Get all events within a time range"""
        with self.lock:
            return [event for event in self.events_by_timestamp 
                    if start_time <= event.timestamp <= end_time]
    
    def get_entity_timeline(
        self, 
        entity_type: str, 
        entity_id: str
    ) -> List[Dict]:
        """Get the visualization timeline for a specific entity"""
        with self.lock:
            if entity_type not in self.entity_timelines:
                return []
            return self.entity_timelines[entity_type].get(entity_id, [])
    
    def get_all_entity_timelines(self) -> Dict[str, Dict[str, List[Dict]]]:
        """Get all entity timelines for visualization"""
        with self.lock:
            # Create a deep copy to avoid external modifications
            return copy.deepcopy(self.entity_timelines)
    
    def save_to_files(self) -> None:
        """Save both debug and visualization data to files"""
        self._save_debug_events()
        self._save_visualization_events()
    
    def _save_debug_events(self) -> None:
        """Save debug-oriented event history to file"""
        try:
            # Use JSONL format for debug events (one event per line)
            with open(self.debug_file_path, 'w') as f:
                for event in self.events_by_timestamp:
                    # Convert each event to JSON and write as a line
                    f.write(json.dumps(event.to_dict(), cls=SimulationEncoder) + '\n')
                    
            logger.info(f"Saved {len(self.events_by_timestamp)} debug events to {self.debug_file_path}")
        except Exception as e:
            logger.error(f"Error saving debug events: {str(e)}")
    
    def _save_visualization_events(self) -> None:
        """Save visualization-oriented event data to file"""
        try:
            # Structure the visualization data for efficient replay
            viz_data = {
                "metadata": {
                    "event_count": self.event_count,
                    "generated_at": datetime.now().isoformat(),
                    "version": "1.0"
                },
                "timelines": self.entity_timelines
            }
            
            with open(self.viz_file_path, 'w') as f:
                json.dump(viz_data, f, indent=2, cls=SimulationEncoder)
                
            logger.info(f"Saved visualization data to {self.viz_file_path}")
        except Exception as e:
            logger.error(f"Error saving visualization events: {str(e)}")
    
    def load_from_files(self) -> None:
        """Load event history from files"""
        self._load_debug_events()
        # We don't typically need to load visualization events as they're derived from debug events
    
    def _load_debug_events(self) -> None:
        """Load debug events from file and rebuild indexes"""
        if not os.path.exists(self.debug_file_path):
            logger.warning(f"Debug events file {self.debug_file_path} not found")
            return
            
        try:
            # Clear current data
            with self.lock:
                self.events_by_id.clear()
                self.events_by_timestamp.clear()
                self.events_by_type.clear()
                self.event_lifecycle.clear()
                for entity_type in self.events_by_entity:
                    self.events_by_entity[entity_type].clear()
                for entity_type in self.entity_timelines:
                    self.entity_timelines[entity_type].clear()
                
            # Load events and rebuild indexes
            with open(self.debug_file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        event_dict = json.loads(line)
                        event = Event.from_dict(event_dict)
                        self.add_event(event)
                        
            logger.info(f"Loaded {len(self.events_by_timestamp)} events from {self.debug_file_path}")
        except Exception as e:
            logger.error(f"Error loading debug events: {str(e)}")