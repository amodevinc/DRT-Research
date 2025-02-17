from datetime import timedelta
from typing import Dict, Any, List, Optional

from drt_sim.core.state.manager import StateManager
from drt_sim.core.simulation.context import SimulationContext
from drt_sim.core.monitoring.visualization.manager import VisualizationManager
from drt_sim.models.event import Event, EventType, EventPriority
from drt_sim.models.location import Location
from drt_sim.config.config import ParameterSet
from drt_sim.models.request import Request
from drt_sim.models.stop import StopStatus
from drt_sim.models.metrics import StopMetrics
from drt_sim.algorithms.base_interfaces.stop_selector_base import StopSelector, StopSelectorConfig
from drt_sim.algorithms.base_interfaces.stop_assigner_base import StopAssigner, StopAssignerConfig, StopAssignment
from drt_sim.algorithms.stop.selector.coverage_based import CoverageBasedStopSelector
from drt_sim.algorithms.stop.selector.demand_based import DemandBasedStopSelector
from drt_sim.algorithms.stop.assigner.nearest import NearestStopAssigner
from drt_sim.algorithms.stop.assigner.multi_objective import MultiObjectiveStopAssigner
from drt_sim.algorithms.stop.assigner.accessibility import AccessibilityFocusedAssigner
from drt_sim.network.manager import NetworkManager
import logging
logger = logging.getLogger(__name__)

class StopHandler:
    """
    Handles stop management in the DRT system.
    Manages stop activation, capacity, congestion, and stop-related events.
    Includes background stop selection for dynamic stop network optimization.
    """
    
    def __init__(
        self,
        config: ParameterSet,
        context: SimulationContext,
        state_manager: StateManager,
        network_manager: NetworkManager,
        visualization_manager: Optional[VisualizationManager] = None
    ):
        self.config = config
        self.context = context
        self.state_manager = state_manager
        self.network_manager = network_manager
        self.visualization_manager = visualization_manager
        
        # Initialize core components
        self.stop_thresholds = self._setup_stop_thresholds()
        self.stop_selector = self._init_stop_selector()
        self.stop_assigner = self._init_stop_assigner()
        
        # Selection interval configuration
        self.selection_interval = timedelta(
            seconds=self.config.stop.selection_interval
        ) if hasattr(self.config.stop, 'selection_interval') else timedelta(seconds=300)
        
        # Register with visualization manager if available
        if self.visualization_manager:
            self.visualization_manager.register_component(
                component_id='stop_handler',
                component_type='handler',
                modules=['stop_selector', 'stop_assigner'],
                metadata={
                    'selection_interval': self.selection_interval.total_seconds(),
                    'stop_thresholds': self.stop_thresholds,
                    'selector_type': self.config.algorithm.stop_selector,
                    'assigner_type': self.config.algorithm.stop_assigner
                }
            )
        
        # Schedule initial selection
        self._schedule_background_selection()
        
    def _setup_stop_thresholds(self) -> Dict[str, Any]:
        """Initialize stop operation thresholds from config"""
        return {
            'max_occupancy': self.config.stop.max_occupancy,
            'congestion_threshold': self.config.stop.congestion_threshold,
            'max_vehicle_queue': self.config.stop.max_vehicle_queue,
            'max_passenger_queue': self.config.stop.max_passenger_queue,
            'min_service_interval': self.config.stop.min_service_interval,
            'max_dwell_time': self.config.stop.max_dwell_time
        }

    def _init_stop_selector(self) -> StopSelector:
        """Initialize stop selector based on algorithm configuration"""
        logger.info("Initializing stop selector")
        logger.info(f"Algorithm config: {self.config.algorithm}")
        
        selector_params = self.config.algorithm.stop_selector_params
        logger.info(f"Stop selector params: {selector_params}")
        
        if selector_params is None:
            raise ValueError("Stop selector parameters are missing")
            
        logger.info("Creating StopSelectorConfig")
        selector_config = StopSelectorConfig(**selector_params)
        logger.info(f"Created selector config: {selector_config}")

        if self.config.algorithm.stop_selector == "coverage_based":
            logger.info("Initializing CoverageBasedStopSelector")
            return CoverageBasedStopSelector(
                sim_context=self.context,
                config=selector_config,
                network_manager=self.network_manager,
                state_manager=self.state_manager,
                visualization_manager=self.visualization_manager
            )
        elif self.config.algorithm.stop_selector == "demand_based":
            logger.info("Initializing DemandBasedStopSelector")
            return DemandBasedStopSelector(
                sim_context=self.context,
                config=selector_config,
                network_manager=self.network_manager,
                visualization_manager=self.visualization_manager
            )
        else:
            raise ValueError(f"Invalid stop selector: {self.config.algorithm.stop_selector}")

    def _init_stop_assigner(self) -> StopAssigner:
        """Initialize stop assigner based on algorithm configuration"""
        logger.info("Initializing stop assigner")
        logger.info(f"Algorithm config: {self.config.algorithm}")
        
        assigner_params = self.config.algorithm.stop_assigner_params
        logger.info(f"Stop assigner params: {assigner_params}")
        
        if assigner_params is None:
            raise ValueError("Stop assigner parameters are missing")
            
        logger.info("Creating StopAssignerConfig")
        assigner_config = StopAssignerConfig(**assigner_params)
        logger.info(f"Created assigner config: {assigner_config}")

        if self.config.algorithm.stop_assigner == "nearest":
            logger.info("Initializing NearestStopAssigner")
            return NearestStopAssigner(
                sim_context=self.context,
                config=assigner_config,
                network_manager=self.network_manager,
                state_manager=self.state_manager,
            )
        elif self.config.algorithm.stop_assigner == "multi_objective":
            logger.info("Initializing MultiObjectiveStopAssigner")
            return MultiObjectiveStopAssigner(
                sim_context=self.context,
                config=assigner_config,
                network_manager=self.network_manager,
                state_manager=self.state_manager,
            )
        elif self.config.algorithm.stop_assigner == "accessibility":
            logger.info("Initializing AccessibilityFocusedAssigner")
            return AccessibilityFocusedAssigner(
                sim_context=self.context,
                config=assigner_config,
                network_manager=self.network_manager,
                state_manager=self.state_manager,
            )
        else:
            raise ValueError(f"Invalid stop assigner: {self.config.algorithm.stop_assigner}")

    def _schedule_background_selection(self) -> None:
        """Schedule periodic stop selection check"""
        event = Event(
            event_type=EventType.STOP_SELECTION_TICK,
            priority=EventPriority.LOW,
            timestamp=self.context.current_time + self.selection_interval,
            data={'interval': self.selection_interval.total_seconds()}
        )
        self.context.event_manager.publish_event(event)

    def handle_stop_selection_tick(self, event: Event) -> None:
        """Handle periodic stop selection check"""
        try:
            self.state_manager.begin_transaction()
            #TODO: Implement stop selection for background selection
            self.state_manager.commit_transaction()
            
        except Exception as e:
            self.state_manager.rollback_transaction()
            logger.error(f"Error in background stop selection: {str(e)}")
            self._handle_stop_error(event, str(e))

    def _analyze_demand_patterns(self) -> Dict[str, float]:
        """Analyze demand patterns for stop selection"""
        selection_window = timedelta(hours=1)  # Default 2-hour window
        if hasattr(self.config.stop, 'selection_window'):
            selection_window = timedelta(seconds=self.config.stop.selection_window)
        
        # Get recent requests and usage data
        request_history = self.state_manager.request_worker.get_recent_requests(
            time_window=selection_window
        )
        stop_usage = self.state_manager.stop_worker.get_recent_requests(
            time_window=selection_window
        )
        
        # Process demand patterns
        demand_patterns = {}
        
        # Analyze request patterns
        for request in request_history:
            pickup_area = self._get_area_id(request.origin)
            dropoff_area = self._get_area_id(request.destination)
            
            demand_patterns[pickup_area] = demand_patterns.get(pickup_area, 0) + 1
            demand_patterns[dropoff_area] = demand_patterns.get(dropoff_area, 0) + 1
            
        # Include stop usage
        for stop_id, usage_count in stop_usage.items():
            stop = self.state_manager.stop_worker.get_stop(stop_id)
            if stop:
                area_id = self._get_area_id(stop.location)
                demand_patterns[area_id] = demand_patterns.get(area_id, 0) + usage_count
            
        # Calculate relative changes from baseline
        baseline = self._get_baseline_demand()
        return {
            area_id: count / max(baseline.get(area_id, 1), 1)
            for area_id, count in demand_patterns.items()
        }

    def _get_baseline_demand(self) -> Dict[str, float]:
        """Get baseline demand levels for each area"""
        # In a real implementation, this would likely come from historical data
        # or predictions. For now, we'll use a simple rolling average.
        baseline_window = timedelta(days=1)  # Use 24-hour baseline
        
        historical_requests = self.state_manager.request_worker.get_historical_requests(
            time_window=baseline_window
        )
        
        baseline = {}
        for request in historical_requests:
            area_id = self._get_area_id(request.origin)
            baseline[area_id] = baseline.get(area_id, 0) + 1
            
        return baseline

    def _get_area_id(self, location: Location) -> str:
        """Get area identifier for a location"""
        # This is a simplified grid-based area identification
        # In practice, you might use actual service area definitions
        grid_size = 0.01  # Roughly 1km grid cells
        lat_grid = int(location.lat / grid_size)
        lon_grid = int(location.lon / grid_size)
        return f"area_{lat_grid}_{lon_grid}"

    async def handle_determine_virtual_stops(self, event: Event) -> None:
        """Handle determination of optimal virtual stops for pickup and dropoff"""
        try:
            logger.info(f"Handling determine virtual stops for request {event.request_id}")
            self.state_manager.begin_transaction()
            
            request = self.state_manager.request_worker.get_request(event.request_id)
            if not request:
                raise ValueError(f"Request {event.request_id} not found")
            
            # Get available stops
            available_stops = self.state_manager.stop_worker.get_active_stops()
            
            try:
                # First try normal stop assignment
                assignment = await self.stop_assigner.assign_stops(
                    request=request,
                    available_stops=available_stops,
                )
                logger.info("Assignment from stop assigner: %s", assignment)
                
                # Add normal assignment to state
                self.state_manager.stop_assignment_worker.add_assignment(assignment)
                
            except ValueError as e:
                # If no viable stops found, create virtual stops using the selector
                origin_stop, dest_stop = await self.stop_selector.create_virtual_stops_for_request(
                    request=request
                )
                
                # Add virtual stops to state
                # Verify stops don't already exist
                if not self.state_manager.stop_worker.get_stop(origin_stop.id):
                    self.state_manager.stop_worker.create_new_stop(origin_stop)
                if not self.state_manager.stop_worker.get_stop(dest_stop.id):
                    self.state_manager.stop_worker.create_new_stop(dest_stop)


                # Calculate walking distances and times
                distance_to_origin_stop = await self.network_manager.calculate_distance(
                    request.origin, 
                    origin_stop.location, 
                    network_type='walk'
                )
                distance_from_destination_stop = await self.network_manager.calculate_distance(
                    dest_stop.location, 
                    request.destination, 
                    network_type='walk'
                )
                walking_speed = self.config.network.walking_speed
                walking_time_to_origin_stop = distance_to_origin_stop / walking_speed
                walking_time_from_destination_stop = distance_from_destination_stop / walking_speed
                
                # Create assignment with virtual stops
                assignment = StopAssignment(
                    request_id=request.id,
                    origin_stop=origin_stop,
                    destination_stop=dest_stop,
                    walking_distance_origin=distance_to_origin_stop,
                    walking_distance_destination=distance_from_destination_stop,
                    walking_time_origin=walking_time_to_origin_stop,
                    walking_time_destination=walking_time_from_destination_stop,
                    expected_passenger_origin_stop_arrival_time = self.context.current_time + timedelta(walking_time_to_origin_stop),
                    total_score=1.0,
                    alternative_origins=[],
                    alternative_destinations=[],
                    assignment_time=self.context.current_time,
                    metadata={
                        'virtual_stops_created': True,
                        'assignment_type': 'virtual',
                        'origin_stop_id': origin_stop.id,
                        'destination_stop_id': dest_stop.id
                    }
                )
                
                # Add virtual assignment to state
                self.state_manager.stop_assignment_worker.add_assignment(assignment)
            
            logger.info(f"Creating match request event for request: {request.id}, stop assignment: {assignment.id}")
            # Create event to proceed with vehicle matching
            self._create_match_request_event(assignment.id)
            
            self.state_manager.commit_transaction()
            
        except Exception as e:
            import traceback
            self.state_manager.rollback_transaction()
            logger.error(f"Error determining virtual stops: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self._handle_stop_error(event, str(e))
    
    def _handle_stop_congestion(self, stop_id: str, metrics: StopMetrics) -> None:
        """Handle stop entering congested state"""
        self.state_manager.stop_worker.update_stop_status(
            stop_id,
            StopStatus.CONGESTED,
            {
                'congestion_start': self.context.current_time,
                'vehicles_queued': metrics.vehicles_queued,
                'waiting_passengers': metrics.waiting_passengers
            }
        )
        
        metrics.record_congestion_event()
        
        # Create congestion event
        event = Event(
            event_type=EventType.STOP_CONGESTED,
            priority=EventPriority.HIGH,
            timestamp=self.context.current_time,
            stop_id=stop_id,
            data={
                'metrics': metrics.to_dict()
            }
        )
        self.context.event_manager.publish_event(event)

    def _handle_stop_congestion_cleared(self, stop_id: str) -> None:
        """Handle stop exiting congested state"""
        self.state_manager.stop_worker.update_stop_status(
            stop_id,
            StopStatus.ACTIVE,
            {
                'congestion_end': self.context.current_time,
                'congestion_duration': (
                    self.context.current_time - 
                    self.state_manager.stop_worker.get_stop(stop_id).congestion_start
                ).total_seconds()
            }
        )

    def _handle_no_viable_stops(self, request: Request) -> None:
        """Handle case when no viable stops are found for a request"""
        event = Event(
            event_type=EventType.REQUEST_NO_VEHICLE,
            priority=EventPriority.HIGH,
            timestamp=self.context.current_time,
            request_id=request.id,
            data={
                'reason': 'no_viable_stops',
                'request_data': request.to_dict()
            }
        )
        self.context.event_manager.publish_event(event)

    def handle_stop_activation_request(self, event: Event) -> None:
        """Handle request to activate a stop"""
        try:
            self.state_manager.begin_transaction()
            
            stop_id = event.stop_id
            stop = self.state_manager.stop_worker.get_stop(stop_id)
            
            if not stop:
                raise ValueError(f"Stop {stop_id} not found")
            
            if stop.status == StopStatus.INACTIVE:
                activation_metadata = {
                    'activation_time': self.context.current_time,
                    'activation_reason': event.data.get('reason', 'Scheduled activation'),
                    'expected_duration': event.data.get('duration')
                }
                
                self.state_manager.stop_worker.update_stop_status(
                    stop_id,
                    StopStatus.ACTIVE,
                    activation_metadata
                )
                
                self._create_stop_activated_event(stop_id, activation_metadata)
            
            self.state_manager.commit_transaction()
            
        except Exception as e:
            self.state_manager.rollback_transaction()
            logger.error(f"Error handling stop activation: {str(e)}")
            self._handle_stop_error(event, str(e))

    def handle_stop_deactivation_request(self, event: Event) -> None:
        """Handle request to deactivate a stop"""
        try:
            self.state_manager.begin_transaction()
            
            stop_id = event.stop_id
            stop = self.state_manager.stop_worker.get_stop(stop_id)
            if not stop:
                raise ValueError(f"Stop {stop_id} not found")
            
            if stop.status not in [StopStatus.ACTIVE, StopStatus.CONGESTED]:
                logger.warning(f"Stop {stop_id} is not active or congested")
                self.state_manager.commit_transaction()
                return
            
            # Check if stop can be safely deactivated
            metrics = self.state_manager.stop_worker.get_stop_metrics(stop_id)
            if metrics.current_occupancy > 0 or metrics.vehicles_queued > 0:
                logger.warning(f"Cannot deactivate stop {stop_id} - stop is in use")
                self.state_manager.commit_transaction()
                return
                
            # Check for active assignments using this stop
            stop_assignments = self.state_manager.stop_assignment_worker.get_assignments_for_stop(stop_id)
            if stop_assignments:
                logger.error(f"Cannot deactivate stop {stop_id} - has active assignments")
                self.state_manager.commit_transaction()
                return
            
            deactivation_metadata = {
                'deactivation_time': self.context.current_time,
                'deactivation_reason': event.data.get('reason', 'Scheduled deactivation'),
                'final_metrics': metrics.to_dict()
            }
            
            self.state_manager.stop_worker.update_stop_status(
                stop_id,
                StopStatus.INACTIVE,
                deactivation_metadata
            )
            
            self._create_stop_deactivated_event(stop_id, deactivation_metadata)
            
            self.state_manager.commit_transaction()
            
        except Exception as e:
            self.state_manager.rollback_transaction()
            logger.error(f"Error handling stop deactivation: {str(e)}")
            self._handle_stop_error(event, str(e))

    def _create_match_request_event(self, stop_assignment_id: str) -> None:
        """Create event to match request to vehicle"""
        event = Event(
            event_type=EventType.MATCH_REQUEST_TO_VEHICLE,
            priority=EventPriority.HIGH,
            timestamp=self.context.current_time,
            data={
                'stop_assignment_id': stop_assignment_id
            }
        )
        self.context.event_manager.publish_event(event)

    def _create_stops_updated_event(self, modified_ids: List[str]) -> None:
        """Create event for stop updates"""
        event = Event(
            event_type=EventType.STOPS_UPDATED,
            priority=EventPriority.HIGH,
            timestamp=self.context.current_time,
            data={
                'modified_stops': modified_ids,
                'update_type': 'background_selection'
            }
        )
        self.context.event_manager.publish_event(event)

    def _create_stop_activated_event(self, stop_id: str, metadata: Dict[str, Any]) -> None:
        """Create event for stop activation"""
        event = Event(
            event_type=EventType.STOP_ACTIVATED,
            priority=EventPriority.HIGH,
            timestamp=self.context.current_time,
            stop_id=stop_id,
            data=metadata
        )
        self.context.event_manager.publish_event(event)

    def _create_stop_deactivated_event(self, stop_id: str, metadata: Dict[str, Any]) -> None:
        """Create event for stop deactivation"""
        event = Event(
            event_type=EventType.STOP_DEACTIVATED,
            priority=EventPriority.HIGH,
            timestamp=self.context.current_time,
            stop_id=stop_id,
            data=metadata
        )
        self.context.event_manager.publish_event(event)

    def _create_stop_capacity_exceeded_event(self, stop_id: str, current_occupancy: int) -> None:
        """Create event for stop capacity violation"""
        event = Event(
            event_type=EventType.STOP_CAPACITY_EXCEEDED,
            priority=EventPriority.HIGH,
            timestamp=self.context.current_time,
            stop_id=stop_id,
            data={
                'current_occupancy': current_occupancy,
                'max_occupancy': self.stop_thresholds['max_occupancy']
            }
        )
        self.context.event_manager.publish_event(event)

    def _is_valid_virtual_stop_location(self, location: Location) -> bool:
        """Validate location for virtual stop creation"""
        if not self._location_in_service_area(location):
            return False
        
        # Check minimum distance from existing stops
        active_stops = self.state_manager.stop_worker.get_active_stops()
        for stop in active_stops:
            distance = self._calculate_distance(location, stop.location)
            if distance < self.config.stop.min_stop_spacing:
                return False
        
        return True

    def _location_in_service_area(self, location: Location) -> bool:
        """Check if location is within service area"""
        # TODO: Implement actual service area check using config boundaries
        return True  # Placeholder implementation

    def _calculate_distance(self, loc1: Location, loc2: Location) -> float:
        """Calculate distance between two locations"""
        # TODO: Implement actual distance calculation (e.g., Haversine)
        return 0.0  # Placeholder implementation

    def _handle_stop_error(self, event: Event, error_msg: str) -> None:
        """Handle errors in stop event processing"""
        logger.error(f"Error processing stop event {event.id}: {error_msg}")
        error_event = Event(
            event_type=EventType.SIMULATION_ERROR,
            priority=EventPriority.CRITICAL,
            timestamp=self.context.current_time,
            stop_id=event.stop_id if hasattr(event, 'stop_id') else None,
            data={
                'error': error_msg,
                'original_event': event.to_dict(),
                'error_type': 'stop_processing_error'
            }
        )
        self.context.event_manager.publish_event(error_event)

    def cleanup(self) -> None:
        """Clean up handler resources"""
        if self.visualization_manager:
            self.visualization_manager.flush_component('stop_handler')