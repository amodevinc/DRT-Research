from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, Any
import traceback
from drt_sim.core.monitoring.types.metrics import MetricName
from drt_sim.core.state.manager import StateManager
from drt_sim.core.simulation.context import SimulationContext
from drt_sim.models.event import Event, EventType, EventPriority
from drt_sim.models.request import Request, RequestStatus
from drt_sim.models.location import Location
from drt_sim.config.config import ScenarioConfig
from drt_sim.core.logging_config import setup_logger
from drt_sim.network.manager import NetworkManager
from drt_sim.models.matching import Assignment
logger = setup_logger(__name__)

@dataclass
class RequestValidationResult:
    """Results of request validation"""
    is_valid: bool
    errors: list[str]
    warnings: list[str]

class RequestHandler:
    """
    Handles the complete lifecycle of transportation requests in the DRT system.
    """
    
    def __init__(
        self,
        config: ScenarioConfig,
        context: SimulationContext,
        state_manager: StateManager,
        network_manager: NetworkManager
    ):
        self.config = config
        self.context = context
        self.state_manager = state_manager
        self.network_manager = network_manager
        self.validation_rules = self._setup_validation_rules()
        self.initialized = True
        
    def _setup_validation_rules(self) -> Dict[str, Any]:
        """Initialize request validation rules from config"""
        return {
            'max_walking_distance': self.config.service.max_walking_distance,
        }

    def handle_request_received(self, event: Event) -> None:
        """Handle initial receipt of a new transportation request."""
        if not self.initialized:
            raise RuntimeError("Handler not initialized")
            
        try:
            self.state_manager.begin_transaction()
            
            # Extract request data and create request object
            request: Request = event.data['request']
            
            # Add request to state management
            self.state_manager.request_worker.add_request(request)
            
            # Validate request
            validation_result = self._validate_request(request)
            
            if validation_result.is_valid:
                # Update status and create validation event
                self.state_manager.request_worker.update_request_status(
                    request.id,
                    RequestStatus.VALIDATED
                )
                self._create_determine_virtual_stops_event(request)
            else:
                # Update status and create rejection event
                self.state_manager.request_worker.update_request_status(
                    request.id,
                    RequestStatus.REJECTED
                )
                self._create_request_validation_failed_event(request, validation_result)
            
            self.state_manager.commit_transaction()
            logger.info(f"Processed new request {request.id}")
            
        except Exception as e:
            self.state_manager.rollback_transaction()
            logger.error(f"Error processing request: {str(e)}\n{traceback.format_exc()}")
            self._handle_request_error(event, str(e))
    
    def handle_request_rejected(self, event: Event) -> None:
        """Handle request rejection."""
        if not self.initialized:
            raise RuntimeError("Handler not initialized")
            
        try:
            self.state_manager.begin_transaction()
            
            request = self.state_manager.request_worker.get_request(event.request_id)
            if not request:
                raise ValueError(f"Request {event.request_id} not found")
            
            rejection_metadata = {
                'rejection_time': self.context.current_time,
                'rejection_reason': event.data.get('reason', 'Unknown reason'),
                'rejection_stage': request.status.value
            }
            
            self.state_manager.request_worker.update_request_status(
                request.id,
                RequestStatus.REJECTED,
            )
            
            self.state_manager.commit_transaction()
            logger.info(f"Request {request.id} rejected: {rejection_metadata['rejection_reason']}")
            
            # Log rejection metrics
            if self.context.metrics_collector:
                self.context.metrics_collector.log(
                    MetricName.REQUEST_REJECTED,
                    1,
                    {
                        'request_id': request.id,
                        'rejection_reason': rejection_metadata['rejection_reason'],
                        'rejection_time': self.context.current_time.isoformat()
                    }
                )
            
        except Exception as e:
            self.state_manager.rollback_transaction()
            logger.error(f"Error handling request rejection: {str(e)}")
            self._handle_request_error(event, str(e))

    def handle_request_validation_failed(self, event: Event) -> None:
        """Handle failed request validation."""
        try:
            self.state_manager.begin_transaction()
            
            request = self.state_manager.request_worker.get_request(event.request_id)
            if not request:
                raise ValueError(f"Request {event.request_id} not found")
            
            self.state_manager.request_worker.update_request_status(
                request.id,
                RequestStatus.REJECTED,
            )
            
            self.state_manager.commit_transaction()
            
        except Exception as e:
            self.state_manager.rollback_transaction()
            logger.error(f"Error handling validation failure: {str(e)}")
            self._handle_request_error(event, str(e))

    def handle_request_assigned(self, event: Event) -> None:
        """Handle successful request assignment to vehicle."""
        try:
            self.state_manager.begin_transaction()
            assignment: Assignment | None = event.data.get('assignment', None)
            if not assignment:
                raise ValueError("Assignment not found in event data")
            
            request = self.state_manager.request_worker.get_request(assignment.request_id)
            if not request:
                raise ValueError(f"Request {assignment.request_id} not found")
            
            self.state_manager.request_worker.update_request_status(
                request.id,
                RequestStatus.ASSIGNED,
            )
            
            # Create passenger journey start event
            self._create_passenger_journey_start_event(assignment)
            self._create_update_route_request_event(assignment)
            
            self.state_manager.commit_transaction()
            
            # Calculate and log assignment delay
            if self.context.metrics_collector:
                stop_assignment = self.state_manager.stop_assignment_worker.get_assignment(assignment.stop_assignment_id)
                self.context.metrics_collector.log(
                    MetricName.REQUEST_ASSIGNED,
                    1,
                    {
                        'request_id': assignment.request_id,
                        'vehicle_id': assignment.vehicle_id,
                        'assignment_time': self.context.current_time.isoformat(),
                        'origin_stop': stop_assignment.origin_stop.location.to_dict(),
                        'destination_stop': stop_assignment.destination_stop.location.to_dict()
                    }
                )
            
        except Exception as e:
            self.state_manager.rollback_transaction()
            logger.error(f"Error handling request assignment: {str(e)}\n{traceback.format_exc()}")
            self._handle_request_error(event, str(e))

    def handle_request_cancelled(self, event: Event) -> None:
        """Handle request cancellation."""
        try:
            self.state_manager.begin_transaction()
            
            request = self.state_manager.request_worker.get_request(event.request_id)
            if not request:
                raise ValueError(f"Request {event.request_id} not found")
            
            cancellation_metadata = {
                'cancellation_time': self.context.current_time,
                'cancellation_reason': event.data.get('reason'),
                'cancellation_stage': request.status.value
            }
            
            self.state_manager.request_worker.update_request_status(
                request.id,
                RequestStatus.CANCELLED,
                cancellation_metadata
            )
            
            self._create_vehicle_request_cancelled_event(request, cancellation_metadata)
            
            self.state_manager.commit_transaction()
            
        except Exception as e:
            self.state_manager.rollback_transaction()
            logger.error(f"Error handling request cancellation: {str(e)}")
            self._handle_request_error(event, str(e))

    def _validate_request(self, request: Request) -> RequestValidationResult:
        """Validate request against business rules."""
        errors = []
        warnings = []
        
        # Service hours validation
        if not self._is_within_service_hours(request):
            errors.append("Requested time outside service hours")
        
        # Advance booking validation
        if not self._is_valid_booking_time(request):
            errors.append("Request exceeds maximum advance booking period")
        
        # Service area validation
        if not self._is_within_service_area(request):
            errors.append("Location outside service area")
        
        # Walking distance validation
        pickup_walk = self._validate_walking_distance(request.origin)
        dropoff_walk = self._validate_walking_distance(request.destination)
        
        if pickup_walk > self.validation_rules['max_walking_distance']:
            errors.append("Pickup location too far from nearest stop")
        if dropoff_walk > self.validation_rules['max_walking_distance']:
            errors.append("Dropoff location too far from nearest stop")
        
        # Minimum notice period validation
        if not self._has_minimum_notice(request):
            errors.append("Request does not meet minimum notice period")
        
        return RequestValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def _is_within_service_hours(self, request: Request) -> bool:
        """Check if request time falls within service hours."""
        service_hours = self.validation_rules['service_hours'] if 'service_hours' in self.validation_rules else None
        if service_hours:
            request_time = request.request_time.time()
            return service_hours['start'] <= request_time <= service_hours['end']
        return True

    def _is_valid_booking_time(self, request: Request) -> bool:
        """Validate advance booking time."""
        if not request.request_time:
            return True
        
        max_days = self.validation_rules['max_advance_booking_days'] if 'max_advance_booking_days' in self.validation_rules else None
        if max_days:
            max_future = self.context.current_time + timedelta(days=max_days)
            return request.request_time <= max_future
        return True

    def _is_within_service_area(self, request: Request) -> bool:
        """Check if locations are within service area."""
        service_area = self.validation_rules['service_area'] if 'service_area' in self.validation_rules else None
        if service_area:
            return (
                self._location_in_polygon(request.origin, service_area) and
                self._location_in_polygon(request.destination, service_area)
            )
        return True

    def _location_in_polygon(self, location: Location, polygon: list) -> bool:
        """Check if location falls within a polygon."""
        # Implement point-in-polygon check
        return True  # Placeholder implementation

    def _validate_walking_distance(self, location: Location) -> float:
        """Calculate walking distance to nearest stop."""
        # Implement actual distance calculation
        return 0.0  # Placeholder implementation

    def _has_minimum_notice(self, request: Request) -> bool:
        """Check if request meets minimum notice period."""
        min_notice = timedelta(minutes=self.validation_rules['min_notice_minutes']) if 'min_notice_minutes' in self.validation_rules else None
        if not request.request_time:
            return True
        if min_notice:
            return request.request_time >= self.context.current_time + min_notice
        return True

    def _create_request_validated_event(self, request: Request, validation_result: RequestValidationResult) -> None:
        """Create REQUEST_VALIDATED event."""
        event = Event(
            event_type=EventType.REQUEST_VALIDATED,
            priority=EventPriority.HIGH,
            timestamp=self.context.current_time,
            request_id=request.id,
            passenger_id=request.passenger_id,
            data={
                'warnings': validation_result.warnings,
                'origin': request.origin.to_dict(),
                'destination': request.destination.to_dict(),
                'requested_pickup_time': request.request_time
            }
        )
        self.context.event_manager.publish_event(event)

    def _create_request_validation_failed_event(self, request: Request, validation_result: RequestValidationResult) -> None:
        """Create REQUEST_VALIDATION_FAILED event."""
        event = Event(
            event_type=EventType.REQUEST_VALIDATION_FAILED,
            priority=EventPriority.HIGH,
            timestamp=self.context.current_time,
            request_id=request.id,
            passenger_id=request.passenger_id,
            data={
                'errors': validation_result.errors,
                'origin': request.origin.to_dict(),
                'destination': request.destination.to_dict(),
                'requested_pickup_time': request.pickup_time
            }
        )
        self.context.event_manager.publish_event(event)
    
    def _create_determine_virtual_stops_event(self, request: Request) -> None:
        """Create DETERMINE_VIRTUAL_STOPS event."""
        event = Event(
            event_type=EventType.DETERMINE_VIRTUAL_STOPS,
            priority=EventPriority.HIGH,
            timestamp=self.context.current_time,
            request_id=request.id,
            passenger_id=request.passenger_id,
            data={
                'origin': request.origin.to_dict(),
                'destination': request.destination.to_dict(),
            }
        )
        logger.info(f"Creating DETERMINE_VIRTUAL_STOPS event for request {request.id}")
        self.context.event_manager.publish_event(event)

    def _create_passenger_journey_start_event(self, assignment: Assignment) -> None:
        """Create event for passenger to start journey to pickup."""
        event = Event(
            event_type=EventType.START_PASSENGER_JOURNEY,
            priority=EventPriority.HIGH,
            timestamp=self.context.current_time,
            data={
                'assignment': assignment
            }
        )
        self.context.event_manager.publish_event(event)

    def _create_vehicle_request_cancelled_event(self, request: Request, cancellation_metadata: Dict[str, Any]) -> None:
        """Create event to notify vehicle of cancellation."""
        event = Event(
            event_type=EventType.REQUEST_CANCELLED,
            priority=EventPriority.HIGH,
            timestamp=self.context.current_time,
            request_id=request.id,
            data={
                'cancellation_metadata': cancellation_metadata,
            }
        )
        self.context.event_manager.publish_event(event)

    def _handle_request_error(self, event: Event, error_msg: str) -> None:
        """Handle errors in request processing."""
        logger.error(f"Error processing request event {event.id}: {error_msg}")
        error_event = Event(
            event_type=EventType.SIMULATION_ERROR,
            priority=EventPriority.CRITICAL,
            timestamp=self.context.current_time,
            request_id=event.request_id,
            data={
                'error': error_msg,
                'original_event': event.to_dict(),
                'error_type': 'request_processing_error'
            }
        )
        self.context.event_manager.publish_event(error_event)

    def handle_request_expired(self, event: Event) -> None:
        """Handle request expiration."""
        try:
            self.state_manager.begin_transaction()
            
            request = self.state_manager.request_worker.get_request(event.request_id)
            if not request:
                raise ValueError(f"Request {event.request_id} not found")
            
            expiration_metadata = {
                'expiration_time': self.context.current_time,
                'expiration_reason': event.data.get('reason', 'Request timed out')
            }
            
            self.state_manager.request_worker.update_request_status(
                request.id,
                RequestStatus.EXPIRED,
                expiration_metadata
            )
            
            self.state_manager.commit_transaction()
            
        except Exception as e:
            self.state_manager.rollback_transaction()
            logger.error(f"Error handling request expiration: {str(e)}")
            self._handle_request_error(event, str(e))

    def handle_request_no_vehicle(self, event: Event) -> None:
        """Handle case when no suitable vehicle is found."""
        try:
            self.state_manager.begin_transaction()
            
            request = self.state_manager.request_worker.get_request(event.request_id)
            if not request:
                raise ValueError(f"Request {event.request_id} not found")
            
            no_vehicle_metadata = {
                'no_vehicle_time': self.context.current_time,
                'reason': event.data.get('reason', 'No suitable vehicle available')
            }
            
            self.state_manager.request_worker.update_request_status(
                request.id,
                RequestStatus.REJECTED,
                no_vehicle_metadata
            )
            
            self.state_manager.commit_transaction()
            
        except Exception as e:
            self.state_manager.rollback_transaction()
            logger.error(f"Error handling no vehicle available: {str(e)}")
            self._handle_request_error(event, str(e))

    def _create_update_route_request_event(self, assignment: Assignment) -> None:
        """Create event to update vehicle route after request assignment."""
        event = Event(
            event_type=EventType.ROUTE_UPDATE_REQUEST,
            priority=EventPriority.NORMAL,
            timestamp=self.context.current_time,
            request_id=assignment.request_id,
            data={
                'assignment': assignment
            }
        )
        self.context.event_manager.publish_event(event)