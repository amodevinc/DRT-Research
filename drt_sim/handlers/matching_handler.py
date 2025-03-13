from typing import List, Optional
from datetime import datetime
import asyncio

from drt_sim.models.event import Event, EventType, EventPriority
from drt_sim.models.request import Request, RequestStatus
from drt_sim.models.vehicle import Vehicle, VehicleStatus
from drt_sim.models.passenger import PassengerState, PassengerStatus
from drt_sim.models.route import RouteStatus
from drt_sim.config.config import ParameterSet
from drt_sim.core.simulation.context import SimulationContext
from drt_sim.core.state.manager import StateManager
from drt_sim.network.manager import NetworkManager
from drt_sim.algorithms.base_interfaces.matching_base import (
    MatchingStrategy, Assignment
)
from drt_sim.algorithms.optimization.global_optimizer import GlobalSystemOptimizer
from drt_sim.core.user.user_profile_manager import UserProfileManager
from drt_sim.core.services.route_service import RouteService
from drt_sim.models.rejection import RejectionReason, RejectionMetadata
from drt_sim.core.user.user_acceptance_manager import UserAcceptanceManager
from drt_sim.core.monitoring.types.metrics import MetricName
import logging
import traceback
logger = logging.getLogger(__name__)

class MatchingHandler:
    """Handles request-vehicle matching in the DRT system."""
    
    def __init__(
        self,
        config: ParameterSet,
        context: SimulationContext,
        state_manager: StateManager,
        network_manager: NetworkManager,
        user_profile_manager: UserProfileManager,
        route_service: RouteService,
        user_acceptance_manager: UserAcceptanceManager
    ):
        self.config = config
        self.context = context
        self.state_manager = state_manager
        self.network_manager = network_manager
        self.user_profile_manager = user_profile_manager
        self.route_service = route_service
        self.user_acceptance_manager = user_acceptance_manager
        # Initialize matching strategy
        self.matching_strategy = self._initialize_matching_strategy()
        
        # Initialize global optimizer
        self.global_optimizer = GlobalSystemOptimizer(
            config=self.config.matching.optimization_config,
            context=self.context,
            state_manager=self.state_manager,
            network_manager=self.network_manager
        )
        
        # Performance tracking
        self.matching_metrics = {
            'total_requests_processed': 0,
            'successful_matches': 0,
            'failed_matches': 0,
            'average_matching_time': 0.0,
            'average_waiting_time': 0.0,
            'user_accepted_matches': 0,
            'user_rejected_matches': 0,
            'user_acceptance_rate': 0.0
        }
        
        # Lock for synchronizing vehicle assignments
        self._assignment_lock = asyncio.Lock()
        
        # Schedule periodic optimization
        self._schedule_periodic_optimization()
    
    def _initialize_matching_strategy(self) -> MatchingStrategy:
        """Initialize the matching strategy based on configuration."""
        
        return MatchingStrategy(
            self.context,
            self.config.matching,
            self.network_manager,
            self.state_manager,
            self.user_profile_manager,
            self.route_service
        )
    
    def _schedule_periodic_optimization(self) -> None:
        """Schedule periodic optimization using new recurring event system"""
        optimization_interval = self.config.matching.optimization_config.optimization_interval
        
        self.context.event_manager.schedule_recurring_event(
            event_type=EventType.SCHEDULED_GLOBAL_OPTIMIZATION,
            start_time=self.context.current_time,
            interval_seconds=optimization_interval,
            data={'scheduled': True}
        )
        
        logger.info(f"Scheduled periodic optimization every {optimization_interval} seconds")
    
    
    async def handle_match_request_to_vehicle(self, event: Event) -> None:
        """Handle immediate dispatch request event."""
        try:
            stop_assignment_id = event.data.get('stop_assignment_id')

            if not stop_assignment_id:
                raise ValueError("Stop assignment ID is missing")
            stop_assignment = self.state_manager.stop_assignment_worker.get_assignment(stop_assignment_id)

            if not stop_assignment:
                raise ValueError("Stop assignment not found")
            request_id = stop_assignment.request_id
            
            logger.debug(f"Processing match to vehicle for request: {request_id}, stop assignment: {stop_assignment_id}")
            
            # Acquire lock before starting the matching process
            async with self._assignment_lock:
                self.state_manager.begin_transaction()
                
                start_time = datetime.now()
                
                # Get request and validate
                request = self.state_manager.request_worker.get_request(request_id)
                if not request:
                    raise ValueError(f"Request {request_id} not found")
                    
                if request.status not in [RequestStatus.VALIDATED, RequestStatus.PENDING]:
                    raise ValueError(f"Invalid request status: {request.status}")
                
                # Get available vehicles
                available_vehicles = self._get_available_vehicles()
                if not available_vehicles or len(available_vehicles) == 0:
                    await self._handle_no_vehicles_available(request)
                    return
                
                # Perform quick matching
                assignment, rejection_metadata = await self.matching_strategy.match_stop_assignment_to_vehicle(
                    stop_assignment,
                )
                
                if assignment:
                    await self._process_assignment(assignment)
                else:
                    await self._handle_matching_failed(request, rejection_metadata)
                
                self.state_manager.commit_transaction()
                
                # Update metrics
                computation_time = (datetime.now() - start_time).total_seconds()
                self._update_metrics(bool(assignment), computation_time)
            
        except Exception as e:
            self.state_manager.rollback_transaction()
            import traceback
            logger.error(f"Error in dispatch request handling: {str(e)}\n{traceback.format_exc()}")
            await self._handle_matching_error(event, str(e))
            raise
    async def handle_dispatch_optimization_started(self, event: Event) -> None:
        """Handle optimization event (remains largely the same)"""
        try:
            # Only process if this is a scheduled event
            if not event.data.get('scheduled', False):
                return
                
            logger.info("Starting scheduled global optimization")
            await self.global_optimizer.optimize_system_state()
            
        except Exception as e:
            logger.error(f"Error in dispatch optimization: {str(e)}")
            await self._create_optimization_error_event(str(e))
    
    async def handle_dispatch_optimization_completed(self, event: Event) -> None:
        """Handle completion of dispatch optimization."""
        try:
            improvement = event.data.get('improvement', 0.0)
            logger.info(f"Dispatch optimization completed with {improvement:.2f}% improvement")
            
            # Log optimization results
            if improvement > 0:
                logger.info("Applying optimized assignments")
                optimized_assignments = event.data.get('optimized_assignments', [])
                if optimized_assignments:
                    await self._process_assignment(optimized_assignments)
            
        except Exception as e:
            logger.error(f"Error handling optimization completion: {str(e)}")
    
    async def _process_assignment(self, assignment: Assignment) -> None:
        try:
            """Process successful assignments."""
            # Get the request
            request = self.state_manager.request_worker.get_request(assignment.request_id)
            if not request:
                raise ValueError(f"Request {assignment.request_id} not found")
            
            # Get the vehicle
            vehicle = self.state_manager.vehicle_worker.get_vehicle(assignment.vehicle_id)
            if not vehicle:
                raise ValueError(f"Vehicle {assignment.vehicle_id} not found")
            
            # Calculate proposed pickup time and travel time
            proposed_pickup_time = None
            proposed_travel_time = None
            cost = 0.0
            service_attributes = {}
            
            # Extract pickup and dropoff times from the route
            for stop in assignment.route.stops:
                if assignment.request_id in stop.pickup_passengers:
                    proposed_pickup_time = stop.estimated_arrival_time
                    service_attributes["pickup_location"] = stop.location
                    service_attributes["pickup_stop_id"] = stop.stop_id
                
                if assignment.request_id in stop.dropoff_passengers:
                    dropoff_time = stop.estimated_arrival_time
                    if proposed_pickup_time:
                        proposed_travel_time = dropoff_time - proposed_pickup_time
                    service_attributes["dropoff_location"] = stop.location
                    service_attributes["dropoff_stop_id"] = stop.stop_id
            
            # Add additional service attributes
            service_attributes["vehicle_id"] = vehicle.id
            service_attributes["vehicle_type"] = vehicle.vehicle_type
            service_attributes["route_id"] = assignment.route.id
            service_attributes["request_time"] = request.request_time
            service_attributes["waiting_time"] = (proposed_pickup_time - request.request_time).total_seconds() / 60 if proposed_pickup_time else 0
            service_attributes["travel_time"] = proposed_travel_time.total_seconds() / 60 if proposed_travel_time else 0
            service_attributes["detour_ratio"] = assignment.detour_ratio if hasattr(assignment, "detour_ratio") else 0.0
            service_attributes["cost"] = cost
            
            # Check if user will accept this assignment
            user_accepted, acceptance_probability = False, 0.0
            if proposed_pickup_time and proposed_travel_time:
                user_accepted, acceptance_probability = self.user_acceptance_manager.decide_acceptance(
                    request=request,
                    proposed_pickup_time=proposed_pickup_time,
                    proposed_travel_time=proposed_travel_time,
                    cost=cost,
                    service_attributes=service_attributes
                )
                
                # Log user acceptance metrics
                self.context.metrics_collector.log(
                    MetricName.USER_ACCEPTANCE_PROBABILITY,
                    acceptance_probability,
                    self.context.current_time,
                    {
                        'request_id': request.id,
                        'user_id': getattr(request, "user_id", "unknown"),
                        'vehicle_id': vehicle.id,
                        'waiting_time': service_attributes["waiting_time"],
                        'travel_time': service_attributes["travel_time"],
                        'accepted': user_accepted
                    }
                )
                
                # Update matching metrics
                if user_accepted:
                    self.matching_metrics['user_accepted_matches'] += 1
                else:
                    self.matching_metrics['user_rejected_matches'] += 1
                
                total_user_decisions = (
                    self.matching_metrics['user_accepted_matches'] + 
                    self.matching_metrics['user_rejected_matches']
                )
                if total_user_decisions > 0:
                    self.matching_metrics['user_acceptance_rate'] = (
                        self.matching_metrics['user_accepted_matches'] / total_user_decisions
                    )
            
            # If user rejected, handle rejection
            if not user_accepted:
                rejection_metadata = RejectionMetadata(
                    reason=RejectionReason.USER_REJECTED,
                    details="User rejected the proposed service",
                    timestamp=self.context.current_time,
                    additional_data={
                        "acceptance_probability": acceptance_probability,
                        "proposed_pickup_time": proposed_pickup_time.isoformat() if proposed_pickup_time else None,
                        "proposed_travel_time": proposed_travel_time.total_seconds() if proposed_travel_time else None,
                        "service_attributes": service_attributes
                    }
                )
                
                # Update the user acceptance model
                self.user_acceptance_manager.update_model(
                    request=request,
                    accepted=False,
                    service_attributes=service_attributes
                )
                
                await self._handle_matching_failed(request, rejection_metadata)
                return
            
            # User accepted, proceed with assignment
            
            # 1. Update request status
            self.state_manager.request_worker.update_request_status(
                assignment.request_id, 
                RequestStatus.ASSIGNED
            )
            
            # Update the user acceptance model
            self.user_acceptance_manager.update_model(
                request=request,
                accepted=True,
                service_attributes=service_attributes
            )
            
            # 2. Update vehicle state immediately
            # Add and update route with proper status
            route_exists = bool(self.state_manager.route_worker.get_route(assignment.route.id))
            if not route_exists:
                logger.debug(f"Creating new route: stops={len(assignment.route.stops)}, "
                        f"total_distance={assignment.route.total_distance:.2f}m, "
                        f"total_duration={assignment.route.total_duration:.2f}s")
                assignment.route.status = RouteStatus.CREATED
                self.state_manager.route_worker.add_route(assignment.route)
            else:
                existing_route = self.state_manager.route_worker.get_route(assignment.route.id)
                # Keep existing status if route exists
                assignment.route.status = existing_route.status
                self.state_manager.route_worker.update_route(assignment.route)
                
            # Validate route consistency before proceeding
            is_consistent, consistency_error = assignment.route.validate_passenger_consistency()
            if not is_consistent:
                logger.warning(f"Route {assignment.route.id} has passenger consistency issues: {consistency_error}")
                logger.warning(f"Route details: vehicle={assignment.vehicle_id}, stops={len(assignment.route.stops)}")
                
                # Log detailed information about each stop for debugging
                for i, stop in enumerate(assignment.route.stops):
                    logger.warning(f"Stop {i}: pickup={stop.pickup_passengers}, dropoff={stop.dropoff_passengers}, "
                                f"current_load={stop.current_load}")
                
                # Track all pickups and dropoffs across stops for detailed diagnostics
                all_pickups = {}  # request_id -> stop_index
                all_dropoffs = {}  # request_id -> stop_index
                
                # Collect all pickups and dropoffs
                for i, stop in enumerate(assignment.route.stops):
                    for request_id in stop.pickup_passengers:
                        if request_id in all_pickups:
                            logger.warning(f"Duplicate pickup for request {request_id} at stops {all_pickups[request_id]} and {i}")
                        all_pickups[request_id] = i
                        
                    for request_id in stop.dropoff_passengers:
                        if request_id in all_dropoffs:
                            logger.warning(f"Duplicate dropoff for request {request_id} at stops {all_dropoffs[request_id]} and {i}")
                        all_dropoffs[request_id] = i
                
                # Check for passengers picked up but not dropped off
                for request_id in all_pickups:
                    if request_id not in all_dropoffs:
                        logger.warning(f"Passenger with request {request_id} was picked up but not dropped off")
                
                # Check for passengers dropped off but not picked up
                for request_id in all_dropoffs:
                    if request_id not in all_pickups:
                        logger.warning(f"Passenger with request {request_id} was dropped off but not picked up")
                
                # Check for pickup after dropoff
                for request_id in set(all_pickups.keys()) & set(all_dropoffs.keys()):
                    if all_pickups[request_id] > all_dropoffs[request_id]:
                        logger.warning(f"Pickup for request {request_id} occurs after dropoff (pickup at {all_pickups[request_id]}, dropoff at {all_dropoffs[request_id]})")
            
            # Validate capacity constraints
            is_capacity_valid, capacity_error = assignment.route.validate_capacity(vehicle.capacity)
            if not is_capacity_valid:
                logger.warning(f"Route {assignment.route.id} has capacity issues: {capacity_error}")
                logger.warning(f"Vehicle capacity: {vehicle.capacity}")
                
                # Log detailed information about each stop for debugging
                current_load = 0
                for i, stop in enumerate(assignment.route.stops):
                    expected_load = current_load + len(stop.pickup_passengers) - len(stop.dropoff_passengers)
                    logger.warning(f"Stop {i}: pickup={len(stop.pickup_passengers)}, "
                                f"dropoff={len(stop.dropoff_passengers)}, "
                                f"current_load={stop.current_load}, "
                                f"expected_load={expected_load}")
                    if expected_load < 0:
                        logger.warning(f"Negative load detected at stop {i}: {expected_load}")
                    if expected_load > vehicle.capacity:
                        logger.warning(f"Capacity exceeded at stop {i}: {expected_load} vs {vehicle.capacity}")
                    current_load = expected_load
                
            # Update vehicle's active route
            self.state_manager.vehicle_worker.update_vehicle_active_route_id(
                vehicle.id,
                assignment.route.id
            )
            
            if vehicle.current_state.status == VehicleStatus.IDLE:
                # Create dispatch event for idle vehicle
                dispatch_event = Event(
                    event_type=EventType.VEHICLE_DISPATCH_REQUEST,
                    priority=EventPriority.HIGH,
                    timestamp=self.context.current_time,
                    vehicle_id=vehicle.id,
                    data={
                        'dispatch_type': 'initial',
                        'timestamp': self.context.current_time,
                        'route_id': assignment.route.id,
                        'route_version': assignment.route.version
                    }
                )
                self.context.event_manager.publish_event(dispatch_event)
            else:
                # Create reroute event for active vehicle
                reroute_event = Event(
                    event_type=EventType.VEHICLE_REROUTE_REQUEST,
                    priority=EventPriority.HIGH,
                    timestamp=self.context.current_time,
                    vehicle_id=vehicle.id,
                    data={
                        'dispatch_type': 'reroute',
                        'timestamp': self.context.current_time,
                        'route_id': assignment.route.id
                    }
                )
                self.context.event_manager.publish_event(reroute_event)
            
            # 3. Update stop assignment state
            stop_assignment = self.state_manager.stop_assignment_worker.get_assignment(assignment.stop_assignment_id)
            if not stop_assignment:
                raise ValueError(f"Stop assignment {assignment.stop_assignment_id} not found")
            
            passenger_journey_event = Event(
                event_type=EventType.START_PASSENGER_JOURNEY,
                priority=EventPriority.HIGH,
                timestamp=self.context.current_time,
                passenger_id=request.passenger_id,
                request_id=request.id,
                data={
                    'assignment': assignment
                }
            )
            logger.debug(f"Event Queue Size Before Publishing: {self.context.event_manager.get_queue_size()}")
            logger.debug(f"Publishing START_PASSENGER_JOURNEY event for request={request.id}, passenger={request.passenger_id}, "
             f"stop_assignment_id={assignment.stop_assignment_id}, vehicle={assignment.vehicle_id}")
            self.context.event_manager.publish_event(passenger_journey_event)
            
            logger.debug(f"START_PASSENGER_JOURNEY event published with ID {passenger_journey_event.id}. "
             f"Current event queue size: {self.context.event_manager.get_queue_size()}")
            
            # Now that all critical state is updated, create event for non-critical updates
            assignment_event = Event(
                event_type=EventType.REQUEST_ASSIGNED,
                timestamp=self.context.current_time,
                priority=EventPriority.HIGH,
                data={
                    'assignment': assignment
                }
            )
            self.context.event_manager.publish_event(assignment_event)
        except Exception as e:
            logger.error(f"Error in _process_assignment: {str(e)}")
            logger.error(f"Assignment details: request_id={assignment.request_id if assignment else 'unknown'}, "
                    f"vehicle_id={assignment.vehicle_id if assignment else 'unknown'}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
    
    async def _handle_no_vehicles_available(self, request: Request) -> None:
        """Handle case when no vehicles are available."""
        rejection_metadata = RejectionMetadata(
            reason=RejectionReason.NO_VEHICLES_AVAILABLE,
            timestamp=self.context.current_time.isoformat(),
            stage="matching",
            details={
                "available_vehicles": 0,
                "total_vehicles": len(self.state_manager.vehicle_worker.get_all_vehicles())
            }
        )
        
        event = Event(
            event_type=EventType.REQUEST_REJECTED,
            timestamp=self.context.current_time,
            priority=EventPriority.HIGH,
            request_id=request.id,
            data=rejection_metadata.to_dict()
        )
        self.context.event_manager.publish_event(event)
    
    async def _handle_matching_failed(self, request: Request, rejection_metadata: Optional[RejectionMetadata] = None) -> None:
        """Handle case when matching fails."""
        if not rejection_metadata:
            rejection_metadata = RejectionMetadata(
                reason=RejectionReason.UNKNOWN,
                timestamp=self.context.current_time.isoformat(),
                stage="matching",
                details={}
            )
        
        event = Event(
            event_type=EventType.REQUEST_REJECTED,
            timestamp=self.context.current_time,
            priority=EventPriority.HIGH,
            request_id=request.id,
            data=rejection_metadata.to_dict()
        )
        self.context.event_manager.publish_event(event)
    
    async def _handle_matching_error(self, event: Event, error_msg: str) -> None:
        """Handle errors in matching process."""
        error_event = Event(
            event_type=EventType.SIMULATION_ERROR,
            timestamp=self.context.current_time,
            priority=EventPriority.CRITICAL,
            request_id=event.request_id if hasattr(event, 'request_id') else None,
            data={
                'error': error_msg,
                'original_event': event.to_dict(),
                'error_type': 'matching_error'
            }
        )
        self.context.event_manager.publish_event(error_event)
    
    def _get_available_vehicles(self) -> List[Vehicle]:
        """Get list of available vehicles."""
        return self.state_manager.vehicle_worker.get_available_vehicles()
    
    def _get_unassigned_requests(self) -> List[Request]:
        """Get list of unassigned requests."""
        return [
            request for request in self.state_manager.request_worker.get_active_requests()
            if request.status in [RequestStatus.VALIDATED, RequestStatus.PENDING]
        ]
    
    async def _create_optimization_error_event(self, error_msg: str) -> None:
        """Create and publish optimization error event."""
        error_event = Event(
            event_type=EventType.SIMULATION_ERROR,
            timestamp=self.context.current_time,
            priority=EventPriority.CRITICAL,
            data={
                'error': error_msg,
                'error_type': 'optimization_error'
            }
        )
        self.context.event_manager.publish_event(error_event)
    
    def _update_metrics(self, success: bool, computation_time: float) -> None:
        """Update matching performance metrics."""
        metrics = self.matching_metrics
        metrics['total_requests_processed'] += 1
        
        if success:
            metrics['successful_matches'] += 1
        else:
            metrics['failed_matches'] += 1
            
        # Update average computation time
        old_count = metrics['total_requests_processed'] - 1
        if old_count > 0:
            metrics['average_matching_time'] = (
                (metrics['average_matching_time'] * old_count + computation_time) /
                metrics['total_requests_processed']
            )
        else:
            metrics['average_matching_time'] = computation_time