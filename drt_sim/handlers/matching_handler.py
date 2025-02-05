from typing import List
from datetime import datetime

from drt_sim.models.event import Event, EventType, EventPriority
from drt_sim.models.request import Request, RequestStatus
from drt_sim.models.vehicle import Vehicle
from drt_sim.config.config import ScenarioConfig
from drt_sim.core.simulation.context import SimulationContext
from drt_sim.core.state.manager import StateManager
from drt_sim.network.manager import NetworkManager
from drt_sim.algorithms.base_interfaces.matching_base import (
    MatchingStrategy, Assignment
)
from drt_sim.algorithms.optimization.global_optimizer import GlobalSystemOptimizer
from drt_sim.core.logging_config import setup_logger
from drt_sim.core.user.manager import UserProfileManager
from drt_sim.core.services.route_service import RouteService
logger = setup_logger(__name__)

class MatchingHandler:
    """Handles request-vehicle matching in the DRT system."""
    
    def __init__(
        self,
        config: ScenarioConfig,
        context: SimulationContext,
        state_manager: StateManager,
        network_manager: NetworkManager,
        user_profile_manager: UserProfileManager,
        route_service: RouteService
    ):
        self.config = config
        self.context = context
        self.state_manager = state_manager
        self.network_manager = network_manager
        self.user_profile_manager = user_profile_manager
        self.route_service = route_service
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
            'average_waiting_time': 0.0
        }
        
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
            
            logger.info(f"Processing match to vehicle for request: {request_id}, stop assignment: {stop_assignment_id}")
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
            if not available_vehicles:
                await self._handle_no_vehicles_available(request)
                return
            
            # Perform quick matching
            assignment = await self.matching_strategy.match_stop_assignment_to_vehicle(
                stop_assignment,
                available_vehicles
            )
            
            if assignment:
                await self._process_assignment(assignment)
            else:
                await self._handle_matching_failed(request)
            
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
        """Process successful assignments."""
        # Update request status
        self.state_manager.request_worker.update_request_status(
            assignment.request_id, 
            RequestStatus.ASSIGNED
        )
        
        # Create assignment event
        assignment_event = Event(
            event_type=EventType.REQUEST_ASSIGNED,
            timestamp=self.context.current_time,
            priority=EventPriority.HIGH,
            data={
                'assignment': assignment
            }
        )
        self.context.event_manager.publish_event(assignment_event)
    
    async def _handle_no_vehicles_available(self, request: Request) -> None:
        """Handle case when no vehicles are available."""
        # Create no vehicle event
        event = Event(
            event_type=EventType.REQUEST_REJECTED,
            timestamp=self.context.current_time,
            priority=EventPriority.HIGH,
            request_id=request.id,
            data={'reason': 'No vehicles available'}
        )
        self.context.event_manager.publish_event(event)
    
    async def _handle_matching_failed(self, request: Request) -> None:
        """Handle case when matching fails."""
        # Create matching failed event
        event = Event(
            event_type=EventType.REQUEST_REJECTED,
            timestamp=self.context.current_time,
            priority=EventPriority.HIGH,
            request_id=request.id,
            data={'reason': 'No feasible match found'}
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