import traceback
import asyncio

from drt_sim.core.state.manager import StateManager
from drt_sim.core.simulation.context import SimulationContext
from drt_sim.models.event import Event, EventType, EventPriority
from drt_sim.models.route import Route, RouteStatus, RouteSegment, RouteStop
from drt_sim.config.config import ScenarioConfig
from drt_sim.models.vehicle import VehicleStatus, Vehicle
from drt_sim.models.matching import Assignment
from drt_sim.core.logging_config import setup_logger
from drt_sim.network.manager import NetworkManager

logger = setup_logger(__name__)

class RouteHandler:
    """
    Handles route lifecycle and operations in the DRT system.
    Manages route creation, modifications, and coordinates with VehicleHandler.
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

    async def handle_route_update_request(self, event: Event) -> None:
        """Handle route update request"""
        try:
            logger.info(f"Handling route update request: {event}")
            self.state_manager.begin_transaction()
            
            assignment = event.data['assignment']
            if not assignment:
                raise ValueError(f"Assignment not found")
                
            vehicle: Vehicle = self.state_manager.vehicle_worker.get_vehicle(assignment.vehicle_id)
                
            # Update or create route
            route_exists = bool(self.state_manager.route_worker.get_route(assignment.route.id))
            if not route_exists:
                self.state_manager.route_worker.add_route(assignment.route)
            else:
                self.state_manager.route_worker.update_route(assignment.route)
            
            # Check vehicle's current status to determine appropriate dispatch
            if vehicle.current_state.status == VehicleStatus.IDLE:
                # Initial dispatch - vehicle hasn't started yet
                self._create_update_vehicle_active_route_event(assignment.vehicle_id, assignment.route.id)
                self._create_initial_dispatch_event(assignment.vehicle_id)
            else:
                # Vehicle is already in service - needs rerouting
                self._create_reroute_dispatch_event(assignment.vehicle_id)
                
            self.state_manager.commit_transaction()
            
        except Exception as e:
            self.state_manager.rollback_transaction()
            logger.error(f"Error handling route update request: {str(e)}")
            await self._handle_route_error(event, str(e))

    async def handle_stop_service_completed(self, event: Event) -> None:
        """Handle completion of service at a stop."""
        try:
            self.state_manager.begin_transaction()
            
            route = self.state_manager.route_worker.get_route(event.data['route_id'])
            if not route:
                raise ValueError(f"Route {event.data['route_id']} not found")
            
            segment_index = event.data['segment_index']
            current_segment: RouteSegment = route.segments[segment_index]
            
            # Mark current segment as completed
            current_segment.completed = True
            
            # Check if this was the last segment
            if segment_index == len(route.segments) - 1:
                route.status = RouteStatus.COMPLETED
                self._create_route_completed_event(route)
            else:
                # Move to next segment
                route.current_segment_index += 1
                next_segment: RouteSegment = route.segments[route.current_segment_index]
                
                # Create next segment start event
                next_segment_event = Event(
                    event_type=EventType.ROUTE_SEGMENT_STARTED,
                    priority=EventPriority.HIGH,
                    timestamp=self.context.current_time,
                    vehicle_id=route.vehicle_id,
                    data={
                        'route_id': route.id,
                        'segment_index': route.current_segment_index,
                        'origin': next_segment.origin,
                        'destination': next_segment.destination,
                        'estimated_duration': next_segment.estimated_duration,
                        'estimated_distance': next_segment.estimated_distance
                    }
                )
                self.context.event_manager.publish_event(next_segment_event)
            
            self.state_manager.route_worker.update_route(route)
            self.state_manager.commit_transaction()
            
        except Exception as e:
            self.state_manager.rollback_transaction()
            logger.error(f"Error handling stop service completion: {str(e)}")
            await self._handle_route_error(event, str(e))

    def _create_route_completed_event(self, route: Route) -> None:
        """Create event for route completion."""
        event = Event(
            event_type=EventType.ROUTE_COMPLETED,
            priority=EventPriority.HIGH,
            timestamp=self.context.current_time,
            vehicle_id=route.vehicle_id,
            data={
                'route_id': route.id,
                'completion_time': self.context.current_time
            }
        )
        self.context.event_manager.publish_event(event)

    async def _handle_route_error(self, event: Event, error_msg: str) -> None:
        """Handle errors in route processing."""
        error_event = Event(
            event_type=EventType.SIMULATION_ERROR,
            priority=EventPriority.CRITICAL,
            timestamp=self.context.current_time,
            vehicle_id=event.vehicle_id if hasattr(event, 'vehicle_id') else None,
            data={
                'error': error_msg,
                'original_event': event.to_dict(),
                'error_type': 'route_processing_error'
            }
        )
        self.context.event_manager.publish_event(error_event)
    
    def _create_initial_dispatch_event(self, vehicle_id: str) -> None:
        """Create dispatch event for initial vehicle start"""
        event = Event(
            event_type=EventType.VEHICLE_DISPATCH_REQUEST,
            priority=EventPriority.HIGH,
            timestamp=self.context.current_time,
            vehicle_id=vehicle_id,
            data={
                'dispatch_type': 'initial',
                'timestamp': self.context.current_time
            }
        )
        self.context.event_manager.publish_event(event)
    
    def _create_update_vehicle_active_route_event(self, vehicle_id: str, route_id: str) -> None:
        """Create event to update vehicle's active route"""
        event = Event(
            event_type=EventType.VEHICLE_ACTIVE_ROUTE_UPDATE,
            priority=EventPriority.HIGH,
            timestamp=self.context.current_time,
            vehicle_id=vehicle_id,
            data={'route_id': route_id}
        )
        self.context.event_manager.publish_event(event)

    def _create_reroute_dispatch_event(self, vehicle_id: str) -> None:
        """Create dispatch event for rerouting an active vehicle"""
        event = Event(
            event_type=EventType.VEHICLE_REROUTE_REQUEST,
            priority=EventPriority.HIGH,
            timestamp=self.context.current_time,
            vehicle_id=vehicle_id,
            data={
                'dispatch_type': 'reroute',
                'timestamp': self.context.current_time
            }
        )
        self.context.event_manager.publish_event(event)