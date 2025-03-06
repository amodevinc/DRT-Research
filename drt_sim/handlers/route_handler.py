import traceback
import asyncio

from drt_sim.core.state.manager import StateManager
from drt_sim.core.simulation.context import SimulationContext
from drt_sim.models.event import Event, EventType, EventPriority
from drt_sim.models.route import Route, RouteStatus
from drt_sim.config.config import ParameterSet
from drt_sim.models.vehicle import VehicleStatus, Vehicle
from drt_sim.models.matching import Assignment
from drt_sim.network.manager import NetworkManager
import logging
logger = logging.getLogger(__name__)

class RouteHandler:
    """
    Handles route lifecycle and operations in the DRT system.
    Manages route creation, modifications, and coordinates with VehicleHandler.
    """
    
    def __init__(
        self,
        config: ParameterSet,
        context: SimulationContext,
        state_manager: StateManager,
        network_manager: NetworkManager
    ):
        self.config = config
        self.context = context
        self.state_manager = state_manager
        self.network_manager = network_manager

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
    
    def _create_initial_dispatch_event(self, vehicle_id: str, route_id: str) -> None:
        """Create dispatch event for initial vehicle start"""
        # Get the route to include its version
        route = self.state_manager.route_worker.get_route(route_id)
        if not route:
            logger.warning(f"Route {route_id} not found when creating dispatch event")
            return
            
        event = Event(
            event_type=EventType.VEHICLE_DISPATCH_REQUEST,
            priority=EventPriority.HIGH,
            timestamp=self.context.current_time,
            vehicle_id=vehicle_id,
            data={
                'dispatch_type': 'initial',
                'timestamp': self.context.current_time,
                'route_id': route_id,
                'route_version': route.version  # Include route version for version checking
            }
        )
        self.context.event_manager.publish_event(event)

    def _create_reroute_dispatch_event(self, vehicle_id: str, route_id: str) -> None:
        """Create dispatch event for rerouting an active vehicle"""
        # Get the route to include its version
        route = self.state_manager.route_worker.get_route(route_id)
        if not route:
            logger.warning(f"Route {route_id} not found when creating reroute event")
            return
            
        event = Event(
            event_type=EventType.VEHICLE_REROUTE_REQUEST,
            priority=EventPriority.HIGH,
            timestamp=self.context.current_time,
            vehicle_id=vehicle_id,
            data={
                'dispatch_type': 'reroute',
                'timestamp': self.context.current_time,
                'route_id': route_id,
                'route_version': route.version  # Include route version for version checking
            }
        )
        self.context.event_manager.publish_event(event)