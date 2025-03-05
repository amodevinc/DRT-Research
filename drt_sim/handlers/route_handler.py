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
    Simplified to work without segments.
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

    async def handle_route_update_request(self, event: Event) -> None:
        """Handle route update request"""
        try:
            logger.info(f"Handling route update request for event: {event.id}")
            self.state_manager.begin_transaction()
            
            assignment = event.data['assignment']
            if not assignment:
                raise ValueError(f"Assignment not found")
                
            vehicle: Vehicle = self.state_manager.vehicle_worker.get_vehicle(assignment.vehicle_id)
            if not vehicle:
                raise ValueError(f"Vehicle {assignment.vehicle_id} not found")

            logger.info(f"Processing route update for vehicle {vehicle.id}: status={vehicle.current_state.status}, "
                     f"location={vehicle.current_state.current_location}")
                
            # Update or create route
            route_exists = bool(self.state_manager.route_worker.get_route(assignment.route.id))
            logger.info(f"Route {assignment.route.id} exists: {route_exists}")
            
            if not route_exists:
                logger.debug(f"Creating new route: stops={len(assignment.route.stops)}, "
                          f"total_distance={assignment.route.total_distance:.2f}m, "
                          f"total_duration={assignment.route.total_duration:.2f}s")
                logger.debug(f"Route stops sequence: {[str(stop) for stop in assignment.route.stops]}")
                assignment.route.status = RouteStatus.CREATED
                self.state_manager.route_worker.add_route(assignment.route)
                logger.info(f"Created new route {assignment.route.id} for vehicle {vehicle.id}")
            else:
                existing_route = self.state_manager.route_worker.get_route(assignment.route.id)
                logger.debug(f"Updating existing route {existing_route.id}:")
                logger.debug(f"Old route: {str(existing_route)}")
                logger.debug(f"New route: {str(assignment.route)}")
                logger.debug(f"New route stops sequence: {[str(stop) for stop in assignment.route.stops]}")
                
                # Keep existing status if route exists
                assignment.route.status = existing_route.status
                self.state_manager.route_worker.update_route(assignment.route)
                logger.info(f"Updated existing route {assignment.route.id} for vehicle {vehicle.id}")
            
            # Check vehicle's current status to determine appropriate dispatch
            if vehicle.current_state.status == VehicleStatus.IDLE:
                logger.debug(f"Vehicle {vehicle.id} is idle, creating initial dispatch")
                self._create_initial_dispatch_event(assignment.vehicle_id, assignment.route.id)
            else:
                logger.debug(f"Vehicle {vehicle.id} is in service, creating reroute dispatch")
                self._create_reroute_dispatch_event(assignment.vehicle_id, assignment.route.id)
                
            self.state_manager.commit_transaction()
            logger.info(f"Successfully created/updated route for vehicle {vehicle.id}")
            
        except Exception as e:
            self.state_manager.rollback_transaction()
            logger.error(f"Error handling route update request: {str(e)}")
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