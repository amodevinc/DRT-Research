from typing import Optional, List, Dict
from datetime import datetime, timedelta

from .base import BaseHandler
from drt_sim.models.event import Event, EventType, EventPriority
from drt_sim.models.route import Route, RouteStatus, RouteStop
from drt_sim.models.request import Request, RequestStatus
from drt_sim.models.vehicle import VehicleStatus
from drt_sim.models.location import Location

class RouteHandler(BaseHandler):
    """
    Handles all route-related events in the DRT simulation, managing route
    creation, updates, optimization, and execution tracking.
    """
    
    def handle_route_created(self, event: Event) -> Optional[List[Event]]:
        """Handle creation of a new route"""
        vehicle_id = event.vehicle_id
        stops = event.data.get('stops', [])
        
        try:
            # Create and validate route
            route = Route(
                id=event.data.get('route_id'),
                vehicle_id=vehicle_id,
                stops=stops,
                creation_time=self.context.current_time
            )
            
            if not self._validate_route(route):
                return [self._create_route_invalid_event(route.id, "Route validation failed")]
            
            # Calculate timing for each stop
            route_schedule = self._calculate_stop_timings(route)
            route.scheduled_stops = route_schedule
            
            # Add to state manager
            self.state_manager.route_worker.add_route(route)
            
            # Update vehicle status
            self.state_manager.vehicle_worker.update_vehicle_state(
                vehicle_id,
                VehicleStatus.ASSIGNED,
                {'current_route': route.id}
            )
            
            self.logger.info(f"Created route {route.id} for vehicle {vehicle_id}")
            
            # Create route start event
            return [Event(
                event_type=EventType.VEHICLE_ROUTE_STARTED,
                timestamp=self.context.current_time,
                priority=EventPriority.HIGH,
                vehicle_id=vehicle_id,
                data={
                    'route_id': route.id,
                    'first_stop': route_schedule[0]
                }
            )]
            
        except Exception as e:
            self.logger.error(f"Failed to create route: {str(e)}")
            raise

    def handle_route_stop_reached(self, event: Event) -> Optional[List[Event]]:
        """Handle vehicle arrival at route stop"""
        vehicle_id = event.vehicle_id
        route_id = event.data.get('route_id')
        stop_id = event.data.get('stop_id')
        
        try:
            route = self.state_manager.route_worker.get_route(route_id)
            current_stop = next(stop for stop in route.stops if stop.id == stop_id)
            
            follow_up_events = []
            
            # Update route status
            self.state_manager.route_worker.update_stop_status(
                route_id,
                stop_id,
                'reached',
                {
                    'arrival_time': self.context.current_time,
                    'actual_load': event.data.get('current_load', 0)
                }
            )
            
            # Handle pickups at this stop
            for request_id in current_stop.pickups:
                follow_up_events.append(Event(
                    event_type=EventType.REQUEST_PICKUP_STARTED,
                    timestamp=self.context.current_time,
                    priority=EventPriority.HIGH,
                    request_id=request_id,
                    vehicle_id=vehicle_id,
                    data={'stop_id': stop_id}
                ))
            
            # Handle dropoffs at this stop
            for request_id in current_stop.dropoffs:
                follow_up_events.append(Event(
                    event_type=EventType.REQUEST_DROPOFF_STARTED,
                    timestamp=self.context.current_time,
                    priority=EventPriority.HIGH,
                    request_id=request_id,
                    vehicle_id=vehicle_id,
                    data={'stop_id': stop_id}
                ))
            
            # Check if this is the last stop
            if self._is_last_stop(route, stop_id):
                follow_up_events.append(Event(
                    event_type=EventType.ROUTE_COMPLETED,
                    timestamp=self.context.current_time,
                    priority=EventPriority.NORMAL,
                    vehicle_id=vehicle_id,
                    data={'route_id': route_id}
                ))
            else:
                # Schedule movement to next stop
                next_stop = self._get_next_stop(route, stop_id)
                follow_up_events.append(Event(
                    event_type=EventType.VEHICLE_DEPARTED,
                    timestamp=self.context.current_time + self.config.vehicle.service_time,
                    priority=EventPriority.HIGH,
                    vehicle_id=vehicle_id,
                    data={
                        'destination': next_stop.location,
                        'route_id': route_id,
                        'next_stop_id': next_stop.id
                    }
                ))
            
            return follow_up_events
            
        except Exception as e:
            self.logger.error(f"Failed to handle route stop: {str(e)}")
            raise

    def handle_route_updated(self, event: Event) -> Optional[List[Event]]:
        """Handle route update (new stops/removals)"""
        route_id = event.data.get('route_id')
        updates = event.data.get('updates', {})
        
        try:
            # Get current route
            route = self.state_manager.route_worker.get_route(route_id)
            
            # Apply updates
            updated_route = self._apply_route_updates(route, updates)
            if not updated_route:
                return [self._create_route_update_failed_event(route_id, "Failed to apply updates")]
            
            # Recalculate timings
            new_schedule = self._calculate_stop_timings(updated_route)
            updated_route.scheduled_stops = new_schedule
            
            # Update route in state manager
            self.state_manager.route_worker.update_route(route_id, updated_route)
            
            # Check for significant timing changes
            if self._has_significant_delays(route.scheduled_stops, new_schedule):
                return [Event(
                    event_type=EventType.ROUTE_DELAYED,
                    timestamp=self.context.current_time,
                    priority=EventPriority.HIGH,
                    data={
                        'route_id': route_id,
                        'delays': self._calculate_delays(route.scheduled_stops, new_schedule)
                    }
                )]
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to update route: {str(e)}")
            raise

    def handle_route_optimization_needed(self, event: Event) -> Optional[List[Event]]:
        """Handle route optimization request"""
        route_id = event.data.get('route_id')
        trigger = event.data.get('trigger', 'manual')
        
        try:
            # Get current route
            route = self.state_manager.route_worker.get_route(route_id)
            
            # Attempt optimization
            optimized_route = self._optimize_route(route)
            if not optimized_route:
                return None
                
            # If optimization improved the route
            if self._is_route_improved(route, optimized_route):
                return [Event(
                    event_type=EventType.ROUTE_UPDATED,
                    timestamp=self.context.current_time,
                    priority=EventPriority.HIGH,
                    data={
                        'route_id': route_id,
                        'updates': {
                            'stops': optimized_route.stops,
                            'optimization_time': self.context.current_time
                        }
                    }
                )]
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to optimize route: {str(e)}")
            raise

    def handle_route_detour_needed(self, event: Event) -> Optional[List[Event]]:
        """Handle request for route detour"""
        route_id = event.data.get('route_id')
        reason = event.data.get('reason')
        new_stop = event.data.get('new_stop')
        
        try:
            # Get current route
            route = self.state_manager.route_worker.get_route(route_id)
            
            # Calculate best detour
            detour_route = self._calculate_detour(route, new_stop)
            if not detour_route:
                return [self._create_detour_failed_event(route_id, "No valid detour found")]
            
            # Validate the detour doesn't violate constraints
            if not self._validate_detour_constraints(detour_route):
                return [self._create_detour_failed_event(route_id, "Detour violates constraints")]
            
            # Create route update event
            return [Event(
                event_type=EventType.ROUTE_UPDATED,
                timestamp=self.context.current_time,
                priority=EventPriority.HIGH,
                data={
                    'route_id': route_id,
                    'updates': {
                        'stops': detour_route.stops,
                        'detour_reason': reason
                    }
                }
            )]
            
        except Exception as e:
            self.logger.error(f"Failed to handle detour request: {str(e)}")
            raise

    def _validate_route(self, route: Route) -> bool:
        """Validate route parameters and constraints"""
        try:
            # Check basic route properties
            if not route.stops:
                self.logger.error(f"Route {route.id} has no stops")
                return False
            
            # Validate stop sequence
            if not self._validate_stop_sequence(route.stops):
                self.logger.error(f"Invalid stop sequence in route {route.id}")
                return False
            
            # Check time windows
            if not self._validate_time_windows(route):
                self.logger.error(f"Time window violations in route {route.id}")
                return False
            
            # Check vehicle constraints
            if not self._validate_vehicle_constraints(route):
                self.logger.error(f"Vehicle constraint violations in route {route.id}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Route validation error: {str(e)}")
            return False

    def _calculate_stop_timings(self, route: Route) -> List[RouteStop]:
        """Calculate expected timing for each stop"""
        # Implementation depends on routing system
        pass

    def _optimize_route(self, route: Route) -> Optional[Route]:
        """Optimize route stop sequence"""
        # Implementation depends on optimization strategy
        pass

    def _calculate_detour(self, route: Route, new_stop: RouteStop) -> Optional[Route]:
        """Calculate best detour incorporation"""
        # Implementation depends on routing strategy
        pass

    def _validate_detour_constraints(self, route: Route) -> bool:
        """Validate if detour meets all constraints"""
        # Implementation depends on constraint definitions
        pass

    def _is_last_stop(self, route: Route, stop_id: str) -> bool:
        """Check if stop is last in route"""
        return route.stops[-1].id == stop_id

    def _get_next_stop(self, route: Route, current_stop_id: str) -> RouteStop:
        """Get next stop in route"""
        current_idx = next(i for i, stop in enumerate(route.stops) 
                         if stop.id == current_stop_id)
        return route.stops[current_idx + 1]

    def _has_significant_delays(self, original: List[RouteStop], 
                              updated: List[RouteStop]) -> bool:
        """Check if updates cause significant timing changes"""
        # Implementation depends on delay threshold definitions
        pass

    def _calculate_delays(self, original: List[RouteStop], 
                         updated: List[RouteStop]) -> Dict[str, timedelta]:
        """Calculate delays for affected stops"""
        delays = {}
        for orig, upd in zip(original, updated):
            if upd.scheduled_arrival > orig.scheduled_arrival:
                delays[upd.id] = upd.scheduled_arrival - orig.scheduled_arrival
        return delays