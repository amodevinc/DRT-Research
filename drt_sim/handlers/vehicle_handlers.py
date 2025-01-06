from typing import Optional, List
from datetime import datetime

from .base import BaseHandler
from drt_sim.models.event import Event, EventType, EventPriority
from drt_sim.models.vehicle import Vehicle, VehicleStatus
from drt_sim.models.route import Route
from drt_sim.models.location import Location

class VehicleHandler(BaseHandler):
    """
    Handles all vehicle-related events in the DRT simulation, managing vehicle
    lifecycle, movements, and state transitions.
    """
    
    def handle_vehicle_created(self, event: Event) -> Optional[List[Event]]:
        """Handle new vehicle creation"""
        try:
            vehicle_data = event.data.get('vehicle')
            if not vehicle_data:
                raise ValueError(f"No vehicle data in event {event.id}")
                
            # Create and validate vehicle
            vehicle = Vehicle(**vehicle_data)
            if not self._validate_vehicle(vehicle):
                self.logger.error(f"Vehicle validation failed: {vehicle.id}")
                return None
                
            # Add to state manager
            self.state_manager.vehicle_worker.add_vehicle(vehicle)
            
            # Create activation event
            return [Event(
                event_type=EventType.VEHICLE_ACTIVATED,
                timestamp=self.context.current_time,
                priority=EventPriority.NORMAL,
                vehicle_id=vehicle.id,
                data={'initial_location': vehicle.current_location}
            )]
            
        except Exception as e:
            self.logger.error(f"Failed to create vehicle: {str(e)}")
            raise

    def handle_vehicle_activated(self, event: Event) -> Optional[List[Event]]:
        """Handle vehicle activation"""
        vehicle_id = event.vehicle_id
        
        try:
            # Update vehicle status
            self.state_manager.vehicle_worker.update_vehicle_state(
                vehicle_id,
                VehicleStatus.IDLE,
                {
                    'activation_time': self.context.current_time,
                    'available_capacity': self.config.vehicle.capacity
                }
            )
            
            self.logger.info(f"Activated vehicle {vehicle_id}")
            
            # Check for immediate assignment needs
            return [Event(
                event_type=EventType.VEHICLE_AVAILABLE,
                timestamp=self.context.current_time,
                priority=EventPriority.NORMAL,
                vehicle_id=vehicle_id
            )]
            
        except Exception as e:
            self.logger.error(f"Failed to activate vehicle {vehicle_id}: {str(e)}")
            raise

    def handle_vehicle_departed(self, event: Event) -> Optional[List[Event]]:
        """Handle vehicle departure"""
        vehicle_id = event.vehicle_id
        destination = event.data.get('destination')
        route = event.data.get('route')
        purpose = event.data.get('purpose', 'movement')
        
        try:
            # Update vehicle status
            self.state_manager.vehicle_worker.update_vehicle_state(
                vehicle_id,
                VehicleStatus.MOVING,
                {
                    'current_route': route,
                    'movement_start_time': self.context.current_time,
                    'movement_purpose': purpose
                }
            )
            
            # Register route if provided
            if route:
                self.state_manager.route_worker.add_route(Route(
                    vehicle_id=vehicle_id,
                    stops=route,
                    start_time=self.context.current_time
                ))
            
            # Calculate arrival time
            arrival_time = self._calculate_arrival_time(
                event.data.get('origin'),
                destination,
                self.context.current_time
            )
            
            # Schedule arrival event
            return [Event(
                event_type=EventType.VEHICLE_ARRIVED,
                timestamp=arrival_time,
                priority=EventPriority.HIGH,
                vehicle_id=vehicle_id,
                data={
                    'location': destination,
                    'purpose': purpose,
                    'route': route
                }
            )]
            
        except Exception as e:
            self.logger.error(f"Failed to handle vehicle departure: {str(e)}")
            raise

    def handle_vehicle_arrived(self, event: Event) -> Optional[List[Event]]:
        """Handle vehicle arrival at destination"""
        vehicle_id = event.vehicle_id
        location = event.data.get('location')
        purpose = event.data.get('purpose', 'movement')
        
        try:
            vehicle = self.state_manager.vehicle_worker.get_vehicle(vehicle_id)
            follow_up_events = []
            
            # Update vehicle location
            self.state_manager.vehicle_worker.update_vehicle_location(
                vehicle_id, 
                location
            )
            
            # Handle different arrival purposes
            if purpose == 'pickup':
                follow_up_events.append(Event(
                    event_type=EventType.REQUEST_PICKUP_STARTED,
                    timestamp=self.context.current_time,
                    priority=EventPriority.HIGH,
                    vehicle_id=vehicle_id,
                    request_id=vehicle.assigned_request
                ))
                
            elif purpose == 'dropoff':
                follow_up_events.append(Event(
                    event_type=EventType.REQUEST_DROPOFF_STARTED,
                    timestamp=self.context.current_time,
                    priority=EventPriority.HIGH,
                    vehicle_id=vehicle_id,
                    request_id=vehicle.assigned_request
                ))
                
            elif purpose == 'rebalancing':
                follow_up_events.append(Event(
                    event_type=EventType.VEHICLE_AVAILABLE,
                    timestamp=self.context.current_time,
                    priority=EventPriority.NORMAL,
                    vehicle_id=vehicle_id
                ))
            
            return follow_up_events
            
        except Exception as e:
            self.logger.error(f"Failed to handle vehicle arrival: {str(e)}")
            raise

    def handle_vehicle_rerouted(self, event: Event) -> Optional[List[Event]]:
        """Handle vehicle rerouting"""
        vehicle_id = event.vehicle_id
        new_route = event.data.get('new_route')
        reason = event.data.get('reason', 'unspecified')
        
        try:
            # Update vehicle route
            self.state_manager.vehicle_worker.update_vehicle_state(
                vehicle_id,
                VehicleStatus.REROUTING,
                {'current_route': new_route}
            )
            
            # Update route in route worker
            self.state_manager.route_worker.update_route(
                vehicle_id,
                new_route,
                {'reroute_reason': reason}
            )
            
            self.logger.info(f"Rerouted vehicle {vehicle_id}: {reason}")
            
            # Calculate new arrival time
            new_arrival_time = self._calculate_arrival_time(
                event.data.get('current_location'),
                new_route[-1].location,
                self.context.current_time
            )
            
            # Create new arrival event
            return [Event(
                event_type=EventType.VEHICLE_ARRIVED,
                timestamp=new_arrival_time,
                priority=EventPriority.HIGH,
                vehicle_id=vehicle_id,
                data={
                    'location': new_route[-1].location,
                    'purpose': event.data.get('purpose', 'movement')
                }
            )]
            
        except Exception as e:
            self.logger.error(f"Failed to handle vehicle rerouting: {str(e)}")
            raise

    def handle_vehicle_breakdown(self, event: Event) -> Optional[List[Event]]:
        """Handle vehicle breakdown"""
        vehicle_id = event.vehicle_id
        reason = event.data.get('reason', 'unspecified')
        
        try:
            vehicle = self.state_manager.vehicle_worker.get_vehicle(vehicle_id)
            follow_up_events = []
            
            # Update vehicle status
            self.state_manager.vehicle_worker.update_vehicle_state(
                vehicle_id,
                VehicleStatus.BREAKDOWN,
                {
                    'breakdown_time': self.context.current_time,
                    'breakdown_reason': reason,
                    'last_location': vehicle.current_location
                }
            )
            
            # Handle any ongoing requests
            if vehicle.assigned_request:
                follow_up_events.append(Event(
                    event_type=EventType.REQUEST_REASSIGNMENT_NEEDED,
                    timestamp=self.context.current_time,
                    priority=EventPriority.HIGH,
                    request_id=vehicle.assigned_request,
                    data={'reason': f"Vehicle breakdown: {reason}"}
                ))
            
            # Handle any scheduled requests
            scheduled_requests = self.state_manager.schedule_worker.get_vehicle_schedule(vehicle_id)
            for request in scheduled_requests:
                follow_up_events.append(Event(
                    event_type=EventType.REQUEST_REASSIGNMENT_NEEDED,
                    timestamp=self.context.current_time,
                    priority=EventPriority.HIGH,
                    request_id=request.id,
                    data={'reason': f"Vehicle breakdown: {reason}"}
                ))
            
            return follow_up_events
            
        except Exception as e:
            self.logger.error(f"Failed to handle vehicle breakdown: {str(e)}")
            raise

    def handle_vehicle_at_capacity(self, event: Event) -> Optional[List[Event]]:
        """Handle vehicle reaching capacity"""
        vehicle_id = event.vehicle_id
        
        try:
            # Update vehicle status
            self.state_manager.vehicle_worker.update_vehicle_state(
                vehicle_id,
                VehicleStatus.AT_CAPACITY,
                {'capacity_reached_time': self.context.current_time}
            )
            
            self.logger.info(f"Vehicle {vehicle_id} reached capacity")
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to handle vehicle capacity: {str(e)}")
            raise

    def _validate_vehicle(self, vehicle: Vehicle) -> bool:
        """Validate vehicle parameters"""
        try:
            # Check capacity
            if vehicle.capacity <= 0:
                self.logger.error(f"Invalid capacity for vehicle {vehicle.id}")
                return False
                
            # Check initial location
            if not self._is_valid_location(vehicle.current_location):
                self.logger.error(f"Invalid location for vehicle {vehicle.id}")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Vehicle validation error: {str(e)}")
            return False

    def _calculate_arrival_time(
        self,
        origin: Location,
        destination: Location,
        departure_time: datetime
    ) -> datetime:
        """Calculate expected arrival time"""
        # Implementation depends on routing system
        pass

    def _is_valid_location(self, location: Location) -> bool:
        """Check if location is valid"""
        # Implementation depends on service area definition
        pass