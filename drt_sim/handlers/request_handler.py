from typing import Optional, List
from datetime import datetime, timedelta

from .base import BaseHandler
from drt_sim.models.event import Event, EventType, EventPriority
from drt_sim.models.request import Request, RequestStatus
from drt_sim.models.vehicle import VehicleStatus

class RequestHandler(BaseHandler):
    """
    Handles all request-related events in the DRT simulation, managing the complete
    lifecycle of passenger requests from creation to completion.
    """
    
    def handle_request_created(self, event: Event) -> Optional[List[Event]]:
        """
        Handle new request creation. Validates the request and initiates the dispatch process.
        """
        try:
            # Extract request from event data
            request_data = event.data.get('request')
            if not request_data:
                raise ValueError(f"No request data found in event {event.id}")
            
            # Create and validate request
            request = Request(**request_data)
            if not self._validate_request(request):
                return [self._create_request_rejected_event(
                    request.id,
                    "Request validation failed"
                )]
            
            # Add request to state manager
            self.state_manager.request_worker.add_request(request)
            
            self.logger.info(f"Created new request: {request.id}")
            
            # Create dispatch request event
            return [Event(
                event_type=EventType.DISPATCH_REQUESTED,
                timestamp=self.context.current_time,
                priority=EventPriority.HIGH,
                request_id=request.id,
                data={'request': request_data}
            )]
            
        except Exception as e:
            self.logger.error(f"Failed to handle request creation: {str(e)}")
            raise
            
    def handle_request_rejected(self, event: Event) -> Optional[List[Event]]:
        """Handle request rejection"""
        request_id = event.request_id
        reason = event.data.get('reason', 'Unspecified')
        
        try:
            # Update request status
            self.state_manager.request_worker.update_request_status(
                request_id,
                RequestStatus.REJECTED,
                {
                    'rejection_time': self.context.current_time,
                    'rejection_reason': reason
                }
            )
            
            self.logger.info(f"Request {request_id} rejected: {reason}")
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to handle request rejection: {str(e)}")
            raise

    def handle_request_assigned(self, event: Event) -> Optional[List[Event]]:
        """Handle request assignment to vehicle"""
        request_id = event.request_id
        vehicle_id = event.data.get('vehicle_id')
        pickup_time = event.data.get('estimated_pickup_time')
        dropoff_time = event.data.get('estimated_dropoff_time')
        
        try:
            # Update request status
            self.state_manager.request_worker.update_request_status(
                request_id,
                RequestStatus.ASSIGNED,
                {
                    'assigned_vehicle': vehicle_id,
                    'assignment_time': self.context.current_time,
                    'estimated_pickup_time': pickup_time,
                    'estimated_dropoff_time': dropoff_time
                }
            )
            
            # Update vehicle status
            self.state_manager.vehicle_worker.update_vehicle_state(
                vehicle_id,
                VehicleStatus.ASSIGNED,
                {
                    'assigned_request': request_id,
                    'next_pickup_time': pickup_time
                }
            )
            
            self.logger.info(f"Request {request_id} assigned to vehicle {vehicle_id}")
            
            # Create vehicle movement event if needed
            vehicle = self.state_manager.vehicle_worker.get_vehicle(vehicle_id)
            request = self.state_manager.request_worker.get_request(request_id)
            
            if vehicle.current_location != request.pickup_location:
                return [Event(
                    event_type=EventType.VEHICLE_DEPARTED,
                    timestamp=self.context.current_time,
                    priority=EventPriority.HIGH,
                    vehicle_id=vehicle_id,
                    request_id=request_id,
                    data={
                        'destination': request.pickup_location,
                        'purpose': 'pickup'
                    }
                )]
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to handle request assignment: {str(e)}")
            raise

    def handle_request_pickup_started(self, event: Event) -> Optional[List[Event]]:
        """Handle start of request pickup process"""
        request_id = event.request_id
        vehicle_id = event.data.get('vehicle_id')
        
        try:
            # Update request status
            self.state_manager.request_worker.update_request_status(
                request_id,
                RequestStatus.PICKUP_STARTED,
                {
                    'pickup_start_time': self.context.current_time,
                }
            )
            
            # Update vehicle status
            self.state_manager.vehicle_worker.update_vehicle_state(
                vehicle_id,
                VehicleStatus.LOADING,
                {'current_action': 'pickup'}
            )
            
            self.logger.info(f"Started pickup for request {request_id}")
            
            # Schedule pickup completion based on boarding time
            return [Event(
                event_type=EventType.REQUEST_PICKUP_COMPLETED,
                timestamp=self.context.current_time + timedelta(
                    seconds=self.config.vehicle.boarding_time
                ),
                priority=EventPriority.HIGH,
                request_id=request_id,
                vehicle_id=vehicle_id
            )]
            
        except Exception as e:
            self.logger.error(f"Failed to handle pickup start: {str(e)}")
            raise

    def handle_request_pickup_completed(self, event: Event) -> Optional[List[Event]]:
        """Handle completion of request pickup"""
        request_id = event.request_id
        vehicle_id = event.data.get('vehicle_id')
        
        try:
            request = self.state_manager.request_worker.get_request(request_id)
            
            # Update request status
            self.state_manager.request_worker.update_request_status(
                request_id,
                RequestStatus.IN_VEHICLE,
                {
                    'pickup_time': self.context.current_time,
                    'pickup_location': request.pickup_location
                }
            )
            
            # Update vehicle status and start journey to dropoff
            self.state_manager.vehicle_worker.update_vehicle_state(
                vehicle_id,
                VehicleStatus.OCCUPIED,
                {
                    'current_passengers': [request_id],
                    'next_destination': request.dropoff_location
                }
            )
            
            self.logger.info(f"Completed pickup for request {request_id}")
            
            # Create vehicle departure event for dropoff
            return [Event(
                event_type=EventType.VEHICLE_DEPARTED,
                timestamp=self.context.current_time,
                priority=EventPriority.HIGH,
                vehicle_id=vehicle_id,
                request_id=request_id,
                data={
                    'destination': request.dropoff_location,
                    'purpose': 'dropoff'
                }
            )]
            
        except Exception as e:
            self.logger.error(f"Failed to handle pickup completion: {str(e)}")
            raise

    def handle_request_dropoff_started(self, event: Event) -> Optional[List[Event]]:
        """Handle start of request dropoff process"""
        request_id = event.request_id
        vehicle_id = event.data.get('vehicle_id')
        
        try:
            # Update request status
            self.state_manager.request_worker.update_request_status(
                request_id,
                RequestStatus.DROPOFF_STARTED,
                {
                    'dropoff_start_time': self.context.current_time,
                }
            )
            
            # Update vehicle status
            self.state_manager.vehicle_worker.update_vehicle_state(
                vehicle_id,
                VehicleStatus.UNLOADING,
                {'current_action': 'dropoff'}
            )
            
            self.logger.info(f"Started dropoff for request {request_id}")
            
            # Schedule dropoff completion based on alighting time
            return [Event(
                event_type=EventType.REQUEST_DROPOFF_COMPLETED,
                timestamp=self.context.current_time + timedelta(
                    seconds=self.config.vehicle.alighting_time
                ),
                priority=EventPriority.HIGH,
                request_id=request_id,
                vehicle_id=vehicle_id
            )]
            
        except Exception as e:
            self.logger.error(f"Failed to handle dropoff start: {str(e)}")
            raise

    def handle_request_dropoff_completed(self, event: Event) -> Optional[List[Event]]:
        """Handle completion of request dropoff"""
        request_id = event.request_id
        vehicle_id = event.data.get('vehicle_id')
        
        try:
            request = self.state_manager.request_worker.get_request(request_id)
            
            # Update request status with final metrics
            self.state_manager.request_worker.update_request_status(
                request_id,
                RequestStatus.COMPLETED,
                {
                    'dropoff_time': self.context.current_time,
                    'dropoff_location': request.dropoff_location,
                    'service_duration': (self.context.current_time - 
                                      request.creation_time).total_seconds(),
                    'final_cost': self._calculate_final_cost(request)
                }
            )
            
            # Update vehicle status
            self.state_manager.vehicle_worker.update_vehicle_state(
                vehicle_id,
                VehicleStatus.IDLE,
                {
                    'current_passengers': [],
                    'last_service': request_id
                }
            )
            
            self.logger.info(f"Completed dropoff for request {request_id}")
            
            # Create rebalancing evaluation event
            return [Event(
                event_type=EventType.REBALANCING_NEEDED,
                timestamp=self.context.current_time,
                priority=EventPriority.LOW,
                vehicle_id=vehicle_id,
                data={'last_service_location': request.dropoff_location}
            )]
            
        except Exception as e:
            self.logger.error(f"Failed to handle dropoff completion: {str(e)}")
            raise

    def handle_request_cancelled(self, event: Event) -> Optional[List[Event]]:
        """Handle request cancellation"""
        request_id = event.request_id
        reason = event.data.get('reason', 'Unspecified')
        
        try:
            request = self.state_manager.request_worker.get_request(request_id)
            
            # Update request status
            self.state_manager.request_worker.update_request_status(
                request_id,
                RequestStatus.CANCELLED,
                {
                    'cancellation_time': self.context.current_time,
                    'cancellation_reason': reason,
                    'cancellation_penalty': self._calculate_cancellation_penalty(request)
                }
            )
            
            # If request was assigned, free up vehicle
            if request.assigned_vehicle:
                self.state_manager.vehicle_worker.update_vehicle_state(
                    request.assigned_vehicle,
                    VehicleStatus.IDLE,
                    {
                        'current_passengers': [],
                        'last_cancelled_service': request_id
                    }
                )
            
            self.logger.info(f"Cancelled request {request_id}: {reason}")
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to handle request cancellation: {str(e)}")
            raise

    def _validate_request(self, request: Request) -> bool:
        """
        Validate request parameters
        """
        try:
            # Check pickup time is in the future
            if request.pickup_time < self.context.current_time:
                self.logger.error(f"Request {request.id} pickup time is in the past")
                return False
                
            # Check locations are within service area
            if not self._is_in_service_area(request.pickup_location):
                self.logger.error(f"Request {request.id} pickup location outside service area")
                return False
                
            if not self._is_in_service_area(request.dropoff_location):
                self.logger.error(f"Request {request.id} dropoff location outside service area")
                return False
                
            # Additional validations as needed...
            return True
            
        except Exception as e:
            self.logger.error(f"Request validation error: {str(e)}")
            return False

    def _calculate_final_cost(self, request: Request) -> float:
        """Calculate final cost for completed request"""
        # Implementation depends on pricing model
        pass

    def _calculate_cancellation_penalty(self, request: Request) -> float:
        """Calculate penalty for cancelled request"""
        # Implementation depends on cancellation policy
        pass

    def _is_in_service_area(self, location) -> bool:
        """Check if location is within service area"""
        # Implementation depends on service area definition
        pass

    def _create_request_rejected_event(self, request_id: str, reason: str) -> Event:
        """Create rejection event"""
        return Event(
            event_type=EventType.REQUEST_REJECTED,
            timestamp=self.context.current_time,
            priority=EventPriority.HIGH,
            request_id=request_id,
            data={'reason': reason}
        )