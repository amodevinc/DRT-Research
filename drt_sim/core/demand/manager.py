from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from drt_sim.config.config import DemandConfig
from drt_sim.models.event import Event, EventType
from drt_sim.models.request import Request, RequestStatus
from drt_sim.core.demand.generators import (
    BaseDemandGenerator,
    CSVDemandGenerator,
    RandomDemandGenerator,
)
import logging
logger = logging.getLogger(__name__)

class DemandManager:
    """
    Manages request generation and demand patterns for the simulation.
    Supports multiple demand sources and generation strategies.
    """
    
    def __init__(self, config: DemandConfig):
        """
        Initialize demand manager.
        
        Args:
            config: Demand configuration
        """
        self.config = config
        logger.info(f"Demand manager initialized with config: {self.config}")
        
        # Initialize counters and state
        self.request_counter = 0
        self.active_requests: Dict[str, Request] = {}
        self.completed_requests: Dict[str, Request] = {}
        
        # Initialize demand generator based on config
        self.generator = self._create_generator()
        logger.info(f"Demand generator initialized: {self.generator}")
        
        # Initialize metrics
        self.metrics: Dict[str, float] = {
            'total_requests': 0,
            'active_requests': 0,
            'completed_requests': 0,
            'cancelled_requests': 0,
            'average_wait_time': 0.0
        }
    
    def _create_generator(self) -> BaseDemandGenerator:
        """Create appropriate demand generator based on configuration."""
        try:
            if self.config.generator_type == "csv":
                logger.info(f"CSV Config: {self.config.csv_config}")
                return CSVDemandGenerator(self.config.csv_config)
            elif self.config.generator_type == "random":
                return RandomDemandGenerator(self.config.random_config)
            else:
                raise ValueError(f"Unknown generator type: {self.config.generator_type}")
                
        except Exception as e:
            logger.error(f"Failed to create demand generator: {str(e)}")
            raise
    
    def generate_demand(
        self,
        current_time: datetime,
        time_step: timedelta
    ) -> Tuple[List[Event], List[Request]]:
        """
        Generate demand events and their associated requests for the current time step.
        Limited by max number of requests specified in config.
        
        Args:
            current_time: Current simulation time
            time_step: Time step duration
            
        Returns:
            Tuple containing (list of events, list of valid requests)
        """
        try:
            # Check if we've reached the maximum number of requests
            if (self.config.num_requests is not None and 
                self.metrics['total_requests'] >= self.config.num_requests):
                return [], []

            # Get events from generator
            events = self.generator.generate(current_time, time_step)
            
            # Process and validate requests from events
            valid_requests = []
            valid_events = []
            
            for event in events:
                # Stop if we've reached the maximum number of requests
                if (self.config.num_requests is not None and 
                    self.metrics['total_requests'] >= self.config.num_requests):
                    logger.info(f"Maximum number of requests reached: {self.metrics['total_requests']}")
                    break
                    
                if event.event_type != EventType.REQUEST_RECEIVED:
                    continue
                    
                request = event.data.get("request")
                if not request or not self._validate_request(request):
                    continue
                
                # Add request ID (if not already present)
                if not request.id:
                    request.id = f"R{self.request_counter}"
                    self.request_counter += 1
                
                # Add to active requests
                self.active_requests[request.id] = request
                valid_requests.append(request)
                valid_events.append(event)
                
                # Update metrics
                self.metrics['total_requests'] += 1
                self.metrics['active_requests'] += 1
            
            return valid_events, valid_requests
            
        except Exception as e:
            logger.error(f"Error generating demand: {str(e)}")
            return [], []
    
    def _validate_request(self, request: Request) -> bool:
        """Validate a generated request."""
        try:
            # Check locations
            if not request.origin or not request.destination:
                return False
                
            # Check timestamps
            if not request.request_time:
                return False
                
            # Check for duplicate pickup/dropoff
            if (request.origin.lat == request.destination.lat and
                request.origin.lon == request.destination.lon):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating request: {str(e)}")
            return False
    
    def update_request_status(
        self,
        request_id: str,
        new_status: RequestStatus,
        timestamp: datetime
    ) -> Optional[Event]:
        """
        Update status of a request and generate appropriate event.
        
        Args:
            request_id: ID of request to update
            new_status: New status to set
            timestamp: Time of status update
            
        Returns:
            Event corresponding to the status update, if applicable
        """
        try:
            if request_id not in self.active_requests:
                raise KeyError(f"Request {request_id} not found in active requests")
                
            request = self.active_requests[request_id]
            old_status = request.status
            request.status = new_status
            
            # Handle completion or cancellation
            if new_status in [RequestStatus.COMPLETED, RequestStatus.CANCELLED]:
                # Move to completed requests
                self.completed_requests[request_id] = request
                del self.active_requests[request_id]
                
                # Update metrics
                self.metrics['active_requests'] -= 1
                if new_status == RequestStatus.COMPLETED:
                    self.metrics['completed_requests'] += 1
                else:
                    self.metrics['cancelled_requests'] += 1
            
            # Log status change
            logger.debug(
                f"Request {request_id} status changed from {old_status} to {new_status}"
            )
            
        except Exception as e:
            logger.error(f"Error updating request status: {str(e)}")
            raise
    
    def get_active_requests(self) -> List[Request]:
        """Get list of currently active requests"""
        return list(self.active_requests.values())
    
    def get_request(self, request_id: str) -> Optional[Request]:
        """Get request by ID from either active or completed requests"""
        return (
            self.active_requests.get(request_id) or 
            self.completed_requests.get(request_id)
        )
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current demand metrics"""
        return self.metrics.copy()
    
    def cleanup(self) -> None:
        """Clean up manager resources"""
        try:
            self.active_requests.clear()
            self.completed_requests.clear()
            self.metrics.clear()
            logger.info("Demand manager cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            raise