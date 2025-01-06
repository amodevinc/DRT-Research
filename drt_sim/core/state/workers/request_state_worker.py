from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass
import logging

from drt_sim.core.state.base import StateWorker, StateContainer
from drt_sim.models.request import Request, RequestStatus

logger = logging.getLogger(__name__)

@dataclass
class RequestMetrics:
    """Metrics tracked for requests"""
    total_requests: int = 0
    active_requests: int = 0
    accepted_requests: int = 0
    rejected_requests: int = 0
    completed_requests: int = 0
    cancelled_requests: int = 0
    average_wait_time: float = 0.0
    average_service_time: float = 0.0
    total_wait_time: float = 0.0
    total_service_time: float = 0.0

class RequestStateWorker(StateWorker):
    """Manages state for passenger requests"""
    
    def __init__(self):
        self.requests = StateContainer[Request]()
        self.metrics = RequestMetrics()
        self.initialized = False
        self.logger = logging.getLogger(__name__)
    
    def initialize(self, config: Optional[Any] = None) -> None:
        """Initialize request state worker"""
        self.initialized = True
        self.logger.info("Request state worker initialized")
    
    def add_request(self, request: Request) -> Request:
        """
        Add a new request to state management.
        
        Args:
            request: Request object to add
            
        Returns:
            Added request object
            
        Raises:
            RuntimeError: If worker not initialized
            ValueError: If request already exists
        """
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
            
        try:
            if request.id in self.requests.items:
                raise ValueError(f"Request {request.id} already exists")
            
            # Add request to state container
            self.requests.add(request.id, request)
            
            # Update metrics
            self.metrics.total_requests += 1
            self.metrics.active_requests += 1
            
            self.logger.debug(f"Added request {request.id} to state management")
            return request
            
        except Exception as e:
            self.logger.error(f"Failed to add request {request.id}: {str(e)}")
            raise
    
    def update_request_status(
        self,
        request_id: str,
        status: RequestStatus,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update request status and metadata.
        
        Args:
            request_id: ID of request to update
            status: New status to set
            metadata: Optional additional data to update
            
        Raises:
            RuntimeError: If worker not initialized
            ValueError: If request not found
        """
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
            
        try:
            request = self.requests.get(request_id)
            if not request:
                raise ValueError(f"Request {request_id} not found")
            
            old_status = request.status
            request.status = status
            
            # Update metrics based on status change
            self._update_metrics_for_status_change(request, old_status, status, metadata)
            
            # Update request metadata
            if metadata:
                for key, value in metadata.items():
                    setattr(request, key, value)
            
            self.logger.debug(
                f"Updated request {request_id} status from {old_status} to {status}"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update request {request_id}: {str(e)}")
            raise
    
    def _update_metrics_for_status_change(
        self,
        request: Request,
        old_status: RequestStatus,
        new_status: RequestStatus,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update metrics based on request status change"""
        try:
            if old_status == new_status:
                return
                
            # Handle status transitions
            if new_status == RequestStatus.ASSIGNED:
                self.metrics.accepted_requests += 1
                
            elif new_status == RequestStatus.COMPLETED:
                self.metrics.completed_requests += 1
                self.metrics.active_requests -= 1
                
                # Calculate and update wait time
                if hasattr(request, 'pickup_time') and hasattr(request, 'request_time'):
                    wait_time = (request.pickup_time - request.request_time).total_seconds()
                    self._update_wait_time_metrics(wait_time)
                
                # Calculate and update service time
                if metadata and 'dropoff_time' in metadata and hasattr(request, 'pickup_time'):
                    service_time = (metadata['dropoff_time'] - request.pickup_time).total_seconds()
                    self._update_service_time_metrics(service_time)
                    
            elif new_status == RequestStatus.CANCELLED:
                self.metrics.cancelled_requests += 1
                self.metrics.active_requests -= 1
                
            elif new_status == RequestStatus.REJECTED:
                self.metrics.rejected_requests += 1
                self.metrics.active_requests -= 1
                
        except Exception as e:
            self.logger.error(f"Error updating metrics for status change: {str(e)}")
            raise
    
    def _update_wait_time_metrics(self, wait_time: float) -> None:
        """Update wait time metrics with new data point"""
        self.metrics.total_wait_time += wait_time
        self.metrics.average_wait_time = (
            self.metrics.total_wait_time / self.metrics.completed_requests
        )
    
    def _update_service_time_metrics(self, service_time: float) -> None:
        """Update service time metrics with new data point"""
        self.metrics.total_service_time += service_time
        self.metrics.average_service_time = (
            self.metrics.total_service_time / self.metrics.completed_requests
        )
    
    def get_request(self, request_id: str) -> Optional[Request]:
        """Get request by ID"""
        return self.requests.get(request_id)
    
    def get_active_requests(self) -> List[Request]:
        """Get all active requests"""
        return [
            request for request in self.requests.items.values()
            if request.status not in [
                RequestStatus.COMPLETED,
                RequestStatus.CANCELLED,
                RequestStatus.REJECTED
            ]
        ]
    
    def take_snapshot(self, timestamp: datetime) -> None:
        """Take snapshot of request states"""
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
        self.requests.take_snapshot(timestamp)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current request metrics"""
        return {
            "total_requests": self.metrics.total_requests,
            "active_requests": self.metrics.active_requests,
            "accepted_requests": self.metrics.accepted_requests,
            "rejected_requests": self.metrics.rejected_requests,
            "completed_requests": self.metrics.completed_requests,
            "cancelled_requests": self.metrics.cancelled_requests,
            "average_wait_time": self.metrics.average_wait_time,
            "average_service_time": self.metrics.average_service_time,
            "total_wait_time": self.metrics.total_wait_time,
            "total_service_time": self.metrics.total_service_time
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get current request states"""
        return {
            request_id: request.to_dict() 
            for request_id, request in self.requests.items.items()
        }
    
    def update_state(self, state: Dict[str, Any]) -> None:
        """Update request states"""
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
            
        for request_id, request_data in state.items():
            if request_id in self.requests.items:
                self.requests.update(request_id, request_data)
            else:
                request = Request.from_dict(request_data)
                self.requests.add(request_id, request)
    
    def restore_state(self, saved_state: Dict[str, Any]) -> None:
        """Restore request states from saved state"""
        self.requests = StateContainer[Request]()
        for request_id, request_data in saved_state.items():
            request = Request.from_dict(request_data)
            self.requests.add(request_id, request)
    
    def begin_transaction(self) -> None:
        """Begin state transaction"""
        self.requests.begin_transaction()
    
    def commit_transaction(self) -> None:
        """Commit current transaction"""
        self.requests.commit_transaction()
    
    def rollback_transaction(self) -> None:
        """Rollback current transaction"""
        self.requests.rollback_transaction()
    
    def cleanup(self) -> None:
        """Clean up resources"""
        self.requests.clear_history()
        self.metrics = RequestMetrics()
        self.initialized = False
        self.logger.info("Request state worker cleaned up")