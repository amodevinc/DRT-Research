from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from drt_sim.core.state.base import StateWorker, StateContainer
from drt_sim.models.request import Request, RequestStatus
from drt_sim.core.logging_config import setup_logger
from collections import defaultdict
from drt_sim.models.simulation import RequestSystemState
class RequestStateWorker(StateWorker):
    """Manages state for passenger requests"""
    
    def __init__(self):
        self.requests = StateContainer[Request]()
        self.initialized = False
        self.logger = setup_logger(__name__)
    
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
            
            self.logger.debug(f"Added request {request.id} to state management")
            return request
            
        except Exception as e:
            self.logger.error(f"Failed to add request {request.id}: {str(e)}")
            raise

    def get_recent_requests(self, time_window: timedelta) -> List[Request]:
        """
        Get requests within recent time window.
        
        Args:
            time_window: Time window to look back from current time
            
        Returns:
            List of requests within the specified time window
            
        Raises:
            RuntimeError: If worker not initialized
        """
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
            
        try:
            current_time = datetime.now()  # In practice, should use simulation time
            cutoff_time = current_time - time_window
            
            recent_requests = [
                request for request in self.requests.items.values()
                if hasattr(request, 'request_time') and 
                request.request_time >= cutoff_time
            ]
            
            self.logger.debug(
                f"Retrieved {len(recent_requests)} requests within last {time_window}"
            )
            return recent_requests
            
        except Exception as e:
            self.logger.error(f"Error getting recent requests: {str(e)}")
            raise
    
    def get_historical_requests(self, time_window: timedelta) -> List[Request]:
        """
        Get historical requests within specified time window.
        
        Args:
            time_window: Time window to analyze
            
        Returns:
            List of historical requests within the time window
            
        Raises:
            RuntimeError: If worker not initialized
        """
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
            
        try:
            current_time = datetime.now()  # In practice, should use simulation time
            cutoff_time = current_time - time_window
            
            historical_requests = [
                request for request in self.requests.items.values()
                if hasattr(request, 'request_time') and 
                request.request_time >= cutoff_time and
                request.status in [
                    RequestStatus.COMPLETED,
                    RequestStatus.CANCELLED,
                    RequestStatus.REJECTED
                ]
            ]
            
            self.logger.debug(
                f"Retrieved {len(historical_requests)} historical requests within {time_window}"
            )
            return historical_requests
            
        except Exception as e:
            self.logger.error(f"Error getting historical requests: {str(e)}")
            raise
    
    def update_request_status(
        self,
        request_id: str,
        status: RequestStatus,
    ) -> None:
        """
        Update request status and metadata.
        
        Args:
            request_id: ID of request to update
            status: New status to set
            metadata: Optional metadata to attach to request
            
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
            
            self.logger.debug(
                f"Updated request {request_id} status from {old_status} to {status}"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update request {request_id}: {str(e)}")
            raise

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
    
    def get_state(self) -> RequestSystemState:
        """Get current state of the request system"""
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
            
        try:
            # Get requests by status
            requests_by_status = defaultdict(list)
            for rid, request in self.requests.items.items():
                requests_by_status[request.status].append(rid)

            return RequestSystemState(
                active_requests={
                    rid: req for rid, req in self.requests.items.items()
                    if req.status not in [
                        RequestStatus.COMPLETED,
                        RequestStatus.CANCELLED,
                        RequestStatus.REJECTED
                    ]
                },
                requests_by_status=dict(requests_by_status),
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get request system state: {str(e)}")
            raise

    def update_state(self, state: RequestSystemState) -> None:
        """Update request system state"""
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
            
        try:
            # Update active requests
            for request_id, request in state.active_requests.items():
                if request_id in self.requests.items:
                    self.requests.update(request_id, request)
                else:
                    self.requests.add(request_id, request)
            
            self.logger.info("Request system state updated successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to update request system state: {str(e)}")
            raise
    
    def restore_state(self, saved_state: Dict[str, Any]) -> None:
        """Restore request states from a saved RequestSystemState"""
        try:
            # Reset current state
            self.requests = StateContainer[Request]()
            # Restore using update_state
            self.update_state(saved_state)
            
            self.logger.debug("Restored request system state")

        except Exception as e:
            self.logger.error(f"Error restoring request system state: {str(e)}")
            raise
    
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
        self.initialized = False
        self.logger.info("Request state worker cleaned up")