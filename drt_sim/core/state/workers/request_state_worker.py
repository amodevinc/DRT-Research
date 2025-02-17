from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from drt_sim.core.state.base import StateWorker, StateContainer
from drt_sim.models.request import Request, RequestStatus
from collections import defaultdict
from drt_sim.models.state import RequestSystemState
import logging
logger = logging.getLogger(__name__)
class RequestStateWorker(StateWorker):
    """Manages state for passenger requests"""
    
    def __init__(self):
        self.requests = StateContainer[Request]()
        self.historical_requests = StateContainer[Request]()
        self.initialized = False
    
    def initialize(self, config: Optional[Any] = None) -> None:
        """Initialize request state worker"""
        self.initialized = True
        logger.info("Request state worker initialized")
    
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
            
            logger.debug(f"Added request {request.id} to state management")
            return request
            
        except Exception as e:
            logger.error(f"Failed to add request {request.id}: {str(e)}")
            raise

    def get_recent_requests(self, time_window: timedelta, current_time: datetime) -> List[Request]:
        """
        Get requests within recent time window.
        
        Args:
            time_window: Time window to look back from current time
            current_time: Current simulation time to measure window from
            
        Returns:
            List of requests within the specified time window
            
        Raises:
            RuntimeError: If worker not initialized
        """
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
            
        try:
            cutoff_time = current_time - time_window
            
            recent_requests = [
                request for request in self.requests.items.values()
                if hasattr(request, 'request_time') and 
                request.request_time >= cutoff_time
            ]
            
            logger.debug(
                f"Retrieved {len(recent_requests)} requests within last {time_window}"
            )
            return recent_requests
            
        except Exception as e:
            logger.error(f"Error getting recent requests: {str(e)}")
            raise

    def get_historical_requests(self, time_window: timedelta, current_time: datetime) -> List[Request]:
        """
        Get historical requests within specified time window.
        
        Args:
            time_window: Time window to analyze
            current_time: Current simulation time to measure window from
            
        Returns:
            List of historical requests within the time window
            
        Raises:
            RuntimeError: If worker not initialized
        """
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
            
        try:
            cutoff_time = current_time - time_window
            
            # First check in the historical_requests container
            historical_requests = [
                request for request in self.historical_requests.items.values()
                if hasattr(request, 'request_time') and 
                request.request_time >= cutoff_time
            ]
            
            logger.debug(
                f"Retrieved {len(historical_requests)} historical requests within {time_window}"
            )
            return historical_requests
            
        except Exception as e:
            logger.error(f"Error getting historical requests: {str(e)}")
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
            
            logger.debug(
                f"Updated request {request_id} status from {old_status} to {status}"
            )
            
        except Exception as e:
            logger.error(f"Failed to update request {request_id}: {str(e)}")
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
    
    def get_all_historical_requests(self) -> List[Request]:
        """Get all historical requests from the historical container"""
        return list(self.historical_requests.items.values())
    
    def load_historical_requests(self, requests: List[Request]) -> None:
        """
        Load a list of requests into the historical requests container.
        
        Args:
            requests: List of Request objects to load into historical section
        
        Raises:
            RuntimeError: If worker not initialized
        """
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
            
        try:
            count = 0
            for request in requests:
                if request.id not in self.historical_requests.items:
                    self.historical_requests.add(request.id, request)
                    count += 1
                    
            logger.info(f"Loaded {count} requests into historical section")
            
        except Exception as e:
            logger.error(f"Failed to load historical requests: {str(e)}")
            raise
    
    def take_snapshot(self, timestamp: datetime) -> None:
        """Take snapshot of request states for both active and historical requests"""
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
        self.requests.take_snapshot(timestamp)
        self.historical_requests.take_snapshot(timestamp)
    
    def get_state(self) -> RequestSystemState:
        """Get current state of the request system, including historical requests"""
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
            
        try:
            # Get requests by status
            requests_by_status = defaultdict(list)
            
            # Process active requests
            for rid, request in self.requests.items.items():
                requests_by_status[request.status].append(rid)
            
            # Process historical requests 
            for rid, request in self.historical_requests.items.items():
                if rid not in requests_by_status[request.status]:
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
                historical_requests={
                    rid: req for rid, req in self.historical_requests.items.items()
                },
            )
            
        except Exception as e:
            logger.error(f"Failed to get request system state: {str(e)}")
            raise

    def update_state(self, state: RequestSystemState) -> None:
        """Update request system state, including historical requests if present"""
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
            
        try:
            # Update active requests
            for request_id, request in state.active_requests.items():
                if request_id in self.requests.items:
                    self.requests.update(request_id, request)
                else:
                    self.requests.add(request_id, request)
            
            # Update historical requests if present in the state
            if hasattr(state, 'historical_requests') and state.historical_requests:
                for request_id, request in state.historical_requests.items():
                    if request_id in self.historical_requests.items:
                        self.historical_requests.update(request_id, request)
                    else:
                        self.historical_requests.add(request_id, request)
            
            logger.info("Request system state updated successfully")
            
        except Exception as e:
            logger.error(f"Failed to update request system state: {str(e)}")
            raise
    
    def restore_state(self, saved_state: Dict[str, Any]) -> None:
        """Restore request states from a saved RequestSystemState"""
        try:
            # Reset current state
            self.requests = StateContainer[Request]()
            self.historical_requests = StateContainer[Request]()
            # Restore using update_state
            self.update_state(saved_state)
            
            logger.debug("Restored request system state")

        except Exception as e:
            logger.error(f"Error restoring request system state: {str(e)}")
            raise
    
    def begin_transaction(self) -> None:
        """Begin state transaction for both containers"""
        self.requests.begin_transaction()
        self.historical_requests.begin_transaction()
    
    def commit_transaction(self) -> None:
        """Commit current transaction for both containers"""
        self.requests.commit_transaction()
        self.historical_requests.commit_transaction()
    
    def rollback_transaction(self) -> None:
        """Rollback current transaction for both containers"""
        self.requests.rollback_transaction()
        self.historical_requests.rollback_transaction()
    
    def cleanup(self) -> None:
        """Clean up resources for both containers"""
        self.requests.clear_history()
        self.historical_requests.clear_history()
        self.initialized = False
        logger.info("Request state worker cleaned up")