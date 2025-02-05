from typing import Dict, Any, Optional, List
from datetime import datetime
from collections import defaultdict
from drt_sim.core.state.base import StateWorker, StateContainer
from drt_sim.core.logging_config import setup_logger
from drt_sim.models.stop import StopAssignment
from drt_sim.models.simulation import StopAssignmentSystemState

class StopAssignmentStateWorker(StateWorker):
    """Manages state for stop assignments"""
    
    def __init__(self):
        self.assignments = StateContainer[StopAssignment]()
        self.initialized = False
        self.logger = setup_logger(__name__)
        # Index for quick lookups
        self.request_to_assignment: Dict[str, str] = {}
        self.stop_to_assignments: Dict[str, List[str]] = defaultdict(list)
        
    def initialize(self, config: Optional[Any] = None) -> None:
        """Initialize stop assignment state worker"""
        self.initialized = True
        self.logger.info("Stop assignment state worker initialized")
    
    def add_assignment(self, assignment: StopAssignment) -> None:
        """
        Add a new stop assignment to state management.
        
        Args:
            assignment: StopAssignment object to add
            
        Raises:
            RuntimeError: If worker not initialized
            ValueError: If assignment already exists
        """
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
            
        try:
            if assignment.id in self.assignments.items:
                raise ValueError(f"Assignment {assignment.id} already exists")
            
            # Add to main container
            self.assignments.add(assignment.id, assignment)
            
            # Update indexes
            self.request_to_assignment[assignment.request_id] = assignment.id
            self.stop_to_assignments[assignment.origin_stop.id].append(assignment.id)
            self.stop_to_assignments[assignment.destination_stop.id].append(assignment.id)
            
            self.logger.debug(
                f"Added assignment {assignment.id} for request {assignment.request_id}"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to add assignment: {str(e)}")
            raise
    
    def get_assignment(self, assignment_id: str) -> Optional[StopAssignment]:
        """Get assignment by ID"""
        return self.assignments.get(assignment_id)
    
    def get_assignment_for_request(self, request_id: str) -> Optional[StopAssignment]:
        """Get stop assignment for a specific request"""
        assignment_id = self.request_to_assignment.get(request_id)
        if assignment_id:
            return self.assignments.get(assignment_id)
        return None
    
    def get_assignments_for_stop(self, stop_id: str) -> List[StopAssignment]:
        """Get all assignments involving a specific stop"""
        assignment_ids = self.stop_to_assignments.get(stop_id, [])
        return [
            assignment for assignment_id in assignment_ids
            if (assignment := self.assignments.get(assignment_id)) is not None
        ]
    
    def get_assignments_in_time_window(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[StopAssignment]:
        """Get all assignments within a specific time window"""
        return [
            assignment for assignment in self.assignments.items.values()
            if start_time <= assignment.assignment_time <= end_time
        ]
    
    def update_assignment(
        self,
        assignment_id: str,
        updates: Dict[str, Any]
    ) -> None:
        """
        Update an existing assignment with new data
        
        Args:
            assignment_id: ID of assignment to update
            updates: Dictionary of fields to update
            
        Raises:
            ValueError: If assignment not found
        """
        assignment = self.assignments.get(assignment_id)
        if not assignment:
            raise ValueError(f"Assignment {assignment_id} not found")
            
        # Update fields
        for key, value in updates.items():
            if hasattr(assignment, key):
                setattr(assignment, key, value)
                
        # Update metadata if provided
        if 'metadata' in updates:
            assignment.metadata.update(updates['metadata'])
    
    def remove_assignment(self, assignment_id: str) -> None:
        """
        Remove an assignment from state management
        
        Args:
            assignment_id: ID of assignment to remove
            
        Raises:
            ValueError: If assignment not found
        """
        assignment = self.assignments.get(assignment_id)
        if not assignment:
            raise ValueError(f"Assignment {assignment_id} not found")
            
        # Remove from indexes
        self.request_to_assignment.pop(assignment.request_id, None)
        self.stop_to_assignments[assignment.origin_stop.id].remove(assignment_id)
        self.stop_to_assignments[assignment.destination_stop.id].remove(assignment_id)
        
        # Remove from container
        self.assignments.remove(assignment_id)

    def take_snapshot(self, timestamp: datetime) -> None:
        """Take snapshot of assignment states"""
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
        self.assignments.take_snapshot(timestamp)
    
    def get_state(self) -> StopAssignmentSystemState:
        """Get current state of the stop assignment system"""
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
            
        try:
            return StopAssignmentSystemState(
                assignments=self.assignments.items,
                assignments_by_request=self.request_to_assignment,
                assignments_by_stop=dict(self.stop_to_assignments),
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get stop assignment system state: {str(e)}")
            raise

    def update_state(self, state: StopAssignmentSystemState) -> None:
        """Update stop assignment system state"""
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
            
        try:
            # Update assignments
            self.assignments.items = state.assignments
            
            # Update indexes
            self.request_to_assignment = state.assignments_by_request
            self.stop_to_assignments = defaultdict(list, state.assignments_by_stop)

            self.logger.info("Stop assignment system state updated successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to update stop assignment system state: {str(e)}")
            raise

    def restore_state(self, saved_state: Dict[str, Any]) -> None:
        """
        Restore stop assignment states from a saved StopAssignmentSystemState
        
        Args:
            saved_state: Dictionary containing saved state data
            
        Raises:
            RuntimeError: If worker not initialized
        """
        try:
            # Reset current state
            self.assignments = StateContainer[StopAssignment]()
            self.request_to_assignment.clear()
            self.stop_to_assignments.clear()
            
            # Restore using update_state
            self.update_state(saved_state)
            
            self.logger.debug("Restored stop assignment system state")
            
        except Exception as e:
            self.logger.error(f"Error restoring stop assignment system state: {str(e)}")
            raise
    
    def cleanup(self) -> None:
        """Clean up resources"""
        self.assignments.clear_history()
        self.request_to_assignment.clear()
        self.stop_to_assignments.clear()
        self.initialized = False
        self.logger.info("Stop assignment state worker cleaned up")
    
    def begin_transaction(self) -> None:
        """Begin state transaction"""
        self.assignments.begin_transaction()
    
    def commit_transaction(self) -> None:
        """Commit current transaction"""
        self.assignments.commit_transaction()
    
    def rollback_transaction(self) -> None:
        """Rollback current transaction"""
        self.assignments.rollback_transaction()