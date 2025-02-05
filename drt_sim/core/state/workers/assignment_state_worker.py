from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict
from drt_sim.core.state.base import StateWorker, StateContainer
from drt_sim.core.logging_config import setup_logger
from drt_sim.models.matching.types import Assignment
from drt_sim.models.simulation import AssignmentSystemState

@dataclass
class AssignmentMetrics:
    """Metrics for request-vehicle assignments"""
    total_assignments: int = 0
    total_waiting_time: float = 0
    total_in_vehicle_time: float = 0
    total_detour_time: float = 0
    average_waiting_time: float = 0
    average_in_vehicle_time: float = 0
    average_detour_time: float = 0
    average_computation_time: float = 0
    total_assignment_score: float = 0
    average_assignment_score: float = 0
    assignments_by_hour: Dict[int, int] = field(default_factory=lambda: defaultdict(int))

class AssignmentStateWorker(StateWorker):
    """Manages state for request-vehicle assignments"""
    
    def __init__(self):
        self.assignments = StateContainer[Assignment]()
        self.metrics = AssignmentMetrics()
        self.initialized = False
        self.logger = setup_logger(__name__)
        # Indexes for quick lookups
        self.request_to_assignment: Dict[str, str] = {}
        self.vehicle_to_assignments: Dict[str, List[str]] = defaultdict(list)
        
    def initialize(self, config: Optional[Any] = None) -> None:
        """Initialize assignment state worker"""
        self.initialized = True
        self.logger.info("Assignment state worker initialized")
    
    def add_assignment(self, assignment: Assignment) -> None:
        """
        Add a new assignment to state management.
        
        Args:
            assignment: Assignment object to add
            
        Raises:
            RuntimeError: If worker not initialized
            ValueError: If assignment already exists
        """
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
            
        try:
            assignment_id = assignment.id
            
            if assignment_id in self.assignments.items:
                raise ValueError(f"Assignment {assignment_id} already exists")
            
            # Add to main container
            self.assignments.add(assignment_id, assignment)
            
            # Update indexes
            self.request_to_assignment[assignment.request_id] = assignment_id
            self.vehicle_to_assignments[assignment.vehicle_id].append(assignment_id)
            
            # Update metrics
            self._update_metrics_for_new_assignment(assignment)
            
            self.logger.debug(
                f"Added assignment for request {assignment.request_id} to vehicle {assignment.vehicle_id}"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to add assignment: {str(e)}")
            raise
    
    def get_assignment(self, assignment_id: str) -> Optional[Assignment]:
        """Get assignment by ID"""
        return self.assignments.get(assignment_id)
    
    def get_assignment_for_request(self, request_id: str) -> Optional[Assignment]:
        """Get assignment for a specific request"""
        assignment_id = self.request_to_assignment.get(request_id)
        if assignment_id:
            return self.assignments.get(assignment_id)
        return None
    
    def get_assignments_for_vehicle(self, vehicle_id: str) -> List[Assignment]:
        """Get all assignments for a specific vehicle"""
        assignment_ids = self.vehicle_to_assignments.get(vehicle_id, [])
        return [
            assignment for assignment_id in assignment_ids
            if (assignment := self.assignments.get(assignment_id)) is not None
        ]
    
    def get_assignments_in_time_window(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[Assignment]:
        """Get all assignments within a specific time window"""
        return [
            assignment for assignment in self.assignments.items.values()
            if start_time <= assignment.assignment_time <= end_time
        ]
    
    def update_assignment(
        self,
        request_id: str,
        vehicle_id: str,
        updates: Dict[str, Any]
    ) -> None:
        """
        Update an existing assignment with new data
        
        Args:
            request_id: ID of request
            vehicle_id: ID of vehicle
            updates: Dictionary of fields to update
            
        Raises:
            ValueError: If assignment not found
        """
        assignment_id = f"{request_id}_{vehicle_id}"
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
    
    def remove_assignment(self, request_id: str, vehicle_id: str) -> None:
        """
        Remove an assignment from state management
        
        Args:
            request_id: ID of request
            vehicle_id: ID of vehicle
            
        Raises:
            ValueError: If assignment not found
        """
        assignment_id = f"{request_id}_{vehicle_id}"
        assignment = self.assignments.get(assignment_id)
        if not assignment:
            raise ValueError(f"Assignment {assignment_id} not found")
            
        # Remove from indexes
        self.request_to_assignment.pop(assignment.request_id, None)
        self.vehicle_to_assignments[assignment.vehicle_id].remove(assignment_id)
        
        # Remove from container
        self.assignments.remove(assignment_id)
    
    def _update_metrics_for_new_assignment(self, assignment: Assignment) -> None:
        """Update metrics when new assignment is added"""
        self.metrics.total_assignments += 1
        self.metrics.total_waiting_time += assignment.waiting_time_mins
        self.metrics.total_in_vehicle_time += assignment.in_vehicle_time_mins
        self.metrics.total_detour_time += assignment.detour_time_mins
        self.metrics.total_assignment_score += assignment.assignment_score
        
        # Update averages
        if self.metrics.total_assignments > 0:
            self.metrics.average_waiting_time = (
                self.metrics.total_waiting_time / self.metrics.total_assignments
            )
            self.metrics.average_in_vehicle_time = (
                self.metrics.total_in_vehicle_time / self.metrics.total_assignments
            )
            self.metrics.average_detour_time = (
                self.metrics.total_detour_time / self.metrics.total_assignments
            )
            self.metrics.average_assignment_score = (
                self.metrics.total_assignment_score / self.metrics.total_assignments
            )
            self.metrics.average_computation_time = (
                sum(a.computation_time for a in self.assignments.items.values()) / 
                self.metrics.total_assignments
            )
            
        # Update hourly metrics
        hour = assignment.assignment_time.hour
        self.metrics.assignments_by_hour[hour] += 1

    
    def take_snapshot(self, timestamp: datetime) -> None:
        """Take snapshot of assignment states"""
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
        self.assignments.take_snapshot(timestamp)
    
    def get_state(self) -> AssignmentSystemState:
        """
        Get current assignment system state
        
        Returns:
            AssignmentSystemState containing current state
            
        Raises:
            RuntimeError: If worker not initialized
        """
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
            
        try:
            return AssignmentSystemState(
                assignments=self.assignments.items,
                assignments_by_request=self.request_to_assignment,
                assignments_by_vehicle=dict(self.vehicle_to_assignments)
            )
            
        except Exception as e:
            self.logger.error(f"Error getting assignment system state: {str(e)}")
            raise
    
    def update_state(self, state: AssignmentSystemState) -> None:
        """
        Update assignment states from an AssignmentSystemState structure
        
        Args:
            state: AssignmentSystemState containing new state data
            
        Raises:
            RuntimeError: If worker not initialized
        """
        if not self.initialized:
            raise RuntimeError("Worker not initialized")
            
        try:
            # Begin transaction
            self.begin_transaction()
            
            try:
                # Clear existing indexes
                self.request_to_assignment.clear()
                self.vehicle_to_assignments.clear()
                
                # Update assignments and rebuild indexes
                for assignment_id, assignment in state.assignments.items():
                    if assignment_id in self.assignments.items:
                        self.assignments.update(assignment_id, assignment)
                    else:
                        self.assignments.add(assignment_id, assignment)
                        
                    # Update indexes
                    self.request_to_assignment[assignment.request_id] = assignment_id
                    self.vehicle_to_assignments[assignment.vehicle_id].append(assignment_id)
                
                # Update metrics
                self.metrics = AssignmentMetrics()
                for assignment in self.assignments.items.values():
                    self._update_metrics_for_new_assignment(assignment)
                
                # Commit transaction
                self.commit_transaction()
                
                self.logger.debug("Updated assignment system state")
                
            except Exception as e:
                self.rollback_transaction()
                raise
                
        except Exception as e:
            self.logger.error(f"Error updating assignment system state: {str(e)}")
            raise
    
    def restore_state(self, saved_state: AssignmentSystemState) -> None:
        """
        Restore assignment states from a saved AssignmentSystemState
        
        Args:
            saved_state: AssignmentSystemState containing saved state data
            
        Raises:
            RuntimeError: If worker not initialized
        """
        try:
            # Reset current state
            self.assignments = StateContainer[Assignment]()
            self.metrics = AssignmentMetrics()
            self.request_to_assignment.clear()
            self.vehicle_to_assignments.clear()
            
            # Restore using update_state
            self.update_state(saved_state)
            
            self.logger.debug("Restored assignment system state")
            
        except Exception as e:
            self.logger.error(f"Error restoring assignment system state: {str(e)}")
            raise
    
    def cleanup(self) -> None:
        """Clean up resources"""
        self.assignments.clear_history()
        self.request_to_assignment.clear()
        self.vehicle_to_assignments.clear()
        self.metrics = AssignmentMetrics()
        self.initialized = False
        self.logger.info("Assignment state worker cleaned up")
    
    def begin_transaction(self) -> None:
        """Begin state transaction"""
        self.assignments.begin_transaction()
    
    def commit_transaction(self) -> None:
        """Commit current transaction"""
        self.assignments.commit_transaction()
    
    def rollback_transaction(self) -> None:
        """Rollback current transaction"""
        self.assignments.rollback_transaction()