# drt_sim/handlers/system_handler.py

from typing import Optional, List
from datetime import datetime

from .base import BaseHandler
from drt_sim.models.event import Event
from drt_sim.models.simulation import SimulationStatus

class SystemHandler(BaseHandler):
    """Handles system-level events in the simulation"""

    def handle_simulation_initialized(self, event: Event) -> Optional[List[Event]]:
        """Handle simulation initialization"""
        try:
            # Initialize all workers
            self.state_manager.initialize_workers()
            
            # Take initial snapshot
            self.state_manager.take_snapshot(self.context.current_time)
            
            self.logger.info("Simulation initialized")
            
            return None
        except Exception as e:
            self.logger.error(f"Failed to initialize simulation: {str(e)}")
            raise

    def handle_warmup_completed(self, event: Event) -> Optional[List[Event]]:
        """Handle warmup period completion"""
        try:
            # Take snapshot at warmup completion
            self.state_manager.take_snapshot(self.context.current_time)
            
            # Update simulation status
            self.context.status = SimulationStatus.RUNNING
            
            self.logger.info("Warmup period completed")
            
            return None
        except Exception as e:
            self.logger.error(f"Failed to handle warmup completion: {str(e)}")
            raise

    def handle_simulation_completed(self, event: Event) -> Optional[List[Event]]:
        """Handle simulation completion"""
        try:
            # Take final snapshot
            self.state_manager.take_snapshot(self.context.current_time)
            
            # Update simulation status
            self.context.status = SimulationStatus.COMPLETED
            
            # Generate final metrics
            metrics = self.state_manager.get_metrics()
            
            self.logger.info("Simulation completed successfully")
            self.logger.info(f"Final metrics: {metrics}")
            
            return None
        except Exception as e:
            self.logger.error(f"Failed to handle simulation completion: {str(e)}")
            raise

    def handle_system_error(self, event: Event) -> Optional[List[Event]]:
        """Handle system error"""
        error_type = event.data.get('error_type', 'Unknown')
        error_message = event.data.get('error_message', 'No details available')
        
        try:
            # Update simulation status
            self.context.status = SimulationStatus.FAILED
            
            # Save error state if configured
            if self.config.save_error_state:
                self.state_manager.save_error_state({
                    'error_type': error_type,
                    'error_message': error_message,
                    'timestamp': self.context.current_time,
                    'system_state': self.state_manager.get_current_state()
                })
            
            self.logger.error(f"System error ({error_type}): {error_message}")
            
            return None
        except Exception as e:
            self.logger.error(f"Failed to handle system error: {str(e)}")
            raise