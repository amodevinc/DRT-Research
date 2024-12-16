# drt_sim/core/hooks.py
from typing import Dict, List, Callable, Any, Optional
from enum import Enum, auto
import logging
from dataclasses import dataclass
from datetime import datetime

class HookType(Enum):
    """Types of hooks available in the simulation"""
    # Simulation lifecycle hooks
    SIMULATION_INIT = auto()
    SIMULATION_START = auto()
    SIMULATION_END = auto()
    
    # Event processing hooks
    PRE_EVENT = auto()
    POST_EVENT = auto()
    
    # Time-based hooks
    TIME_STEP = auto()
    INTERVAL = auto()
    
    # Algorithm hooks
    PRE_DISPATCH = auto()
    POST_DISPATCH = auto()
    PRE_ROUTING = auto()
    POST_ROUTING = auto()
    PRE_MATCHING = auto()
    POST_MATCHING = auto()
    
    # State management hooks
    STATE_CHANGE = auto()
    METRICS_UPDATE = auto()
    
    # Vehicle hooks
    VEHICLE_DEPARTURE = auto()
    VEHICLE_ARRIVAL = auto()
    
    # Request hooks
    REQUEST_RECEIVED = auto()
    REQUEST_ASSIGNED = auto()
    REQUEST_COMPLETED = auto()
    REQUEST_CANCELLED = auto()

@dataclass
class HookRegistration:
    """Registration details for a hook"""
    callback: Callable
    priority: int
    condition: Optional[Callable] = None
    enabled: bool = True
    metadata: Dict[str, Any] = None

class HookManager:
    """Manages hooks and callbacks for the DRT simulation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._hooks: Dict[HookType, List[HookRegistration]] = {
            hook_type: [] for hook_type in HookType
        }
        self._interval_hooks: Dict[datetime, List[HookRegistration]] = {}
        
    def register_hook(self,
                     hook_type: HookType,
                     callback: Callable,
                     priority: int = 0,
                     condition: Optional[Callable] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> None:
        """Register a new hook"""
        if not callable(callback):
            raise ValueError("Callback must be callable")
            
        registration = HookRegistration(
            callback=callback,
            priority=priority,
            condition=condition,
            metadata=metadata or {}
        )
        
        self._hooks[hook_type].append(registration)
        # Sort by priority (higher numbers execute first)
        self._hooks[hook_type].sort(key=lambda x: -x.priority)
        
        self.logger.debug(
            f"Registered hook for {hook_type.name} with priority {priority}"
        )
        
    def register_interval_hook(self,
                             callback: Callable,
                             interval: datetime,
                             priority: int = 0,
                             metadata: Optional[Dict[str, Any]] = None) -> None:
        """Register a hook to be called at specific intervals"""
        if interval not in self._interval_hooks:
            self._interval_hooks[interval] = []
            
        registration = HookRegistration(
            callback=callback,
            priority=priority,
            metadata=metadata or {}
        )
        
        self._interval_hooks[interval].append(registration)
        self._interval_hooks[interval].sort(key=lambda x: -x.priority)
        
    def deregister_hook(self,
                       hook_type: HookType,
                       callback: Callable) -> bool:
        """Remove a registered hook"""
        hooks = self._hooks[hook_type]
        for i, registration in enumerate(hooks):
            if registration.callback == callback:
                hooks.pop(i)
                self.logger.debug(
                    f"Deregistered hook for {hook_type.name}"
                )
                return True
        return False
        
    def disable_hook(self,
                    hook_type: HookType,
                    callback: Callable) -> bool:
        """Temporarily disable a hook"""
        hooks = self._hooks[hook_type]
        for registration in hooks:
            if registration.callback == callback:
                registration.enabled = False
                return True
        return False
        
    def enable_hook(self,
                   hook_type: HookType,
                   callback: Callable) -> bool:
        """Re-enable a disabled hook"""
        hooks = self._hooks[hook_type]
        for registration in hooks:
            if registration.callback == callback:
                registration.enabled = True
                return True
        return False
        
    def call_hooks(self,
                  hook_type: HookType,
                  context: Any,
                  *args,
                  **kwargs) -> None:
        """Call all registered hooks of a specific type"""
        if hook_type not in self._hooks:
            return
            
        for registration in self._hooks[hook_type]:
            if not registration.enabled:
                continue
                
            try:
                # Check condition if specified
                if registration.condition and not registration.condition(context):
                    continue
                    
                registration.callback(context, *args, **kwargs)
                
            except Exception as e:
                self.logger.error(
                    f"Error in {hook_type.name} hook: {str(e)}"
                )
                # Don't raise the exception to avoid breaking the simulation
                # but log it for debugging
                
    def call_interval_hooks(self,
                          current_time: datetime,
                          context: Any) -> None:
        """Call hooks that should execute at the current interval"""
        for interval, hooks in self._interval_hooks.items():
            if self._should_execute_interval(current_time, interval):
                for registration in hooks:
                    if not registration.enabled:
                        continue
                        
                    try:
                        registration.callback(context)
                    except Exception as e:
                        self.logger.error(
                            f"Error in interval hook: {str(e)}"
                        )
                        
    def _should_execute_interval(self,
                               current_time: datetime,
                               interval: datetime) -> bool:
        """Check if an interval hook should execute at the current time"""
        # Implementation depends on how you want to handle intervals
        # This is a simple example
        return current_time.timestamp() % interval.timestamp() == 0
        
    def get_hooks(self,
                 hook_type: HookType) -> List[HookRegistration]:
        """Get all hooks registered for a specific type"""
        return self._hooks[hook_type].copy()
        
    def clear_hooks(self,
                   hook_type: Optional[HookType] = None) -> None:
        """Clear all hooks of a specific type or all hooks if type not specified"""
        if hook_type:
            self._hooks[hook_type].clear()
        else:
            for hook_list in self._hooks.values():
                hook_list.clear()
            self._interval_hooks.clear()