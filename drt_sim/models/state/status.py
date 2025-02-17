from enum import Enum

class SimulationStatus(Enum):
    """Possible states of the simulation"""
    INITIALIZED = "initialized"
    WARMING_UP = "warming_up"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped" 