# drt_sim/models/matching/enums.py
from enum import Enum

class AssignmentMethod(Enum):
    """Available methods for request-vehicle assignment."""
    INSERTION = "insertion"     # Basic insertion heuristic
    AUCTION = "auction"         # Auction-based assignment
    NEAREST = "nearest"         # Nearest vehicle assignment

class OptimizationMethod(Enum):
    """Available methods for assignment optimization."""
    ROLLING_HORIZON = "rolling_horizon"  # Rolling time window optimization
    LOCAL_SEARCH = "local_search"        # Local search improvement
    NONE = "none"                        # No additional optimization