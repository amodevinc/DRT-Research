# drt_sim/utils/random_seed_manager.py
import random
import numpy as np
from typing import Optional, Dict
import json
from pathlib import Path
from datetime import datetime

class RandomSeedManager:
    """Manages random seeds for reproducible experiments"""
    
    def __init__(self, base_seed: Optional[int] = None):
        self.base_seed = base_seed or int(datetime.now().timestamp())
        self.component_seeds: Dict[str, int] = {}
        self.seed_history: Dict[str, list] = {}
        
    def get_seed(self, component: str) -> int:
        """Get a deterministic seed for a specific component"""
        if component not in self.component_seeds:
            # Generate unique seed for component using base_seed
            component_seed = int(hash(f"{self.base_seed}_{component}") % (2**32))
            self.component_seeds[component] = component_seed
            self.seed_history[component] = []
            
        return self.component_seeds[component]
    
    def reset_component(self, component: str) -> None:
        """Reset the random state for a specific component"""
        if component in self.component_seeds:
            random.seed(self.component_seeds[component])
            np.random.seed(self.component_seeds[component])
    
    def log_seed_usage(self, component: str, context: str) -> None:
        """Log when and how seeds are used"""
        if component in self.seed_history:
            self.seed_history[component].append({
                'timestamp': datetime.now().isoformat(),
                'context': context,
                'seed_value': self.component_seeds[component]
            })
    
    def save_state(self, filepath: str) -> None:
        """Save the current seed state"""
        state = {
            'base_seed': self.base_seed,
            'component_seeds': self.component_seeds,
            'seed_history': self.seed_history
        }
        Path(filepath).write_text(json.dumps(state, indent=2))
    
    def load_state(self, filepath: str) -> None:
        """Load a previous seed state"""
        state = json.loads(Path(filepath).read_text())
        self.base_seed = state['base_seed']
        self.component_seeds = state['component_seeds']
        self.seed_history = state['seed_history']
