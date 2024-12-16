# drt_sim/core/logging_config.py
import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Dict
import json
from datetime import datetime

class DRTLogger:
    """Configurable logging system for DRT simulation"""
    
    def __init__(self, log_dir: str = "logs", 
                 log_level: int = logging.INFO,
                 retention_days: int = 30):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create main logger
        self.logger = logging.getLogger('drt_sim')
        self.logger.setLevel(log_level)
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # File handler for general logs
        general_log = self.log_dir / f"drt_sim_{datetime.now():%Y%m%d_%H%M%S}.log"
        file_handler = logging.handlers.TimedRotatingFileHandler(
            general_log, when='midnight', interval=1, backupCount=retention_days
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            '%(levelname)s: %(message)s'
        ))
        self.logger.addHandler(console_handler)
        
        # Create specialized loggers
        self.event_logger = self._create_specialized_logger('events')
        self.metrics_logger = self._create_specialized_logger('metrics')
        self.algorithm_logger = self._create_specialized_logger('algorithm')
        
    def _create_specialized_logger(self, name: str) -> logging.Logger:
        """Create a specialized logger for specific components"""
        logger = logging.getLogger(f'drt_sim.{name}')
        logger.setLevel(logging.INFO)
        
        log_file = self.log_dir / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.log"
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(message)s'
        ))
        logger.addHandler(handler)
        
        return logger
    
    def log_event(self, event_data: Dict):
        """Log simulation events"""
        self.event_logger.info(json.dumps(event_data))
        
    def log_metrics(self, metrics_data: Dict):
        """Log performance metrics"""
        self.metrics_logger.info(json.dumps(metrics_data))
        
    def log_algorithm(self, algorithm_data: Dict):
        """Log algorithm-specific information"""
        self.algorithm_logger.info(json.dumps(algorithm_data))
