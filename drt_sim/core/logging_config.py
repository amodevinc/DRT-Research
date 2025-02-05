import logging
from pathlib import Path
from typing import Optional

def setup_logger(class_name: str, log_level: int = logging.INFO) -> logging.Logger:
    """
    Get or create a logger that inherits from root logger configuration.
    
    Args:
        class_name: Name of the class requesting the logger
        log_level: Logging level for this specific logger (default: logging.INFO)
        
    Returns:
        logging.Logger: Logger instance
    """
    logger = logging.getLogger(class_name)
    logger.setLevel(log_level)
    return logger