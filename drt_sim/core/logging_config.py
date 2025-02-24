import logging
import logging.handlers
from pathlib import Path
from typing import List

def configure_logging(
    base_log_dir: Path,
    log_level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> None:
    """
    Configure global logging for the application.

    This creates a console handler and a module-specific rotating file handler.
    Modules can then simply use logging.getLogger(__name__).

    Args:
        base_log_dir: Base directory for all logs.
        log_level: Logging level (default: INFO).
        max_bytes: Maximum size of each log file before rotation (default: 10MB).
        backup_count: Number of backup files to keep (default: 5).
    """
    base_log_dir = Path(base_log_dir)

    # Create directories for module logs and replications.
    modules_log_dir = base_log_dir / 'modules'
    modules_log_dir.mkdir(parents=True, exist_ok=True)
    replications_log_dir = base_log_dir / 'replications'
    replications_log_dir.mkdir(parents=True, exist_ok=True)

    # Remove any existing handlers on the root logger.
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create a common formatter.
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console handler.
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)

    # Module-specific handler.
    # This handler writes to files under base_log_dir/modules/<module_path>/module.log.
    class ModuleFileHandler(logging.handlers.RotatingFileHandler):
        def __init__(self, base_dir: Path, *args, **kwargs):
            self.base_dir = Path(base_dir)
            # Dummy file path; will be replaced in emit.
            super().__init__(str(base_dir / "dummy.log"), *args, **kwargs)
        
        def emit(self, record):
            # Create a file path based on the module name.
            module_path = record.name.replace('.', '/')
            module_dir = self.base_dir / module_path
            module_dir.mkdir(parents=True, exist_ok=True)
            self.baseFilename = str(module_dir / 'module.log')
            super().emit(record)

    module_handler = ModuleFileHandler(
        base_dir=modules_log_dir,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    module_handler.setFormatter(formatter)
    module_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(module_handler)

def add_replication_file_handler(
    base_log_dir: Path,
    replication_id: str,
    log_level: int = logging.DEBUG
) -> List[logging.Handler]:
    """
    Add file handlers that write logs to a replication-specific directory.

    Creates two types of log files:
    1. A main replication.log containing all logs
    2. Module-specific log files under <module_path>/module.log

    Args:
        base_log_dir: The base directory for logs.
        replication_id: A unique ID for this replication (e.g., "rep_1").
        log_level: The logging level for this handler.

    Returns:
        List of handlers that were added (to be removed when replication ends).
    """
    base_log_dir = Path(base_log_dir)
    replication_dir = base_log_dir / 'replications' / replication_id
    replication_dir.mkdir(parents=True, exist_ok=True)

    handlers = []
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 1. Create main replication log handler
    main_log_file = replication_dir / "replication.log"
    main_handler = logging.handlers.RotatingFileHandler(
        filename=str(main_log_file),
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    main_handler.setFormatter(formatter)
    main_handler.setLevel(log_level)
    logging.getLogger().addHandler(main_handler)
    handlers.append(main_handler)

    # 2. Create module-specific handler
    class ReplicationModuleHandler(logging.handlers.RotatingFileHandler):
        def __init__(self, base_dir: Path, *args, **kwargs):
            self.base_dir = Path(base_dir)
            self.handlers = {}  # Cache for module-specific handlers
            # Dummy file path; will be replaced in emit
            super().__init__(str(base_dir / "dummy.log"), *args, **kwargs)
        
        def emit(self, record):
            try:
                # Get or create handler for this module
                if record.name not in self.handlers:
                    # Create module-specific directory and handler
                    module_path = record.name.replace('.', '/')
                    module_dir = self.base_dir / module_path
                    module_dir.mkdir(parents=True, exist_ok=True)
                    log_file = module_dir / "module.log"
                    
                    # Create new handler for this module
                    handler = logging.handlers.RotatingFileHandler(
                        filename=str(log_file),
                        maxBytes=self.maxBytes,
                        backupCount=self.backupCount,
                        encoding=self.encoding
                    )
                    handler.setFormatter(self.formatter)
                    self.handlers[record.name] = handler
                
                # Use the module-specific handler to emit the record
                self.handlers[record.name].emit(record)
                
            except Exception as e:
                # If there's an error, try to emit to the base handler
                self.handleError(record)
        
        def close(self):
            """Close all handlers"""
            try:
                # Close all module-specific handlers
                for handler in self.handlers.values():
                    handler.close()
                self.handlers.clear()
            finally:
                super().close()

    module_handler = ReplicationModuleHandler(
        base_dir=replication_dir,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    module_handler.setFormatter(formatter)
    module_handler.setLevel(log_level)
    logging.getLogger().addHandler(module_handler)
    handlers.append(module_handler)

    return handlers

def cleanup_logging() -> None:
    """
    Clean up logging by removing and closing all handlers.
    """
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)
