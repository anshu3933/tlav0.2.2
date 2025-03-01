# config/logging_config.py
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Dict, Optional

class LoggerFactory:
    """Factory for creating configured loggers."""
    
    # Dictionary to store logger instances
    _loggers: Dict[str, logging.Logger] = {}
    
    @classmethod
    def get_logger(cls, name: str, log_to_file: bool = True) -> logging.Logger:
        """Get or create a logger with the specified name.
        
        Args:
            name: The name of the logger
            log_to_file: Whether to log to a file
            
        Returns:
            A configured logger
        """
        if name in cls._loggers:
            return cls._loggers[name]
            
        # Create new logger
        logger = logging.getLogger(name)
        
        # Only configure if not already configured
        if not logger.handlers:
            cls._configure_logger(logger, log_to_file)
            
        cls._loggers[name] = logger
        return logger
    
    @staticmethod
    def _configure_logger(logger: logging.Logger, log_to_file: bool):
        """Configure the logger with proper formatting and handlers."""
        # Set level based on environment
        log_level = os.getenv("LOG_LEVEL", "INFO")
        logger.setLevel(getattr(logging, log_level))
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler (optional)
        if log_to_file:
            # Create logs directory if it doesn't exist
            log_dir = "logs"
            os.makedirs(log_dir, exist_ok=True)
            
            # Create file handler
            file_handler = RotatingFileHandler(
                filename=f"{log_dir}/{logger.name}.log",
                maxBytes=10485760,  # 10MB
                backupCount=5
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # Disable propagation to avoid duplicate logs
        logger.propagate = False

# Global logger for general use
logger = LoggerFactory.get_logger("educational_assistant")


class LoggerAdapter(logging.LoggerAdapter):
    """Logger adapter with context information."""
    
    def __init__(self, logger, context=None):
        """Initialize the adapter with a logger and context."""
        super().__init__(logger, context or {})
    
    def process(self, msg, kwargs):
        """Add contextual information to the log message."""
        if self.extra:
            context_str = " ".join(f"{k}={v}" for k, v in self.extra.items())
            return f"{msg} [{context_str}]", kwargs
        return msg, kwargs


def get_module_logger(module_name: str, context: Optional[Dict] = None) -> logging.Logger:
    """Get a logger for a specific module with optional context.
    
    Args:
        module_name: The name of the module
        context: Optional context information
    
    Returns:
        A configured logger
    """
    logger = LoggerFactory.get_logger(module_name)
    
    if context:
        return LoggerAdapter(logger, context)
    
    return logger
