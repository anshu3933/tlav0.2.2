"""Error handling utilities and base exception classes."""

import traceback
from typing import Dict, Any, Optional, Type, List, Callable
from functools import wraps
import logging

# Get the logger
logger = logging.getLogger(__name__)

class AppBaseException(Exception):
    """Base exception class for application-specific exceptions."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize with error message and optional details.
        
        Args:
            message: Error message
            details: Additional error details
        """
        self.message = message
        self.details = details or {}
        super().__init__(message)


class ValidationError(AppBaseException):
    """Exception raised for input validation errors."""
    pass


class ConfigurationError(AppBaseException):
    """Exception raised for configuration errors."""
    pass


class DocumentError(AppBaseException):
    """Exception raised for document processing errors."""
    pass


class VectorStoreError(AppBaseException):
    """Exception raised for vector store errors."""
    pass


class LLMError(AppBaseException):
    """Exception raised for LLM-related errors."""
    pass


class RAGError(AppBaseException):
    """Exception raised for RAG-related errors."""
    pass


class PipelineError(AppBaseException):
    """Exception raised for pipeline-related errors."""
    pass


class UIError(AppBaseException):
    """Exception raised for UI-related errors."""
    pass


def format_exception(exc: Exception) -> Dict[str, Any]:
    """Format an exception into a structured dictionary.
    
    Args:
        exc: Exception to format
        
    Returns:
        Formatted exception dictionary
    """
    exc_type = type(exc).__name__
    exc_message = str(exc)
    exc_traceback = traceback.format_exc()
    
    result = {
        "type": exc_type,
        "message": exc_message,
        "traceback": exc_traceback
    }
    
    # Add details for application exceptions
    if isinstance(exc, AppBaseException) and exc.details:
        result["details"] = exc.details
    
    return result


def handle_exceptions(
    error_map: Dict[Type[Exception], Callable[[Exception], Any]] = None,
    default_handler: Optional[Callable[[Exception], Any]] = None,
    log_exception: bool = True
):
    """Decorator for handling exceptions in functions.
    
    Args:
        error_map: Mapping of exception types to handler functions
        default_handler: Default handler for unspecified exceptions
        log_exception: Whether to log the exception
        
    Returns:
        Decorator function
    """
    error_map = error_map or {}
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log exception if requested
                if log_exception:
                    logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
                
                # Find appropriate handler
                handler = None
                for exc_type, exc_handler in error_map.items():
                    if isinstance(e, exc_type):
                        handler = exc_handler
                        break
                
                # Use default handler if no specific handler found
                if handler is None:
                    if default_handler is not None:
                        return default_handler(e)
                    else:
                        raise
                
                # Call the handler
                return handler(e)
        
        return wrapper
    
    return decorator


def retry(
    max_attempts: int = 3,
    exceptions: List[Type[Exception]] = None,
    delay: float = 1.0,
    backoff: float = 2.0,
    logger: Optional[logging.Logger] = None
):
    """Retry decorator with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        exceptions: List of exceptions to catch and retry
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier
        logger: Logger for retry messages
        
    Returns:
        Decorator function
    """
    import time
    
    exceptions = exceptions or [Exception]
    _logger = logger or logging.getLogger(__name__)
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except tuple(exceptions) as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        _logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed: {str(e)}. "
                            f"Retrying in {_delay:.2f} seconds..."
                        )
                        time.sleep(_delay)
                        _delay *= backoff
                    else:
                        _logger.error(
                            f"All {max_attempts} attempts failed. Last error: {str(e)}"
                        )
            
            # If we get here, all attempts failed
            raise last_exception
        
        return wrapper
    
    return decorator
