"""Rate limiting utilities for API calls."""

import time
import threading
from typing import Callable, List
from config.logging_config import get_module_logger

# Create a logger for this module
logger = get_module_logger("rate_limiter")

class RateLimiter:
    """Thread-safe rate limiter for API calls."""
    
    def __init__(self, max_calls: int, time_period: int = 60):
        """Initialize with rate limit parameters.
        
        Args:
            max_calls: Maximum number of calls allowed in the time period
            time_period: Time period in seconds (default: 60)
        """
        self.max_calls = max_calls
        self.time_period = time_period
        self.calls = []
        self.lock = threading.Lock()
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to rate limit a function.
        
        Args:
            func: Function to rate limit
            
        Returns:
            Rate-limited function
        """
        def wrapper(*args, **kwargs):
            self.wait_if_needed()
            return func(*args, **kwargs)
        return wrapper
    
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded."""
        with self.lock:
            now = time.time()
            
            # Remove calls outside the time window
            self.calls = [t for t in self.calls if now - t < self.time_period]
            
            # If at capacity, wait until we can make a call
            if len(self.calls) >= self.max_calls:
                oldest_call = self.calls[0]
                sleep_time = self.time_period - (now - oldest_call)
                if sleep_time > 0:
                    logger.debug(f"Rate limit reached. Waiting {sleep_time:.2f} seconds")
                    time.sleep(sleep_time)
            
            # Record this call
            self.calls.append(time.time())


class TokenRateLimiter(RateLimiter):
    """Rate limiter based on token usage rather than call count."""
    
    def __init__(self, max_tokens: int, time_period: int = 60):
        """Initialize with token rate limit parameters.
        
        Args:
            max_tokens: Maximum number of tokens allowed in the time period
            time_period: Time period in seconds (default: 60)
        """
        super().__init__(max_tokens, time_period)
        self.tokens = []  # List of (timestamp, token_count) tuples
    
    def add_tokens(self, token_count: int):
        """Record token usage.
        
        Args:
            token_count: Number of tokens used
        """
        with self.lock:
            now = time.time()
            
            # Remove tokens outside the time window
            self.tokens = [(t, count) for t, count in self.tokens 
                          if now - t < self.time_period]
            
            # Add current token usage
            self.tokens.append((now, token_count))
    
    def wait_if_needed(self, estimated_tokens: int = 0):
        """Wait if token rate limit would be exceeded.
        
        Args:
            estimated_tokens: Estimated tokens for the upcoming request
        """
        with self.lock:
            now = time.time()
            
            # Remove tokens outside the time window
            self.tokens = [(t, count) for t, count in self.tokens 
                          if now - t < self.time_period]
            
            # Calculate current token usage
            current_usage = sum(count for _, count in self.tokens)
            
            # If adding estimated tokens would exceed limit, wait
            if current_usage + estimated_tokens > self.max_calls:
                # Sort tokens by timestamp (oldest first)
                self.tokens.sort(key=lambda x: x[0])
                
                # Calculate how many tokens need to expire
                tokens_to_expire = current_usage + estimated_tokens - self.max_calls
                expired_tokens = 0
                
                for i, (timestamp, count) in enumerate(self.tokens):
                    expired_tokens += count
                    if expired_tokens >= tokens_to_expire:
                        # Wait until this token expires
                        wait_time = timestamp + self.time_period - now
                        if wait_time > 0:
                            logger.debug(f"Token limit reached. Waiting {wait_time:.2f} seconds")
                            time.sleep(wait_time)
                        break
