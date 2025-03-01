# core/llm/llm_client.py

import time
import threading
from typing import Dict, Any, Optional, Callable, List
import backoff
import openai
from openai import OpenAI
from config.app_config import config, LLMConfig
from config.logging_config import get_module_logger

# Create a logger for this module
logger = get_module_logger("llm_client")

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


class LLMClient:
    """Client for interacting with LLMs with retry, rate limiting, and model fallbacks."""
    
    # Define fallback model hierarchy
    MODEL_FALLBACKS = {
        "gpt-4o": ["gpt-4o-mini", "gpt-3.5-turbo"],
        "gpt-4": ["gpt-4o", "gpt-3.5-turbo"],
        "gpt-4-turbo": ["gpt-3.5-turbo"],
        "gpt-3.5-turbo": ["gpt-3.5-turbo-16k"],
    }
    
    def __init__(self, llm_config: Optional[LLMConfig] = None):
        """Initialize with configuration.
        
        Args:
            llm_config: LLM configuration (default: from app config)
        """
        self.config = llm_config or config.llm
        self.client = OpenAI(api_key=self.config.api_key)
        self.rate_limiter = RateLimiter(self.config.rate_limit_rpm)
        
        # Configure backoff parameters
        self.max_retries = self.config.max_retries
        
        # Cache for recent API calls to reduce duplicate requests
        self._cache = {}
        self._cache_ttl = 3600  # Cache TTL in seconds
        
        logger.debug(f"Initialized LLM client with model {self.config.model_name}")
    
    @backoff.on_exception(
        backoff.expo, 
        (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError),
        max_tries=5,
        jitter=backoff.full_jitter
    )
    def _call_with_retry(self, func, *args, **kwargs):
        """Call a function with exponential backoff retry.
        
        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
        """
        return func(*args, **kwargs)
    
    def _get_cache_key(self, messages, model, temperature, max_tokens):
        """Generate a cache key for a request.
        
        Args:
            messages: The messages for the request
            model: The model name
            temperature: The temperature setting
            max_tokens: The max tokens setting
            
        Returns:
            A cache key string
        """
        # Simple cache key generation - in production, consider a more robust hashing method
        message_str = str(messages)
        return f"{model}_{temperature}_{max_tokens}_{hash(message_str)}"
    
    def _try_get_from_cache(self, cache_key):
        """Try to get a response from the cache.
        
        Args:
            cache_key: The cache key
            
        Returns:
            Cached response or None
        """
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            if time.time() - entry["timestamp"] < self._cache_ttl:
                logger.debug(f"Cache hit for {cache_key}")
                return entry["response"]
            else:
                # Entry expired
                del self._cache[cache_key]
        return None
    
    def _add_to_cache(self, cache_key, response):
        """Add a response to the cache.
        
        Args:
            cache_key: The cache key
            response: The response to cache
        """
        self._cache[cache_key] = {
            "response": response,
            "timestamp": time.time()
        }
        
        # Cleanup old cache entries
        current_time = time.time()
        self._cache = {
            k: v for k, v in self._cache.items() 
            if current_time - v["timestamp"] < self._cache_ttl
        }
    
    @RateLimiter
    def chat_completion(self, 
                       messages: List[Dict[str, str]], 
                       temperature: Optional[float] = None,
                       max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """Get a chat completion with retry, rate limiting, and model fallbacks.
        
        Args:
            messages: List of message dictionaries
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            
        Returns:
            Completion response
            
        Raises:
            Exception: If the API call fails after retries and fallbacks
        """
        # Use instance defaults if not specified
        temperature = temperature if temperature is not None else self.config.temperature
        max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        
        # Start with the configured model
        model = self.config.model_name
        models_to_try = [model] + self.MODEL_FALLBACKS.get(model, [])
        
        # Check cache first
        cache_key = self._get_cache_key(messages, model, temperature, max_tokens)
        cached_response = self._try_get_from_cache(cache_key)
        if cached_response:
            return cached_response
        
        # Try models in fallback order
        last_exception = None
        for model_name in models_to_try:
            try:
                logger.debug(f"Making chat completion request with model {model_name} and {len(messages)} messages")
                
                # Use retry wrapper
                response = self._call_with_retry(
                    self.client.chat.completions.create,
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=self.config.request_timeout
                )
                
                # Extract and return relevant information
                result = {
                    "content": response.choices[0].message.content,
                    "finish_reason": response.choices[0].finish_reason,
                    "model": response.model,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                }
                
                # If we used a fallback model, log it
                if model_name != model:
                    logger.info(f"Used fallback model {model_name} instead of {model}")
                    result["used_fallback"] = True
                    result["original_model"] = model
                
                # Cache the successful response
                self._add_to_cache(cache_key, result)
                
                return result
                
            except openai.BadRequestError as e:
                # If the model doesn't exist or the request is invalid, log and try the next model
                logger.warning(f"Bad request with model {model_name}: {str(e)}")
                last_exception = e
                continue
                
            except Exception as e:
                # For other exceptions, log and try the next model
                logger.warning(f"Error with model {model_name}: {str(e)}")
                last_exception = e
                continue
        
        # If we get here, all models failed
        logger.error(f"All models failed. Last error: {str(last_exception)}")
        raise last_exception or Exception("All models failed for unknown reasons")
    
    def embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings with retry and rate limiting.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            Exception: If the API call fails after retries
        """
        try:
            logger.debug(f"Making embeddings request for {len(texts)} texts")
            
            # Try to use the smaller, faster embedding model first
            try:
                # Use retry wrapper
                response = self._call_with_retry(
                    self.client.embeddings.create,
                    model="text-embedding-3-small",  # Try the smaller model first
                    input=texts,
                    timeout=self.config.request_timeout
                )
            except Exception as e:
                logger.warning(f"Failed to use text-embedding-3-small, falling back to ada: {str(e)}")
                # Fall back to ada if the 3-small model fails
                response = self._call_with_retry(
                    self.client.embeddings.create,
                    model="text-embedding-ada-002",  # Fallback embedding model
                    input=texts,
                    timeout=self.config.request_timeout
                )
            
            # Extract embeddings from response
            embeddings = [data.embedding for data in response.data]
            
            logger.debug(f"Generated {len(embeddings)} embeddings using model {response.model}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error in embeddings: {str(e)}", exc_info=True)
            raise
