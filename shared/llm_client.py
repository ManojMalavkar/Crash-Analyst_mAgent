"""Unified LLM Client with Model Fallback and Usage Tracking.

Provides a single interface for all agents to interact with LLMs through
Databricks AI Gateway or any OpenAI-compatible endpoint.

Features:
- Model fallback chain (primary → fallback_1 → fallback_2)
- Exponential backoff retry with jitter
- Token usage tracking per session/conversation
- Streaming and non-streaming support
- Tool/function calling support
- Thread-safe usage statistics

Usage:
    from shared.llm_client import LLMClient
    
    client = LLMClient()
    
    # Simple completion
    response = client.chat([
        {"role": "system", "content": "You are a CAE engineer assistant."},
        {"role": "user", "content": "Generate ANSA script to mesh a part"}
    ])
    print(response.content)
    
    # With tool calling
    response = client.chat(messages, tools=tool_specs)
    
    # Check usage
    print(client.get_usage_summary())
"""

import time
import random
import logging
from dataclasses import dataclass, field
from typing import Optional, Generator
from threading import Lock

from openai import OpenAI, APIError, RateLimitError, APITimeoutError

from shared.config import settings


logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TokenUsage:
    """Token usage statistics for a single request."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    latency_ms: float = 0.0


@dataclass
class LLMResponse:
    """Structured response from an LLM call."""
    content: Optional[str] = None
    tool_calls: Optional[list] = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    model: str = ""
    finish_reason: str = ""
    raw_response: Optional[object] = None
    
    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return self.tool_calls is not None and len(self.tool_calls) > 0


@dataclass
class UsageStats:
    """Aggregated usage statistics across multiple requests."""
    total_requests: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    failed_requests: int = 0
    fallback_count: int = 0
    requests_by_model: dict = field(default_factory=dict)


# =============================================================================
# Retry Logic
# =============================================================================

class RetryHandler:
    """Exponential backoff retry with jitter."""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter."""
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        # Add jitter (±25%) to prevent thundering herd
        jitter = delay * 0.25 * (2 * random.random() - 1)
        return max(0, delay + jitter)
    
    def should_retry(self, error: Exception, attempt: int) -> bool:
        """Determine if the error is retryable."""
        if attempt >= self.max_retries:
            return False
        # Retry on rate limits, timeouts, and server errors
        retryable_errors = (RateLimitError, APITimeoutError)
        if isinstance(error, retryable_errors):
            return True
        if isinstance(error, APIError) and error.status_code >= 500:
            return True
        return False


# =============================================================================
# Main LLM Client
# =============================================================================

class LLMClient:
    """Unified LLM client with fallback chain and usage tracking.
    
    Connects to Databricks AI Gateway or any OpenAI-compatible endpoint.
    Automatically handles model fallback, retries, and token tracking.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base_url: Optional[str] = None,
        primary_model: Optional[str] = None,
        fallback_models: Optional[list[str]] = None,
    ):
        """Initialize the LLM client.
        
        Args:
            api_key: API key (defaults to settings.llm.api_key)
            api_base_url: Base URL (defaults to settings.llm.api_base_url)
            primary_model: Primary model name (defaults to settings.llm.primary_model)
            fallback_models: Fallback model list (defaults to settings.llm.fallback_models)
        """
        self._api_key = api_key or settings.llm.api_key
        self._api_base_url = api_base_url or settings.llm.api_base_url
        self._primary_model = primary_model or settings.llm.primary_model
        self._fallback_models = fallback_models or settings.llm.fallback_models
        
        # Build model chain: [primary, fallback_1, fallback_2, ...]
        self._model_chain = [self._primary_model] + self._fallback_models
        
        # Initialize OpenAI client
        self._client = OpenAI(
            api_key=self._api_key,
            base_url=self._api_base_url,
            timeout=settings.llm.request_timeout,
        )
        
        # Retry handler
        self._retry = RetryHandler(
            max_retries=settings.llm.max_retries,
            base_delay=settings.llm.retry_base_delay,
            max_delay=settings.llm.retry_max_delay,
        )
        
        # Usage tracking (thread-safe)
        self._usage_stats = UsageStats()
        self._usage_lock = Lock()
        
        logger.info(
            f"LLMClient initialized | primary={self._primary_model} | "
            f"fallbacks={self._fallback_models}"
        )
    
    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    
    def chat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """Send a chat completion request with automatic fallback.
        
        Args:
            messages: List of message dicts [{"role": ..., "content": ...}]
            tools: Optional tool/function specifications for tool calling
            tool_choice: Tool choice strategy ("auto", "none", or specific)
            temperature: Override default temperature
            max_tokens: Override default max tokens
            model: Force a specific model (skips fallback chain)
            **kwargs: Additional parameters passed to the API
            
        Returns:
            LLMResponse with content, tool_calls, and usage info
            
        Raises:
            LLMClientError: If all models in the chain fail
        """
        # Determine model chain
        models_to_try = [model] if model else self._model_chain
        
        last_error = None
        
        for model_idx, current_model in enumerate(models_to_try):
            if model_idx > 0:
                logger.warning(
                    f"Falling back to model: {current_model} "
                    f"(attempt {model_idx + 1}/{len(models_to_try)})"
                )
                self._track_fallback()
            
            try:
                response = self._call_with_retry(
                    model=current_model,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    temperature=temperature or settings.llm.temperature,
                    max_tokens=max_tokens or settings.llm.max_tokens,
                    **kwargs,
                )
                return response
                
            except Exception as e:
                last_error = e
                logger.error(
                    f"Model {current_model} failed: {type(e).__name__}: {e}"
                )
                continue
        
        # All models failed
        self._track_failure()
        raise LLMClientError(
            f"All models failed. Last error: {last_error}"
        ) from last_error
    
    def chat_stream(
        self,
        messages: list[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> Generator[str, None, None]:
        """Stream a chat completion response token by token.
        
        Args:
            messages: List of message dicts
            temperature: Override default temperature
            max_tokens: Override default max tokens
            model: Force a specific model
            **kwargs: Additional parameters
            
        Yields:
            String chunks as they arrive from the model
        """
        current_model = model or self._primary_model
        
        try:
            stream = self._client.chat.completions.create(
                model=current_model,
                messages=messages,
                temperature=temperature or settings.llm.temperature,
                max_tokens=max_tokens or settings.llm.max_tokens,
                stream=True,
                **kwargs,
            )
            
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Stream failed for {current_model}: {e}")
            raise LLMClientError(f"Stream failed: {e}") from e
    
    def get_usage_summary(self) -> dict:
        """Get aggregated usage statistics."""
        with self._usage_lock:
            return {
                "total_requests": self._usage_stats.total_requests,
                "total_tokens": self._usage_stats.total_tokens,
                "prompt_tokens": self._usage_stats.total_prompt_tokens,
                "completion_tokens": self._usage_stats.total_completion_tokens,
                "avg_latency_ms": (
                    self._usage_stats.total_latency_ms / 
                    max(self._usage_stats.total_requests, 1)
                ),
                "failed_requests": self._usage_stats.failed_requests,
                "fallback_count": self._usage_stats.fallback_count,
                "requests_by_model": dict(self._usage_stats.requests_by_model),
            }
    
    def reset_usage(self) -> None:
        """Reset usage statistics."""
        with self._usage_lock:
            self._usage_stats = UsageStats()
    
    # -------------------------------------------------------------------------
    # Private Methods
    # -------------------------------------------------------------------------
    
    def _call_with_retry(
        self,
        model: str,
        messages: list[dict],
        tools: Optional[list[dict]],
        tool_choice: Optional[str],
        temperature: float,
        max_tokens: int,
        **kwargs,
    ) -> LLMResponse:
        """Execute API call with retry logic."""
        
        for attempt in range(self._retry.max_retries + 1):
            try:
                start_time = time.perf_counter()
                
                # Build request params
                params = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    **kwargs,
                }
                
                if tools:
                    params["tools"] = tools
                    if tool_choice:
                        params["tool_choice"] = tool_choice
                
                # Make the API call
                response = self._client.chat.completions.create(**params)
                
                latency_ms = (time.perf_counter() - start_time) * 1000
                
                # Parse response
                choice = response.choices[0]
                usage = TokenUsage(
                    prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                    completion_tokens=response.usage.completion_tokens if response.usage else 0,
                    total_tokens=response.usage.total_tokens if response.usage else 0,
                    model=model,
                    latency_ms=latency_ms,
                )
                
                # Track usage
                self._track_usage(usage)
                
                return LLMResponse(
                    content=choice.message.content,
                    tool_calls=(
                        [tc.model_dump() for tc in choice.message.tool_calls]
                        if choice.message.tool_calls
                        else None
                    ),
                    usage=usage,
                    model=model,
                    finish_reason=choice.finish_reason or "",
                    raw_response=response,
                )
                
            except Exception as e:
                if self._retry.should_retry(e, attempt):
                    delay = self._retry.calculate_delay(attempt)
                    logger.warning(
                        f"Retry {attempt + 1}/{self._retry.max_retries} "
                        f"for {model} after {delay:.1f}s | {type(e).__name__}: {e}"
                    )
                    time.sleep(delay)
                else:
                    raise
        
        # Should not reach here, but just in case
        raise LLMClientError(f"Max retries exceeded for model {model}")
    
    def _track_usage(self, usage: TokenUsage) -> None:
        """Thread-safe usage tracking."""
        with self._usage_lock:
            self._usage_stats.total_requests += 1
            self._usage_stats.total_prompt_tokens += usage.prompt_tokens
            self._usage_stats.total_completion_tokens += usage.completion_tokens
            self._usage_stats.total_tokens += usage.total_tokens
            self._usage_stats.total_latency_ms += usage.latency_ms
            
            # Track per-model usage
            model_key = usage.model
            if model_key not in self._usage_stats.requests_by_model:
                self._usage_stats.requests_by_model[model_key] = 0
            self._usage_stats.requests_by_model[model_key] += 1
    
    def _track_fallback(self) -> None:
        """Track fallback event."""
        with self._usage_lock:
            self._usage_stats.fallback_count += 1
    
    def _track_failure(self) -> None:
        """Track complete failure (all models exhausted)."""
        with self._usage_lock:
            self._usage_stats.failed_requests += 1


# =============================================================================
# Exceptions
# =============================================================================

class LLMClientError(Exception):
    """Raised when all LLM models fail or an unrecoverable error occurs."""
    pass


# =============================================================================
# Convenience Factory
# =============================================================================

def create_client(**kwargs) -> LLMClient:
    """Factory function to create a configured LLMClient instance.
    
    Args:
        **kwargs: Override any LLMClient constructor parameters
        
    Returns:
        Configured LLMClient instance
    """
    return LLMClient(**kwargs)


if __name__ == "__main__":
    # Quick test / demo
    client = LLMClient()
    print("LLM Client initialized successfully.")
    print(f"Model chain: {client._model_chain}")
    print(f"Usage: {client.get_usage_summary()}")
