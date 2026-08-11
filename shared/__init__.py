"""SafetyAgent Shared Module.

Provides common utilities used across all agents:
- config: Central configuration management
- llm_client: Unified LLM client with model fallback (AI Gateway)
- logger: Enterprise logging framework (JSONL + SQLite) [Day 3]

Usage:
    from shared import settings, LLMClient, create_client
    
    # Access global settings
    model = settings.llm.primary_model
    
    # Create an LLM client
    client = LLMClient()
    response = client.chat([{"role": "user", "content": "Hello"}])
"""

__version__ = "0.1.0"

from shared.config import settings, Settings
from shared.llm_client import LLMClient, LLMResponse, LLMClientError, create_client
from shared.logger import UsageLogger, create_logger, LoggerError

__all__ = [
    # Configuration
    "settings",
    "Settings",
    # LLM Client
    "LLMClient",
    "LLMResponse",
    "LLMClientError",
    "create_client",
    # Logging
    "UsageLogger",
    "create_logger",
    "LoggerError",
]
