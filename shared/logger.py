"""Enterprise Logging Framework with Dual-Write (JSONL + SQLite).

Provides the UsageLogger class that automatically logs all agent interactions
to both human-readable JSONL files (for grep/tail) and a structured SQLite
database (for analytics queries).

Features:
- Dual-write: Every event goes to JSONL + SQLite simultaneously
- Session/conversation tracking with unique IDs
- Automatic integration with LLMClient responses
- Thread-safe operations
- Configurable log rotation
- Context manager for session lifecycle

Usage:
    from shared.logger import UsageLogger
    
    # Create logger for a specific agent
    logger = UsageLogger(agent_name="01_ANSA_ApiAgent")
    
    # Start a session (context manager auto-closes)
    with logger.session() as session:
        conv_id = logger.start_conversation("Mesh generation help")
        
        # Log an LLM interaction (call after client.chat())
        logger.log_llm_request(
            conversation_id=conv_id,
            model="claude-sonnet-4-20250514",
            user_message="Generate mesh script",
            response=llm_response,  # LLMResponse object from client
        )
"""

import json
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any
from contextlib import contextmanager
from threading import Lock

from shared.config import settings
from shared.log_db import LogDatabase


# Standard Python logger for this module
_logger = logging.getLogger(__name__)


# =============================================================================
# JSONL Writer
# =============================================================================

class JSONLWriter:
    """Thread-safe JSONL file writer with rotation support."""
    
    def __init__(self, log_dir: Path, agent_name: str):
        self.log_dir = log_dir
        self.agent_name = agent_name
        self._lock = Lock()
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_log_path(self) -> Path:
        """Get today's log file path (daily rotation)."""
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.log_dir / f"{self.agent_name}_{date_str}.jsonl"
    
    def write(self, event: dict) -> None:
        """Write a single event to the JSONL file."""
        event["timestamp"] = datetime.now(timezone.utc).isoformat()
        event["agent"] = self.agent_name
        
        with self._lock:
            log_path = self._get_log_path()
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, default=str, ensure_ascii=False) + "\n")


# =============================================================================
# Usage Logger (Main Class)
# =============================================================================

class UsageLogger:
    """Dual-write logger for tracking all agent interactions.
    
    Writes every event to both JSONL (human-readable) and SQLite (queryable).
    Manages session and conversation lifecycle automatically.
    """
    
    def __init__(
        self,
        agent_name: str,
        log_dir: Optional[Path] = None,
        db_path: Optional[Path] = None,
    ):
        """Initialize the usage logger.
        
        Args:
            agent_name: Name of the agent (e.g., "01_ANSA_ApiAgent")
            log_dir: Directory for JSONL files (defaults to settings.paths.log_dir)
            db_path: Path to SQLite DB (defaults to settings.paths.log_db)
        """
        self.agent_name = agent_name
        self._log_dir = log_dir or settings.paths.log_dir
        self._db_path = db_path or settings.paths.log_db
        
        # Initialize writers
        self._jsonl = JSONLWriter(self._log_dir, agent_name)
        self._db = LogDatabase(self._db_path)
        self._db.initialize()
        
        # Session state
        self._session_id: Optional[str] = None
        self._current_conversation_id: Optional[str] = None
        
        _logger.info(f"UsageLogger initialized for '{agent_name}'")
    
    # -------------------------------------------------------------------------
    # Session Management
    # -------------------------------------------------------------------------
    
    @contextmanager
    def session(self, user_id: str = "anonymous", metadata: Optional[dict] = None):
        """Context manager for session lifecycle.
        
        Usage:
            with logger.session(user_id="engineer_01") as session_id:
                # ... all logging within this session
        """
        self._session_id = self._db.create_session(
            agent_name=self.agent_name,
            user_id=user_id,
            metadata=json.dumps(metadata) if metadata else None,
        )
        
        self._jsonl.write({
            "event": "session_start",
            "session_id": self._session_id,
            "user_id": user_id,
        })
        
        try:
            yield self._session_id
        finally:
            self._db.end_session(self._session_id)
            self._jsonl.write({
                "event": "session_end",
                "session_id": self._session_id,
            })
            self._session_id = None
            self._current_conversation_id = None
    
    def start_session(self, user_id: str = "anonymous", metadata: Optional[dict] = None) -> str:
        """Manually start a session (use context manager when possible)."""
        self._session_id = self._db.create_session(
            agent_name=self.agent_name,
            user_id=user_id,
            metadata=json.dumps(metadata) if metadata else None,
        )
        self._jsonl.write({
            "event": "session_start",
            "session_id": self._session_id,
            "user_id": user_id,
        })
        return self._session_id
    
    def end_session(self) -> None:
        """Manually end the current session."""
        if self._session_id:
            self._db.end_session(self._session_id)
            self._jsonl.write({
                "event": "session_end",
                "session_id": self._session_id,
            })
            self._session_id = None
    
    # -------------------------------------------------------------------------
    # Conversation Management
    # -------------------------------------------------------------------------
    
    def start_conversation(self, title: Optional[str] = None) -> str:
        """Start a new conversation within the current session."""
        if not self._session_id:
            raise LoggerError("No active session. Call start_session() first.")
        
        self._current_conversation_id = self._db.create_conversation(
            session_id=self._session_id,
            title=title,
        )
        
        self._jsonl.write({
            "event": "conversation_start",
            "session_id": self._session_id,
            "conversation_id": self._current_conversation_id,
            "title": title,
        })
        
        return self._current_conversation_id
    
    # -------------------------------------------------------------------------
    # LLM Request Logging
    # -------------------------------------------------------------------------
    
    def log_llm_request(
        self,
        model: str,
        user_message: str,
        assistant_response: Optional[str] = None,
        conversation_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        finish_reason: Optional[str] = None,
        latency_ms: float = 0.0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        is_fallback: bool = False,
        attempt_number: int = 1,
        response: Optional[Any] = None,  # LLMResponse object
    ) -> str:
        """Log a complete LLM request/response pair.
        
        Can accept individual parameters OR an LLMResponse object from the client.
        
        Args:
            model: Model name used
            user_message: User's input message
            assistant_response: Model's response text
            conversation_id: Override current conversation
            system_prompt: System prompt (truncated in JSONL)
            finish_reason: Stop reason
            latency_ms: Request latency
            prompt_tokens: Input tokens
            completion_tokens: Output tokens
            total_tokens: Total tokens
            is_fallback: Whether this used a fallback model
            attempt_number: Retry attempt number
            response: LLMResponse object (auto-extracts fields)
            
        Returns:
            request_id for correlation with tool calls
        """
        # Auto-extract from LLMResponse if provided
        if response is not None:
            model = response.model or model
            assistant_response = assistant_response or response.content
            finish_reason = finish_reason or response.finish_reason
            latency_ms = latency_ms or response.usage.latency_ms
            prompt_tokens = prompt_tokens or response.usage.prompt_tokens
            completion_tokens = completion_tokens or response.usage.completion_tokens
            total_tokens = total_tokens or response.usage.total_tokens
        
        conv_id = conversation_id or self._current_conversation_id
        if not conv_id:
            # Auto-create a conversation if none exists
            conv_id = self.start_conversation("Auto-created")
        
        # Write to SQLite
        request_id = self._db.log_request(
            conversation_id=conv_id,
            session_id=self._session_id or "no-session",
            agent_name=self.agent_name,
            model=model,
            user_message=user_message,
            assistant_response=assistant_response,
            system_prompt=system_prompt,
            finish_reason=finish_reason,
            latency_ms=latency_ms,
            is_fallback=is_fallback,
            attempt_number=attempt_number,
        )
        
        # Log token usage
        if total_tokens > 0:
            self._db.log_token_usage(
                request_id=request_id,
                session_id=self._session_id or "no-session",
                agent_name=self.agent_name,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
        
        # Write to JSONL
        self._jsonl.write({
            "event": "llm_request",
            "request_id": request_id,
            "session_id": self._session_id,
            "conversation_id": conv_id,
            "model": model,
            "user_message": user_message[:500],  # Truncate for readability
            "response_preview": (assistant_response or "")[:200],
            "finish_reason": finish_reason,
            "latency_ms": round(latency_ms, 1),
            "tokens": {
                "prompt": prompt_tokens,
                "completion": completion_tokens,
                "total": total_tokens,
            },
            "is_fallback": is_fallback,
        })
        
        return request_id
    
    # -------------------------------------------------------------------------
    # Tool Call Logging
    # -------------------------------------------------------------------------
    
    def log_tool_call(
        self,
        request_id: str,
        tool_name: str,
        tool_arguments: Optional[dict] = None,
        tool_result: Optional[Any] = None,
        execution_ms: float = 0.0,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> None:
        """Log a tool/function call made by an agent."""
        args_str = json.dumps(tool_arguments, default=str) if tool_arguments else None
        result_str = str(tool_result)[:2000] if tool_result else None  # Truncate
        
        # Write to SQLite
        self._db.log_tool_call(
            request_id=request_id,
            session_id=self._session_id or "no-session",
            agent_name=self.agent_name,
            tool_name=tool_name,
            tool_arguments=args_str,
            tool_result=result_str,
            execution_ms=execution_ms,
            success=success,
            error_message=error_message,
        )
        
        # Write to JSONL
        self._jsonl.write({
            "event": "tool_call",
            "request_id": request_id,
            "session_id": self._session_id,
            "tool_name": tool_name,
            "arguments": tool_arguments,
            "execution_ms": round(execution_ms, 1),
            "success": success,
            "error": error_message,
        })
    
    # -------------------------------------------------------------------------
    # Error Logging
    # -------------------------------------------------------------------------
    
    def log_error(
        self,
        error: Exception,
        request_id: Optional[str] = None,
        model: Optional[str] = None,
        retry_attempt: Optional[int] = None,
        is_fatal: bool = False,
    ) -> None:
        """Log an error with full context."""
        error_type = type(error).__name__
        error_msg = str(error)
        stack = traceback.format_exc()
        
        # Write to SQLite
        self._db.log_error(
            agent_name=self.agent_name,
            error_type=error_type,
            error_message=error_msg,
            stack_trace=stack,
            session_id=self._session_id,
            conversation_id=self._current_conversation_id,
            request_id=request_id,
            model=model,
            retry_attempt=retry_attempt,
            is_fatal=is_fatal,
        )
        
        # Write to JSONL
        self._jsonl.write({
            "event": "error",
            "session_id": self._session_id,
            "request_id": request_id,
            "error_type": error_type,
            "error_message": error_msg,
            "model": model,
            "is_fatal": is_fatal,
            "stack_trace": stack[:1000],  # Truncate for JSONL
        })
        
        # Also log via standard Python logging
        log_level = logging.CRITICAL if is_fatal else logging.ERROR
        _logger.log(log_level, f"[{self.agent_name}] {error_type}: {error_msg}")
    
    # -------------------------------------------------------------------------
    # Analytics Helpers
    # -------------------------------------------------------------------------
    
    def get_usage_summary(self, since: Optional[str] = None) -> dict:
        """Get usage summary for this agent."""
        return self._db.get_usage_summary(
            agent_name=self.agent_name,
            since=since,
        )
    
    def get_recent_errors(self, limit: int = 10) -> list[dict]:
        """Get recent errors for this agent."""
        return self._db.get_recent_errors(
            limit=limit,
            agent_name=self.agent_name,
        )


# =============================================================================
# Exceptions
# =============================================================================

class LoggerError(Exception):
    """Raised on logging configuration or state errors."""
    pass


# =============================================================================
# Convenience: Pre-configured loggers per agent
# =============================================================================

def create_logger(agent_name: str, **kwargs) -> UsageLogger:
    """Factory function to create a configured UsageLogger."""
    return UsageLogger(agent_name=agent_name, **kwargs)


if __name__ == "__main__":
    import tempfile
    
    # Quick demo with temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        logger = UsageLogger(
            agent_name="demo_agent",
            log_dir=tmp_path / "logs",
            db_path=tmp_path / "usage.db",
        )
        
        with logger.session(user_id="engineer_01") as session_id:
            conv_id = logger.start_conversation("Demo conversation")
            
            req_id = logger.log_llm_request(
                model="claude-sonnet-4-20250514",
                user_message="Generate ANSA mesh script",
                assistant_response="import ansa\n...",
                latency_ms=1250.0,
                prompt_tokens=150,
                completion_tokens=80,
                total_tokens=230,
            )
            
            logger.log_tool_call(
                request_id=req_id,
                tool_name="search_api",
                tool_arguments={"query": "mesh generation"},
                tool_result="Found 5 results",
                execution_ms=45.0,
            )
        
        print("\n=== Demo Complete ===")
        print(f"Session: {session_id}")
        print(f"Usage: {logger.get_usage_summary()}")
        
        # Show JSONL output
        log_files = list((tmp_path / "logs").glob("*.jsonl"))
        if log_files:
            print(f"\n=== JSONL Output ({log_files[0].name}) ===")
            print(log_files[0].read_text())
