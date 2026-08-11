"""SQLite Database Schema and Manager for SafetyAgent Logging.

Defines the database schema (6 tables) for tracking all agent interactions,
token usage, errors, and tool calls. Supports schema migrations for
forward-compatible updates.

Tables:
    sessions        - User sessions (one per app launch / connection)
    conversations   - Conversations within a session (one per chat thread)
    requests        - Individual LLM requests with full metadata
    token_usage     - Token consumption tracking (aggregatable)
    tool_calls      - Tool/function calls made by agents
    errors          - Error events with stack traces

Usage:
    from shared.log_db import LogDatabase
    
    db = LogDatabase("logs/usage.db")
    db.initialize()
    session_id = db.create_session(agent="01_ANSA_ApiAgent")
"""

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from contextlib import contextmanager


# =============================================================================
# Schema Version (increment on breaking changes)
# =============================================================================

SCHEMA_VERSION = 1

# =============================================================================
# SQL Schema Definitions
# =============================================================================

CREATE_TABLES_SQL = """
-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_info (
    version         INTEGER NOT NULL,
    applied_at      TEXT NOT NULL DEFAULT (datetime('now')),
    description     TEXT
);

-- User sessions (one per app launch / connection)
CREATE TABLE IF NOT EXISTS sessions (
    session_id      TEXT PRIMARY KEY,
    agent_name      TEXT NOT NULL,
    user_id         TEXT DEFAULT 'anonymous',
    started_at      TEXT NOT NULL DEFAULT (datetime('now')),
    ended_at        TEXT,
    metadata        TEXT  -- JSON blob for extra context
);

-- Conversations within a session
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL,
    title           TEXT,
    started_at      TEXT NOT NULL DEFAULT (datetime('now')),
    ended_at        TEXT,
    message_count   INTEGER DEFAULT 0,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

-- Individual LLM requests
CREATE TABLE IF NOT EXISTS requests (
    request_id      TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    session_id      TEXT NOT NULL,
    agent_name      TEXT NOT NULL,
    model           TEXT NOT NULL,
    
    -- Request details
    system_prompt   TEXT,
    user_message    TEXT,
    assistant_response TEXT,
    finish_reason   TEXT,
    
    -- Performance
    latency_ms      REAL,
    is_fallback     INTEGER DEFAULT 0,
    attempt_number  INTEGER DEFAULT 1,
    
    -- Timestamps
    requested_at    TEXT NOT NULL DEFAULT (datetime('now')),
    
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

-- Token usage tracking
CREATE TABLE IF NOT EXISTS token_usage (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id      TEXT NOT NULL,
    session_id      TEXT NOT NULL,
    agent_name      TEXT NOT NULL,
    model           TEXT NOT NULL,
    
    prompt_tokens   INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    total_tokens    INTEGER DEFAULT 0,
    
    -- Cost estimation (configurable rates)
    estimated_cost  REAL DEFAULT 0.0,
    
    recorded_at     TEXT NOT NULL DEFAULT (datetime('now')),
    
    FOREIGN KEY (request_id) REFERENCES requests(request_id),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

-- Tool/function calls made by agents
CREATE TABLE IF NOT EXISTS tool_calls (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id      TEXT NOT NULL,
    session_id      TEXT NOT NULL,
    agent_name      TEXT NOT NULL,
    
    tool_name       TEXT NOT NULL,
    tool_arguments  TEXT,   -- JSON string
    tool_result     TEXT,   -- JSON string (truncated if large)
    
    -- Performance
    execution_ms    REAL,
    success         INTEGER DEFAULT 1,
    error_message   TEXT,
    
    called_at       TEXT NOT NULL DEFAULT (datetime('now')),
    
    FOREIGN KEY (request_id) REFERENCES requests(request_id),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

-- Error events with context
CREATE TABLE IF NOT EXISTS errors (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT,
    conversation_id TEXT,
    request_id      TEXT,
    agent_name      TEXT NOT NULL,
    
    error_type      TEXT NOT NULL,   -- Exception class name
    error_message   TEXT NOT NULL,
    stack_trace     TEXT,
    
    -- Context
    model           TEXT,
    retry_attempt   INTEGER,
    is_fatal        INTEGER DEFAULT 0,
    
    occurred_at     TEXT NOT NULL DEFAULT (datetime('now')),
    
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);
"""

# Indexes for common queries
CREATE_INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_requests_session ON requests(session_id);
CREATE INDEX IF NOT EXISTS idx_requests_conversation ON requests(conversation_id);
CREATE INDEX IF NOT EXISTS idx_requests_agent ON requests(agent_name);
CREATE INDEX IF NOT EXISTS idx_requests_model ON requests(model);
CREATE INDEX IF NOT EXISTS idx_requests_timestamp ON requests(requested_at);
CREATE INDEX IF NOT EXISTS idx_token_usage_session ON token_usage(session_id);
CREATE INDEX IF NOT EXISTS idx_token_usage_agent ON token_usage(agent_name);
CREATE INDEX IF NOT EXISTS idx_token_usage_timestamp ON token_usage(recorded_at);
CREATE INDEX IF NOT EXISTS idx_tool_calls_agent ON tool_calls(agent_name);
CREATE INDEX IF NOT EXISTS idx_tool_calls_name ON tool_calls(tool_name);
CREATE INDEX IF NOT EXISTS idx_errors_agent ON errors(agent_name);
CREATE INDEX IF NOT EXISTS idx_errors_timestamp ON errors(occurred_at);
"""

# =============================================================================
# Schema Migrations
# =============================================================================

MIGRATIONS = {
    # version: (description, sql_statements)
    # Example for future migrations:
    # 2: ("Add cost_center column to sessions", """
    #     ALTER TABLE sessions ADD COLUMN cost_center TEXT;
    # """),
}


# =============================================================================
# Database Manager
# =============================================================================

class LogDatabase:
    """SQLite database manager for usage logging.
    
    Thread-safe via connection-per-call pattern with WAL mode.
    """
    
    def __init__(self, db_path: str | Path):
        """Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file (created if not exists)
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent access
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def initialize(self) -> None:
        """Create tables and apply pending migrations."""
        with self._get_connection() as conn:
            conn.executescript(CREATE_TABLES_SQL)
            conn.executescript(CREATE_INDEXES_SQL)
            
            # Record schema version
            cursor = conn.execute(
                "SELECT MAX(version) FROM schema_info"
            )
            current_version = cursor.fetchone()[0] or 0
            
            if current_version == 0:
                conn.execute(
                    "INSERT INTO schema_info (version, description) VALUES (?, ?)",
                    (SCHEMA_VERSION, "Initial schema")
                )
            
            # Apply pending migrations
            self._apply_migrations(conn, current_version)
    
    def _apply_migrations(self, conn: sqlite3.Connection, current_version: int) -> None:
        """Apply any pending schema migrations."""
        for version in sorted(MIGRATIONS.keys()):
            if version > current_version:
                description, sql = MIGRATIONS[version]
                conn.executescript(sql)
                conn.execute(
                    "INSERT INTO schema_info (version, description) VALUES (?, ?)",
                    (version, description)
                )
    
    # -------------------------------------------------------------------------
    # Session Operations
    # -------------------------------------------------------------------------
    
    def create_session(
        self,
        agent_name: str,
        user_id: str = "anonymous",
        metadata: Optional[str] = None,
    ) -> str:
        """Create a new session and return its ID."""
        session_id = str(uuid.uuid4())
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO sessions (session_id, agent_name, user_id, metadata) "
                "VALUES (?, ?, ?, ?)",
                (session_id, agent_name, user_id, metadata)
            )
        return session_id
    
    def end_session(self, session_id: str) -> None:
        """Mark a session as ended."""
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE sessions SET ended_at = datetime('now') WHERE session_id = ?",
                (session_id,)
            )
    
    # -------------------------------------------------------------------------
    # Conversation Operations
    # -------------------------------------------------------------------------
    
    def create_conversation(
        self,
        session_id: str,
        title: Optional[str] = None,
    ) -> str:
        """Create a new conversation within a session."""
        conversation_id = str(uuid.uuid4())
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO conversations (conversation_id, session_id, title) "
                "VALUES (?, ?, ?)",
                (conversation_id, session_id, title)
            )
        return conversation_id
    
    # -------------------------------------------------------------------------
    # Request Logging
    # -------------------------------------------------------------------------
    
    def log_request(
        self,
        conversation_id: str,
        session_id: str,
        agent_name: str,
        model: str,
        user_message: str,
        assistant_response: Optional[str] = None,
        system_prompt: Optional[str] = None,
        finish_reason: Optional[str] = None,
        latency_ms: float = 0.0,
        is_fallback: bool = False,
        attempt_number: int = 1,
    ) -> str:
        """Log an LLM request and return its ID."""
        request_id = str(uuid.uuid4())
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO requests "
                "(request_id, conversation_id, session_id, agent_name, model, "
                "system_prompt, user_message, assistant_response, finish_reason, "
                "latency_ms, is_fallback, attempt_number) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (request_id, conversation_id, session_id, agent_name, model,
                 system_prompt, user_message, assistant_response, finish_reason,
                 latency_ms, int(is_fallback), attempt_number)
            )
            # Update conversation message count
            conn.execute(
                "UPDATE conversations SET message_count = message_count + 1 "
                "WHERE conversation_id = ?",
                (conversation_id,)
            )
        return request_id
    
    # -------------------------------------------------------------------------
    # Token Usage
    # -------------------------------------------------------------------------
    
    def log_token_usage(
        self,
        request_id: str,
        session_id: str,
        agent_name: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        estimated_cost: float = 0.0,
    ) -> None:
        """Log token usage for a request."""
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO token_usage "
                "(request_id, session_id, agent_name, model, "
                "prompt_tokens, completion_tokens, total_tokens, estimated_cost) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (request_id, session_id, agent_name, model,
                 prompt_tokens, completion_tokens, total_tokens, estimated_cost)
            )
    
    # -------------------------------------------------------------------------
    # Tool Calls
    # -------------------------------------------------------------------------
    
    def log_tool_call(
        self,
        request_id: str,
        session_id: str,
        agent_name: str,
        tool_name: str,
        tool_arguments: Optional[str] = None,
        tool_result: Optional[str] = None,
        execution_ms: float = 0.0,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> None:
        """Log a tool/function call."""
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO tool_calls "
                "(request_id, session_id, agent_name, tool_name, "
                "tool_arguments, tool_result, execution_ms, success, error_message) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (request_id, session_id, agent_name, tool_name,
                 tool_arguments, tool_result, execution_ms, int(success), error_message)
            )
    
    # -------------------------------------------------------------------------
    # Error Logging
    # -------------------------------------------------------------------------
    
    def log_error(
        self,
        agent_name: str,
        error_type: str,
        error_message: str,
        stack_trace: Optional[str] = None,
        session_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        request_id: Optional[str] = None,
        model: Optional[str] = None,
        retry_attempt: Optional[int] = None,
        is_fatal: bool = False,
    ) -> None:
        """Log an error event."""
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO errors "
                "(session_id, conversation_id, request_id, agent_name, "
                "error_type, error_message, stack_trace, model, "
                "retry_attempt, is_fatal) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (session_id, conversation_id, request_id, agent_name,
                 error_type, error_message, stack_trace, model,
                 retry_attempt, int(is_fatal))
            )
    
    # -------------------------------------------------------------------------
    # Query Helpers (for admin/analytics)
    # -------------------------------------------------------------------------
    
    def get_usage_summary(
        self,
        agent_name: Optional[str] = None,
        since: Optional[str] = None,
    ) -> dict:
        """Get aggregated usage statistics."""
        query = """
            SELECT 
                COUNT(*) as total_requests,
                SUM(prompt_tokens) as total_prompt_tokens,
                SUM(completion_tokens) as total_completion_tokens,
                SUM(total_tokens) as total_tokens,
                SUM(estimated_cost) as total_cost
            FROM token_usage
            WHERE 1=1
        """
        params = []
        
        if agent_name:
            query += " AND agent_name = ?"
            params.append(agent_name)
        if since:
            query += " AND recorded_at >= ?"
            params.append(since)
        
        with self._get_connection() as conn:
            row = conn.execute(query, params).fetchone()
            return dict(row) if row else {}
    
    def get_recent_errors(
        self,
        limit: int = 20,
        agent_name: Optional[str] = None,
    ) -> list[dict]:
        """Get recent error events."""
        query = "SELECT * FROM errors"
        params = []
        
        if agent_name:
            query += " WHERE agent_name = ?"
            params.append(agent_name)
        
        query += " ORDER BY occurred_at DESC LIMIT ?"
        params.append(limit)
        
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]
    
    def get_model_distribution(
        self,
        since: Optional[str] = None,
    ) -> list[dict]:
        """Get request distribution by model."""
        query = """
            SELECT model, COUNT(*) as request_count, 
                   SUM(total_tokens) as tokens
            FROM token_usage
        """
        params = []
        
        if since:
            query += " WHERE recorded_at >= ?"
            params.append(since)
        
        query += " GROUP BY model ORDER BY request_count DESC"
        
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]


if __name__ == "__main__":
    # Quick test
    db = LogDatabase(":memory:")
    db.initialize()
    
    session_id = db.create_session(agent_name="test_agent")
    conv_id = db.create_conversation(session_id, title="Test conversation")
    req_id = db.log_request(
        conversation_id=conv_id,
        session_id=session_id,
        agent_name="test_agent",
        model="claude-sonnet-4-20250514",
        user_message="Hello world",
        assistant_response="Hi there!",
        latency_ms=450.0,
    )
    db.log_token_usage(
        request_id=req_id,
        session_id=session_id,
        agent_name="test_agent",
        model="claude-sonnet-4-20250514",
        prompt_tokens=15,
        completion_tokens=8,
        total_tokens=23,
    )
    
    print("Database initialized successfully.")
    print(f"Session: {session_id}")
    print(f"Usage: {db.get_usage_summary()}")
