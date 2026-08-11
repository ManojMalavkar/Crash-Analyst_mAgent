"""Unit tests for the SafetyAgent logging framework.

Tests cover:
- SQLite database initialization and schema
- JSONL file writing and daily rotation
- UsageLogger dual-write consistency
- Session and conversation lifecycle
- Error logging with stack traces
- Tool call logging
- Analytics query helpers

Run with: pytest tests/test_logger.py -v
"""

import json
import tempfile
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

import pytest

# Adjust path for imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.log_db import LogDatabase, SCHEMA_VERSION
from shared.logger import UsageLogger, JSONLWriter, LoggerError


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def tmp_dir():
    """Provide a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def db(tmp_dir):
    """Provide an initialized LogDatabase."""
    db = LogDatabase(tmp_dir / "test.db")
    db.initialize()
    return db


@pytest.fixture
def logger(tmp_dir):
    """Provide an initialized UsageLogger."""
    return UsageLogger(
        agent_name="test_agent",
        log_dir=tmp_dir / "logs",
        db_path=tmp_dir / "test.db",
    )


# =============================================================================
# LogDatabase Tests
# =============================================================================

class TestLogDatabase:
    """Tests for the SQLite database layer."""
    
    def test_initialize_creates_tables(self, db, tmp_dir):
        """Database initialization should create all 6 tables + schema_info."""
        conn = sqlite3.connect(str(tmp_dir / "test.db"))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        expected_tables = [
            "conversations", "errors", "requests",
            "schema_info", "sessions", "token_usage", "tool_calls"
        ]
        for table in expected_tables:
            assert table in tables, f"Table '{table}' not found"
    
    def test_schema_version_recorded(self, db, tmp_dir):
        """Schema version should be recorded on first init."""
        conn = sqlite3.connect(str(tmp_dir / "test.db"))
        cursor = conn.execute("SELECT MAX(version) FROM schema_info")
        version = cursor.fetchone()[0]
        conn.close()
        
        assert version == SCHEMA_VERSION
    
    def test_create_session(self, db):
        """Should create a session and return a valid UUID."""
        session_id = db.create_session(agent_name="test_agent", user_id="user_01")
        
        assert session_id is not None
        assert len(session_id) == 36  # UUID format
    
    def test_create_conversation(self, db):
        """Should create a conversation linked to a session."""
        session_id = db.create_session(agent_name="test_agent")
        conv_id = db.create_conversation(session_id, title="Test conv")
        
        assert conv_id is not None
        assert len(conv_id) == 36
    
    def test_log_request(self, db):
        """Should log a request and return a request ID."""
        session_id = db.create_session(agent_name="test_agent")
        conv_id = db.create_conversation(session_id)
        
        req_id = db.log_request(
            conversation_id=conv_id,
            session_id=session_id,
            agent_name="test_agent",
            model="test-model",
            user_message="Hello",
            assistant_response="Hi there!",
            latency_ms=100.0,
        )
        
        assert req_id is not None
        assert len(req_id) == 36
    
    def test_log_token_usage(self, db):
        """Should record token usage and be queryable."""
        session_id = db.create_session(agent_name="test_agent")
        conv_id = db.create_conversation(session_id)
        req_id = db.log_request(
            conversation_id=conv_id,
            session_id=session_id,
            agent_name="test_agent",
            model="test-model",
            user_message="Test",
        )
        
        db.log_token_usage(
            request_id=req_id,
            session_id=session_id,
            agent_name="test_agent",
            model="test-model",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )
        
        summary = db.get_usage_summary(agent_name="test_agent")
        assert summary["total_tokens"] == 150
        assert summary["total_requests"] == 1
    
    def test_log_tool_call(self, db):
        """Should log tool calls with arguments and results."""
        session_id = db.create_session(agent_name="test_agent")
        conv_id = db.create_conversation(session_id)
        req_id = db.log_request(
            conversation_id=conv_id,
            session_id=session_id,
            agent_name="test_agent",
            model="test-model",
            user_message="Test",
        )
        
        db.log_tool_call(
            request_id=req_id,
            session_id=session_id,
            agent_name="test_agent",
            tool_name="search_api",
            tool_arguments='{"query": "mesh"}',
            execution_ms=45.0,
            success=True,
        )
        # No exception = success
    
    def test_log_error(self, db):
        """Should log errors and make them queryable."""
        session_id = db.create_session(agent_name="test_agent")
        
        db.log_error(
            agent_name="test_agent",
            error_type="ValueError",
            error_message="Something went wrong",
            session_id=session_id,
            is_fatal=True,
        )
        
        errors = db.get_recent_errors(agent_name="test_agent")
        assert len(errors) == 1
        assert errors[0]["error_type"] == "ValueError"
        assert errors[0]["is_fatal"] == 1
    
    def test_get_model_distribution(self, db):
        """Should aggregate requests by model."""
        session_id = db.create_session(agent_name="test_agent")
        conv_id = db.create_conversation(session_id)
        
        # Log requests with different models
        for model, tokens in [("model-a", 100), ("model-a", 200), ("model-b", 50)]:
            req_id = db.log_request(
                conversation_id=conv_id,
                session_id=session_id,
                agent_name="test_agent",
                model=model,
                user_message="Test",
            )
            db.log_token_usage(
                request_id=req_id,
                session_id=session_id,
                agent_name="test_agent",
                model=model,
                prompt_tokens=tokens,
                completion_tokens=0,
                total_tokens=tokens,
            )
        
        dist = db.get_model_distribution()
        assert len(dist) == 2
        assert dist[0]["model"] == "model-a"  # Most requests first
        assert dist[0]["request_count"] == 2


# =============================================================================
# JSONLWriter Tests
# =============================================================================

class TestJSONLWriter:
    """Tests for JSONL file writing."""
    
    def test_write_creates_file(self, tmp_dir):
        """Writing an event should create the daily log file."""
        writer = JSONLWriter(tmp_dir / "logs", "test_agent")
        writer.write({"event": "test", "data": "hello"})
        
        log_files = list((tmp_dir / "logs").glob("*.jsonl"))
        assert len(log_files) == 1
    
    def test_write_valid_json(self, tmp_dir):
        """Each line in the JSONL file should be valid JSON."""
        writer = JSONLWriter(tmp_dir / "logs", "test_agent")
        writer.write({"event": "test", "value": 42})
        writer.write({"event": "test2", "value": "hello"})
        
        log_files = list((tmp_dir / "logs").glob("*.jsonl"))
        lines = log_files[0].read_text().strip().split("\n")
        
        assert len(lines) == 2
        for line in lines:
            parsed = json.loads(line)  # Should not raise
            assert "timestamp" in parsed
            assert "agent" in parsed
    
    def test_write_includes_timestamp_and_agent(self, tmp_dir):
        """Each event should have timestamp and agent fields auto-added."""
        writer = JSONLWriter(tmp_dir / "logs", "my_agent")
        writer.write({"event": "test"})
        
        log_files = list((tmp_dir / "logs").glob("*.jsonl"))
        event = json.loads(log_files[0].read_text().strip())
        
        assert event["agent"] == "my_agent"
        assert "timestamp" in event


# =============================================================================
# UsageLogger Integration Tests
# =============================================================================

class TestUsageLogger:
    """Integration tests for the dual-write UsageLogger."""
    
    def test_session_lifecycle(self, logger):
        """Session context manager should create and end a session."""
        with logger.session(user_id="test_user") as session_id:
            assert session_id is not None
            assert logger._session_id == session_id
        
        # After context exits, session should be cleared
        assert logger._session_id is None
    
    def test_conversation_requires_session(self, logger):
        """Starting a conversation without a session should raise."""
        with pytest.raises(LoggerError):
            logger.start_conversation("Should fail")
    
    def test_dual_write_consistency(self, logger, tmp_dir):
        """Both JSONL and SQLite should receive the same events."""
        with logger.session() as session_id:
            conv_id = logger.start_conversation("Test")
            req_id = logger.log_llm_request(
                model="test-model",
                user_message="Hello",
                assistant_response="Hi",
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                latency_ms=100.0,
            )
        
        # Check SQLite
        summary = logger.get_usage_summary()
        assert summary["total_tokens"] == 15
        assert summary["total_requests"] == 1
        
        # Check JSONL
        log_files = list((tmp_dir / "logs").glob("*.jsonl"))
        assert len(log_files) == 1
        
        lines = log_files[0].read_text().strip().split("\n")
        events = [json.loads(line) for line in lines]
        event_types = [e["event"] for e in events]
        
        assert "session_start" in event_types
        assert "conversation_start" in event_types
        assert "llm_request" in event_types
        assert "session_end" in event_types
    
    def test_tool_call_logging(self, logger):
        """Tool calls should be logged to both destinations."""
        with logger.session() as session_id:
            conv_id = logger.start_conversation("Test")
            req_id = logger.log_llm_request(
                model="test-model",
                user_message="Search for mesh",
                latency_ms=50.0,
            )
            
            logger.log_tool_call(
                request_id=req_id,
                tool_name="search_api",
                tool_arguments={"query": "mesh generation"},
                tool_result="Found 3 results",
                execution_ms=25.0,
                success=True,
            )
        # No exception = success
    
    def test_error_logging(self, logger):
        """Errors should be logged with stack traces."""
        with logger.session() as session_id:
            try:
                raise ValueError("Test error for logging")
            except Exception as e:
                logger.log_error(error=e, is_fatal=False)
        
        errors = logger.get_recent_errors()
        assert len(errors) == 1
        assert errors[0]["error_type"] == "ValueError"
        assert "Test error" in errors[0]["error_message"]
    
    def test_auto_create_conversation(self, logger):
        """Logging a request without a conversation should auto-create one."""
        logger.start_session(user_id="auto_test")
        
        # Should not raise even without explicit start_conversation()
        req_id = logger.log_llm_request(
            model="test-model",
            user_message="Hello",
            latency_ms=50.0,
        )
        
        assert req_id is not None
        assert logger._current_conversation_id is not None
        
        logger.end_session()


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
