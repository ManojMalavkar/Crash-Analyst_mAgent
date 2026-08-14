"""Admin Query Functions for Usage Analytics.

Provides SQL query wrappers over the SQLite usage database (log_db)
for monitoring agent performance, costs, and error rates.

Used by:
- log_agent.py (NL-to-SQL agent)
- app_log_agent.py (admin dashboard)
- analytics.py (nightly reports)

Usage:
    from bin.admin_tools import get_usage_summary, get_top_queries
    
    summary = get_usage_summary(days=7)
    top = get_top_queries(limit=10)
"""

import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from shared.config import settings


logger = logging.getLogger(__name__)


# =============================================================================
# Database Connection
# =============================================================================

def _get_db_path() -> Path:
    """Get the log database path."""
    return Path(settings.paths.log_db)


def _query(sql: str, params: tuple = ()) -> list[dict]:
    """Execute a SQL query and return results as list of dicts."""
    db_path = _get_db_path()
    if not db_path.exists():
        logger.warning(f"Log database not found: {db_path}")
        return []
    
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]
    except sqlite3.Error as e:
        logger.error(f"SQL error: {e}")
        return []
    finally:
        conn.close()


def run_sql(sql: str) -> list[dict]:
    """Run an arbitrary SQL query against the log database.
    
    Args:
        sql: SQL SELECT query to execute
    
    Returns:
        List of result dicts
    """
    # Safety: only allow SELECT queries
    if not sql.strip().upper().startswith("SELECT"):
        return [{"error": "Only SELECT queries are allowed."}]
    return _query(sql)


# =============================================================================
# Usage Statistics
# =============================================================================

def get_usage_summary(days: int = 7) -> dict:
    """Get overall usage summary for the last N days.
    
    Args:
        days: Number of days to look back
    
    Returns:
        Dict with total_requests, total_tokens, avg_latency, unique_sessions, etc.
    """
    since = (datetime.now() - timedelta(days=days)).isoformat()
    
    rows = _query("""
        SELECT 
            COUNT(*) as total_requests,
            COUNT(DISTINCT session_id) as unique_sessions,
            SUM(prompt_tokens) as total_prompt_tokens,
            SUM(completion_tokens) as total_completion_tokens,
            SUM(prompt_tokens + completion_tokens) as total_tokens,
            AVG(latency_ms) as avg_latency_ms,
            MIN(timestamp) as first_request,
            MAX(timestamp) as last_request
        FROM requests
        WHERE timestamp >= ?
    """, (since,))
    
    if rows:
        return rows[0]
    return {"total_requests": 0}


def get_model_distribution(days: int = 7) -> list[dict]:
    """Get request count by model for the last N days.
    
    Args:
        days: Number of days to look back
    
    Returns:
        List of {model, request_count, total_tokens, avg_latency_ms}
    """
    since = (datetime.now() - timedelta(days=days)).isoformat()
    
    return _query("""
        SELECT 
            model,
            COUNT(*) as request_count,
            SUM(prompt_tokens + completion_tokens) as total_tokens,
            ROUND(AVG(latency_ms), 1) as avg_latency_ms
        FROM requests
        WHERE timestamp >= ?
        GROUP BY model
        ORDER BY request_count DESC
    """, (since,))


def get_hourly_usage(days: int = 1) -> list[dict]:
    """Get hourly request counts for the last N days.
    
    Args:
        days: Number of days to look back
    
    Returns:
        List of {hour, request_count}
    """
    since = (datetime.now() - timedelta(days=days)).isoformat()
    
    return _query("""
        SELECT 
            strftime('%Y-%m-%d %H:00', timestamp) as hour,
            COUNT(*) as request_count
        FROM requests
        WHERE timestamp >= ?
        GROUP BY hour
        ORDER BY hour
    """, (since,))


def get_tool_usage(days: int = 7) -> list[dict]:
    """Get tool call statistics.
    
    Args:
        days: Number of days to look back
    
    Returns:
        List of {tool_name, call_count, avg_execution_ms, success_rate}
    """
    since = (datetime.now() - timedelta(days=days)).isoformat()
    
    return _query("""
        SELECT 
            tool_name,
            COUNT(*) as call_count,
            ROUND(AVG(execution_ms), 1) as avg_execution_ms,
            ROUND(100.0 * SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) / COUNT(*), 1) as success_rate
        FROM tool_calls
        WHERE timestamp >= ?
        GROUP BY tool_name
        ORDER BY call_count DESC
    """, (since,))


def get_recent_errors(limit: int = 10) -> list[dict]:
    """Get the most recent errors.
    
    Args:
        limit: Max number of errors to return
    
    Returns:
        List of {timestamp, model, error_type, error_message, is_fatal}
    """
    return _query("""
        SELECT 
            timestamp,
            model,
            error_type,
            error_message,
            is_fatal
        FROM errors
        ORDER BY timestamp DESC
        LIMIT ?
    """, (limit,))


def get_top_queries(limit: int = 10, days: int = 7) -> list[dict]:
    """Get the most common user queries.
    
    Args:
        limit: Max results
        days: Number of days to look back
    
    Returns:
        List of {user_message, count}
    """
    since = (datetime.now() - timedelta(days=days)).isoformat()
    
    return _query("""
        SELECT 
            user_message,
            COUNT(*) as count
        FROM requests
        WHERE timestamp >= ? AND user_message != '[tool_result]'
        GROUP BY user_message
        ORDER BY count DESC
        LIMIT ?
    """, (since, limit))


def get_session_summary(session_id: str) -> dict:
    """Get summary for a specific session.
    
    Args:
        session_id: Session ID to query
    
    Returns:
        Dict with session details
    """
    rows = _query("""
        SELECT 
            s.session_id,
            s.start_time,
            s.end_time,
            s.user_id,
            COUNT(r.id) as request_count,
            SUM(r.prompt_tokens + r.completion_tokens) as total_tokens
        FROM sessions s
        LEFT JOIN requests r ON r.session_id = s.session_id
        WHERE s.session_id = ?
        GROUP BY s.session_id
    """, (session_id,))
    
    return rows[0] if rows else {"error": "Session not found"}


# =============================================================================
# Schema Info (for NL-to-SQL agent)
# =============================================================================

DB_SCHEMA = """
Tables in the usage log database:

1. sessions (session_id, user_id, start_time, end_time, metadata)
2. conversations (conversation_id, session_id, title, start_time)
3. requests (id, session_id, conversation_id, timestamp, model, user_message,
             assistant_message, prompt_tokens, completion_tokens, latency_ms, finish_reason)
4. token_usage (id, request_id, model, prompt_tokens, completion_tokens, total_tokens, latency_ms)
5. tool_calls (id, request_id, timestamp, tool_name, tool_arguments, tool_result,
              execution_ms, success, error_message)
6. errors (id, request_id, timestamp, model, error_type, error_message,
           retry_attempt, is_fatal)
"""
