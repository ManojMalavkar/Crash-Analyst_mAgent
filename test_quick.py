"""Quick Integration Test - LLM Client + Logging Framework.

Run from project root:
    python test_quick.py

Tests:
1. Config loading & validation
2. LLM Client initialization (no real API call)
3. Logger dual-write (JSONL + SQLite)
4. Full agent session simulation

Requires: pip install openai python-dotenv
(No actual LLM endpoint needed - tests initialization + logging only)
"""

import sys
import json
import tempfile
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))


def test_config():
    """Test 1: Configuration Loading."""
    print("\n" + "=" * 60)
    print("  TEST 1: Configuration Loading")
    print("=" * 60)
    
    from shared.config import settings
    
    # Verify all config sections exist
    assert settings.llm is not None, "LLM config missing"
    assert settings.embedding is not None, "Embedding config missing"
    assert settings.paths is not None, "Paths config missing"
    assert settings.agent is not None, "Agent config missing"
    assert settings.ui is not None, "UI config missing"
    
    # Print status
    settings.print_status()
    
    # Verify defaults
    assert settings.llm.max_retries == 3
    assert settings.llm.temperature == 0.1
    assert settings.agent.retrieval_top_k == 5
    assert settings.embedding.model_name == "BAAI/bge-small-en-v1.5"
    
    print("\n  \u2705 Config loading: PASSED")


def test_llm_client():
    """Test 2: LLM Client Initialization."""
    print("\n" + "=" * 60)
    print("  TEST 2: LLM Client Initialization")
    print("=" * 60)
    
    from shared.llm_client import LLMClient, LLMClientError, LLMResponse
    
    # Client should initialize without error
    client = LLMClient()
    
    print(f"  Primary model : {client._primary_model}")
    print(f"  Fallback chain: {client._model_chain}")
    print(f"  Retry config  : max={client._retry.max_retries}, "
          f"base_delay={client._retry.base_delay}s")
    
    # Test retry delay calculation
    delays = [client._retry.calculate_delay(i) for i in range(4)]
    print(f"  Retry delays  : {[f'{d:.1f}s' for d in delays]}")
    
    # Test usage stats (should be empty)
    usage = client.get_usage_summary()
    assert usage["total_requests"] == 0
    assert usage["total_tokens"] == 0
    print(f"  Initial usage : {usage['total_requests']} requests, {usage['total_tokens']} tokens")
    
    # Test reset
    client.reset_usage()
    assert client.get_usage_summary()["total_requests"] == 0
    
    # Test LLMResponse dataclass
    resp = LLMResponse(content="Hello", tool_calls=None)
    assert resp.has_tool_calls is False
    resp2 = LLMResponse(content=None, tool_calls=[{"function": {"name": "test"}}])
    assert resp2.has_tool_calls is True
    
    print("\n  \u2705 LLM Client init: PASSED")
    print("  \u2139\ufe0f  (No API call made - set LLM_API_KEY in .env to test live)")


def test_logging():
    """Test 3: Logging Framework (Dual-Write)."""
    print("\n" + "=" * 60)
    print("  TEST 3: Logging Framework (Dual-Write)")
    print("=" * 60)
    
    from shared.logger import UsageLogger, LoggerError
    from shared.log_db import LogDatabase, SCHEMA_VERSION
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Initialize logger
        logger = UsageLogger(
            agent_name="01_ANSA_ApiAgent",
            log_dir=tmp_path / "logs",
            db_path=tmp_path / "usage.db",
        )
        print("  \u2705 Logger created")
        
        # === Full Session Simulation ===
        with logger.session(user_id="engineer_01") as session_id:
            print(f"  \u2705 Session: {session_id[:8]}...")
            
            conv_id = logger.start_conversation("ANSA mesh generation help")
            print(f"  \u2705 Conversation: {conv_id[:8]}...")
            
            # Simulate 3 LLM interactions
            request_ids = []
            for i, (msg, resp, pt, ct) in enumerate([
                ("Generate mesh script for hood", "import ansa\nfrom ansa import base, mesh...", 150, 80),
                ("Change element size to 3mm", "# Updated mesh parameters\nelem_size = 3.0...", 120, 60),
                ("Add jacobian quality check", "# Quality criteria\nbase.SetQualityCriteria...", 130, 70),
            ], 1):
                req_id = logger.log_llm_request(
                    model="claude-sonnet-4-20250514" if i < 3 else "llama-3-70b-instruct",
                    user_message=msg,
                    assistant_response=resp,
                    latency_ms=1000.0 + i * 200,
                    prompt_tokens=pt,
                    completion_tokens=ct,
                    total_tokens=pt + ct,
                    is_fallback=(i == 3),
                )
                request_ids.append(req_id)
                
                # Tool call for each request
                logger.log_tool_call(
                    request_id=req_id,
                    tool_name="search_ansa_api",
                    tool_arguments={"query": msg, "top_k": 5},
                    tool_result=f"Found {3+i} relevant API functions",
                    execution_ms=30.0 + i * 15,
                    success=True,
                )
            
            # Simulate an error
            try:
                raise TimeoutError("Model endpoint timed out after 120s")
            except Exception as e:
                logger.log_error(
                    error=e,
                    model="llama-3-70b-instruct",
                    retry_attempt=3,
                    is_fatal=False,
                )
            
            print(f"  \u2705 Logged: 3 requests, 3 tool calls, 1 error")
        
        print(f"  \u2705 Session ended cleanly")
        
        # === Verify SQLite ===
        print("\n  --- SQLite Results ---")
        summary = logger.get_usage_summary()
        print(f"  Total requests     : {summary['total_requests']}")
        print(f"  Total tokens       : {summary['total_tokens']}")
        print(f"  Prompt tokens      : {summary['total_prompt_tokens']}")
        print(f"  Completion tokens  : {summary['total_completion_tokens']}")
        
        assert summary["total_requests"] == 3, f"Expected 3 requests, got {summary['total_requests']}"
        assert summary["total_tokens"] == 610, f"Expected 610 tokens, got {summary['total_tokens']}"
        
        errors = logger.get_recent_errors()
        assert len(errors) == 1
        assert errors[0]["error_type"] == "TimeoutError"
        print(f"  Errors logged      : {len(errors)} ({errors[0]['error_type']})")
        
        # === Verify JSONL ===
        print("\n  --- JSONL Output ---")
        log_files = list((tmp_path / "logs").glob("*.jsonl"))
        assert len(log_files) == 1, "Expected 1 log file"
        
        lines = log_files[0].read_text().strip().split("\n")
        events = [json.loads(line) for line in lines]
        
        event_types = [e["event"] for e in events]
        print(f"  Log file           : {log_files[0].name}")
        print(f"  Total events       : {len(events)}")
        print(f"  Event types        : {set(event_types)}")
        
        # Verify all event types present
        assert "session_start" in event_types
        assert "conversation_start" in event_types
        assert "llm_request" in event_types
        assert "tool_call" in event_types
        assert "error" in event_types
        assert "session_end" in event_types
        
        # Show sample events
        print("\n  --- Sample JSONL Events ---")
        for event in events[:3]:
            compact = {k: v for k, v in event.items() if v and k != "timestamp"}
            print(f"  {json.dumps(compact, default=str)[:100]}...")
        
        print("\n  \u2705 Logging dual-write: PASSED")
        print("  \u2705 JSONL + SQLite consistency: PASSED")


def test_conversation_auto_create():
    """Test 4: Auto-create conversation when none exists."""
    print("\n" + "=" * 60)
    print("  TEST 4: Auto-Create Conversation")
    print("=" * 60)
    
    from shared.logger import UsageLogger, LoggerError
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        logger = UsageLogger(
            agent_name="test",
            log_dir=tmp_path / "logs",
            db_path=tmp_path / "test.db",
        )
        
        # Should raise without session
        try:
            logger.start_conversation("Should fail")
            assert False, "Should have raised LoggerError"
        except LoggerError:
            print("  \u2705 Correctly raises error without session")
        
        # With session, logging without conversation should auto-create
        logger.start_session()
        req_id = logger.log_llm_request(
            model="test-model",
            user_message="Hello",
            latency_ms=50.0,
        )
        assert req_id is not None
        assert logger._current_conversation_id is not None
        print("  \u2705 Auto-creates conversation when needed")
        
        logger.end_session()
        print("  \u2705 Manual session start/end works")


# =============================================================
# Main
# =============================================================

if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("  SafetyAgent - Quick Integration Test")
    print("  Testing: Config + LLM Client + Logging")
    print("#" * 60)
    
    try:
        test_config()
        test_llm_client()
        test_logging()
        test_conversation_auto_create()
        
        print("\n" + "=" * 60)
        print("  \U0001f389 ALL 4 TESTS PASSED")
        print("=" * 60)
        print("\n  Next: Set LLM_API_KEY in .env to test live API calls.")
        print("  Run: python test_quick.py")
        print()
        
    except Exception as e:
        print(f"\n  \u274c TEST FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
