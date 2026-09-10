"""PyDyna Agent — LS-DYNA Keyword & Solver Setup Assistant.

AI agent that helps CAE engineers with LS-DYNA model setup:
1. Receives natural language query about LS-DYNA keywords or PyDyna API
2. Selects and calls appropriate retrieval tools (vector search, exact lookup)
3. Synthesises retrieved context into accurate keyword definitions or PyDyna code
4. Validates generated parameters against keyword specifications
5. Supports multi-turn conversation with context management

Architecture follows the same tool-calling loop as 01_ANSA_ApiAgent:
  User message → LLM + tools → tool calls → tool results → LLM → final response

Usage:
    from bin.agent import PyDynaAgent

    agent = PyDynaAgent()
    response = agent.chat("Create MAT_024 for mild steel with yield 250 MPa")
    print(response)
"""

import json
import time
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from shared import LLMClient, create_client, create_logger, settings
from bin.tools import create_tool_registry
from bin.pydyna_tools import ALL_TOOLS


logger = logging.getLogger(__name__)


# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = """You are an expert LS-DYNA and PyDyna assistant. You help CAE safety engineers
set up crash simulation models by providing accurate keyword definitions, material selections,
contact configurations, and PyDyna Python code.

You have access to the following tools to retrieve LS-DYNA keyword documentation:

## Tool Selection Guide

| User Intent | Tool to Use | Example |
|-------------|-------------|----------|
| General keyword search | search_keyword | "elastic material for steel" |
| Know exact keyword name | lookup_keyword | "*MAT_024", "*CONTROL_TIMESTEP" |
| PyDyna Python API help | search_pydyna_api | "how to set timestep in PyDyna" |
| Need a material model | get_material_model | "material for crash steel" |
| Need a contact type | get_contact_type | "contact between bumper and barrier" |
| Verify parameters | validate_keyword | check *MAT_024 params are correct |

## Rules

1. ALWAYS call at least one tool before generating keyword definitions or PyDyna code.
   Never guess at keyword parameters or field names.
2. Use lookup_keyword when you know the exact keyword (most accurate, returns all cards).
3. Use search_keyword for open-ended questions (semantic search).
4. Use get_material_model / get_contact_type for recommendation questions.
5. After generating keyword parameters, use validate_keyword to verify correctness.
6. When generating PyDyna code:
   - Use correct imports: from ansys.dyna.core.keywords import *
   - Match class names exactly as returned by tools (e.g., Mat024, not MAT024)
   - Set all required fields (those without defaults)
   - Include inline comments with field descriptions
7. When generating keyword cards (raw .k format):
   - Use fixed-width fields (8 or 10 characters per field)
   - Include the *KEYWORD header line
   - Add $ comment lines explaining each card
8. If tools return no results, say so honestly. Do NOT hallucinate keyword parameters.
9. Format code in ```python blocks and keyword cards in ```text blocks.

## LS-DYNA Context

- Keywords use *KEYWORD_NAME format (e.g., *MAT_024, *CONTACT_AUTOMATIC_SINGLE_SURFACE)
- Common material models: *MAT_001 (elastic), *MAT_020 (rigid), *MAT_024 (piecewise linear plasticity)
- Common contacts: *CONTACT_AUTOMATIC_SINGLE_SURFACE, *CONTACT_AUTOMATIC_SURFACE_TO_SURFACE
- Control cards: *CONTROL_TIMESTEP, *CONTROL_TERMINATION, *CONTROL_ENERGY
- Output: *DATABASE_BINARY_D3PLOT, *DATABASE_BINARY_D3THDT
- Units: consistent system required (e.g., mm/s/tonne/N/MPa or m/s/kg/N/Pa)

## PyDyna Context

- PyDyna SDK: ansys.dyna.core
- Keyword classes: from ansys.dyna.core.keywords.keyword_classes.auto.mat.mat_024 import Mat024
- Deck management: from ansys.dyna.core.lib.deck import Deck
- Properties map to keyword fields: mat.ro = density, mat.e = Young's modulus
"""


# =============================================================================
# Agent Configuration
# =============================================================================

@dataclass
class AgentConfig:
    """Configuration for the PyDyna agent."""
    max_tool_calls: int = 10
    max_context_messages: int = 20
    temperature: float = 0.1
    max_tokens: int = 4096
    tool_call_timeout: float = 30.0


# =============================================================================
# PyDyna Agent
# =============================================================================

class PyDynaAgent:
    """LS-DYNA Keyword & Solver Setup Agent with RAG tool-calling loop."""

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        llm_client: Optional[LLMClient] = None,
    ):
        self.config = config or AgentConfig()
        self.client = llm_client or create_client()
        self.logger = create_logger("pydyna_agent")

        # Build tool registry from pydyna_tools.ALL_TOOLS
        registry = create_tool_registry(ALL_TOOLS)
        self.tool_specs = registry["specs"]
        self.tool_dispatch = registry["dispatch"]

        # Start logging session (non-fatal)
        try:
            self.logger.start_session(user_id="cli_user")
        except Exception:
            pass

        # Conversation state
        self.messages: list[dict] = []
        self._init_conversation()

    def _init_conversation(self):
        """Initialize conversation with system prompt."""
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def chat(self, user_message: str) -> str:
        """Send a message and get a response via the tool-calling loop.

        1. Send user message + tools to LLM
        2. If LLM requests tool calls, execute them
        3. Send tool results back to LLM
        4. Repeat until LLM produces final response

        Args:
            user_message: The user's question or request

        Returns:
            The agent's final text response
        """
        self.messages.append({"role": "user", "content": user_message})
        self._trim_context()

        try:
            self.logger.start_conversation(title=user_message[:50])
        except Exception:
            pass

        tool_calls_made = 0

        while tool_calls_made < self.config.max_tool_calls:
            # Call LLM with tools
            response = self.client.chat(
                messages=self.messages,
                tools=self.tool_specs,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            # Log request
            self.logger.log_llm_request(
                model=response.model,
                user_message=user_message if tool_calls_made == 0 else "[tool_result]",
                response=response,
            )

            # No tool calls → final response
            if not response.has_tool_calls:
                assistant_message = response.content or ""
                self.messages.append({"role": "assistant", "content": assistant_message})
                return assistant_message

            # Process tool calls
            assistant_msg = {
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": response.tool_calls,
            }
            self.messages.append(assistant_msg)

            for tool_call in response.tool_calls:
                tool_calls_made += 1
                tool_result = self._execute_tool(tool_call)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": tool_result,
                })

        # Exceeded max tool calls
        logger.warning(f"Max tool calls ({self.config.max_tool_calls}) reached.")
        response = self.client.chat(
            messages=self.messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        final = response.content or "Unable to generate a complete response. Please try rephrasing."
        self.messages.append({"role": "assistant", "content": final})
        return final

    def reset(self):
        """Reset conversation history."""
        self._init_conversation()

    def get_usage_summary(self) -> dict:
        """Get token usage and cost summary."""
        return self.client.get_usage_summary()

    # -------------------------------------------------------------------------
    # Tool Execution
    # -------------------------------------------------------------------------

    def _execute_tool(self, tool_call: dict) -> str:
        """Execute a single tool call and return the result."""
        func_name = tool_call["function"]["name"]

        try:
            arguments = json.loads(tool_call["function"]["arguments"])
        except (json.JSONDecodeError, KeyError) as e:
            error_msg = f"Invalid tool arguments: {e}"
            logger.error(error_msg)
            return json.dumps({"error": error_msg})

        func = self.tool_dispatch.get(func_name)
        if not func:
            error_msg = f"Unknown tool: {func_name}"
            logger.error(error_msg)
            return json.dumps({"error": error_msg})

        start = time.time()
        try:
            result = func(**arguments)
            execution_ms = (time.time() - start) * 1000
            logger.debug(f"Tool {func_name} executed in {execution_ms:.0f}ms")

            try:
                self.logger.log_tool_call(
                    request_id="",
                    tool_name=func_name,
                    tool_arguments=arguments,
                    tool_result=result[:500] if result else "",
                    execution_ms=execution_ms,
                    success=True,
                )
            except Exception:
                pass

            return result

        except Exception as e:
            execution_ms = (time.time() - start) * 1000
            error_msg = f"Tool error ({func_name}): {str(e)}"
            logger.error(error_msg)

            try:
                self.logger.log_tool_call(
                    request_id="",
                    tool_name=func_name,
                    tool_arguments=arguments,
                    tool_result="",
                    execution_ms=execution_ms,
                    success=False,
                    error_message=str(e),
                )
            except Exception:
                pass

            return json.dumps({"error": error_msg})

    # -------------------------------------------------------------------------
    # Context Management
    # -------------------------------------------------------------------------

    def _trim_context(self):
        """Trim conversation history to stay within context limit."""
        max_messages = self.config.max_context_messages
        if len(self.messages) <= max_messages + 1:
            return
        system = self.messages[0]
        recent = self.messages[-(max_messages):]
        self.messages = [system] + recent
        logger.debug(f"Trimmed context to {len(self.messages)} messages")


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    print("\n" + "=" * 60)
    print("  PyDyna Agent — LS-DYNA Keyword Assistant")
    print("=" * 60)
    print("  Type 'quit' to exit, 'reset' to clear history.")
    print("  Examples:")
    print("    'Create MAT_024 for mild steel'")
    print("    'What contact for self-contact in crash?'")
    print("    'Show me *CONTROL_TIMESTEP parameters'")
    print("=" * 60 + "\n")

    agent = PyDynaAgent()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if user_input.lower() == "reset":
            agent.reset()
            print("[Conversation reset]\n")
            continue

        response = agent.chat(user_input)
        print(f"\nAgent: {response}\n")
