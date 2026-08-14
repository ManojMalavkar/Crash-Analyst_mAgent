"""ANSA/META CodeRAG Agent with Tool-Calling Loop.

The core agent that orchestrates retrieval-augmented code generation:
1. Receives user query
2. Selects and calls appropriate RAG tools (vector search, KG, exact lookup)
3. Synthesizes retrieved context into working ANSA/META Python code
4. Supports multi-turn conversation with context management

Usage:
    from bin.agent import CodeRAGAgent
    
    agent = CodeRAGAgent()
    response = agent.chat("How do I mesh a part with 5mm elements?")
    print(response)
"""

import json
import time
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from shared import LLMClient, LLMResponse, create_client, create_logger, settings
from bin.tools import create_tool_registry
from bin.tool_functions import ALL_TOOLS


logger = logging.getLogger(__name__)


# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = """You are an expert ANSA and META Python API assistant. You generate accurate, 
runnable Python code for pre-processing (ANSA) and post-processing (META) tasks in crash simulation.

You have access to the following tools to retrieve API documentation:

## Tool Selection Guide

| User Intent | Tool to Use | Example |
|-------------|-------------|----------|
| General API search | search_api | "how to create shell mesh" |
| Need working code/script | search_code_examples | "mesh quality check script" |
| Know exact function name | get_function_details | "CreateMesh", "Entity" |
| Understand class structure | get_class_hierarchy | "Entity", "ShellElement" |
| Explore API relationships | search_knowledge_graph | "ansa.base", "mesh" |

## Rules

1. ALWAYS call at least one tool before generating code. Never guess at API signatures.
2. Use get_function_details when you know the function name (most accurate).
3. Use search_api for open-ended questions (semantic search).
4. Use get_class_hierarchy when the user asks about class methods or inheritance.
5. Combine multiple tools for complex queries (e.g., search_api + get_function_details).
6. After retrieving context, generate complete, runnable Python code with:
   - Correct imports (from ansa import base, mesh, etc.)
   - Proper function signatures (match retrieved documentation exactly)
   - Error handling where appropriate
   - Brief inline comments explaining key steps
7. If tools return no results, say so honestly. Do NOT hallucinate API calls.
8. Format code in ```python blocks.

## Software Context

- ANSA: Pre-processor for FE model creation (meshing, connections, loads, BCs)
- META: Post-processor for result visualization and extraction
- Both use Python scripting API accessed via: from ansa import base, mesh, connections, etc.
- LS-DYNA keywords use *KEYWORD_NAME format (e.g., *MAT_024, *CONTACT_AUTOMATIC_SURFACE_TO_SURFACE)
"""


# =============================================================================
# Agent Configuration
# =============================================================================

@dataclass
class AgentConfig:
    """Configuration for the CodeRAG agent."""
    max_tool_calls: int = 10            # Max tool calls per user message
    max_context_messages: int = 20      # Max messages in conversation history
    temperature: float = 0.1            # Low temperature for code generation
    max_tokens: int = 4096              # Max response tokens
    tool_call_timeout: float = 30.0     # Timeout per tool call (seconds)


# =============================================================================
# CodeRAG Agent
# =============================================================================

class CodeRAGAgent:
    """ANSA/META Code Generation Agent with RAG tool-calling loop."""
    
    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        llm_client: Optional[LLMClient] = None,
    ):
        """Initialize the CodeRAG agent.
        
        Args:
            config: Agent configuration (uses defaults if None)
            llm_client: Pre-configured LLM client (creates one if None)
        """
        self.config = config or AgentConfig()
        self.client = llm_client or create_client()
        self.logger = create_logger("ansa_agent")
        
        # Build tool registry
        registry = create_tool_registry(ALL_TOOLS)
        self.tool_specs = registry["specs"]
        self.tool_dispatch = registry["dispatch"]
        
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
        """Send a message to the agent and get a response.
        
        Executes the full tool-calling loop:
        1. Send user message + tools to LLM
        2. If LLM requests tool calls, execute them
        3. Send tool results back to LLM
        4. Repeat until LLM produces final response (no more tool calls)
        
        Args:
            user_message: The user's question or request
        
        Returns:
            The agent's final text response (with generated code)
        """
        # Add user message
        self.messages.append({"role": "user", "content": user_message})
        self._trim_context()
        
        # Start logging session
        self.logger.start_conversation(title=user_message[:50])
        
        # Tool-calling loop
        tool_calls_made = 0
        
        while tool_calls_made < self.config.max_tool_calls:
            # Call LLM
            response = self.client.chat(
                messages=self.messages,
                tools=self.tool_specs,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            
            # Log the request
            self.logger.log_llm_request(
                model=response.model,
                user_message=user_message if tool_calls_made == 0 else "[tool_result]",
                response=response,
            )
            
            # If no tool calls, we have the final response
            if not response.has_tool_calls:
                assistant_message = response.content or ""
                self.messages.append({"role": "assistant", "content": assistant_message})
                return assistant_message
            
            # Process tool calls
            # Add assistant message with tool_calls
            assistant_msg = {
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": response.tool_calls,
            }
            self.messages.append(assistant_msg)
            
            # Execute each tool call
            for tool_call in response.tool_calls:
                tool_calls_made += 1
                tool_result = self._execute_tool(tool_call)
                
                # Add tool result to messages
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": tool_result,
                })
        
        # Exceeded max tool calls — ask LLM for final response without tools
        logger.warning(f"Max tool calls ({self.config.max_tool_calls}) reached. Generating final response.")
        response = self.client.chat(
            messages=self.messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        
        final = response.content or "I was unable to generate a complete response. Please try rephrasing."
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
        """Execute a single tool call and return the result.
        
        Args:
            tool_call: Tool call dict from LLM response
                       {"id": "...", "function": {"name": "...", "arguments": "..."}}
        
        Returns:
            Tool result as string (JSON)
        """
        func_name = tool_call["function"]["name"]
        
        # Parse arguments
        try:
            arguments = json.loads(tool_call["function"]["arguments"])
        except (json.JSONDecodeError, KeyError) as e:
            error_msg = f"Invalid tool arguments: {e}"
            logger.error(error_msg)
            return json.dumps({"error": error_msg})
        
        # Dispatch to function
        func = self.tool_dispatch.get(func_name)
        if not func:
            error_msg = f"Unknown tool: {func_name}"
            logger.error(error_msg)
            return json.dumps({"error": error_msg})
        
        # Execute with timing
        start = time.time()
        try:
            result = func(**arguments)
            execution_ms = (time.time() - start) * 1000
            
            logger.debug(f"Tool {func_name} executed in {execution_ms:.0f}ms")
            
            # Log tool call
            self.logger.log_tool_call(
                request_id="",
                tool_name=func_name,
                tool_arguments=arguments,
                tool_result=result[:500] if result else "",
                execution_ms=execution_ms,
                success=True,
            )
            
            return result
            
        except Exception as e:
            execution_ms = (time.time() - start) * 1000
            error_msg = f"Tool error ({func_name}): {str(e)}"
            logger.error(error_msg)
            
            self.logger.log_tool_call(
                request_id="",
                tool_name=func_name,
                tool_arguments=arguments,
                tool_result="",
                execution_ms=execution_ms,
                success=False,
                error_message=str(e),
            )
            
            return json.dumps({"error": error_msg})
    
    # -------------------------------------------------------------------------
    # Context Management
    # -------------------------------------------------------------------------
    
    def _trim_context(self):
        """Trim conversation history to stay within context limit.
        
        Keeps system prompt + last N messages.
        """
        max_messages = self.config.max_context_messages
        
        if len(self.messages) <= max_messages + 1:  # +1 for system prompt
            return
        
        # Keep system prompt + last N messages
        system = self.messages[0]
        recent = self.messages[-(max_messages):]
        self.messages = [system] + recent
        
        logger.debug(f"Trimmed context to {len(self.messages)} messages")


# =============================================================================
# CLI Entry Point (for testing)
# =============================================================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    
    print("\n" + "=" * 60)
    print("  ANSA/META CodeRAG Agent (Interactive Mode)")
    print("=" * 60)
    print("  Type your question. Type 'quit' to exit, 'reset' to clear history.")
    print()
    
    agent = CodeRAGAgent()
    
    while True:
        try:
            user_input = input("  You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if user_input.lower() == "reset":
            agent.reset()
            print("  [Conversation reset]\n")
            continue
        if user_input.lower() == "usage":
            print(f"  {agent.get_usage_summary()}\n")
            continue
        
        print()
        response = agent.chat(user_input)
        print(f"  Agent: {response}\n")
    
    print("\n  Goodbye!")
    usage = agent.get_usage_summary()
    print(f"  Session usage: {usage}")
    print()
