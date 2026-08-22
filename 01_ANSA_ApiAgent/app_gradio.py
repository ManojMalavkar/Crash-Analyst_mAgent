"""ANSA/META CodeRAG Agent — Gradio Web Interface.

Main copilot UI for interacting with the CodeRAG agent.

Usage:
    python app_gradio.py
    # Opens at http://localhost:7860
"""

import sys
import logging
from pathlib import Path

# Project root path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr

from shared import settings
from bin.agent import CodeRAGAgent, AgentConfig


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Agent Singleton
# =============================================================================

agent: CodeRAGAgent = None


def get_agent() -> CodeRAGAgent:
    """Get or create the global agent instance."""
    global agent
    if agent is None:
        config = AgentConfig(
            max_tool_calls=settings.agent.max_tool_calls,
            max_context_messages=settings.agent.max_context_messages,
        )
        agent = CodeRAGAgent(config=config)
        logger.info("CodeRAG Agent initialized")
    return agent


# =============================================================================
# Chat Handler
# =============================================================================

# =============================================================================
# Gradio Interface (compatible with all Gradio versions)
# =============================================================================

EXAMPLE_QUERIES = [
    "How to create shell mesh on a part with 5mm element size?",
    "Write a script to delete duplicate nodes with 0.01 tolerance",
    "Show me how to export model to LS-DYNA keyword format",
    "What methods does the Entity class have?",
    "How to apply SPC boundary condition to selected nodes?",
    "Create a contact between two part sets",
    "Check element quality for jacobian > 0.3",
]


def submit(message, chat_history, agent_state):
    """Handle chat submission."""
    chat_history = chat_history or []
    agent_state = agent_state or []
    
    if not message.strip():
        return "", chat_history, agent_state
    
    current_agent = get_agent()
    
    try:
        response = current_agent.chat(message)
    except Exception as e:
        logger.error(f"Agent error: {e}")
        response = f"Error: {str(e)}\n\nPlease try again or reset the conversation."
    
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": response})
    
    return "", chat_history, agent_state


def reset(chat_history, agent_state):
    """Reset conversation."""
    current_agent = get_agent()
    current_agent.reset()
    return [], []


# =============================================================================
# Build UI
# =============================================================================

with gr.Blocks(title="SafetyAgent — ANSA/META CodeRAG") as demo:
    
    gr.Markdown("# SafetyAgent — ANSA/META Code Assistant")
    gr.Markdown("Generate Python code for ANSA pre-processing and META post-processing.")
    
    chatbot = gr.Chatbot(height=600)
    
    with gr.Row():
        msg_input = gr.Textbox(
            placeholder="Ask about ANSA/META API, e.g., 'How to mesh a part?'",
            show_label=False,
            scale=5,
        )
        send_btn = gr.Button("Send", variant="primary", scale=1)
    
    with gr.Row():
        reset_btn = gr.Button("Reset Conversation", size="sm")
    
    agent_state = gr.State([])
    
    # Event handlers
    send_btn.click(
        fn=submit,
        inputs=[msg_input, chatbot, agent_state],
        outputs=[msg_input, chatbot, agent_state],
    )
    
    msg_input.submit(
        fn=submit,
        inputs=[msg_input, chatbot, agent_state],
        outputs=[msg_input, chatbot, agent_state],
    )
    
    reset_btn.click(
        fn=reset,
        inputs=[chatbot, agent_state],
        outputs=[chatbot, agent_state],
    )
    
    # Example queries
    gr.Markdown("### Example Queries")
    gr.Examples(
        examples=EXAMPLE_QUERIES,
        inputs=msg_input,
    )


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  SafetyAgent — ANSA/META CodeRAG Agent")
    print("=" * 60)
    print(f"  LLM Model  : {settings.llm.primary_model}")
    print(f"  API Base   : {settings.llm.api_base_url}")
    print(f"  UI Port    : {settings.ui.port}")
    print()
    
    demo.launch(
        server_name=settings.ui.host,
        server_port=settings.ui.port,
        share=settings.ui.share,
    )
