"""ANSA/META CodeRAG Agent — Gradio Web Interface.

Main copilot UI for interacting with the CodeRAG agent.
Features:
- Chat interface with streaming responses
- Code syntax highlighting
- Session management (reset, export)
- Usage statistics panel
- Knowledge base status indicator

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

def chat_handler(message: str, history: list[list]) -> tuple[str, list[list]]:
    """Handle a user message and return the agent response.
    
    Args:
        message: User's input message
        history: Gradio chat history [[user, assistant], ...]
    
    Returns:
        Tuple of (empty string to clear input, updated history)
    """
    if not message.strip():
        return "", history
    
    current_agent = get_agent()
    
    try:
        response = current_agent.chat(message)
    except Exception as e:
        logger.error(f"Agent error: {e}")
        response = f"Error: {str(e)}\n\nPlease try again or reset the conversation."
    
    history.append([message, response])
    return "", history


def reset_handler() -> tuple[list, str]:
    """Reset the conversation."""
    current_agent = get_agent()
    current_agent.reset()
    return [], "Conversation reset."


def get_usage_info() -> str:
    """Get formatted usage statistics."""
    current_agent = get_agent()
    usage = current_agent.get_usage_summary()
    
    lines = ["## Usage Statistics\n"]
    for key, value in usage.items():
        lines.append(f"- **{key}**: {value}")
    
    return "\n".join(lines) if usage else "No usage data yet."


def get_kb_status() -> str:
    """Get knowledge base status."""
    try:
        from bin.build_vector_db import VectorStoreBuilder
        builder = VectorStoreBuilder(persist_dir="vector_db")
        stats = builder.get_stats()
        
        if stats.get("status") == "not_found":
            return "Knowledge base not built. Run `python setup.py` first."
        
        return (
            f"Collection: **{stats['collection_name']}**\n\n"
            f"Documents: **{stats['total_documents']:,}**\n\n"
            f"Location: `{stats['persist_dir']}`"
        )
    except Exception as e:
        return f"Error reading KB: {e}"


# =============================================================================
# Gradio Interface
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

CSS = """
.code-output pre {
    background-color: #1e1e1e;
    color: #d4d4d4;
    padding: 12px;
    border-radius: 8px;
    overflow-x: auto;
}
.status-box {
    font-size: 0.85em;
    padding: 8px;
}
"""


def build_ui() -> gr.Blocks:
    """Build the Gradio Blocks interface."""
    
    with gr.Blocks(
        title="SafetyAgent — ANSA/META CodeRAG",
        theme=gr.themes.Soft(),
        css=CSS,
    ) as app:
        
        # Header
        gr.Markdown(
            "# SafetyAgent — ANSA/META Code Assistant\n"
            "Generate Python code for ANSA pre-processing and META post-processing. "
            "Powered by RAG over 23K+ API symbols."
        )
        
        with gr.Row():
            # Main chat area (left, wider)
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=500,
                    show_copy_button=True,
                    render_markdown=True,
                    avatar_images=(None, "https://img.icons8.com/color/48/robot-2.png"),
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Ask about ANSA/META API, e.g., 'How to mesh a part?'",
                        label="Your Question",
                        scale=5,
                        lines=1,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                
                with gr.Row():
                    reset_btn = gr.Button("Reset Conversation", size="sm")
                    status_text = gr.Textbox(
                        label="Status",
                        interactive=False,
                        scale=3,
                        lines=1,
                    )
            
            # Side panel (right, narrower)
            with gr.Column(scale=1):
                gr.Markdown("### Knowledge Base")
                kb_status = gr.Markdown(value=get_kb_status, every=30)
                
                gr.Markdown("### Example Queries")
                for example in EXAMPLE_QUERIES:
                    gr.Button(
                        example[:50] + "..." if len(example) > 50 else example,
                        size="sm",
                    ).click(
                        fn=lambda e=example: (e,),
                        outputs=[msg_input],
                    )
                
                gr.Markdown("### Usage")
                usage_display = gr.Markdown(value="No usage data yet.")
                refresh_usage_btn = gr.Button("Refresh Stats", size="sm")
        
        # Event handlers
        send_btn.click(
            fn=chat_handler,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot],
        )
        
        msg_input.submit(
            fn=chat_handler,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot],
        )
        
        reset_btn.click(
            fn=reset_handler,
            outputs=[chatbot, status_text],
        )
        
        refresh_usage_btn.click(
            fn=get_usage_info,
            outputs=[usage_display],
        )
    
    return app


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
    
    app = build_ui()
    app.launch(
        server_name=settings.ui.host,
        server_port=settings.ui.port,
        share=settings.ui.share,
    )
