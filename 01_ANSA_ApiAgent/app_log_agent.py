"""Log Analytics Admin Dashboard — Gradio Interface.

Separate UI for monitoring agent usage, performance, and errors.
Talk to the LogAgent in natural language to query usage data.

Usage:
    python app_log_agent.py
    # Opens at http://localhost:7861
"""

import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr

from bin.log_agent import LogAgent
from bin.admin_tools import (
    get_usage_summary,
    get_model_distribution,
    get_tool_usage,
    get_recent_errors,
    get_hourly_usage,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Agent Instance
# =============================================================================

log_agent: LogAgent = None


def get_log_agent() -> LogAgent:
    global log_agent
    if log_agent is None:
        log_agent = LogAgent()
    return log_agent


# =============================================================================
# Handlers
# =============================================================================

def nl_query_handler(question: str) -> tuple[str, str]:
    """Handle natural language query.
    
    Returns:
        Tuple of (SQL query, formatted results)
    """
    if not question.strip():
        return "", ""
    
    agent = get_log_agent()
    result = agent.query(question)
    
    sql = result.get("sql", "")
    formatted = agent.format_results(result)
    
    return sql, formatted


def refresh_dashboard() -> tuple[str, str, str, str, str]:
    """Refresh all dashboard panels."""
    # Usage summary
    summary = get_usage_summary(days=7)
    summary_text = (
        f"### Last 7 Days\n\n"
        f"- **Total Requests**: {summary.get('total_requests', 0):,}\n"
        f"- **Unique Sessions**: {summary.get('unique_sessions', 0)}\n"
        f"- **Total Tokens**: {summary.get('total_tokens', 0):,}\n"
        f"- **Avg Latency**: {summary.get('avg_latency_ms', 0):.0f} ms"
    )
    
    # Model distribution
    models = get_model_distribution(days=7)
    if models:
        model_lines = ["| Model | Requests | Tokens | Avg Latency |\n|---|---|---|---|"]
        for m in models:
            model_lines.append(
                f"| {m['model']} | {m['request_count']} | "
                f"{m.get('total_tokens', 0):,} | {m.get('avg_latency_ms', 0)} ms |"
            )
        model_text = "\n".join(model_lines)
    else:
        model_text = "No data yet."
    
    # Tool usage
    tools = get_tool_usage(days=7)
    if tools:
        tool_lines = ["| Tool | Calls | Avg Time | Success % |\n|---|---|---|---|"]
        for t in tools:
            tool_lines.append(
                f"| {t['tool_name']} | {t['call_count']} | "
                f"{t.get('avg_execution_ms', 0)} ms | {t.get('success_rate', 0)}% |"
            )
        tool_text = "\n".join(tool_lines)
    else:
        tool_text = "No data yet."
    
    # Recent errors
    errors = get_recent_errors(limit=5)
    if errors:
        error_lines = []
        for e in errors:
            error_lines.append(
                f"- `{e.get('timestamp', '')[:19]}` — **{e.get('error_type', 'unknown')}**: "
                f"{e.get('error_message', '')[:80]}"
            )
        error_text = "\n".join(error_lines)
    else:
        error_text = "No errors."
    
    # Hourly usage
    hourly = get_hourly_usage(days=1)
    if hourly:
        hourly_lines = [f"- {h['hour']}: {h['request_count']} requests" for h in hourly[-12:]]
        hourly_text = "\n".join(hourly_lines)
    else:
        hourly_text = "No activity today."
    
    return summary_text, model_text, tool_text, error_text, hourly_text


# =============================================================================
# Gradio Interface
# =============================================================================

EXAMPLE_QUESTIONS = [
    "How many requests were made today?",
    "Which model has the highest latency?",
    "Show me the most common user queries this week",
    "What's the tool success rate?",
    "How many errors occurred in the last 24 hours?",
    "Average tokens per request by model",
]


def build_admin_ui() -> gr.Blocks:
    """Build the admin analytics UI."""
    
    with gr.Blocks(
        title="SafetyAgent — Admin Analytics",
        theme=gr.themes.Soft(),
    ) as app:
        
        gr.Markdown(
            "# SafetyAgent — Usage Analytics\n"
            "Monitor agent performance, costs, and errors. "
            "Ask questions in natural language or view the dashboard."
        )
        
        with gr.Tabs():
            # Tab 1: NL Query
            with gr.TabItem("Query"):
                gr.Markdown("### Ask a question about usage data")
                
                with gr.Row():
                    question_input = gr.Textbox(
                        placeholder="e.g., How many requests today?",
                        label="Question",
                        scale=4,
                    )
                    query_btn = gr.Button("Ask", variant="primary", scale=1)
                
                with gr.Row():
                    for example in EXAMPLE_QUESTIONS[:3]:
                        gr.Button(example, size="sm").click(
                            fn=lambda e=example: e,
                            outputs=[question_input],
                        )
                
                sql_output = gr.Code(label="Generated SQL", language="sql")
                result_output = gr.Textbox(label="Results", lines=15)
                
                query_btn.click(
                    fn=nl_query_handler,
                    inputs=[question_input],
                    outputs=[sql_output, result_output],
                )
                question_input.submit(
                    fn=nl_query_handler,
                    inputs=[question_input],
                    outputs=[sql_output, result_output],
                )
            
            # Tab 2: Dashboard
            with gr.TabItem("Dashboard"):
                refresh_btn = gr.Button("Refresh", variant="primary")
                
                with gr.Row():
                    with gr.Column():
                        summary_panel = gr.Markdown(value="Click Refresh to load data.")
                    with gr.Column():
                        hourly_panel = gr.Markdown(value="")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Model Distribution")
                        model_panel = gr.Markdown(value="")
                    with gr.Column():
                        gr.Markdown("### Tool Usage")
                        tool_panel = gr.Markdown(value="")
                
                gr.Markdown("### Recent Errors")
                error_panel = gr.Markdown(value="")
                
                refresh_btn.click(
                    fn=refresh_dashboard,
                    outputs=[summary_panel, model_panel, tool_panel, error_panel, hourly_panel],
                )
    
    return app


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  SafetyAgent — Admin Analytics Dashboard")
    print("=" * 60 + "\n")
    
    app = build_admin_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
    )
