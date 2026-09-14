"""PyDyna Agent — Gradio Web Interface.

Full-featured UI for interacting with the LS-DYNA PyDyna agent:
  1. Chat interface     — natural language queries about LS-DYNA / PyDyna
  2. Model file upload  — load .k/.key/.dyn files for analysis & modification
  3. Keyword card viewer — browse parsed keyword blocks and parts table
  4. Code preview panel  — generated PyDyna code with copy/download support

Usage:
    python 02_PyDyna_Agent/app_gradio.py
    # Opens at http://localhost:7861
"""

import sys
import json
import logging
import tempfile
import re
from pathlib import Path
from typing import Optional

# Project root path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr

from shared import settings
from bin.agent import PyDynaAgent, AgentConfig
from bin.model_tools import (
    read_keyword_file,
    list_parts,
    get_model_summary,
    _get_model_state,
    KeywordFileParser,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Agent Singleton
# =============================================================================

agent: Optional[PyDynaAgent] = None


def get_agent() -> PyDynaAgent:
    """Get or create the global PyDyna agent instance."""
    global agent
    if agent is None:
        config = AgentConfig(
            max_tool_calls=settings.agent.max_tool_calls,
            max_context_messages=settings.agent.max_context_messages,
        )
        agent = PyDynaAgent(config=config)
        logger.info("PyDyna Agent initialized")
    return agent


# =============================================================================
# Code Extraction Helper
# =============================================================================

def extract_code_blocks(response: str) -> str:
    """Extract Python/text code blocks from agent response."""
    blocks = []
    pattern = re.compile(r"```(?:python|text|lsdyna)?\n(.*?)```", re.DOTALL)
    for match in pattern.finditer(response):
        blocks.append(match.group(1).strip())
    return "\n\n# " + "=" * 60 + "\n\n".join(blocks) if blocks else ""


# =============================================================================
# Chat Handler
# =============================================================================

def chat_submit(message: str, chat_history: list, code_output: str):
    """Handle chat submission with code extraction."""
    chat_history = chat_history or []

    if not message.strip():
        return "", chat_history, code_output

    current_agent = get_agent()

    try:
        response = current_agent.chat(message)
    except Exception as e:
        logger.error(f"Agent error: {e}")
        response = f"Error: {str(e)}\n\nPlease try again or reset the conversation."

    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": response})

    # Extract code blocks for code preview panel
    extracted = extract_code_blocks(response)
    if extracted:
        code_output = extracted

    return "", chat_history, code_output


def chat_reset():
    """Reset conversation and all panels."""
    current_agent = get_agent()
    current_agent.reset()
    return [], "", "", "", []


# =============================================================================
# Model Upload Handler
# =============================================================================

def handle_model_upload(file_obj):
    """Handle .k/.key/.dyn file upload.

    Parses the file and returns:
    - Summary text (for info panel)
    - Keywords table (for keyword viewer)
    - Parts table (for parts viewer)
    """
    if file_obj is None:
        return "No file uploaded.", [], []

    filepath = file_obj.name if hasattr(file_obj, "name") else str(file_obj)

    # Validate extension
    ext = Path(filepath).suffix.lower()
    if ext not in (".k", ".key", ".dyn"):
        return f"Unsupported file type: {ext}. Expected .k, .key, or .dyn", [], []

    try:
        # Load model via model_tools
        result_json = read_keyword_file(filepath)
        result = json.loads(result_json)

        if result.get("status") == "error":
            return f"Parse error: {result.get('message', 'Unknown error')}", [], []

        # Build summary text
        summary_lines = [
            f"**File:** `{Path(filepath).name}`",
            f"**Total keyword blocks:** {result.get('total_blocks', 0)}",
            f"**Parts:** {result.get('parts_count', 0)}",
            f"**Materials:** {result.get('materials_count', 0)}",
            f"**Contacts:** {result.get('contacts_count', 0)}",
            f"**Sections:** {result.get('sections_count', 0)}",
            f"**Control cards:** {result.get('controls_count', 0)}",
            "",
            "**Keyword categories:**",
        ]
        for cat, count in sorted(
            result.get("keyword_categories", {}).items(),
            key=lambda x: -x[1],
        ):
            summary_lines.append(f"  - `*{cat}`: {count}")

        summary = "\n".join(summary_lines)

        # Build keyword blocks table
        state = _get_model_state()
        keyword_rows = []
        for block in state.blocks:
            first_field = ""
            if block.lines:
                fields = KeywordFileParser.parse_fixed_fields(block.lines[0])
                first_field = fields[0] if fields else ""
            keyword_rows.append([
                block.full_keyword,
                first_field,
                len(block.lines),
                block.line_start,
            ])

        # Build parts table
        parts_json = json.loads(list_parts())
        parts_rows = []
        for p in parts_json.get("parts", []):
            parts_rows.append([
                p.get("pid", ""),
                p.get("name", ""),
                p.get("section_id", ""),
                p.get("material_id", ""),
                p.get("hourglass_id", ""),
            ])

        return summary, keyword_rows, parts_rows

    except Exception as e:
        logger.error(f"Model upload error: {e}")
        return f"Error loading model: {str(e)}", [], []


# =============================================================================
# Keyword Detail Viewer
# =============================================================================

def view_keyword_detail(keyword_table, evt: gr.SelectData):
    """Show detailed content of a selected keyword block."""
    if evt is None:
        return "Select a keyword row to view details."

    row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    state = _get_model_state()

    if not state.is_loaded or row_idx >= len(state.blocks):
        return "Model not loaded or invalid selection."

    block = state.blocks[row_idx]
    lines = [
        f"### {block.full_keyword}",
        f"**Line start:** {block.line_start}",
        f"**Data cards:** {len(block.lines)}",
        "",
    ]

    if block.comment_lines:
        lines.append("**Comments:**")
        lines.append("```")
        for c in block.comment_lines[:10]:
            lines.append(c)
        lines.append("```")
        lines.append("")

    lines.append("**Data lines (raw):**")
    lines.append("```")
    for data_line in block.lines[:20]:
        lines.append(data_line)
    if len(block.lines) > 20:
        lines.append(f"... ({len(block.lines) - 20} more lines)")
    lines.append("```")

    # Parse fields from first data line
    if block.lines:
        fields = KeywordFileParser.parse_fixed_fields(block.lines[0])
        lines.append("")
        lines.append("**Parsed fields (card 1):**")
        for i, f in enumerate(fields):
            if f:
                lines.append(f"  - Field {i}: `{f}`")

    return "\n".join(lines)


# =============================================================================
# Export Handler
# =============================================================================

def export_model():
    """Export the current (possibly modified) model to a downloadable file."""
    state = _get_model_state()
    if not state.is_loaded:
        return None

    try:
        tmp = tempfile.NamedTemporaryFile(
            suffix=".key", delete=False, mode="w", encoding="utf-8",
        )
        output_path = state.write(tmp.name)
        return output_path
    except Exception as e:
        logger.error(f"Export error: {e}")
        return None


# =============================================================================
# Example Queries
# =============================================================================

EXAMPLE_QUERIES = [
    "Create *MAT_024 for mild steel with yield 250 MPa",
    "What contact type for self-contact in a crash model?",
    "Show me *CONTROL_TIMESTEP parameters and recommended values",
    "Write PyDyna code to define a rigid wall material",
    "What material for aluminum honeycomb barrier?",
    "Explain the difference between *CONTACT_AUTOMATIC_SINGLE_SURFACE and _SURFACE_TO_SURFACE",
    "Generate a complete keyword card for *SECTION_SHELL with 4 integration points",
]


# =============================================================================
# Custom CSS
# =============================================================================

CUSTOM_CSS = """
.model-info { font-size: 14px; line-height: 1.6; }
.keyword-detail { font-size: 13px; max-height: 400px; overflow-y: auto; }
#code-preview textarea { font-family: 'Fira Code', 'Consolas', monospace !important; }
"""


# =============================================================================
# Build UI
# =============================================================================

with gr.Blocks(
    title="SafetyAgent — PyDyna LS-DYNA Assistant",
    css=CUSTOM_CSS,
    theme=gr.themes.Soft(),
) as demo:

    # ─── Header ───────────────────────────────────────────────────────────
    gr.Markdown(
        "# SafetyAgent — PyDyna LS-DYNA Assistant\n"
        "AI-powered keyword setup, material selection, and model file manipulation "
        "for LS-DYNA crash simulations."
    )

    with gr.Tabs():

        # ═══════════════════════════════════════════════════════════════════
        # Tab 1: Chat Interface
        # ═══════════════════════════════════════════════════════════════════
        with gr.Tab("Chat", id="chat"):

            with gr.Row():
                # Left column: Chat
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        height=550,
                        show_copy_button=True,
                        placeholder="Ask about LS-DYNA keywords, materials, contacts, "
                                    "or PyDyna Python code...",
                    )

                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="e.g., 'Create *MAT_024 for mild steel'",
                            show_label=False,
                            scale=5,
                            lines=1,
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)

                    with gr.Row():
                        reset_btn = gr.Button("Reset Conversation", size="sm")
                        usage_info = gr.Markdown("", visible=True)

                # Right column: Code Preview
                with gr.Column(scale=2):
                    gr.Markdown("### Generated Code Preview")
                    code_preview = gr.Code(
                        label="Latest Code Output",
                        language="python",
                        lines=25,
                        interactive=False,
                        elem_id="code-preview",
                    )
                    with gr.Row():
                        copy_hint = gr.Markdown(
                            "*Code blocks from agent responses appear here automatically.*"
                        )

            # Example queries
            gr.Markdown("### Quick Start Examples")
            gr.Examples(
                examples=EXAMPLE_QUERIES,
                inputs=msg_input,
            )

        # ═══════════════════════════════════════════════════════════════════
        # Tab 2: Model File Upload & Viewer
        # ═══════════════════════════════════════════════════════════════════
        with gr.Tab("Model Viewer", id="model"):

            with gr.Row():
                # Left: Upload + Summary
                with gr.Column(scale=1):
                    gr.Markdown("### Upload LS-DYNA Model")
                    file_upload = gr.File(
                        label="Upload .k / .key / .dyn file",
                        file_types=[".k", ".key", ".dyn"],
                        type="filepath",
                    )
                    model_summary = gr.Markdown(
                        value="*Upload a model file to see its summary.*",
                        elem_classes="model-info",
                    )
                    export_btn = gr.Button(
                        "Export Modified Model", variant="secondary", size="sm",
                    )
                    export_file = gr.File(label="Download", visible=True)

                # Right: Keyword detail
                with gr.Column(scale=1):
                    gr.Markdown("### Keyword Detail")
                    keyword_detail = gr.Markdown(
                        value="*Click a row in the Keywords table to see details.*",
                        elem_classes="keyword-detail",
                    )

            # Keywords table
            gr.Markdown("### Keyword Blocks")
            keyword_table = gr.Dataframe(
                headers=["Keyword", "ID / First Field", "Data Lines", "Line #"],
                datatype=["str", "str", "number", "number"],
                interactive=False,
                height=300,
            )

            # Parts table
            gr.Markdown("### Parts List")
            parts_table = gr.Dataframe(
                headers=["PID", "Name", "Section ID", "Material ID", "Hourglass ID"],
                datatype=["number", "str", "number", "number", "number"],
                interactive=False,
                height=250,
            )

        # ═══════════════════════════════════════════════════════════════════
        # Tab 3: Help / Reference
        # ═══════════════════════════════════════════════════════════════════
        with gr.Tab("Help", id="help"):
            gr.Markdown("""
### How to Use This Tool

**Chat Tab:**
- Ask natural language questions about LS-DYNA keywords, materials, contacts
- Request PyDyna Python code generation
- The agent uses RAG retrieval over 3,148 keyword definitions
- Generated code appears automatically in the Code Preview panel

**Model Viewer Tab:**
- Upload an LS-DYNA keyword file (.k, .key, .dyn)
- Browse all keyword blocks in the model
- Click any row to see its raw content and parsed fields
- View the parts table with PID, section, and material assignments
- Export the model after modifications (via chat commands)

### Supported Question Types

| Question Type | Example |
|---|---|
| Material selection | "What material for crash steel?" |
| Keyword lookup | "Show me *MAT_024 parameters" |
| Contact setup | "Contact for self-contact in crash?" |
| PyDyna code | "Write PyDyna code for rigid wall" |
| Parameter validation | "Validate my *MAT_024 with RO=7.85e-9" |
| Model analysis | "Summarize the uploaded model" |

### LS-DYNA Unit Systems

| System | Length | Time | Mass | Force | Stress |
|---|---|---|---|---|---|
| **mm-s-tonne** | mm | s | tonne | N | MPa |
| **mm-ms-kg** | mm | ms | kg | kN | GPa |
| **m-s-kg** | m | s | kg | N | Pa |

### Architecture

```
User Query → PyDyna Agent → RAG Tools (ChromaDB) → LLM → Response
                          → Model Tools (file I/O) →
```
            """)

    # ─── Event Handlers ───────────────────────────────────────────────────

    # Chat
    send_btn.click(
        fn=chat_submit,
        inputs=[msg_input, chatbot, code_preview],
        outputs=[msg_input, chatbot, code_preview],
    )
    msg_input.submit(
        fn=chat_submit,
        inputs=[msg_input, chatbot, code_preview],
        outputs=[msg_input, chatbot, code_preview],
    )
    reset_btn.click(
        fn=chat_reset,
        inputs=[],
        outputs=[chatbot, code_preview, model_summary, keyword_detail, parts_table],
    )

    # Model upload
    file_upload.change(
        fn=handle_model_upload,
        inputs=[file_upload],
        outputs=[model_summary, keyword_table, parts_table],
    )

    # Keyword detail on row select
    keyword_table.select(
        fn=view_keyword_detail,
        inputs=[keyword_table],
        outputs=[keyword_detail],
    )

    # Export
    export_btn.click(
        fn=export_model,
        inputs=[],
        outputs=[export_file],
    )


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  SafetyAgent — PyDyna LS-DYNA Assistant")
    print("=" * 60)
    print(f"  LLM Model  : {settings.llm.primary_model}")
    print(f"  API Base   : {settings.llm.api_base_url}")
    print(f"  UI Port    : 7861")
    print()

    demo.launch(
        server_name=settings.ui.host,
        server_port=7861,
        share=settings.ui.share,
    )
