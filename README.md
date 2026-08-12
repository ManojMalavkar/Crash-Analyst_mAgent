# SafetyAgent — AI-Powered CAE Safety Engineering Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

> An AI-powered platform that automates the complete crash simulation workflow:
> **Pre-processing (ANSA) → Solving (LS-DYNA) → Post-processing (META) → HPC Orchestration**

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SafetyAgent Platform                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ 01_ANSA      │  │ 02_PyDyna    │  │ 03_ModelCheck│  │ 05_HPC       │    │
│  │ ApiAgent     │  │ Agent        │  │ Agent        │  │ Orchestrator │    │
│  │              │  │              │  │              │  │              │    │
│  │ Code gen for │  │ LS-DYNA via  │  │ Automated QA │  │ Job submit + │    │
│  │ ANSA + META  │  │ PyDyna SDK   │  │ checklist    │  │ post-process │    │
│  │ Python API   │  │              │  │ validation   │  │ pipeline     │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
│         │                  │                  │                  │            │
│         └──────────┬───────┴──────────┬───────┘                  │            │
│                    ▼                  ▼                          ▼            │
│         ┌────────────────────────────────────────────────────────────┐       │
│         │              04_Toolbar_Plugin                               │       │
│         │   Integrated ANSA/META toolbar (all agents accessible)      │       │
│         └────────────────────────────────────────────────────────────┘       │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
User Query → Agent Router → Specialized Agent → Tool Execution → Response
     │                            │                    │
     │                            ▼                    ▼
     │                     Vector DB (RAG)      Code Generation
     │                            │                    │
     ▼                            ▼                    ▼
  Gradio UI ◄──────── LLM (Claude/Llama) ◄──── Knowledge Base
```

---

## Project Structure

```
SafetyAgent/
├── README.md                    # This file
├── DEVELOPMENT_PLAN.md          # 40-day development roadmap
├── .gitignore
├── requirements.txt             # Global dependencies
├── LICENSE
├── shared/                      # Shared utilities across agents
│   ├── __init__.py
│   ├── llm_client.py           # Unified LLM client (AI Gateway)
│   ├── logger.py               # Shared logging framework
│   └── config.py               # Central configuration
├── 01_ANSA_ApiAgent/           # ANSA/META code generation agent
│   ├── bin/                    # Core agent logic + tools
│   └── knowledge-base/         # API documentation vectors
├── 02_PyDyna_Agent/            # LS-DYNA manipulation via PyDyna
│   ├── bin/                    # Agent + keyword retriever
│   ├── knowledge-base/         # PyDyna docs + keyword reference
│   └── vector_db/              # ChromaDB collections
├── 03_ModelCheck_Agent/        # CAE model & include validation
│   ├── bin/                    # Comparison engine + checkers
│   ├── checklists/             # Reference include definitions (YAML)
│   └── reports/                # Deviation reports (diff output)
├── 04_Toolbar_Plugin/          # ANSA/META integrated toolbar
│   ├── ansa_plugin/            # ANSA GUI toolbar registration
│   ├── meta_plugin/            # META GUI toolbar registration
│   └── shared/                 # API client + config
└── 05_HPC_Orchestrator/        # HPC job management + automation
    ├── bin/                    # Job manager + scheduler
    ├── templates/              # PBS/SLURM job templates
    └── pipelines/              # YAML pipeline definitions
```

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **LLM** | Claude Sonnet 4 / Llama 70B | Code generation & reasoning |
| **LLM Gateway** | Databricks AI Gateway | Model routing, fallback, rate limiting |
| **Vector DB** | ChromaDB | Persistent multi-collection embeddings |
| **Embeddings** | BAAI/bge-small-en-v1.5 | Document & code embedding |
| **Knowledge Graph** | NetworkX | API relationship traversal |
| **Database** | SQLite | Usage logging & analytics |
| **Web UI** | Gradio | Interactive agent interfaces |
| **Solver** | LS-DYNA + PyDyna | Crash simulation |
| **Pre-processor** | ANSA Python API | FE model preparation |
| **Post-processor** | META Python API | Result visualization |
| **HPC** | PBS Pro / SLURM | Job scheduling |
| **CI/CD** | GitHub Actions | Automated testing & deployment |
| **Container** | Docker + docker-compose | Service orchestration |
| **Language** | Python 3.9+ | Primary development language |

---

## Key Features

### 01 — ANSA/META Code Generation Agent (CodeRAG)
- **Hybrid retrieval**: Vector search + Knowledge Graph over 23K+ API symbols
- **Tool-calling loop**: Autonomous multi-step code generation
- **Session commands**: Parameterized META post-processing workflows
- **Admin analytics**: Usage tracking with NL→SQL query agent

### 02 — PyDyna Agent
- **Keyword expertise**: Exact lookup for 500+ LS-DYNA keyword cards
- **Model manipulation**: Read/write/modify .k/.key files programmatically
- **Code templates**: Jinja2-based generation for common workflows
- **Validation**: Keyword syntax and parameter checking

### 03 — Model Check Agent (Include Validation)
- **Reference comparison**: Compare model includes against reference include library
- **Deviation detection**: Find missing, modified, or outdated include files
- **Include hierarchy**: Validate include structure and dependencies
- **YAML-driven checklists**: Euro NCAP 2024, company-standard reference templates
- **Reports**: Deviation reports with diff highlights and severity levels

### 04 — Toolbar Plugin
- **Native integration**: Registers directly in ANSA/META GUI
- **Zero context-switch**: All agents accessible without leaving the tool
- **Safe execution**: Sandboxed code execution with undo support

### 05 — HPC Orchestrator
- **Multi-scheduler**: PBS Pro and SLURM support
- **Pipeline automation**: YAML-defined multi-stage workflows
- **Auto post-processing**: Triggered on job completion
- **Natural language**: "Submit crash job with 64 cores, post-process when done"

---

## Quick Start

### Prerequisites

- Python 3.10+
- ANSA/META installation (for API documentation)
- Access to an LLM endpoint (Databricks AI Gateway or OpenAI-compatible)

### Setup (3 commands)

```bash
git clone https://github.com/yourusername/SafetyAgent.git
cd SafetyAgent

python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate        # Linux/Mac

pip install -r requirements.txt
python setup.py
```

The setup wizard will:
1. Install all dependencies
2. Create `.env` from `.env.example`
3. Ask for your ANSA documentation path (e.g. `C:\BETA_CAE_Systems\ANSA_v2025.2.2\python\doc`)
4. Build the knowledge base (ingest docs + embed into ChromaDB)
5. Verify the installation

### Running an Agent

```bash
# Start the ANSA CodeRAG agent
python 01_ANSA_ApiAgent/app_gradio.py

# Start the PyDyna agent
python 02_PyDyna_Agent/app_gradio.py

# Start the Model Check agent
python 03_ModelCheck_Agent/app_gradio.py

# Start the HPC Orchestrator
python 05_HPC_Orchestrator/app_gradio.py
```

### Rebuild Knowledge Base

If you upgrade ANSA or add new documentation:

```bash
python 01_ANSA_ApiAgent/bin/build_vector_db.py --source "<new_docs_path>" --rebuild
```

### Docker Deployment

```bash
docker-compose up -d
# Access unified dashboard at http://localhost:7860
```

---

## Development

See [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) for the complete 40-day roadmap.

```bash
# Run linter
ruff check .

# Run tests
pytest tests/ -v

# Type checking
mypy shared/ --ignore-missing-imports
```

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [BETA CAE Systems](https://www.beta-cae.com/) — ANSA/META software
- [Ansys](https://www.ansys.com/) — PyDyna SDK
- [LSTC/Ansys](https://www.lstc.com/) — LS-DYNA solver
- [Databricks](https://www.databricks.com/) — AI Gateway infrastructure
