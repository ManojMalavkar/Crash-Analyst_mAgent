# SafetyAgent — Daily Development Plan (GitHub Portfolio)

## Project Vision

An **AI-powered CAE Safety Engineering Platform** that automates the complete crash simulation workflow:
Pre-processing (ANSA) → Solving (LS-DYNA) → Post-processing (META) → HPC Orchestration

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      SafetyAgent Platform                                     │
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

## Repository Structure

```
SafetyAgent/
├── README.md                    # Project overview + architecture diagram
├── DEVELOPMENT_PLAN.md          # This file
├── .gitignore
├── requirements.txt             # Global requirements
├── shared/                      # Shared utilities across agents
│   ├── __init__.py
│   ├── llm_client.py           # Unified LLM client (AI Gateway)
│   ├── logger.py               # Shared logging framework
│   └── config.py               # Central configuration
├── 01_ANSA_ApiAgent/           # ✅ DONE — ANSA/META code generation
├── 02_PyDyna_Agent/            # LS-DYNA manipulation via PyDyna
│   ├── app_gradio.py
│   ├── bin/
│   │   ├── agent.py
│   │   ├── pydyna_tools.py
│   │   ├── keyword_retriever.py
│   │   └── build_vector_db.py
│   ├── knowledge-base/
│   │   ├── pydyna_docs.jsonl
│   │   └── keyword_reference.jsonl
│   └── vector_db/
├── 03_ModelCheck_Agent/         # Model quality validation
│   ├── app_gradio.py
│   ├── bin/
│   │   ├── agent.py
│   │   ├── checklist_engine.py
│   │   ├── ansa_checks.py
│   │   └── pydyna_checks.py
│   ├── checklists/
│   │   ├── euro_ncap_2024.yaml
│   │   ├── company_standard.yaml
│   │   └── custom_template.yaml
│   └── reports/
├── 04_Toolbar_Plugin/           # ANSA/META integrated toolbar
│   ├── ansa_plugin/
│   │   ├── __init__.py
│   │   ├── toolbar.py
│   │   └── ui_panels.py
│   ├── meta_plugin/
│   │   ├── __init__.py
│   │   ├── toolbar.py
│   │   └── ui_panels.py
│   └── shared/
│       ├── api_client.py
│       └── config.py
└── 05_HPC_Orchestrator/         # HPC job management + automation
    ├── app_gradio.py
    ├── bin/
    │   ├── agent.py
    │   ├── job_manager.py
    │   ├── scheduler.py
    │   └── post_processor.py
    ├── templates/
    │   ├── pbs_template.sh
    │   └── slurm_template.sh
    └── pipelines/
        ├── crash_pipeline.yaml
        └── optimization_pipeline.yaml
```

---

## Daily Commit Plan (40 days)

### Week 1: Foundation & Repository Setup

---

#### Day 1 — Repository Setup + README
```
commit: "Initial project setup with architecture documentation"
```
- [ ] Create GitHub repo `SafetyAgent`
- [ ] Write comprehensive `README.md` with:
  - Project vision, architecture diagram
  - Tech stack table
  - Setup instructions
- [ ] Add `.gitignore` (Python, IDE, vector_db/, *.pkl, *.db)
- [ ] Add `requirements.txt` (global deps)
- [ ] Create folder structure (empty `__init__.py` files)
- [ ] Add LICENSE (MIT or Apache 2.0)

---

#### Day 2 — Shared Module: LLM Client + Config
```
commit: "Add shared LLM client with model fallback and config management"
```
- [ ] `shared/config.py` — Central configuration (paths, models, tokens)
- [ ] `shared/llm_client.py` — Unified OpenAI-compatible client with:
  - Model fallback chain
  - Retry logic with exponential backoff
  - Token usage tracking
  - Works with Databricks AI Gateway
- [ ] `shared/__init__.py` — Clean exports

---

#### Day 3 — Shared Module: Logging Framework
```
commit: "Add enterprise logging framework with dual-write (JSONL + SQLite)"
```
- [ ] `shared/logger.py` — UsageLogger class
  - Dual-write: JSONL + SQLite
  - Schema migration support
  - Session/conversation tracking
- [ ] `shared/log_db.py` — SQLite schema (6 tables)
- [ ] Unit tests for logger

---

#### Day 4 — 01_ANSA_ApiAgent: Core Agent
```
commit: "Add ANSA/META CodeRAG agent with tool-calling loop"
```
- [ ] `01_ANSA_ApiAgent/bin/agent.py` — CodeRAG_Agent class
- [ ] `01_ANSA_ApiAgent/bin/tools.py` — Auto tool spec generator
- [ ] `01_ANSA_ApiAgent/bin/tool_functions.py` — 5 RAG tools
- [ ] System prompt with tool selection guide

---

#### Day 5 — 01_ANSA_ApiAgent: Knowledge Base & Vector DB
```
commit: "Add knowledge base ingestion and ChromaDB vector store builder"
```
- [ ] `01_ANSA_ApiAgent/bin/ingest.py` — Parse HTML/JSON/Python sources
- [ ] `01_ANSA_ApiAgent/bin/build_vector_db.py` — ChromaDB collection builder
- [ ] `01_ANSA_ApiAgent/bin/kg_retriever.py` — Knowledge Graph class
- [ ] Document the ingestion pipeline in README

---

#### Day 6 — 01_ANSA_ApiAgent: Session Commands
```
commit: "Add META session command tools with vector search + parameterization"
```
- [ ] `01_ANSA_ApiAgent/bin/session_tools.py` — 4 session tools
- [ ] `01_ANSA_ApiAgent/bin/build_session_vector_db.py` — Session collection builder
- [ ] `01_ANSA_ApiAgent/knowledge-base/session_commands.json` (or sample subset)

---

#### Day 7 — 01_ANSA_ApiAgent: Gradio UI + Admin Tools
```
commit: "Add Gradio web UI, admin tools, and log analytics agent"
```
- [ ] `01_ANSA_ApiAgent/app_gradio.py` — Main copilot UI
- [ ] `01_ANSA_ApiAgent/app_log_agent.py` — Log analytics UI
- [ ] `01_ANSA_ApiAgent/bin/admin_tools.py` — Admin query functions
- [ ] `01_ANSA_ApiAgent/bin/log_agent.py` — NL → SQL log agent
- [ ] `01_ANSA_ApiAgent/bin/analytics.py` — Nightly report generation

---

### Week 2: PyDyna Agent (LS-DYNA)

---

#### Day 8 — PyDyna Research + Knowledge Base Design
```
commit: "Add PyDyna documentation and keyword reference knowledge base"
```
- [ ] Research PyDyna API: `ansys.dyna.core` module structure
- [ ] `02_PyDyna_Agent/knowledge-base/pydyna_docs.jsonl` — API documentation
  - Extract from PyDyna official docs
  - Classes: `DynaBase`, `DynaSolution`, `DynaMech`, etc.
  - Keywords: `*KEYWORD`, `*SECTION`, `*MAT`, `*CONTACT`, etc.
- [ ] `02_PyDyna_Agent/knowledge-base/keyword_reference.jsonl`
  - LS-DYNA keyword cards with parameters
  - Cross-reference to PyDyna methods

---

#### Day 9 — PyDyna Vector DB + Retriever
```
commit: "Build PyDyna vector database and keyword retriever"
```
- [ ] `02_PyDyna_Agent/bin/build_vector_db.py` — Index PyDyna docs
- [ ] `02_PyDyna_Agent/bin/keyword_retriever.py`
  - Keyword-based lookup (exact match for `*MAT_024`)
  - Semantic search for "how to define contact between parts"
  - Parameter validation against keyword spec

---

#### Day 10 — PyDyna Agent Core
```
commit: "Add PyDyna agent with tool-calling for LS-DYNA model manipulation"
```
- [ ] `02_PyDyna_Agent/bin/agent.py` — PyDyna_Agent class
- [ ] `02_PyDyna_Agent/bin/pydyna_tools.py` — Tools:
  - `search_keyword(query)` — find LS-DYNA keywords
  - `lookup_keyword(name)` — exact keyword lookup with params
  - `search_pydyna_api(query)` — search PyDyna Python API
  - `get_material_model(name)` — material model parameters
  - `get_contact_type(description)` — contact algorithm selection
  - `validate_keyword(card_content)` — validate keyword syntax

---

#### Day 11 — PyDyna Agent: Model Read/Write Tools
```
commit: "Add LS-DYNA model file read/write and modification tools"
```
- [ ] `02_PyDyna_Agent/bin/model_tools.py` — Tools:
  - `read_keyword_file(filepath)` — parse .k/.key file
  - `modify_material(mat_id, params)` — change material props
  - `modify_contact(contact_id, params)` — change contact settings
  - `add_keyword(keyword_type, params)` — insert new keyword
  - `list_parts()` — list all parts in model
  - `get_model_summary()` — overview of model content

---

#### Day 12 — PyDyna Agent: Gradio UI
```
commit: "Add PyDyna agent Gradio interface with model upload support"
```
- [ ] `02_PyDyna_Agent/app_gradio.py` — Web UI
  - Chat interface for PyDyna questions
  - Model file upload (.k, .key, .dyn)
  - Keyword card viewer/editor
  - Generated code preview panel

---

#### Day 13 — PyDyna Agent: Code Generation Templates
```
commit: "Add PyDyna code generation templates for common workflows"
```
- [ ] `02_PyDyna_Agent/templates/`
  - `create_material.py.j2` — Material definition template
  - `define_contact.py.j2` — Contact definition template  
  - `boundary_conditions.py.j2` — SPC/load curves
  - `control_cards.py.j2` — Time step, termination, output
- [ ] Template engine in `bin/template_engine.py`

---

#### Day 14 — PyDyna Agent: Testing + Documentation
```
commit: "Add PyDyna agent tests and usage documentation"
```
- [ ] `02_PyDyna_Agent/tests/` — Unit tests
  - Test keyword parsing
  - Test tool functions
  - Test code generation
- [ ] `02_PyDyna_Agent/README.md` — Agent documentation
- [ ] Example conversations/demos

---

### Week 3: Model Check Agent

---

#### Day 15 — Checklist Schema Design
```
commit: "Define model check checklist YAML schema with Euro NCAP rules"
```
- [ ] `03_ModelCheck_Agent/checklists/schema.yaml` — Checklist format:
  ```yaml
  name: "Euro NCAP 2024 Front Crash"
  checks:
    - id: mesh_quality
      description: "Element quality > 0.3"
      tool: ansa  # or pydyna
      method: check_element_quality
      params: {min_jacobian: 0.3, max_aspect: 5.0}
      severity: error
  ```
- [ ] `03_ModelCheck_Agent/checklists/euro_ncap_2024.yaml`
- [ ] `03_ModelCheck_Agent/checklists/company_standard.yaml`

---

#### Day 16 — Checklist Engine
```
commit: "Add checklist execution engine with ANSA and PyDyna runners"
```
- [ ] `03_ModelCheck_Agent/bin/checklist_engine.py`
  - Load YAML checklist
  - Execute checks sequentially
  - Generate pass/fail report
  - Support auto-fix for certain failures
- [ ] `03_ModelCheck_Agent/bin/check_result.py` — Result data classes

---

#### Day 17 — ANSA-Based Checks
```
commit: "Add ANSA script-based model quality checks"
```
- [ ] `03_ModelCheck_Agent/bin/ansa_checks.py` — ANSA script generators:
  - `check_element_quality()` — Jacobian, aspect ratio, warpage
  - `check_connectivity()` — Free edges, T-connections
  - `check_penetrations()` — Initial penetrations
  - `check_mass_properties()` — Mass/CG validation
  - `check_materials()` — Material card completeness
  - `check_contacts()` — Contact definitions
- [ ] Each check generates an ANSA Python script using CodeRAG

---

#### Day 18 — PyDyna-Based Checks
```
commit: "Add PyDyna-based direct keyword file validation checks"
```
- [ ] `03_ModelCheck_Agent/bin/pydyna_checks.py`:
  - `check_timestep()` — Minimum time step estimation
  - `check_energy_balance()` — Energy parameters set correctly
  - `check_hourglass()` — Hourglass control present
  - `check_damping()` — Appropriate damping values
  - `check_output_requests()` — D3PLOT/D3THDT intervals
  - `check_termination()` — Simulation time set
  - `check_units_consistency()` — Unit system validation

---

#### Day 19 — Model Check Agent Core
```
commit: "Add model check AI agent with auto-fix capabilities"
```
- [ ] `03_ModelCheck_Agent/bin/agent.py` — ModelCheck_Agent
  - Tools: `run_checklist`, `get_check_status`, `auto_fix`, `generate_report`
  - Explains failures in plain language
  - Suggests fixes using CodeRAG or PyDyna agent
  - Can auto-apply fixes for known issues

---

#### Day 20 — Model Check: Report Generator
```
commit: "Add HTML/PDF model check report generator"
```
- [ ] `03_ModelCheck_Agent/bin/report_generator.py`
  - HTML report with pass/fail table
  - Color-coded severity (error/warning/info)
  - Screenshots of problem areas (via ANSA script)
  - Summary statistics
- [ ] `03_ModelCheck_Agent/templates/report.html.j2`

---

#### Day 21 — Model Check: Gradio UI
```
commit: "Add model check Gradio UI with checklist selector and live results"
```
- [ ] `03_ModelCheck_Agent/app_gradio.py`
  - Upload model file
  - Select checklist (dropdown)
  - Run checks with progress bar
  - Display results with auto-fix buttons
  - Download report

---

### Week 4: Toolbar Plugin (ANSA/META Integration)

---

#### Day 22 — ANSA Plugin Architecture
```
commit: "Add ANSA toolbar plugin with agent communication layer"
```
- [ ] `04_Toolbar_Plugin/ansa_plugin/__init__.py`
- [ ] `04_Toolbar_Plugin/ansa_plugin/toolbar.py`
  - Register toolbar in ANSA GUI
  - Buttons: CodeRAG, Model Check, PyDyna, HPC
  - Uses `ansa.guitk` for native widgets
- [ ] `04_Toolbar_Plugin/ansa_plugin/ui_panels.py`
  - Chat panel (embedded)
  - Code preview + execute panel
  - Check results panel

---

#### Day 23 — ANSA Plugin: Agent Client
```
commit: "Add ANSA plugin API client connecting to agent backend"
```
- [ ] `04_Toolbar_Plugin/shared/api_client.py`
  - HTTP client to agent server (local or remote)
  - Streaming response support
  - File upload (model files)
- [ ] `04_Toolbar_Plugin/shared/config.py`
  - Agent URLs, authentication
  - User preferences persistence

---

#### Day 24 — META Plugin Architecture
```
commit: "Add META toolbar plugin with post-processing agent integration"
```
- [ ] `04_Toolbar_Plugin/meta_plugin/__init__.py`
- [ ] `04_Toolbar_Plugin/meta_plugin/toolbar.py`
  - Register toolbar in META GUI
  - Buttons: Session Record → Parameterize, Post-Process, Report
- [ ] `04_Toolbar_Plugin/meta_plugin/ui_panels.py`
  - Session recorder helper
  - Parameterization dialog
  - Result comparison panel

---

#### Day 25 — Plugin: Code Execution Engine
```
commit: "Add safe code execution engine for running agent-generated scripts"
```
- [ ] `04_Toolbar_Plugin/shared/executor.py`
  - Sandboxed execution of generated Python code
  - Capture stdout/stderr
  - Error recovery and reporting
  - Undo support (via ANSA undo stack)
- [ ] `04_Toolbar_Plugin/shared/code_preview.py`
  - Syntax highlighting
  - Diff view (before/after)
  - User approval before execution

---

#### Day 26 — Plugin: Integration Tests
```
commit: "Add plugin integration tests and installation documentation"
```
- [ ] `04_Toolbar_Plugin/tests/`
- [ ] `04_Toolbar_Plugin/README.md`
  - Installation guide (ANSA/META user scripts folder)
  - Configuration
  - Screenshots / demo GIF
- [ ] `04_Toolbar_Plugin/install.py` — Auto-installer script

---

### Week 5: HPC Orchestrator

---

#### Day 27 — HPC Job Manager
```
commit: "Add HPC job manager with PBS/SLURM support"
```
- [ ] `05_HPC_Orchestrator/bin/job_manager.py`
  - Submit jobs (PBS `qsub`, SLURM `sbatch`)
  - Monitor job status (`qstat`, `squeue`)
  - Cancel jobs
  - Parse job output logs
  - Resource estimation (CPUs, memory, wall time)
- [ ] `05_HPC_Orchestrator/templates/pbs_template.sh`
- [ ] `05_HPC_Orchestrator/templates/slurm_template.sh`

---

#### Day 28 — HPC Scheduler
```
commit: "Add intelligent job scheduler with dependency chains"
```
- [ ] `05_HPC_Orchestrator/bin/scheduler.py`
  - Job dependency graph (DAG)
  - Auto-submit next job when previous completes
  - Resource-aware scheduling
  - Priority queue management
  - Failure handling (retry, notify, skip)

---

#### Day 29 — Post-Processing Pipeline
```
commit: "Add automated post-processing pipeline triggered on job completion"
```
- [ ] `05_HPC_Orchestrator/bin/post_processor.py`
  - Monitor job completion (poll or callback)
  - Auto-launch META post-processing scripts:
    - Load results
    - Apply standard contour plots
    - Capture screenshots
    - Generate animations
    - Run model checks on results
  - Uses CodeRAG-generated META scripts

---

#### Day 30 — Pipeline Definition
```
commit: "Add YAML pipeline definitions for crash simulation workflows"
```
- [ ] `05_HPC_Orchestrator/pipelines/crash_pipeline.yaml`:
  ```yaml
  name: "Front Crash Full Pipeline"
  stages:
    - name: model_check
      agent: model_check
      input: model.key
      on_fail: stop
    - name: solve
      type: hpc_job
      solver: ls-dyna
      cpus: 64
      walltime: "8:00:00"
    - name: post_process
      agent: ansa_codegen
      trigger: solve.complete
      scripts: [contour_plots, animation, report]
    - name: results_check
      agent: model_check
      checklist: results_validation
  ```
- [ ] `05_HPC_Orchestrator/pipelines/optimization_pipeline.yaml`

---

#### Day 31 — HPC Agent Core
```
commit: "Add HPC orchestrator AI agent with natural language job management"
```
- [ ] `05_HPC_Orchestrator/bin/agent.py` — HPC_Agent
  - Tools: `submit_job`, `check_status`, `cancel_job`,
    `run_pipeline`, `get_results`, `schedule_post_processing`
  - Natural language: "Submit bumper crash with 64 cores, run post-processing when done"

---

#### Day 32 — HPC Agent: Gradio UI
```
commit: "Add HPC orchestrator Gradio UI with pipeline visualizer"
```
- [ ] `05_HPC_Orchestrator/app_gradio.py`
  - Job queue dashboard
  - Pipeline status visualization
  - Chat interface for job management
  - Log viewer

---

### Week 6: Integration & Polish

---

#### Day 33 — Unified Agent Router
```
commit: "Add multi-agent router that delegates to specialized agents"
```
- [ ] `shared/agent_router.py`
  - Classifies user intent → routes to correct agent
  - Handles cross-agent workflows
  - Maintains conversation context across agents
  - Example: "Check my model and submit to HPC" → ModelCheck → HPC

---

#### Day 34 — Unified Gradio Dashboard
```
commit: "Add unified dashboard with all agents accessible from one interface"
```
- [ ] `app.py` (root level) — Master Gradio app
  - Tab per agent (CodeRAG, PyDyna, ModelCheck, HPC)
  - Unified chat with auto-routing
  - Shared session/context

---

#### Day 35 — Docker Deployment
```
commit: "Add Docker containerization for all services"
```
- [ ] `Dockerfile`
- [ ] `docker-compose.yml` — All agents as services
- [ ] `docs/deployment.md`
- [ ] Health checks, logging, monitoring

---

#### Day 36 — CI/CD Pipeline
```
commit: "Add GitHub Actions CI with linting, tests, and build verification"
```
- [ ] `.github/workflows/ci.yml`
  - Lint (flake8/ruff)
  - Type check (mypy)
  - Unit tests (pytest)
  - Build Docker image
- [ ] `pyproject.toml` or `setup.py`

---

#### Day 37 — Documentation: Architecture Deep-Dive
```
commit: "Add comprehensive architecture documentation with diagrams"
```
- [ ] `docs/architecture.md` — System design
- [ ] `docs/data-flow.md` — Data flow diagrams
- [ ] `docs/api-reference.md` — Internal APIs
- [ ] Mermaid diagrams for GitHub rendering

---

#### Day 38 — Demo Notebooks
```
commit: "Add Jupyter demo notebooks for each agent"
```
- [ ] `demos/01_codegen_demo.ipynb`
- [ ] `demos/02_pydyna_demo.ipynb`
- [ ] `demos/03_modelcheck_demo.ipynb`
- [ ] `demos/04_hpc_pipeline_demo.ipynb`
- [ ] Each with markdown explanations + live code

---

#### Day 39 — Performance & Evaluation
```
commit: "Add evaluation framework with benchmarks for all agents"
```
- [ ] `evaluation/codegen_eval.py` — Code generation accuracy
- [ ] `evaluation/retrieval_eval.py` — RAG retrieval recall
- [ ] `evaluation/modelcheck_eval.py` — Check coverage
- [ ] Results summary in `evaluation/RESULTS.md`

---

#### Day 40 — Final Polish + Release
```
commit: "v1.0 release: Final polish, badges, and demo video link"
```
- [ ] Add GitHub badges (CI status, Python version, license)
- [ ] Record demo video (Loom/YouTube) → link in README
- [ ] Add `CHANGELOG.md`
- [ ] Tag release: `v1.0.0`
- [ ] Clean up all TODO comments

---

## Interview Talking Points by Day Range

| Days | Topic | Key Takeaway |
|------|-------|--------------|
| 1-3 | Foundation | "I designed a modular architecture from day 1 — shared components, clean separation" |
| 4-7 | CodeRAG Agent | "RAG with hybrid retrieval (vector + knowledge graph) for 23K API symbols" |
| 8-14 | PyDyna Agent | "Extended to LS-DYNA domain — same architecture, different knowledge base" |
| 15-21 | Model Check | "Automated quality assurance with YAML-driven checklists and auto-fix" |
| 22-26 | Toolbar Plugin | "Embedded AI directly in the engineer's daily tools — zero context switching" |
| 27-32 | HPC Orchestrator | "End-to-end automation: submit, monitor, post-process — all from natural language" |
| 33-40 | Integration | "Multi-agent routing, Docker, CI/CD, evaluation — production-grade engineering" |

---

## Git Commit Best Practices (for interview impression)

1. **Atomic commits** — Each commit does ONE thing well
2. **Meaningful messages** — Start with verb: "Add", "Fix", "Refactor", "Update"
3. **No "WIP" commits** — Squash before push if needed
4. **README updates** — Update README with each new feature
5. **Tests alongside code** — Show testing discipline
6. **Documentation** — Docstrings in every file, module docstrings

## Daily Routine

```
Morning:
  1. Review DEVELOPMENT_PLAN.md — what's today's task
  2. Create branch: feature/day-XX-description
  3. Code the feature
  4. Write tests (if applicable)
  5. Update README section

Evening:
  1. Run linter (ruff/flake8)
  2. Commit with meaningful message
  3. Push to GitHub
  4. Mark day as complete in this plan
```

---

## Priority Adjustments

If time is limited, prioritize in this order:
1. **Days 1-7** (MUST) — Shows complete working agent
2. **Days 8-14** (HIGH) — Shows you can extend to new domain
3. **Days 15-21** (HIGH) — Shows real engineering value (QA automation)
4. **Days 27-32** (MEDIUM) — Shows system thinking (orchestration)
5. **Days 22-26** (MEDIUM) — Shows UX thinking (embedded in tool)
6. **Days 33-40** (NICE) — Shows production readiness

---

## Technology Stack (Full Project)

| Layer | Technology |
|-------|-----------|
| LLM | Claude Sonnet 4 / Llama 70B (via AI Gateway) |
| Vector DB | ChromaDB (persistent, multi-collection) |
| Embeddings | BAAI/bge-small-en-v1.5 |
| Knowledge Graph | NetworkX (pickle-cached) |
| Database | SQLite (logging) |
| Web UI | Gradio |
| Solver Integration | PyDyna (ansys.dyna.core) |
| Pre-processor | ANSA Python API |
| Post-processor | META Python API |
| HPC | PBS Pro / SLURM |
| CI/CD | GitHub Actions |
| Containerization | Docker + docker-compose |
| Language | Python 3.9+ |
