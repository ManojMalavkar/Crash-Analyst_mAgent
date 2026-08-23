# Project Overview — SafetyAgent (CAE Analyst Agent)

## What Is This Project?

An **AI-powered multi-agent platform** that automates crash simulation workflows in the automotive industry. It uses LLMs (Large Language Models) combined with RAG (Retrieval-Augmented Generation) to help CAE engineers write code, run simulations, and analyze results — replacing hours of manual scripting with natural language conversations.

**One-liner:** "I built a multi-agent AI system that generates Python code for crash simulation pre/post-processing by retrieving relevant API documentation from a vector database."

---

## Problem Statement

CAE (Computer-Aided Engineering) engineers in automotive crash simulation spend significant time:
- Writing repetitive Python scripts for ANSA (pre-processor) and META (post-processor)
- Looking up API documentation across 13,000+ functions
- Setting up LS-DYNA solver simulations manually
- Managing HPC (High-Performance Computing) job submissions

**Solution:** An AI agent that understands the entire ANSA/META/LS-DYNA API and generates production-ready code from natural language instructions.

---

## Architecture — 5 Agents (Monorepo)

```
SafetyAgent/
├── shared/                    # Common: LLM client, logging, config
├── 01_ANSA_ApiAgent/          # ANSA/META code generation (CodeRAG)
├── 02_PyDyna_Agent/           # LS-DYNA solver via PyDyna SDK
├── 03_ModelCheck_Agent/       # CAE include file validation
├── 04_Toolbar_Plugin/         # ANSA/META GUI toolbar integration
└── 05_HPC_Orchestrator/       # PBS/SLURM job management
```

### Agent 1: ANSA/META CodeRAG Agent (Completed)

The core agent. Uses RAG + Knowledge Graph to generate Python code:

```
User: "change color of all curves from blue to red in META"
  ↓
[1] Embed query → Vector Search (ChromaDB)
  ↓
[2] Retrieve top-5 relevant API docs
  ↓
[3] Knowledge Graph lookup (class hierarchy, related functions)
  ↓
[4] LLM generates code with retrieved context
  ↓
Agent: "from meta import plot2d\ncurves = plot2d.GetCurves()..."
```

---

## How RAG Works in This Project

### Ingestion Pipeline (Offline, Run Once)

```
HTML API Docs (96 files) → Parse & Extract → JSONL (13,597 records)
                                                    ↓
                                        ChromaDB Vector DB (embeddings)
                                                    ↓
                                        NetworkX Knowledge Graph (28K edges)
```

1. **`ingest.py`** — Parses HTML API documentation, extracts functions/classes/attributes with signatures, descriptions, docstrings, deprecation status
2. **`build_vector_db.py`** — Embeds each record into a 384-dim vector (BGE-small) and stores in ChromaDB
3. **`kg_retriever.py`** — Builds a knowledge graph of relationships (inheritance, module membership, function-of-class)

### Retrieval Pipeline (Online, Per Query)

```
User Query → Embedding → Cosine Similarity Search → Top-K docs
                                                        ↓
                                              Knowledge Graph expansion
                                                        ↓
                                              LLM prompt with context
```

---

## Key Technical Decisions

| Decision | Reasoning |
|----------|----------|
| **Single `api` collection** (not separate ansa/meta) | Unified search across all 3 APIs; user doesn't need to specify software |
| **ChromaDB** (not FAISS/Pinecone) | Persistent, zero-config, runs on HPC nodes without cloud |
| **BGE-small-en-v1.5** embedding | Best speed/quality trade-off for 13K docs; 384 dims = fast |
| **NetworkX** knowledge graph | Lightweight, pickle-serializable, no Neo4j dependency |
| **JSONL intermediate format** | Inspectable, debuggable, decouples ingestion from embedding |
| **Tool-calling loop** (not single-shot) | Agent can call multiple tools iteratively for complex queries |
| **SQLite WAL + JSONL dual logging** | Structured queries + human-readable audit trail |

---

## Data Flow

```
┌─────────────────────────────────────────────────────┐
│                    RAW DATA                          │
├─────────────────────────────────────────────────────┤
│ api_ref_ansa/generated/   (29 HTML files)           │
│ api_ref_meta/generated/   (47 HTML files)           │
│ api_ref_komvos/generated/ (20 HTML files)           │
└─────────────────────────┬───────────────────────────┘
                          │ ingest.py
                          ▼
┌─────────────────────────────────────────────────────┐
│              KNOWLEDGE BASE (JSONL)                  │
├─────────────────────────────────────────────────────┤
│ coderag_documents.jsonl  (13,597 records)            │
│ kg_nodes.jsonl           (13,722 nodes)              │
│ kg_edges.jsonl           (28,728 edges)              │
│ session_commands.json    (25 commands)               │
└─────────────────────────┬───────────────────────────┘
                          │ build_vector_db.py / kg_retriever.py
                          ▼
┌─────────────────────────────────────────────────────┐
│                 VECTOR DB                            │
├─────────────────────────────────────────────────────┤
│ ChromaDB collection: "api" (13,597 embeddings)      │
│ ChromaDB collection: "meta_session_commands" (25)    │
│ Knowledge Graph: knowledge_graph.pkl (NetworkX)      │
└─────────────────────────┬───────────────────────────┘
                          │ tool_functions.py
                          ▼
┌─────────────────────────────────────────────────────┐
│                   AGENT                              │
├─────────────────────────────────────────────────────┤
│ 5 Tools: search_api, search_code_examples,          │
│          get_function_details, get_class_hierarchy,  │
│          search_knowledge_graph                      │
│ LLM: Claude Sonnet 4 / Llama 70B                    │
│ UI: Gradio (web) + CLI (terminal)                   │
└─────────────────────────────────────────────────────┘
```

---

## Numbers to Remember

| Metric | Value |
|--------|-------|
| Total API functions/classes documented | 13,597 |
| Deprecated (filtered out) | 1,520 |
| Knowledge Graph nodes | 13,722 |
| Knowledge Graph edges | 28,728 |
| Embedding dimension | 384 |
| Embedding model | BAAI/bge-small-en-v1.5 |
| HTML source files | 96 (29 ANSA + 47 META + 20 Komvos/SDM) |
| Session commands | 25 |
| Agent tools | 5 |
| LLM fallback chain | Claude Sonnet → Llama 70B |
