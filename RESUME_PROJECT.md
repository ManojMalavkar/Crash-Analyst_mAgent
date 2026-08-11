# Resume / CV — Project Section

---

## Option 1: Full Project Entry (for CV / Portfolio Page)

---

### CAE Analyst Agent — AI-Powered Crash Simulation Automation Platform
**Personal Project | Python, LLM, RAG, ChromaDB, LS-DYNA, ANSA/META**

*Multi-agent AI platform that automates the complete crash simulation lifecycle — from FE model pre-processing through HPC job orchestration to post-processing — using LLM tool-calling agents with domain-specific retrieval-augmented generation (RAG).*

**Key Contributions:**
- Architected a **5-agent system** (CodeRAG, PyDyna, ModelCheck, Toolbar Plugin, HPC Orchestrator) with shared infrastructure, unified LLM client with model fallback chain, and enterprise-grade logging
- Built a **RAG pipeline over 23,000+ ANSA/META API symbols** using ChromaDB vector search combined with NetworkX knowledge graphs for hybrid retrieval
- Developed an **LS-DYNA keyword manipulation agent** with exact-match lookup for 500+ keyword cards and semantic search for natural language queries via PyDyna SDK
- Implemented **automated CAE model validation** by comparing simulation includes against reference libraries, supporting Euro NCAP 2024 and company-standard checklists with deviation reporting
- Created **native ANSA/META toolbar plugins** embedding AI copilot directly in the engineer's daily tools — zero context switching
- Designed **HPC job orchestration** with PBS/SLURM support, YAML-defined multi-stage pipelines, and automated post-processing triggered on job completion
- Engineered a **production-grade shared module** with OpenAI-compatible LLM client (Databricks AI Gateway + OpenAI), exponential backoff retry, token tracking, and dual-write logging (JSONL + SQLite)

**Tech Stack:** Python 3.9+ | OpenAI API | ChromaDB | NetworkX | Gradio | PyDyna (ansys.dyna.core) | ANSA/META Python API | PBS/SLURM | SQLite | Docker | GitHub Actions

---

## Option 2: Concise Resume Bullets (2-3 lines for space-limited CV)

---

**CAE Analyst Agent** — AI Crash Simulation Platform | *Python, LLM, RAG, LS-DYNA*
- Built multi-agent AI platform automating crash simulation workflows (ANSA → LS-DYNA → META → HPC) using RAG over 23K+ API symbols, reducing manual scripting effort by ~80%
- Engineered LLM tool-calling agents with ChromaDB vector search, knowledge graphs, model fallback chains, and enterprise logging (JSONL + SQLite)
- Developed automated model validation comparing FE includes against reference libraries, supporting Euro NCAP 2024 and company standards

---

## Option 3: LinkedIn Project Description

---

**CAE Analyst Agent** — AI-Powered Crash Simulation Automation

Built a multi-agent AI platform that brings copilot capabilities to crash safety engineers. The system automates the complete simulation lifecycle:

→ Pre-processing (ANSA code generation via RAG over 23K+ API symbols)
→ Solver setup (LS-DYNA keyword manipulation via PyDyna)
→ Quality validation (automated include comparison against reference libraries)
→ HPC orchestration (PBS/SLURM job submission + automated post-processing)
→ Native toolbar integration (embedded AI inside ANSA/META GUI)

Tech: Python | OpenAI/Databricks AI Gateway | ChromaDB | NetworkX | Gradio | Docker

Highlights:
• 5 specialized agents with shared LLM infrastructure and model fallback
• Hybrid retrieval: vector search + knowledge graph for code generation
• Enterprise logging with dual-write (JSONL + SQLite) and usage analytics
• YAML-driven checklists for Euro NCAP 2024 compliance validation
• Multi-stage pipeline orchestration with automated post-processing

---

## Option 4: Interview Talking Points

---

### "Tell me about a project you're proud of"

> "I built a multi-agent AI platform called CAE Analyst Agent that automates crash simulation workflows end-to-end. In automotive safety engineering, analysts spend significant time writing repetitive scripts for mesh generation, setting up LS-DYNA models, validating includes against standards, and managing HPC jobs.
>
> I designed 5 specialized AI agents — each with its own knowledge domain — that communicate through a shared infrastructure layer. The CodeRAG agent uses hybrid retrieval (vector search + knowledge graph) over 23,000 ANSA/META API symbols to generate Python code. The PyDyna agent handles LS-DYNA keyword files. The ModelCheck agent validates model includes against reference libraries for Euro NCAP compliance.
>
> What I'm most proud of is the architecture decisions: a unified LLM client with automatic model fallback (Claude → Llama 70B → Llama 8B), exponential backoff with jitter, token usage tracking, and dual-write logging to both JSONL and SQLite — so you get both grep-able logs and queryable analytics.
>
> The system reduced manual scripting time by approximately 80% in my testing."

### Key Technical Questions & Answers

**Q: Why multi-agent instead of one large agent?**
> Each domain (ANSA API, LS-DYNA keywords, model validation) has unique knowledge retrieval needs. A single agent would require an impossibly large context window. Specialized agents keep retrieval focused and responses accurate.

**Q: How do you handle LLM failures in production?**
> Three layers: (1) exponential backoff with jitter for transient errors, (2) model fallback chain — if Claude is down, automatically switch to Llama 70B then 8B, (3) all failures logged with full context for diagnostics.

**Q: What's the RAG approach?**
> Hybrid retrieval. Vector search via ChromaDB (BGE embeddings) finds semantically similar API functions. Knowledge graph via NetworkX handles structural queries like "what methods does this class have" or "what imports are needed." Combined, retrieval recall is significantly higher than vector-only.

**Q: How do you validate code quality before execution?**
> The ModelCheck agent compares model includes against a reference library — detecting missing, modified, or outdated includes before HPC submission. It uses YAML-driven checklists that can encode Euro NCAP 2024 rules or company-specific standards.

**Q: Why dual-write logging?**
> Different consumers need different formats. Ops teams `grep` and `tail -f` JSONL files for real-time monitoring. The admin analytics agent queries SQLite for reports like "token usage by agent this week" or "which tools are called most often."

---

## Option 5: Skills Tag Cloud (for ATS / Keywords)

---

**Languages:** Python, SQL, Bash, YAML
**AI/ML:** LLM, RAG, Embeddings, Tool-Calling, Agent Architecture, Prompt Engineering
**LLM Infra:** OpenAI API, Databricks AI Gateway, Model Fallback, Token Optimization
**Vector DB:** ChromaDB, Sentence-Transformers, BGE Embeddings
**Knowledge Graph:** NetworkX, Hybrid Retrieval
**CAE/FEA:** ANSA Python API, META Python API, LS-DYNA, PyDyna SDK, Crash Simulation
**HPC:** PBS Pro, SLURM, Job Orchestration, Pipeline Automation
**Standards:** Euro NCAP 2024, Automotive Safety
**Web/UI:** Gradio, REST API, Streaming
**DevOps:** Docker, GitHub Actions, CI/CD, pytest
**Database:** SQLite, JSONL Logging
**Architecture:** Multi-Agent Systems, Microservices, Event-Driven, Plugin Architecture

---

## Option 6: GitHub Profile README Snippet

---

```markdown
### 🚗 CAE Analyst Agent
AI-powered multi-agent platform for crash simulation automation.
5 specialized agents | RAG over 23K+ API symbols | LS-DYNA + ANSA/META

[![Python](https://img.shields.io/badge/Python-3.9+-blue)]() 
[![Agents](https://img.shields.io/badge/Agents-5-orange)]() 
[![API Symbols](https://img.shields.io/badge/API%20Coverage-23K+-green)]()
```
