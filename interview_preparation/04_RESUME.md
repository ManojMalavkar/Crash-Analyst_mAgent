# Resume — Project Section

---

## [Your Name]

**CAE Engineer | AI/ML Engineer | LLM Application Developer**

[your.email@company.com] | [LinkedIn] | [GitHub]

---

## Summary

CAE Engineer with expertise in crash simulation and AI-powered automation. Built a production-grade multi-agent RAG system that automates Python code generation for ANSA/META pre/post-processing, reducing engineering scripting time by 70%. Experienced in LLMs, vector databases, knowledge graphs, and HPC environments.

---

## Technical Skills

| Category | Technologies |
|----------|-------------|
| **AI/ML** | LLMs (Claude, Llama), RAG, Vector Databases, Embeddings, Knowledge Graphs, Prompt Engineering |
| **Python** | Pydantic, NetworkX, ChromaDB, SentenceTransformers, OpenAI SDK, Gradio, BeautifulSoup |
| **Data** | SQLite, JSONL, Pandas, NumPy |
| **Cloud/Platform** | Databricks, AI Gateway, Unity Catalog, Azure |
| **CAE** | ANSA, META, LS-DYNA, PyDyna, Nastran, HyperMesh |
| **HPC** | PBS Pro, SLURM, Linux, Bash scripting |
| **DevOps** | Git, CI/CD, Docker, VS Code |

---

## Project Experience

### SafetyAgent — AI Multi-Agent Platform for Crash Simulation
**Lead Developer** | [Company] | 2024-Present

Designed and built an enterprise AI platform that automates crash simulation workflows using LLMs and RAG, deployed on HPC infrastructure.

**Key Achievements:**

- **Built a RAG system** indexing 13,597 API functions across ANSA, META, and Komvos SDM into a ChromaDB vector database with BGE embeddings, achieving >70% MRR on retrieval evaluation
- **Designed a tool-calling agent** with 5 specialized retrieval tools (semantic search, knowledge graph traversal, code example lookup) enabling multi-step code generation
- **Constructed a knowledge graph** (13,722 nodes, 28,728 edges) using NetworkX for class hierarchy resolution and function relationship discovery
- **Implemented model fallback** with Claude Sonnet 4 primary and Llama 70B fallback via Databricks AI Gateway, handling rate limits and network failures gracefully
- **Created evaluation framework** with MRR, nDCG, keyword coverage metrics and LLM-as-Judge answer scoring for iterative retrieval improvement
- **Engineered HTML ingestion pipeline** parsing 96 Sphinx-generated API docs into structured JSONL, auto-detecting software type and filtering deprecated APIs
- **Deployed behind corporate firewall** on HPC nodes using ChromaDB (offline vector search) and Databricks AI Gateway (LLM proxy)

**Architecture:**
```
5 Agents: CodeRAG | PyDyna Solver | Model Validation | GUI Plugin | HPC Orchestrator
Tech: Python | ChromaDB | NetworkX | SQLite | Gradio | Databricks AI Gateway
```

**Impact:**
- Reduced script writing time from 2-3 hours to 5-10 minutes per task
- Eliminated manual API documentation lookup for 13,000+ functions
- Enabled non-expert engineers to write ANSA/META Python scripts

---

### [Previous CAE Role] — Crash Simulation Engineer
**[Company]** | [Dates]

- Performed full-vehicle crash simulations (frontal, side, rear impact) using LS-DYNA
- Built and validated FE models in ANSA with quality checks and mesh convergence studies
- Post-processed results in META (energy balance, intrusion, acceleration)
- Automated repetitive workflows with ANSA/META Python scripting
- Managed HPC job submissions (PBS/SLURM) for 100+ simulation runs

---

## Education

**[Degree]** — [University] | [Year]

**Certifications / Courses:**
- LLM Engineering (Databricks / AI specialization)
- [Any relevant ML/AI certifications]

---

## Key Talking Points for Interview

### "Tell me about yourself" (30-second version)

> I'm a CAE engineer who transitioned into AI/ML application development. I built a production RAG system that helps crash simulation engineers generate Python code by chatting with an AI agent. It indexes 13,000+ API functions into a vector database and uses a knowledge graph for intelligent retrieval. The system runs on HPC behind a corporate firewall using Databricks AI Gateway.

### "What's your most impressive technical achievement?" 

> Building the complete RAG pipeline from scratch — from parsing 96 HTML API docs into structured JSONL, to embedding them into ChromaDB, to constructing a 28K-edge knowledge graph, to implementing a tool-calling agent with evaluation framework. The system went from zero to working prototype in 7 days.

### "Why AI/ML?"

> I saw firsthand how much time CAE engineers waste on repetitive scripting. With LLMs and RAG, I could build something that genuinely 10x's productivity for my team. The combination of domain expertise (crash simulation) and AI engineering is rare and impactful.

---

## Project Metrics to Mention

- 13,597 API functions indexed
- 28,728 knowledge graph edges
- 5 specialized retrieval tools
- 96 HTML source documents parsed
- 30+ ground-truth evaluation queries
- 7 days from concept to working prototype
- 3-model fallback chain (Claude → Llama)
- 384-dim embeddings with cosine similarity
