# Interview Q&A — Expected Questions & Answers

---

## General Project Questions

### Q: Tell me about your project.

**A:** I built a multi-agent AI platform called SafetyAgent for automotive crash simulation workflows. The core agent uses RAG (Retrieval-Augmented Generation) to generate Python code for ANSA and META — the industry-standard pre/post-processing software. It searches through 13,597 API functions using vector embeddings and a knowledge graph, then generates production-ready code from natural language queries. The system runs on HPC nodes behind a corporate firewall, using Databricks AI Gateway for LLM access.

### Q: Why did you build this?

**A:** CAE engineers spend 30-40% of their time writing repetitive scripts and looking up API documentation. With 13,000+ functions across ANSA, META, and Komvos SDM, finding the right API is a significant bottleneck. This agent reduces that to a conversational interface — ask in natural language, get working code.

### Q: What makes this different from just using ChatGPT?

**A:** Three key differences:
1. **Domain-specific retrieval** — ChatGPT doesn't know ANSA/META APIs. My agent retrieves actual documentation from a curated vector database.
2. **Knowledge graph** — Understands class hierarchies and function relationships, so it can suggest related functions.
3. **Runs on-premise** — Works behind corporate firewalls on HPC nodes, no data leaves the company network.

---

## RAG-Specific Questions

### Q: Explain how RAG works in your system.

**A:** Two phases:

**Offline (build once):**
1. Parse 96 HTML API docs → extract 13,597 structured records (JSONL)
2. Embed each record into a 384-dim vector using BGE-small
3. Store in ChromaDB with metadata (symbol, module, type)
4. Build NetworkX knowledge graph (28K edges)

**Online (per query):**
1. User asks: "how to create shell mesh"
2. Embed the query with same model
3. Cosine similarity search → top-5 most relevant API docs
4. Optionally expand via knowledge graph (related functions)
5. Pass retrieved context + query to LLM
6. LLM generates code using the context

### Q: Why ChromaDB and not FAISS or Pinecone?

**A:**
- **ChromaDB** — Persistent (survives restarts), zero configuration, runs offline on HPC without cloud access, built-in metadata filtering
- **FAISS** — Faster for millions of vectors, but no built-in persistence or metadata filtering. Overkill for 13K docs.
- **Pinecone** — Cloud-only, can't run behind corporate firewall on HPC nodes

### Q: Why BGE-small and not a larger model?

**A:** Speed/quality trade-off:
- 384 dims vs 768 (bge-base) or 1024 (bge-large)
- For 13K documents, retrieval quality difference is <5% MRR but speed is 3x faster
- Runs on CPU without GPU (HPC nodes may not have GPU for embedding)
- We have an evaluation framework to measure this — can switch if needed

### Q: How do you evaluate retrieval quality?

**A:** Three metrics:
1. **MRR (Mean Reciprocal Rank)** — How high does the correct result rank? 1/rank averaged.
2. **nDCG** — Rank-aware quality score that penalizes correct results at low positions.
3. **Keyword coverage** — What % of expected keywords appear in top-K results.

We have 30 ground-truth test queries with expected symbols. Run the eval, see which queries miss, diagnose why, tune one variable, rebuild, re-evaluate.

### Q: What embedding text do you use?

**A:** Currently: `symbol + signature + description`. This is tunable — the evaluation framework measures impact of adding docstring, module prefix, or code examples to the embed text.

### Q: How do you handle the cold start / no relevant results?

**A:** The agent has multiple tools. If `search_api` returns low-confidence results (high distance), it can:
1. Try `search_knowledge_graph` for name-based lookup
2. Try `get_function_details` for exact name match
3. Fall back to the LLM's general knowledge with a disclaimer

---

## Architecture Questions

### Q: Why a tool-calling agent and not a simple RAG chain?

**A:** A simple chain does: retrieve → generate. But real queries need multiple lookups:
- "Create mesh and check quality" → needs 2 separate API searches
- "What methods does Entity have?" → needs knowledge graph, not vector search
- "Show me example code for CollectEntities" → needs filtered search (code examples only)

The tool-calling loop lets the LLM decide which tools to call and in what order, handling complex multi-step queries.

### Q: Explain the model fallback chain.

**A:** Primary: Claude Sonnet 4 (best quality) → Fallback: Llama 70B (open-source, always available).

Retry logic:
- Rate limit, timeout, connection errors → retry with exponential backoff (max 3)
- After 3 failures on primary → switch to fallback model
- 4xx errors (bad request) → don't retry, fail fast

### Q: Why dual logging (SQLite + JSONL)?

**A:**
- **SQLite** — Structured queries: "show me all failed tool calls in the last hour", analytics, dashboards
- **JSONL** — Append-only, never corrupts, human-readable, easy to grep, works as audit trail
- Together: best of both worlds, no single point of failure

### Q: How do you handle the knowledge graph?

**A:** NetworkX DiGraph with 13,722 nodes and 28,728 edges:
- **Nodes** = functions, classes, modules, attributes
- **Edges** = belongs_to, has_method, inherits_from, related_to
- Serialized as pickle for fast load (~100ms)
- Used for: class hierarchy lookup, finding related functions, module exploration

---

## LLM / AI Questions

### Q: How do you connect to the LLM?

**A:** Through **Databricks AI Gateway** — a proxy that:
- Handles authentication (Databricks token)
- Provides model routing (claude, llama, etc.)
- Adds rate limiting and logging
- Works behind corporate firewall (no direct OpenAI/Anthropic access)

The client uses the OpenAI SDK with a custom `base_url` pointing to the gateway.

### Q: How do you handle prompt engineering?

**A:** System prompt defines the agent's role:
- "You are an ANSA/META Python API expert"
- Tool descriptions with clear parameter schemas
- Context injection: retrieved docs are passed as user messages
- Few-shot examples embedded in tool descriptions

### Q: What about hallucination?

**A:** Mitigated by:
1. **RAG** — LLM only generates code from retrieved documentation, not from memory
2. **Tool results** — Agent shows which API docs it found, user can verify
3. **Evaluation** — LLM-as-Judge catches wrong answers with accuracy scoring
4. **Knowledge graph** — Validates that referenced functions actually exist

---

## Challenges & Solutions

### Q: What was the hardest technical challenge?

**A:** Three main ones:

1. **HTML parsing** — Sphinx-generated docs have inconsistent structure across modules. Solved with flexible extractors that handle multiple HTML patterns.

2. **Corporate firewall** — HPC nodes can't reach external APIs. Solved with Databricks AI Gateway as a proxy, and ChromaDB for offline vector search.

3. **Retrieval quality** — Generic embeddings don't understand domain-specific terms ("jacobian", "CONTACT_AUTOMATIC_SURFACE_TO_SURFACE"). Solved with evaluation framework + iterative tuning of embed text template.

### Q: If you had more time, what would you improve?

**A:**
1. **Fine-tune embeddings** on ANSA/META domain vocabulary
2. **Hybrid search** — combine vector search with BM25 keyword search
3. **User feedback loop** — use thumbs up/down to improve retrieval over time
4. **Multi-agent orchestration** — coordinate all 5 agents for end-to-end simulation workflows
5. **Streaming responses** — show code generation token by token

---

## Python / Software Engineering Questions

### Q: What design patterns did you use?

**A:**
1. **Singleton** — Vector store and KG loaded once, shared across requests
2. **Strategy pattern** — Model fallback chain (swap models without changing code)
3. **Builder pattern** — VectorStoreBuilder configures and builds the DB step by step
4. **Registry pattern** — Tools auto-registered from function definitions
5. **Observer pattern** — Dual-write logger observes all agent activity

### Q: How do you handle errors?

**A:** Layered approach:
- **Tool level** — try/except returns JSON `{"error": "..."}`, agent continues
- **LLM level** — Retry with backoff, then fallback model
- **Logger level** — Non-fatal (wrapped in try/except), never crashes the agent
- **Graceful degradation** — If vector DB missing, agent says so; if KG missing, skips hierarchy lookup

### Q: How do you test this system?

**A:**
1. **Unit tests** — `test_tools.py` tests each RAG tool independently
2. **Retrieval evaluation** — 30 ground-truth queries with MRR/nDCG metrics
3. **LLM-as-Judge** — Automated answer quality scoring (accuracy, completeness, relevance)
4. **Integration** — `app_cli.py` for manual end-to-end testing

---

## Behavioral / Situational

### Q: How did you decide on the architecture?

**A:** Started with constraints:
- Must run offline (HPC, no internet) → ChromaDB, not cloud vector DB
- Must handle 13K+ docs → needed vector search, not keyword lookup
- Must support multiple query types → tool-calling agent, not simple chain
- Must be extensible → monorepo with shared components, 5 agents

### Q: How do you measure success?

**A:** Quantitative:
- Retrieval MRR > 0.7 (correct function in top-3 results)
- Keyword coverage > 80%
- LLM judge scores > 3.5/5

Qualitative:
- Engineers can get working code without reading docs
- Reduces script writing time from hours to minutes
