# Concepts & Glossary

Everything you need to understand to explain this project confidently.

---

## Core AI/ML Concepts

### RAG (Retrieval-Augmented Generation)

**What:** Instead of relying on LLM's training data (which doesn't know your private APIs), you retrieve relevant documents first, then give them to the LLM as context.

**Why:** LLMs hallucinate when asked about domain-specific APIs they weren't trained on. RAG grounds the response in actual documentation.

**How (in our project):**
```
Query → Embed → Vector Search → Top-K docs → LLM(context + query) → Answer
```

---

### Vector Embeddings

**What:** Converting text into a fixed-size array of numbers (vector) that captures semantic meaning.

**Example:**
- "create mesh" → [0.23, -0.15, 0.87, ...] (384 numbers)
- "generate mesh elements" → [0.21, -0.14, 0.85, ...] (similar vector = similar meaning)

**In our project:** We use `BAAI/bge-small-en-v1.5` model that produces 384-dimensional vectors.

---

### Cosine Similarity

**What:** Measures how similar two vectors are. Range: -1 (opposite) to 1 (identical).

**Formula:** cos(θ) = (A · B) / (|A| × |B|)

**In our project:** ChromaDB uses cosine distance (1 - cosine_similarity) to find the most similar API docs to a user's query.

---

### ChromaDB

**What:** An open-source vector database. Stores documents + their embeddings + metadata.

**Key features we use:**
- **Persistent storage** — Survives process restarts
- **Metadata filtering** — Filter by software ("ansa"/"meta")
- **HNSW index** — Approximate nearest neighbor for fast search

**Alternative to:** FAISS (no metadata), Pinecone (cloud-only), Weaviate (heavy)

---

### Knowledge Graph

**What:** A graph where nodes = entities and edges = relationships between them.

**In our project:**
- Nodes: `ansa.base.CollectEntities` (function), `Entity` (class), `ansa.base` (module)
- Edges: `CollectEntities` --belongs_to--> `ansa.base`, `Entity` --has_method--> `get_property`

**Why needed:** Vector search finds semantically similar docs, but doesn't understand structure. KG answers: "What methods does Entity have?" or "What class does this inherit from?"

---

### Tool-Calling Agent

**What:** An LLM that can decide to call external functions (tools) and use their results.

**How it works:**
```python
# LLM sees tool definitions:
tools = [
    {"name": "search_api", "parameters": {"query": "string", "top_k": "int"}},
    {"name": "get_function_details", "parameters": {"function_name": "string"}},
]

# LLM decides: "I need to search for mesh creation functions"
# Returns: {"tool_call": {"name": "search_api", "arguments": {"query": "create mesh"}}}
# System executes the tool, returns results to LLM
# LLM generates final answer using the results
```

**vs Simple RAG:** Simple RAG always does one search. Agent can do multiple searches, different tools, iteratively.

---

### LLM-as-Judge

**What:** Using one LLM to evaluate the output of another LLM.

**In our project:** After the agent generates code, we ask a judge LLM:
- "Is this code accurate compared to the reference?" → Score 1-5
- "Is it complete?" → Score 1-5
- "Is it relevant?" → Score 1-5

**Why:** Manual evaluation doesn't scale. LLM judges correlate well with human evaluation.

---

## Retrieval Metrics

### MRR (Mean Reciprocal Rank)

**What:** Average of 1/rank of the first correct result.

**Example:**
- Query 1: correct at rank 1 → 1/1 = 1.0
- Query 2: correct at rank 3 → 1/3 = 0.33
- Query 3: correct at rank 2 → 1/2 = 0.5
- MRR = (1.0 + 0.33 + 0.5) / 3 = 0.61

**Target:** MRR > 0.7 means correct result is usually in top-2.

---

### nDCG (Normalized Discounted Cumulative Gain)

**What:** Measures ranking quality, penalizing correct results that appear too low.

**Intuition:** A correct result at rank 1 is much better than at rank 10. nDCG captures this with logarithmic discounting.

**Formula:** DCG = Σ relevance_i / log2(rank_i + 1)

**Target:** nDCG > 0.7

---

### Keyword Coverage

**What:** % of expected keywords that appear anywhere in the top-K retrieved documents.

**Example:** Expected: ["CollectEntities", "ansa.base"] → Both found in top-5 → Coverage = 100%

---

## Infrastructure Concepts

### Databricks AI Gateway

**What:** A managed proxy for LLM APIs. Sits between your code and OpenAI/Anthropic/Meta.

**Why we use it:**
- Corporate firewall blocks direct access to api.openai.com
- Provides authentication via Databricks token
- Rate limiting and usage tracking built-in
- Can switch models without code changes

**How:** Set `base_url` in OpenAI client to gateway URL instead of api.openai.com.

---

### HPC (High-Performance Computing)

**What:** Cluster of powerful compute nodes for running simulations.

**Relevant context:**
- Our agent runs ON HPC nodes (no internet access)
- Simulations run via PBS Pro / SLURM job schedulers
- ChromaDB chosen specifically because it works offline

---

### ANSA / META / LS-DYNA

| Tool | Role | Analogy |
|------|------|--------|
| **ANSA** | Pre-processor (build FE model) | Like SolidWorks but for FE meshes |
| **LS-DYNA** | Solver (run crash simulation) | The "calculator" that solves physics |
| **META** | Post-processor (analyze results) | Like Excel for simulation results |

**Python APIs:** Both ANSA and META expose Python APIs (13,000+ functions) for automation.

---

## Software Engineering Concepts

### Pydantic BaseSettings

**What:** Type-safe configuration that loads from `.env` files and environment variables.

**Why:** No more `os.environ.get("KEY", "default")` scattered everywhere. One place for all config, validated on startup.

---

### WAL Mode (SQLite)

**What:** Write-Ahead Logging — allows concurrent reads while writing.

**Why:** Without WAL, SQLite locks the entire database during writes. With WAL, the agent can log and query simultaneously.

---

### JSONL Format

**What:** One JSON object per line. Not a JSON array.

```jsonl
{"symbol": "CreateMesh", "type": "function"}
{"symbol": "Entity", "type": "class"}
```

**Why:** Append-friendly (no need to parse entire file), streamable, debuggable with `head -5 file.jsonl`.

---

### Monorepo

**What:** All 5 agents in one Git repository with shared code.

**Why:** Shared utilities (LLM client, logging, config) used by all agents. One PR, one CI pipeline, consistent versions.

---

## Abbreviations

| Abbreviation | Full Form |
|-------------|-----------|
| RAG | Retrieval-Augmented Generation |
| LLM | Large Language Model |
| MRR | Mean Reciprocal Rank |
| nDCG | Normalized Discounted Cumulative Gain |
| KG | Knowledge Graph |
| HPC | High-Performance Computing |
| CAE | Computer-Aided Engineering |
| FE | Finite Element |
| SDK | Software Development Kit |
| HNSW | Hierarchical Navigable Small World (index algorithm) |
| WAL | Write-Ahead Logging |
| JSONL | JSON Lines |
| BGE | BAAI General Embedding |
