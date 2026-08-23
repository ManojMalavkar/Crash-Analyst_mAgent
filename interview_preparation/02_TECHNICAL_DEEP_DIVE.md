# Technical Deep Dive

## Technology Stack

| Layer | Technology | Why This Choice |
|-------|-----------|----------------|
| LLM | Claude Sonnet 4 / Llama 70B | Sonnet for accuracy, Llama as fallback (free tier) |
| LLM Gateway | Databricks AI Gateway | Corporate proxy, rate limiting, model switching |
| Vector DB | ChromaDB (persistent, cosine) | Zero-config, runs offline on HPC, no cloud dependency |
| Embeddings | BAAI/bge-small-en-v1.5 (384d) | Best speed/quality for domain-specific retrieval |
| Knowledge Graph | NetworkX (pickle-cached) | Lightweight, no Neo4j server, instant load |
| Logging | SQLite (WAL) + JSONL | Structured queries + human-readable audit |
| Web UI | Gradio | Quick prototyping, compatible with HPC nodes |
| Config | Pydantic BaseSettings + .env | Type-safe, environment-aware |

---

## RAG Pipeline — Detailed Breakdown

### Stage 1: Document Ingestion (`ingest.py`)

**Input:** 96 HTML files from Sphinx-generated API documentation

**Processing:**
- Parses HTML with BeautifulSoup4
- Extracts structured records: symbol, signature, description, docstring, notes, deprecation status
- Auto-detects software (ansa/meta/sdm) from folder path
- Filters noise directories (`_static`, `_images`, `.doctrees`)
- Skips deprecated APIs (1,520 filtered)

**Output:** 3 JSONL files:
- `coderag_documents.jsonl` — 13,597 records (one per function/class/attribute)
- `kg_nodes.jsonl` — 13,722 graph nodes
- `kg_edges.jsonl` — 28,728 graph edges

**Record schema:**
```json
{
    "symbol": "ansa.base.CollectEntities",
    "type": "function",
    "module": "ansa.base",
    "software": "ansa",
    "signature": "CollectEntities(deck, type_name, search_set=None, ...)",
    "description": "Collects all entities of a given type...",
    "docstring": "Full documentation text...",
    "notes": "Example code if available...",
    "deprecated": false
}
```

### Stage 2: Vector Store (`build_vector_db.py`)

**Embedding model:** `BAAI/bge-small-en-v1.5`
- 384 dimensions
- Cosine similarity
- Batch size: 64
- Normalized embeddings

**Embed text construction:**
```python
def _build_embed_text(record):
    parts = [record['symbol']]
    if record.get('signature'):
        parts.append(record['signature'])
    parts.append(record.get('description', ''))
    return '\n'.join(parts)
```

**ChromaDB configuration:**
- Persistent client (survives restarts)
- HNSW index with cosine distance
- Collection: `"api"` (unified across all software)
- Metadata stored: symbol, module, type, software

### Stage 3: Knowledge Graph (`kg_retriever.py`)

**Graph structure (NetworkX DiGraph):**
- **Nodes:** Functions, classes, modules, attributes
- **Edges:** belongs_to, has_method, inherits_from, related_to
- **Serialization:** Pickle for fast load (~100ms for 28K edges)

**Use cases:**
- "What methods does Entity have?" → traverse has_method edges
- "What inherits from Base?" → traverse inherits_from edges
- "Related functions to MergeNodes" → 1-hop neighbors

---

## Agent Architecture (`agent.py`)

### Tool-Calling Loop

```python
while True:
    response = llm.chat(messages, tools=tools)
    
    if response.has_tool_calls:
        for tool_call in response.tool_calls:
            result = execute_tool(tool_call)  # search_api, get_details, etc.
            messages.append(tool_result)
    else:
        return response.content  # Final answer
```

**Key design:** Agent can call multiple tools per turn. Example:
1. `search_api("mesh creation")` → finds `ansa.mesh.Create`
2. `get_function_details("Create")` → gets full signature + docstring
3. `get_class_hierarchy("Entity")` → gets base class methods
4. Final answer with complete code

### 5 Agent Tools

| Tool | Purpose | Database |
|------|---------|----------|
| `search_api` | Semantic search over all APIs | ChromaDB `api` |
| `search_code_examples` | Find functions with code examples | ChromaDB `api` (filtered) |
| `get_function_details` | Exact lookup by name | ChromaDB `api` (exact match) |
| `get_class_hierarchy` | Inheritance chain + methods | NetworkX KG |
| `search_knowledge_graph` | Graph traversal + related functions | NetworkX KG |

---

## LLM Client (`shared/llm_client.py`)

### Model Fallback Chain

```
Primary: databricks-claude-sonnet-4
    ↓ (on failure)
Fallback 1: databricks-meta-llama-3-3-70b-instruct
    ↓ (on failure)
Fallback 2: databricks-meta-llama-3-3-70b-instruct
```

### Retry Logic

- Retryable: `RateLimitError`, `APITimeoutError`, `APIConnectionError`, 5xx errors
- Non-retryable: 4xx (bad request), auth errors
- Exponential backoff with jitter
- Max 3 retries per model, then fallback

---

## Logging Framework (`shared/log_db.py` + `shared/logger.py`)

### Dual-Write Architecture

```
Agent Activity → Logger → SQLite DB (structured, queryable)
                      → JSONL file (human-readable, append-only)
```

### SQLite Schema (WAL mode)

- `sessions` → user sessions
- `conversations` → chat threads within a session
- `requests` → individual LLM API calls
- `tool_calls` → tool execution records
- `feedback` → user ratings

---

## Configuration Management (`shared/config.py`)

**Pydantic BaseSettings pattern:**

```python
class Settings(BaseSettings):
    class Config:
        env_file = ".env"
    
    class LLM(BaseModel):
        primary_model: str = "databricks-claude-sonnet-4"
        fallback_1: str = "databricks-meta-llama-3-3-70b-instruct"
    
    class Embedding(BaseModel):
        model_name: str = "BAAI/bge-small-en-v1.5"
        batch_size: int = 64
```

---

## Evaluation Framework (`evaluation/`)

### Two-Level Evaluation

**Level 1: Retrieval Quality (no LLM needed)**
- MRR (Mean Reciprocal Rank)
- nDCG (Normalized Discounted Cumulative Gain)
- Keyword coverage

**Level 2: Answer Quality (LLM-as-Judge)**
- Accuracy (1-5)
- Completeness (1-5)
- Relevance (1-5)

### Improvement Loop

```
Baseline → Diagnose → Tune (1 variable) → Rebuild → Re-evaluate → Accept/Reject
```

Tunable variables (priority order):
1. Embed text template (biggest impact)
2. Embedding model
3. top_k
4. Distance function
5. Metadata filtering

---

## Session Commands (`session_tools.py`)

**Separate from API docs.** Handles META GUI workflow commands:
- Menu navigation paths ("Results > Contour > vonMises")
- Toolbar button sequences
- Parameterized command templates

**Own collection:** `meta_session_commands` (25 commands)
**Own search tool:** Used for "how do I do X in META GUI" questions

---

## Key Design Patterns Used

1. **Singleton lazy-loading** — Vector store and KG loaded once on first use
2. **Tool-calling agent loop** — LLM decides which tools to call iteratively
3. **Dual-write logging** — Structured (SQLite) + append-only (JSONL)
4. **Model fallback chain** — Graceful degradation across LLM providers
5. **Configuration via environment** — Same code, different .env per deployment
6. **JSONL intermediate format** — Decouples ingestion from embedding
7. **Absolute path resolution** — All paths relative to script location, not CWD
