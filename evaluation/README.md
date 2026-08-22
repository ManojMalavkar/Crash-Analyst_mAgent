# Evaluation — RAG Retrieval Improvement

Iterative evaluation framework for optimizing ANSA/META CodeRAG retrieval quality.

---

## File Structure

```
evaluation/
├── README.md               ← This file (improvement flow guide)
├── rag_eval.py             ← Main evaluator (retrieval + LLM-as-judge)
├── embedding_eval.py       ← Embedding model comparison (standalone)
├── test_tools.py           ← Test individual RAG tools (debug)
├── tests_api.jsonl         ← Ground truth: ANSA/META Python API (30 queries)
└── tests_session.jsonl     ← Ground truth: META session commands (25 queries)
```

---

## Two Databases, Two Test Files

| Test File | Collection | Records | What it evaluates |
|-----------|-----------|---------|-------------------|
| `tests_api.jsonl` | `api` | 13,597 | Python API retrieval (functions, classes, modules) |
| `tests_session.jsonl` | `meta_session_commands` | 25 | META GUI session commands (menus, tools, workflows) |

---

## Metrics

| Metric | What it measures | Target |
|--------|-----------------|--------|
| **MRR** (Mean Reciprocal Rank) | How high does the correct result rank? | > 0.7 |
| **nDCG** (Normalized Discounted Cumulative Gain) | Rank-aware quality score | > 0.7 |
| **Keyword Coverage** | % of expected keywords found in top-K | > 80% |
| **Accuracy** (LLM judge) | Is the generated code correct? (1-5) | > 3.5 |
| **Completeness** (LLM judge) | Does it cover all aspects? (1-5) | > 3.5 |
| **Relevance** (LLM judge) | Is it on-point, no extra? (1-5) | > 4.0 |

---

## Iterative Improvement Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                RETRIEVAL IMPROVEMENT CYCLE                       │
│                                                                 │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌─────────┐ │
│   │ BASELINE │───>│ DIAGNOSE │───>│   TUNE   │───>│ REBUILD │ │
│   └──────────┘    └──────────┘    └──────────┘    └────┬────┘ │
│        ▲                                                │      │
│        │           ┌──────────┐    ┌──────────┐         │      │
│        └───────────│  ACCEPT  │<───│RE-EVALUATE│<────────┘      │
│                    └──────────┘    └──────────┘                 │
└─────────────────────────────────────────────────────────────────┘
```

### Step 1: BASELINE — Measure current performance

```bash
# Verify DB connection first
python evaluation/test_tools.py --tool db

# Run retrieval evaluation against API database
python evaluation/rag_eval.py \
    --eval-file evaluation/tests_api.jsonl \
    --collection api

# Run retrieval evaluation against session database
python evaluation/rag_eval.py \
    --eval-file evaluation/tests_session.jsonl \
    --collection meta_session_commands
```

**Record baseline:** MRR = ?, nDCG = ?, Coverage = ?%

---

### Step 2: DIAGNOSE — Find what's failing

```bash
# Run a single failing test to see details
python evaluation/rag_eval.py \
    --eval-file evaluation/tests_api.jsonl \
    --test 5
```

**Output tells you:**
- What was retrieved (top-5 symbols + distances)
- What was expected (keywords from ground truth)
- WHY it missed (wrong module? wrong function? synonym mismatch?)

**Common failure patterns:**

| Pattern | Root Cause | Fix |
|---------|-----------|-----|
| Correct module, wrong function | Embed text too generic | Add signature to embed text |
| Completely unrelated results | Embedding model weak on code | Try bge-base or bge-large |
| Right function at rank 6-10 | top_k too low | Increase top_k |
| Synonym mismatch ("delete" vs "remove") | Model doesn't understand domain | Add more context to embed text |
| META results when ANSA expected | No software filtering | Add metadata filter in search |

---

### Step 3: TUNE — Change ONE variable

**Rule: Change ONE variable at a time. Never two.**

#### Variable A: Embed Text Template (Biggest Impact)

Edit `01_ANSA_ApiAgent/bin/build_vector_db.py` → `_build_embed_text()` method:

```python
# Current (minimal):
def _build_embed_text(record):
    return f"{record['symbol']}\n{record['description']}"

# Try: Add signature
def _build_embed_text(record):
    parts = [record['symbol']]
    if record.get('signature'):
        parts.append(record['signature'])
    parts.append(record.get('description', ''))
    return '\n'.join(parts)

# Try: Add module prefix + docstring
def _build_embed_text(record):
    parts = []
    if record.get('module'):
        parts.append(f"Module: {record['module']}")
    parts.append(record.get('symbol', ''))
    if record.get('signature'):
        parts.append(record['signature'])
    parts.append(record.get('description', ''))
    if record.get('docstring'):
        parts.append(record['docstring'][:200])
    return '\n'.join(parts)
```

#### Variable B: Embedding Model

Edit `shared/config.py` → `embedding.model_name`:

| Model | Dim | Speed | Quality | Best for |
|-------|-----|-------|---------|----------|
| `BAAI/bge-small-en-v1.5` | 384 | Fast | Good | Quick iteration |
| `BAAI/bge-base-en-v1.5` | 768 | Medium | Better | Production candidate |
| `BAAI/bge-large-en-v1.5` | 1024 | Slow | Best | If accuracy matters most |
| `nomic-ai/nomic-embed-text-v1.5` | 768 | Medium | Good for code | Code-heavy docs |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | Fastest | Baseline | Quick sanity check |

#### Variable C: top_k (Retrieval Window)

Edit `01_ANSA_ApiAgent/bin/tool_functions.py` → `search_api()` default:

| top_k | Trade-off |
|-------|-----------|
| 3 | Fast, but may miss relevant results |
| 5 | Default — good balance |
| 10 | More context for LLM, but more noise |
| 15 | Only if MRR is good but Hit@5 is low |

#### Variable D: Distance Function

Edit `01_ANSA_ApiAgent/bin/build_vector_db.py` → collection creation:

```python
# In build_from_jsonl():
collection = self._client.get_or_create_collection(
    name=self.collection_name,
    metadata={"hnsw:space": "cosine"},  # Try: "l2" or "ip"
)
```

#### Variable E: Metadata Filtering

Add software filter in `tool_functions.py` → `search_api()`:

```python
# Filter by software when user specifies
where_filter = None
if software:  # "ansa" or "meta"
    where_filter = {"software": software}

results = collection.query(
    query_embeddings=...,
    n_results=top_k,
    where=where_filter,  # Only search within that software's docs
)
```

---

### Step 4: REBUILD — Apply the change

```bash
# Rebuild vector DB with new configuration
python bin/build_vector_db.py --rebuild

# Rebuild knowledge graph (if embed text changed)
python bin/kg_retriever.py
```

---

### Step 5: RE-EVALUATE — Measure improvement

```bash
# Same command as Step 1
python evaluation/rag_eval.py \
    --eval-file evaluation/tests_api.jsonl \
    --collection api
```

**Compare:**
- MRR improved? → Keep the change
- MRR same or worse? → Revert, try next variable

---

### Step 6: ACCEPT or ITERATE

| Condition | Action |
|-----------|--------|
| MRR improved | Keep change. Go to Step 3 with next variable |
| MRR same/worse | Revert. Go to Step 3 with different variable |
| MRR > 0.7 AND Coverage > 80% | Move to Step 7 (answer quality) |
| Stuck after all variables | Add more ground truth tests, review embed text |

---

### Step 7: ANSWER QUALITY — End-to-end validation

Only run after retrieval is good (MRR > 0.7):

```bash
# Full evaluation with LLM-as-Judge
python evaluation/rag_eval.py \
    --eval-file evaluation/tests_api.jsonl \
    --collection api \
    --full
```

**This evaluates:**
1. Retrieval (MRR, nDCG) — same as before
2. Code generation — does the agent produce correct code?
3. LLM Judge scores — accuracy, completeness, relevance (1-5)

---

## Priority Order for Tuning

```
1. Embed text template     ← Biggest impact, free, fast to test
2. Embedding model         ← Medium impact, one-time rebuild cost
3. top_k value             ← Small impact, no rebuild needed
4. Distance function       ← Small impact, rebuild needed
5. Metadata filtering      ← Situational, no rebuild needed
6. Add more test queries   ← Improves evaluation coverage
```

---

## Quick Reference Commands

```bash
# Test tools independently
python evaluation/test_tools.py --tool db
python evaluation/test_tools.py --tool search_api --query "change curve color"
python evaluation/test_tools.py --tool kg --query "Entity"

# Evaluate API retrieval
python evaluation/rag_eval.py --eval-file evaluation/tests_api.jsonl --collection api

# Evaluate session retrieval
python evaluation/rag_eval.py --eval-file evaluation/tests_session.jsonl --collection meta_session_commands

# Debug single test
python evaluation/rag_eval.py --eval-file evaluation/tests_api.jsonl --test 0

# Full eval (retrieval + LLM judge)
python evaluation/rag_eval.py --eval-file evaluation/tests_api.jsonl --collection api --full

# Compare embedding models
python evaluation/embedding_eval.py --models bge-small bge-base bge-large

# Rebuild after changes
python bin/build_vector_db.py --rebuild
python bin/kg_retriever.py
```

---

## Tracking Results

Keep a log of your experiments:

| # | Variable Changed | Value | MRR | nDCG | Coverage | Notes |
|---|-----------------|-------|-----|------|----------|-------|
| 0 | Baseline | bge-small, symbol+desc | ? | ? | ? | Initial |
| 1 | Embed text | +signature | ? | ? | ? | |
| 2 | Embed text | +signature +docstring | ? | ? | ? | |
| 3 | Model | bge-base | ? | ? | ? | |
| 4 | top_k | 10 | ? | ? | ? | |

---

## Adding Ground Truth Tests

The more tests you add to `tests_api.jsonl` and `tests_session.jsonl`, the more reliable your evaluation becomes.

**Format (JSONL — one JSON per line):**

```json
{"question": "your natural language query", "keywords": ["expected", "symbols", "in", "results"], "reference_answer": "ideal code answer", "category": "mesh"}
```

**Good test queries:**
- Real questions you'd ask the agent
- Mix of semantic ("how to mesh a part") and exact ("ansa.base.CollectEntities")
- Cover all categories: mesh, entity, connections, io, meta_results, meta_plot

**Target:** 50+ tests per file for reliable metrics.
