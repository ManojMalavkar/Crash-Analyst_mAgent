# 01_ANSA_ApiAgent — CodeRAG Agent

AI agent that generates ANSA and META Python code using **Retrieval-Augmented Generation** with hybrid search (vector + knowledge graph) over 23K+ API symbols.

---

## Architecture

```
User Query
    │
    ▼
┌───────────────────────┐
│   CodeRAG Agent       │  (agent.py — Day 4)
│   Tool-Calling Loop   │
└───────────┬───────────┘
            │
    ┌───────┼───────┐
    ▼       ▼       ▼
┌──────┐ ┌──────┐ ┌──────┐
│Vector│ │ KG   │ │Direct│
│Search│ │Query │ │Lookup│
└───┬──┘ └──┬───┘ └──┬───┘
    │       │       │
    ▼       ▼       ▼
┌───────────────────────┐
│  ChromaDB  │ NetworkX  │
│  (23K+     │ Knowledge │
│  embeddings)│ Graph    │
└───────────────────────┘
```

---

## Ingestion Pipeline

### Step 1: Source Documentation

Place your ANSA/META API documentation in `knowledge-base/raw/`:

```
knowledge-base/raw/
├── ansa_api_reference.html     # ANSA HTML docs
├── meta_api_reference.html     # META HTML docs
├── ansa_functions.jsonl        # Structured API data
├── ansa_source/                # Python source files
│   ├── base.py
│   ├── mesh.py
│   └── ...
└── tutorials/                  # Markdown guides
    ├── meshing_guide.md
    └── scripting_intro.md
```

Supported formats: `.py`, `.json`, `.jsonl`, `.html`, `.md`

### Step 2: Build Vector Database

```bash
cd 01_ANSA_ApiAgent

# Full rebuild
python bin/build_vector_db.py --source knowledge-base/raw --rebuild

# Incremental update (only new/changed docs)
python bin/build_vector_db.py --source knowledge-base/raw

# Build for META specifically
python bin/build_vector_db.py --source knowledge-base/raw --software meta
```

### Step 3: Build Knowledge Graph

```bash
python bin/kg_retriever.py --source knowledge-base/raw --output vector_db/knowledge_graph.pkl
```

---

## Retrieval Strategy

| Query Type | Tool | Example |
|-----------|------|--------|
| Semantic search | Vector DB | "how to create shell mesh" |
| Exact function lookup | Knowledge Graph | "get methods of Entity class" |
| Class hierarchy | Knowledge Graph | "what does ShellElement inherit from" |
| Code examples | Vector DB (filtered) | "example of quality check script" |
| Module contents | Knowledge Graph | "what's in ansa.base module" |

### Hybrid Retrieval Flow

1. **Vector search** finds top-k semantically similar documents
2. **Knowledge graph** enriches results with:
   - Required imports for found functions
   - Related methods in the same class
   - Inheritance chain for context
3. **LLM** synthesizes all retrieved context into working code

---

## File Structure

```
01_ANSA_ApiAgent/
├── __init__.py
├── app_gradio.py           # Gradio web UI (Day 7)
├── bin/
│   ├── __init__.py
│   ├── agent.py            # CodeRAG agent with tool loop (Day 4)
│   ├── tools.py            # Auto tool spec generator (Day 4)
│   ├── tool_functions.py   # 5 RAG tools (Day 4)
│   ├── ingest.py           # Source parser pipeline (Day 5)
│   ├── build_vector_db.py  # ChromaDB builder (Day 5)
│   └── kg_retriever.py     # Knowledge graph (Day 5)
├── knowledge-base/
│   └── raw/                # Source documentation
└── vector_db/              # ChromaDB + graph pickle
```

---

## Dependencies

```
chromadb>=0.4.22
sentence-transformers>=2.2.0
networkx>=3.1
```
