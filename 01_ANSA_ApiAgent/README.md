# 01_ANSA_ApiAgent — CodeRAG Agent

AI agent that generates ANSA and META Python code using **Retrieval-Augmented Generation** with hybrid search (vector + knowledge graph) over 23K+ API symbols.

---

## Quick Start (After Cloning)

### Prerequisites

- Python 3.10+
- ANSA installed (any version: 24.x, 2025.x, 2026.x)

### 1. Install dependencies

```bash
cd Crash-Analyst_mAgent
pip install -r requirements.txt
```

### 2. Build the knowledge base (one command)

```bash
python 01_ANSA_ApiAgent/bin/build_vector_db.py --ansa-path "<YOUR_ANSA_INSTALL_DIR>"
```

**Examples:**

```bash
# Windows — typical ANSA installation
python 01_ANSA_ApiAgent/bin/build_vector_db.py --ansa-path "C:\BETA_CAE_Systems\ANSA_v2025.2.2"

# Linux — typical ANSA installation
python 01_ANSA_ApiAgent/bin/build_vector_db.py --ansa-path /opt/BETA_CAE_Systems/ansa_v2025.2.2

# Custom docs directory (if you extracted docs separately)
python 01_ANSA_ApiAgent/bin/build_vector_db.py --source /path/to/your/docs
```

### 3. That's it. Start the agent

```bash
python 01_ANSA_ApiAgent/app_gradio.py
# Opens at http://localhost:7860
```

---

## Where Does the Script Find Documentation?

The ingestion pipeline automatically scans these locations inside your ANSA installation:

```
<ANSA_INSTALL_DIR>/
├── python/
│   ├── doc/                    # HTML API reference
│   │   ├── ansa/
│   │   │   ├── base.html
│   │   │   ├── mesh.html
│   │   │   ├── connections.html
│   │   │   └── ...
│   │   └── meta/
│   │       ├── post.html
│   │       └── ...
│   └── lib/                    # Python source stubs
│       ├── ansa/
│       │   ├── base.py
│       │   ├── mesh.py
│       │   └── ...
│       └── meta/
│           └── ...
├── scripts/                    # Example scripts (.py)
│   ├── mesh_example.py
│   └── ...
└── documentation/              # PDF/HTML user guides (optional)
    └── scripting_guide.html
```

The script recursively finds all `.py`, `.html`, `.json`, `.jsonl`, and `.md` files — no manual copying needed.

---

## What Happens Behind the Scenes

```
┌─────────────────────────────────────────────────────────────────────┐
│  python bin/build_vector_db.py --ansa-path "C:\...\ANSA_v2025.2.2"  │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
         Step 1: DISCOVER     │  Scans ANSA install dir
                              │  Finds: python/doc/, python/lib/, scripts/
                              ▼
         Step 2: INGEST       │  Parses all files:
                              │  • .py  → AST parser (classes, functions, signatures)
                              │  • .html → HTML parser (API reference)
                              │  • .json → JSON parser (structured API data)
                              │  • .md  → Markdown parser (guides, tutorials)
                              ▼
         Step 3: EMBED        │  Encodes documents with sentence-transformers
                              │  Model: BAAI/bge-small-en-v1.5 (384 dims)
                              ▼
         Step 4: STORE        │  Writes to ChromaDB (vector_db/)
                              │  + Builds NetworkX knowledge graph
                              ▼
         ✅ DONE              │  Ready for queries!
                              │  ~23K documents, ~2-5 minutes on first run
└─────────────────────────────┘
```

---

## Architecture

```
User Query: "How to mesh a part with 5mm elements?"
    │
    ▼
┌───────────────────────┐
│   CodeRAG Agent       │  (agent.py)
│   Tool-Calling Loop   │
└───────────┬───────────┘
            │
    ┌───────┼───────┐
    ▼       ▼       ▼
┌──────┐ ┌──────┐ ┌──────┐
│Vector│ │ KG   │ │Direct│
│Search│ │Query │ │Lookup│
└───┬──┘ └──┬───┘ └──┬───┘
    │       │        │
    ▼       ▼        ▼
┌───────────────────────┐
│  ChromaDB  │ NetworkX │
│  (23K+     │ Knowledge│
│  embeddings)│ Graph   │
└───────────────────────┘
    │
    ▼
┌───────────────────────┐
│  Generated Code:      │
│  from ansa import base│
│  base.CreateMesh(...) │
└───────────────────────┘
```

---

## Advanced Usage

### Rebuild from scratch (if you update ANSA version)

```bash
python 01_ANSA_ApiAgent/bin/build_vector_db.py --ansa-path "C:\...\ANSA_v2026.1.0" --rebuild
```

### Ingest only (without building vector DB)

```bash
python 01_ANSA_ApiAgent/bin/ingest.py "C:\BETA_CAE_Systems\ANSA_v2025.2.2\python"
```

### Build knowledge graph separately

```bash
python 01_ANSA_ApiAgent/bin/kg_retriever.py --source "C:\...\ANSA_v2025.2.2\python"
```

### Use META docs (post-processor)

```bash
python 01_ANSA_ApiAgent/bin/build_vector_db.py --ansa-path "C:\...\ANSA_v2025.2.2" --software meta
```

### Custom source directory

If you've extracted or organized docs elsewhere:

```bash
python 01_ANSA_ApiAgent/bin/build_vector_db.py --source /my/custom/docs/folder --rebuild
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
│   └── kg_retriever.py     # Knowledge graph builder (Day 5)
├── knowledge-base/         # (auto-populated after build)
└── vector_db/              # ChromaDB + knowledge_graph.pkl
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Source directory not found` | Check your ANSA path — use quotes if it has spaces |
| `No documents found` | Verify ANSA install has `python/doc/` or `python/lib/` |
| `CUDA out of memory` | Embedding runs on CPU by default — no GPU needed |
| `Permission denied` | Run terminal as admin, or copy docs to a local folder |
| Slow first run (~5 min) | Normal — embedding 23K docs. Subsequent runs are incremental |

---

## Dependencies

```
chromadb>=0.4.22
sentence-transformers>=2.2.0
networkx>=3.1
beautifulsoup4>=4.12.0
```
