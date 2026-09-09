"""ChromaDB Vector Store Builder for PyDyna Keyword Knowledge Base.

Reads keyword_reference.jsonl and pydyna_docs.jsonl (produced by
ingest_keywords.py) and embeds them into persistent ChromaDB collections.

Collections:
  - pydyna_keywords : LS-DYNA keyword cards with parameters (*MAT_024, etc.)
  - pydyna_api      : PyDyna Python API docs (classes, methods, properties)

Embedding model: BAAI/bge-small-en-v1.5 (384 dims, from shared/config.py).

Pipeline:
  Step 1: python bin/ingest_keywords.py --source <pydyna_auto_dir>
  Step 2: python bin/build_vector_db.py --rebuild

Usage:
    python bin/build_vector_db.py --rebuild
    python bin/build_vector_db.py --stats
    python bin/build_vector_db.py --source knowledge-base/keyword_reference.jsonl
"""

import json
import hashlib
import logging
import time
import sys
import argparse
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.config import settings


logger = logging.getLogger(__name__)

_AGENT_DIR = Path(__file__).resolve().parent.parent          # 02_PyDyna_Agent/
_KB_DIR = _AGENT_DIR / "knowledge-base"
_VECTOR_DB_DIR = _AGENT_DIR / "vector_db"

# Collection names
KEYWORD_COLLECTION = "pydyna_keywords"
API_COLLECTION = "pydyna_api"


# =============================================================================
# Embedding Model Wrapper
# =============================================================================

class EmbeddingModel:
    """Wrapper around sentence-transformers (same pattern as Agent 01)."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.embedding.model_name
        self.batch_size = settings.embedding.batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            if SentenceTransformer is None:
                raise ImportError(
                    "sentence-transformers required. "
                    "Install: pip install sentence-transformers"
                )
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts in batches."""
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            embeddings = self.model.encode(
                batch, show_progress_bar=False, normalize_embeddings=True,
            )
            all_embeddings.extend(embeddings.tolist())
            if i > 0 and i % (self.batch_size * 10) == 0:
                logger.info(f"  Embedded {i}/{len(texts)} documents...")
        return all_embeddings


# =============================================================================
# Embed-Text Builders (what goes into the vector)
# =============================================================================

def _build_keyword_embed_text(record: dict) -> str:
    """Build rich embedding text for a keyword record.

    Combines keyword name, category, description, and field names
    to enable both exact and semantic search.
    """
    parts = [record["keyword"]]                              # *MAT_024
    parts.append(record.get("class_name", ""))                # Mat024
    parts.append(f"category: {record.get('category', '')}")

    desc = record.get("description", "")
    if desc:
        parts.append(desc)

    # Add field names with descriptions for semantic matching
    for param in record.get("params", [])[:20]:  # cap at 20 fields
        p_text = param["name"]
        if param.get("description"):
            p_text += f" ({param['description']})"
        parts.append(p_text)

    # Add link fields (references to curves, sets, etc.)
    for field_name, link_type in record.get("link_fields", {}).items():
        parts.append(f"{field_name} references {link_type}")

    return " | ".join(parts)


def _build_api_embed_text(record: dict) -> str:
    """Build embedding text for a PyDyna API record."""
    parts = [record.get("symbol", "")]                       # Mat024
    parts.append(f"keyword: {record.get('keyword_ref', '')}")
    parts.append(f"module: {record.get('module', '')}")
    parts.append(f"category: {record.get('category', '')}")

    desc = record.get("description", "")
    if desc:
        parts.append(desc)

    docstring = record.get("docstring", "")
    if docstring and docstring != desc:
        parts.append(docstring)

    # Property names as searchable terms
    methods = record.get("methods", [])[:15]
    if methods:
        parts.append(f"properties: {', '.join(methods)}")

    return " | ".join(parts)


def _doc_id(text: str) -> str:
    """Deterministic document ID from content hash."""
    return hashlib.md5(text.encode()).hexdigest()


# =============================================================================
# Vector Store Builder
# =============================================================================

class VectorStoreBuilder:
    """Manages ChromaDB collections for PyDyna keyword knowledge base."""

    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = persist_dir or str(_VECTOR_DB_DIR)
        self._client = None
        self._embedder = None

    @property
    def client(self) -> chromadb.ClientAPI:
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            logger.info(f"ChromaDB opened at {self.persist_dir}")
        return self._client

    @property
    def embedder(self) -> EmbeddingModel:
        if self._embedder is None:
            self._embedder = EmbeddingModel()
        return self._embedder

    # ----- Collection Management -----

    def get_or_create_collection(self, name: str) -> chromadb.Collection:
        """Get or create a collection with cosine distance."""
        return self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def delete_collection(self, name: str) -> None:
        """Delete a collection if it exists."""
        try:
            self.client.delete_collection(name)
            logger.info(f"Deleted collection: {name}")
        except Exception:
            pass

    # ----- Build from JSONL -----

    def build_keywords(
        self, source: Optional[Path] = None, rebuild: bool = False,
    ) -> int:
        """Build the pydyna_keywords collection from keyword_reference.jsonl."""
        source = source or (_KB_DIR / "keyword_reference.jsonl")
        if not source.exists():
            logger.error(f"Source not found: {source}")
            return 0

        if rebuild:
            self.delete_collection(KEYWORD_COLLECTION)

        collection = self.get_or_create_collection(KEYWORD_COLLECTION)
        existing = collection.count()

        records = self._load_jsonl(source)
        if not records:
            return 0

        # Filter out already-indexed docs
        if existing > 0 and not rebuild:
            logger.info(f"Collection has {existing} docs. Running incremental.")

        # Build embed texts + metadata
        ids, texts, metadatas = [], [], []
        for rec in records:
            embed_text = _build_keyword_embed_text(rec)
            doc_id = _doc_id(embed_text)
            ids.append(doc_id)
            texts.append(embed_text)
            metadatas.append({
                "keyword": rec["keyword"],
                "class_name": rec.get("class_name", ""),
                "category": rec.get("category", ""),
                "param_count": rec.get("param_count", 0),
                "card_count": rec.get("card_count", 0),
                "module": rec.get("module", ""),
                "has_links": bool(rec.get("link_fields")),
                "source": "keyword_reference",
            })

        return self._upsert_batch(collection, ids, texts, metadatas)

    def build_api(
        self, source: Optional[Path] = None, rebuild: bool = False,
    ) -> int:
        """Build the pydyna_api collection from pydyna_docs.jsonl."""
        source = source or (_KB_DIR / "pydyna_docs.jsonl")
        if not source.exists():
            logger.error(f"Source not found: {source}")
            return 0

        if rebuild:
            self.delete_collection(API_COLLECTION)

        collection = self.get_or_create_collection(API_COLLECTION)

        records = self._load_jsonl(source)
        if not records:
            return 0

        ids, texts, metadatas = [], [], []
        for rec in records:
            embed_text = _build_api_embed_text(rec)
            doc_id = _doc_id(embed_text)
            ids.append(doc_id)
            texts.append(embed_text)
            metadatas.append({
                "symbol": rec.get("symbol", ""),
                "type": rec.get("type", ""),
                "module": rec.get("module", ""),
                "category": rec.get("category", ""),
                "keyword_ref": rec.get("keyword_ref", ""),
                "signature": rec.get("signature", ""),
                "source": "pydyna_docs",
            })

        return self._upsert_batch(collection, ids, texts, metadatas)

    def build_all(self, rebuild: bool = False) -> dict:
        """Build both collections."""
        t0 = time.time()
        kw_count = self.build_keywords(rebuild=rebuild)
        api_count = self.build_api(rebuild=rebuild)
        elapsed = time.time() - t0
        logger.info(
            f"Build complete: {kw_count} keywords + {api_count} API docs "
            f"in {elapsed:.1f}s"
        )
        return {"keywords": kw_count, "api": api_count, "elapsed": elapsed}

    # ----- Search -----

    def search(
        self,
        query: str,
        collection_name: str = KEYWORD_COLLECTION,
        top_k: int = 5,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None,
    ) -> list[dict]:
        """Search a collection by semantic similarity."""
        try:
            collection = self.client.get_collection(collection_name)
        except Exception:
            logger.error(f"Collection {collection_name} not found.")
            return []

        embeddings = self.embedder.embed_documents([query])

        kwargs = {
            "query_embeddings": embeddings,
            "n_results": min(top_k, 20),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        if where_document:
            kwargs["where_document"] = where_document

        results = collection.query(**kwargs)

        formatted = []
        for i in range(len(results["ids"][0])):
            formatted.append({
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1.0 - results["distances"][0][i],  # cosine → similarity
            })
        return formatted

    # ----- Stats -----

    def get_stats(self) -> dict:
        """Return collection statistics."""
        stats = {}
        for name in [KEYWORD_COLLECTION, API_COLLECTION]:
            try:
                col = self.client.get_collection(name)
                stats[name] = {"count": col.count()}
            except Exception:
                stats[name] = {"count": 0, "status": "not found"}
        return stats

    # ----- Internal -----

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict]:
        """Load records from a JSONL file."""
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        logger.info(f"Loaded {len(records)} records from {path.name}")
        return records

    def _upsert_batch(
        self,
        collection: chromadb.Collection,
        ids: list[str],
        texts: list[str],
        metadatas: list[dict],
    ) -> int:
        """Embed and upsert documents in batches."""
        batch_size = self.embedder.batch_size
        total = len(ids)
        count = 0

        logger.info(f"Embedding {total} documents into '{collection.name}'...")
        t0 = time.time()

        for i in range(0, total, batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_texts = texts[i:i + batch_size]
            batch_meta = metadatas[i:i + batch_size]

            embeddings = self.embedder.embed_documents(batch_texts)

            collection.upsert(
                ids=batch_ids,
                embeddings=embeddings,
                documents=batch_texts,
                metadatas=batch_meta,
            )
            count += len(batch_ids)

            if i > 0 and i % (batch_size * 5) == 0:
                logger.info(f"  Upserted {count}/{total}...")

        elapsed = time.time() - t0
        logger.info(
            f"  Done: {count} docs in '{collection.name}' ({elapsed:.1f}s)"
        )
        return count


# =============================================================================
# CLI
# =============================================================================

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Build ChromaDB vector collections for PyDyna keyword KB"
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Delete and rebuild collections from scratch",
    )
    parser.add_argument(
        "--source", type=Path, default=None,
        help="Build from a specific JSONL file (auto-detects keyword vs API)",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Print collection statistics and exit",
    )
    parser.add_argument(
        "--persist-dir", type=str, default=None,
        help="ChromaDB persist directory (default: 02_PyDyna_Agent/vector_db/)",
    )
    args = parser.parse_args()

    builder = VectorStoreBuilder(persist_dir=args.persist_dir)

    if args.stats:
        stats = builder.get_stats()
        print(f"\n{'='*50}")
        print(f"  PyDyna Vector DB Statistics")
        print(f"{'='*50}")
        for name, info in stats.items():
            print(f"  {name:<25} {info.get('count', 0):>6} docs")
        print(f"{'='*50}")
        return

    if args.source:
        # Auto-detect: keyword_reference → keywords, pydyna_docs → api
        if "keyword" in args.source.name:
            count = builder.build_keywords(source=args.source, rebuild=args.rebuild)
            print(f"Built {count} keyword docs")
        else:
            count = builder.build_api(source=args.source, rebuild=args.rebuild)
            print(f"Built {count} API docs")
    else:
        result = builder.build_all(rebuild=args.rebuild)
        print(f"\n{'='*50}")
        print(f"  Build Complete")
        print(f"{'='*50}")
        print(f"  pydyna_keywords : {result['keywords']:>6} docs")
        print(f"  pydyna_api      : {result['api']:>6} docs")
        print(f"  Elapsed         : {result['elapsed']:>6.1f}s")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
