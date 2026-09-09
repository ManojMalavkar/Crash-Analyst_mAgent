"""Qdrant Vector Store Builder for ANSA/META CodeRAG.

Reads normalized API records from api_normalized.jsonl + examples.jsonl
(produced by ingest.py) and embeds them into Qdrant collections.

Embedding model: OpenAI text-embedding-3-small (1536 dims).
Requires: OPENAI_API_KEY environment variable.

Collections:
  - api_docs:    API documentation chunks (function, method, class, attribute)
  - api_examples: Code example chunks (independent, high-value for generation)
  - api_migration: Deprecated API migration chunks

Chunk Strategy (500-1200 tokens per chunk):
  - Module Chunk:   Module-level overview
  - Class Chunk:    Class + attributes + method list
  - Function Chunk: Full function/method doc + signature + params + returns
  - Example Chunk:  Standalone code examples
  - Migration Chunk: Deprecated API + replacement info

Metadata per vector point:
  - module:      str  (e.g. "meta.annotations")
  - api_name:    str  (e.g. "AnnotationById")
  - api_type:    str  (function|method|class|attribute)
  - return_type: str  (e.g. "Annotation")
  - deprecated:  bool
  - replacement: str | None
  - version:     str | None
  - software:    str  (ansa|meta)
  - class_name:  str | None
  - chunk_type:  str  (module|class|function|method|example|migration)

Pipeline:
  Step 1: python bin/ingest.py <docs_path>         -> api_normalized.jsonl + examples.jsonl
  Step 2: python bin/build_vector_db.py            -> Qdrant collections

Usage:
    # Build all collections
    python bin/build_vector_db.py --rebuild

    # Build from specific JSONL
    python bin/build_vector_db.py --source knowledge-base/api_normalized.jsonl

    # Programmatic
    from bin.build_vector_db import VectorStoreBuilder
    builder = VectorStoreBuilder(qdrant_url="http://localhost:6333")
    builder.build_from_jsonl("knowledge-base/api_normalized.jsonl")
"""

import json
import logging
import time
import uuid
import sys
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct,
        Filter, FieldCondition, MatchValue,
        PayloadSchemaType,
    )
except ImportError:
    QdrantClient = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


logger = logging.getLogger(__name__)


# =============================================================================
# Embedding Model
# =============================================================================

# Dimension lookup for OpenAI embedding models
_OPENAI_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

# text-embedding-3-small max is 8191 tokens; ~4 chars/token is a safe estimate
_MAX_CHARS_PER_INPUT = 28_000


class EmbeddingModel:
    """Wrapper around OpenAI Embeddings API (text-embedding-3-small).

    Features:
        - Automatic retry with exponential backoff on rate limits
        - Input truncation to stay within token limits
        - Batch processing with progress logging

    Requires:
        pip install openai
        export OPENAI_API_KEY=sk-...
    """

    def __init__(self, model_name: str = "text-embedding-3-small", batch_size: int = 256):
        self.model_name = model_name
        self.batch_size = batch_size  # OpenAI supports up to 2048 inputs per call
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if OpenAI is None:
                raise ImportError(
                    "openai required. Install: pip install openai"
                )
            import os
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable is required.\n"
                    "  export OPENAI_API_KEY=sk-..."
                )
            self._client = OpenAI(api_key=api_key)
            logger.info(f"OpenAI embedding model: {self.model_name}")
        return self._client

    @property
    def dimension(self) -> int:
        return _OPENAI_DIMS.get(self.model_name, 1536)

    @staticmethod
    def _truncate(text: str) -> str:
        """Truncate text to stay within OpenAI token limit."""
        if not text or not text.strip():
            return " "
        if len(text) > _MAX_CHARS_PER_INPUT:
            return text[:_MAX_CHARS_PER_INPUT]
        return text

    def _embed_with_retry(self, batch: list[str], max_retries: int = 5):
        """Call OpenAI embeddings API with exponential backoff on rate limits."""
        for attempt in range(max_retries):
            try:
                return self.client.embeddings.create(
                    model=self.model_name,
                    input=batch,
                )
            except Exception as exc:
                exc_name = exc.__class__.__name__
                # Retry on rate limit or transient server errors
                if "RateLimit" in exc_name or "APITimeout" in exc_name or "APIConnectionError" in exc_name:
                    wait = min(2 ** attempt * 2, 60)  # 2s, 4s, 8s, 16s, 32s
                    logger.warning(
                        f"OpenAI {exc_name} (attempt {attempt+1}/{max_retries}). "
                        f"Retrying in {wait}s..."
                    )
                    time.sleep(wait)
                elif hasattr(exc, 'status_code') and exc.status_code in (429, 500, 502, 503):
                    wait = min(2 ** attempt * 2, 60)
                    logger.warning(
                        f"OpenAI HTTP {exc.status_code} (attempt {attempt+1}/{max_retries}). "
                        f"Retrying in {wait}s..."
                    )
                    time.sleep(wait)
                else:
                    raise
        # Final attempt without catching
        return self.client.embeddings.create(
            model=self.model_name,
            input=batch,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts using OpenAI API with retry and truncation."""
        all_embeddings = []
        total = len(texts)
        for i in range(0, total, self.batch_size):
            batch = [self._truncate(t) for t in texts[i:i + self.batch_size]]
            response = self._embed_with_retry(batch)
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            processed = min(i + self.batch_size, total)
            if processed % (self.batch_size * 4) == 0 or processed == total:
                logger.info(f"  Embedded {processed}/{total} documents...")
        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query."""
        response = self._embed_with_retry([self._truncate(query)])
        return response.data[0].embedding


# =============================================================================
# Chunk Builders
# =============================================================================

def build_function_chunk(rec: dict) -> str:
    """Build embed text for a function/method record (500-1200 tokens target)."""
    parts = []

    full_name = rec.get("full_name") or rec.get("name", "")
    module = rec.get("module") or ""
    api_type = rec.get("type") or "function"
    signature = rec.get("signature") or ""
    description = rec.get("description") or ""
    docstring = rec.get("docstring") or ""
    returns = rec.get("returns") or ""
    software = rec.get("software") or ""
    params = rec.get("parameters") or []

    # Header
    parts.append(f"[{software.upper()}] {api_type}: {full_name}")
    if module:
        parts.append(f"Module: {module}")

    # Signature
    if signature:
        parts.append(f"Signature: {signature}")

    # Parameters
    if params:
        param_lines = []
        for p in params:
            pstr = p.get("name", "?")
            if p.get("type"):
                pstr += f": {p['type']}"
            if p.get("optional"):
                pstr += f" = {p.get('default', 'None')}"
            param_lines.append(f"  - {pstr}")
        parts.append("Parameters:\n" + "\n".join(param_lines))

    # Return type
    if returns:
        parts.append(f"Returns: {returns}")

    # Description
    if description:
        parts.append(f"Description: {description}")
    if docstring and docstring != description:
        parts.append(docstring[:600])

    # Deprecation
    if rec.get("deprecated"):
        depr_text = "DEPRECATED"
        if rec.get("deprecated_version"):
            depr_text += f" since {rec['deprecated_version']}"
        if rec.get("replacement"):
            depr_text += f". Use {rec['replacement']} instead."
        parts.append(depr_text)

    return "\n".join(parts)


def build_class_chunk(rec: dict, methods: list[dict] = None) -> str:
    """Build embed text for a class record."""
    parts = []
    full_name = rec.get("full_name") or rec.get("name", "")
    module = rec.get("module") or ""
    software = rec.get("software") or ""

    parts.append(f"[{software.upper()}] class: {full_name}")
    if module:
        parts.append(f"Module: {module}")

    if rec.get("description"):
        parts.append(rec["description"])
    if rec.get("docstring") and rec["docstring"] != rec.get("description"):
        parts.append(rec["docstring"][:400])

    # List method names for discoverability
    if methods:
        method_names = [m.get("name", "") for m in methods[:20]]
        parts.append(f"Methods: {', '.join(method_names)}")

    # Navigation targets
    if rec.get("navigation_targets"):
        parts.append(f"Can navigate to: {', '.join(rec['navigation_targets'])}")

    return "\n".join(parts)


def build_example_chunk(example: dict) -> str:
    """Build embed text for a code example."""
    parts = []
    content = example.get("content", "")
    api_calls = example.get("api_calls", [])
    software = example.get("software", "")

    if software:
        parts.append(f"[{software.upper()}] Code Example")
    if api_calls:
        parts.append(f"APIs used: {', '.join(api_calls[:10])}")

    # Truncate content to fit chunk budget
    if content:
        parts.append(content[:1500])

    return "\n".join(parts)


def build_migration_chunk(rec: dict) -> str:
    """Build embed text for a deprecated API migration chunk."""
    parts = []
    full_name = rec.get("full_name") or rec.get("name", "")
    software = rec.get("software") or ""

    parts.append(f"[{software.upper()}] DEPRECATED API: {full_name}")
    if rec.get("deprecated_version"):
        parts.append(f"Deprecated in version: {rec['deprecated_version']}")
    if rec.get("replacement"):
        parts.append(f"Replacement: {rec['replacement']}")
    if rec.get("description"):
        parts.append(f"Original: {rec['description']}")
    if rec.get("signature"):
        parts.append(f"Old signature: {rec['signature']}")

    return "\n".join(parts)


# =============================================================================
# Vector Store Builder (Qdrant)
# =============================================================================

class VectorStoreBuilder:
    """Builds and manages Qdrant vector collections for ANSA/META CodeRAG."""

    COLLECTION_DOCS = "api_docs"
    COLLECTION_EXAMPLES = "api_examples"
    COLLECTION_MIGRATION = "api_migration"

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        qdrant_api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
        collection_prefix: str = "coderag",
        local_path: Optional[str] = None,
    ):
        """Initialize vector store builder with Qdrant connection.

        Args:
            qdrant_url: Qdrant server URL (ignored if local_path is set)
            qdrant_api_key: Optional API key for Qdrant Cloud
            embedding_model: OpenAI embedding model name (default: text-embedding-3-small)
            collection_prefix: Prefix for collection names
            local_path: Path for local on-disk Qdrant storage (no server needed).
                        If set, uses embedded Qdrant instead of connecting to a server.
        """
        if QdrantClient is None:
            raise ImportError(
                "qdrant-client required. Install: pip install qdrant-client"
            )

        self.prefix = collection_prefix
        self._embedder = EmbeddingModel(model_name=embedding_model)

        if local_path:
            # Embedded mode: persistent storage on disk, no server required
            Path(local_path).mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=local_path)
            logger.info(f"Qdrant local storage: {local_path}")
        else:
            # Try server connection; auto-fallback to local if unreachable
            try:
                client = QdrantClient(
                    url=qdrant_url,
                    api_key=qdrant_api_key,
                    timeout=10,
                )
                client.get_collections()  # connectivity check
                self._client = client
                logger.info(f"Qdrant connected: {qdrant_url}")
            except Exception as exc:
                fallback_dir = str(
                    Path(__file__).resolve().parent.parent / "knowledge-base" / "qdrant_store"
                )
                logger.warning(
                    f"Qdrant server unavailable at {qdrant_url} "
                    f"({exc.__class__.__name__}). "
                    f"Falling back to local storage: {fallback_dir}"
                )
                Path(fallback_dir).mkdir(parents=True, exist_ok=True)
                self._client = QdrantClient(path=fallback_dir)

    def _collection_name(self, suffix: str) -> str:
        return f"{self.prefix}_{suffix}"

    def _ensure_collection(self, name: str, recreate: bool = False):
        """Create collection if it doesn't exist."""
        full_name = self._collection_name(name)
        if recreate:
            try:
                self._client.delete_collection(full_name)
                logger.info(f"Deleted collection: {full_name}")
            except Exception:
                pass

        collections = [c.name for c in self._client.get_collections().collections]
        if full_name not in collections:
            self._client.create_collection(
                collection_name=full_name,
                vectors_config=VectorParams(
                    size=self._embedder.dimension,
                    distance=Distance.COSINE,
                ),
            )
            # Create payload indexes for filtering
            for field_name in ["module", "api_type", "software", "deprecated", "chunk_type"]:
                self._client.create_payload_index(
                    collection_name=full_name,
                    field_name=field_name,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            logger.info(f"Created collection: {full_name}")

    def _upsert_points(
        self,
        collection_suffix: str,
        texts: list[str],
        payloads: list[dict],
    ) -> int:
        """Embed texts and upsert into Qdrant collection."""
        if not texts:
            return 0

        full_name = self._collection_name(collection_suffix)
        embeddings = self._embedder.embed_documents(texts)

        points = []
        seen_ids = set()
        for i, (embedding, payload) in enumerate(zip(embeddings, payloads)):
            # Build a unique seed: collection + full_name + index as fallback
            # This prevents UUID collisions when multiple records share full_name
            seed = f"{collection_suffix}::{payload.get('full_name', '')}::{i}"
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, seed))
            # Extra safety: detect collision and append counter
            if point_id in seen_ids:
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{seed}::dup{i}"))
            seen_ids.add(point_id)
            points.append(PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload,
            ))

        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self._client.upsert(
                collection_name=full_name,
                points=batch,
            )
            if i > 0 and i % 500 == 0:
                logger.info(f"  Upserted {i}/{len(points)} points...")

        return len(points)

    # ------------------------------------------------------------------
    # Build from JSONL
    # ------------------------------------------------------------------

    def build_from_jsonl(
        self,
        api_jsonl: str | Path,
        examples_jsonl: Optional[str | Path] = None,
        rebuild: bool = False,
    ) -> dict:
        """Build all Qdrant collections from JSONL files.

        Args:
            api_jsonl: Path to api_normalized.jsonl
            examples_jsonl: Path to examples.jsonl (optional)
            rebuild: If True, recreate all collections

        Returns:
            Build statistics dict
        """
        api_jsonl = Path(api_jsonl)
        if not api_jsonl.exists():
            raise FileNotFoundError(f"API JSONL not found: {api_jsonl}")

        start_time = time.time()

        # Load records
        records = []
        with open(api_jsonl, "r", encoding="utf-8") as fp:
            for line in fp:
                if line.strip():
                    records.append(json.loads(line))
        logger.info(f"Loaded {len(records)} API records from {api_jsonl}")

        # Load examples
        examples = []
        if examples_jsonl:
            examples_jsonl = Path(examples_jsonl)
            if examples_jsonl.exists():
                with open(examples_jsonl, "r", encoding="utf-8") as fp:
                    for line in fp:
                        if line.strip():
                            examples.append(json.loads(line))
                logger.info(f"Loaded {len(examples)} examples from {examples_jsonl}")

        # Ensure collections
        self._ensure_collection("docs", recreate=rebuild)
        self._ensure_collection("examples", recreate=rebuild)
        self._ensure_collection("migration", recreate=rebuild)

        # --- Collection 1: API Documentation ---
        logger.info("Building api_docs collection...")
        doc_texts = []
        doc_payloads = []

        # Group by class for class chunks
        class_methods = {}
        for rec in records:
            if rec.get("class_name") and rec.get("type") in ("method", "attribute"):
                cls_key = f"{rec.get('module', '')}.{rec['class_name']}"
                class_methods.setdefault(cls_key, []).append(rec)

        for rec in records:
            api_type = rec.get("type", "function")

            if api_type == "class":
                # Class chunk with method list
                methods = class_methods.get(rec.get("full_name", ""), [])
                text = build_class_chunk(rec, methods)
                chunk_type = "class"
            else:
                text = build_function_chunk(rec)
                chunk_type = api_type if api_type in ("method", "attribute") else "function"

            payload = {
                "full_name": rec.get("full_name") or "",
                "name": rec.get("name") or "",
                "module": rec.get("module") or "",
                "api_type": api_type,
                "return_type": rec.get("returns") or "",
                "deprecated": str(rec.get("deprecated") or False),
                "replacement": rec.get("replacement") or "",
                "version": rec.get("version") or "",
                "software": rec.get("software") or "",
                "class_name": rec.get("class_name") or "",
                "chunk_type": chunk_type,
                "signature": rec.get("signature") or "",
                "description": (rec.get("description") or "")[:200],
            }

            doc_texts.append(text)
            doc_payloads.append(payload)

        docs_count = self._upsert_points("docs", doc_texts, doc_payloads)

        # --- Collection 2: Examples ---
        logger.info("Building api_examples collection...")
        ex_texts = []
        ex_payloads = []

        # Examples from dedicated example files
        for ex in examples:
            text = build_example_chunk(ex)
            payload = {
                "full_name": f"EXAMPLE::{Path(ex.get('file') or 'unknown').stem}",
                "name": Path(ex.get("file") or "unknown").stem,
                "module": "",
                "api_type": "example",
                "software": ex.get("software") or "",
                "chunk_type": "example",
                "api_calls": ",".join((ex.get("api_calls") or [])[:10]),
            }
            ex_texts.append(text)
            ex_payloads.append(payload)

        # Inline examples from API records
        for rec in records:
            for i, example_code in enumerate(rec.get("examples") or []):
                if len(example_code) > 20:
                    sw = (rec.get("software") or "").upper()
                    fn = rec.get("full_name") or ""
                    text = f"[{sw}] Example for {fn}:\n{example_code}"
                    payload = {
                        "full_name": f"{fn}::example_{i}",
                        "name": rec.get("name") or "",
                        "module": rec.get("module") or "",
                        "api_type": "example",
                        "software": rec.get("software") or "",
                        "chunk_type": "example",
                        "parent_api": fn,
                    }
                    ex_texts.append(text)
                    ex_payloads.append(payload)

        examples_count = self._upsert_points("examples", ex_texts, ex_payloads)

        # --- Collection 3: Migration (deprecated APIs) ---
        logger.info("Building api_migration collection...")
        mig_texts = []
        mig_payloads = []

        for rec in records:
            if rec.get("deprecated"):
                text = build_migration_chunk(rec)
                payload = {
                    "full_name": rec.get("full_name") or "",
                    "name": rec.get("name") or "",
                    "module": rec.get("module") or "",
                    "api_type": rec.get("type") or "function",
                    "deprecated": "True",
                    "deprecated_version": rec.get("deprecated_version") or "",
                    "replacement": rec.get("replacement") or "",
                    "software": rec.get("software") or "",
                    "chunk_type": "migration",
                }
                mig_texts.append(text)
                mig_payloads.append(payload)

        migration_count = self._upsert_points("migration", mig_texts, mig_payloads)

        duration = time.time() - start_time

        stats = {
            "docs_indexed": docs_count,
            "examples_indexed": examples_count,
            "migration_indexed": migration_count,
            "total_points": docs_count + examples_count + migration_count,
            "duration_seconds": round(duration, 1),
        }

        logger.info(
            f"Build complete: {stats['total_points']} total points in {stats['duration_seconds']}s"
        )
        return stats

    # ------------------------------------------------------------------
    # Search Methods
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        collection: str = "docs",
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """Search a Qdrant collection.

        Args:
            query: Natural language query
            collection: Collection suffix (docs|examples|migration)
            top_k: Number of results
            filters: Optional metadata filters {field: value}

        Returns:
            List of result dicts with payload and score
        """
        full_name = self._collection_name(collection)
        query_vector = self._embedder.embed_query(query)

        # Build Qdrant filter
        qdrant_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            qdrant_filter = Filter(must=conditions)

        results = self._client.search(
            collection_name=full_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
        )

        return [
            {
                "id": str(r.id),
                "score": r.score,
                "payload": r.payload,
            }
            for r in results
        ]

    def search_docs(self, query: str, top_k: int = 5, **filters) -> list[dict]:
        """Search API documentation."""
        return self.search(query, collection="docs", top_k=top_k, filters=filters or None)

    def search_examples(self, query: str, top_k: int = 5, **filters) -> list[dict]:
        """Search code examples."""
        return self.search(query, collection="examples", top_k=top_k, filters=filters or None)

    def search_migration(self, query: str, top_k: int = 5, **filters) -> list[dict]:
        """Search deprecated API migration info."""
        return self.search(query, collection="migration", top_k=top_k, filters=filters or None)

    def get_stats(self) -> dict:
        """Get stats for all collections."""
        stats = {}
        for suffix in ["docs", "examples", "migration"]:
            full_name = self._collection_name(suffix)
            try:
                info = self._client.get_collection(full_name)
                stats[full_name] = {
                    "points_count": info.points_count,
                    "vectors_count": info.vectors_count,
                }
            except Exception:
                stats[full_name] = {"status": "not_found"}
        return stats

# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    import os

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Build ANSA/META Qdrant vector database from normalized JSONL",
        epilog="""
Examples:
  # Local mode (no Qdrant server needed, stores in knowledge-base/qdrant_store/):
  python bin/build_vector_db.py --local --rebuild

  # Local mode with custom path:
  python bin/build_vector_db.py --local /path/to/qdrant_data --rebuild

  # Full rebuild with Qdrant server:
  python bin/build_vector_db.py --rebuild

  # With Qdrant Cloud:
  python bin/build_vector_db.py --qdrant-url https://xxx.cloud.qdrant.io --api-key YOUR_KEY
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    kb_dir = Path(__file__).resolve().parent.parent / "knowledge-base"
    default_source = str(kb_dir / "api_normalized.jsonl")
    default_examples = str(kb_dir / "examples.jsonl")

    parser.add_argument("--source", default=default_source,
                        help=f"Path to api_normalized.jsonl (default: {default_source})")
    parser.add_argument("--examples", default=default_examples,
                        help=f"Path to examples.jsonl (default: {default_examples})")
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"),
                        help="Qdrant server URL (env: QDRANT_URL)")
    parser.add_argument("--api-key", default=os.getenv("QDRANT_API_KEY"),
                        help="Qdrant API key (env: QDRANT_API_KEY)")
    parser.add_argument("--embedding-model", default="text-embedding-3-small",
                        help="OpenAI embedding model (default: text-embedding-3-small)")
    parser.add_argument("--prefix", default="coderag",
                        help="Collection name prefix")
    parser.add_argument("--rebuild", action="store_true",
                        help="Rebuild all collections from scratch")

    default_local = str(kb_dir / "qdrant_store")
    parser.add_argument("--local", nargs="?", const=default_local, default=None,
                        metavar="PATH",
                        help=f"Use local on-disk Qdrant (no server needed). "
                             f"Optional path (default: {default_local})")
    args = parser.parse_args()

    source_path = Path(args.source)
    if not source_path.exists():
        print(f"\n  ERROR: Source file not found: {source_path}")
        print(f"  Run ingest.py first: python bin/ingest.py /path/to/docs")
        sys.exit(1)

    examples_path = Path(args.examples) if Path(args.examples).exists() else None
    storage_mode = f"local: {args.local}" if args.local else f"server: {args.qdrant_url}"

    print(f"\n{'='*60}")
    print(f"  Building Qdrant Vector Database")
    print(f"  Source:    {source_path}")
    print(f"  Examples:  {examples_path or 'N/A'}")
    print(f"  Qdrant:    {storage_mode}")
    print(f"  Model:     {args.embedding_model}")
    print(f"  Rebuild:   {args.rebuild}")
    print(f"{'='*60}\n")

    builder = VectorStoreBuilder(
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.api_key,
        embedding_model=args.embedding_model,
        collection_prefix=args.prefix,
        local_path=args.local,
    )

    build_stats = builder.build_from_jsonl(
        api_jsonl=source_path,
        examples_jsonl=examples_path,
        rebuild=args.rebuild,
    )

    print(f"\n{'='*60}")
    print(f"  BUILD COMPLETE")
    print(f"{'='*60}")
    for k, v in build_stats.items():
        print(f"  {k:25s}: {v}")

    # Quick test search
    if build_stats["total_points"] > 0:
        print(f"\n  --- Test Search ---")
        test_queries = ["create annotation", "mesh generation", "deprecated functions"]
        for query in test_queries:
            results = builder.search_docs(query, top_k=2)
            print(f"\n  Query: '{query}'")
            for r in results:
                name = r['payload'].get('full_name', 'unknown')
                score = r['score']
                print(f"    [{score:.3f}] {name} ({r['payload'].get('api_type', '?')})")

    print(f"\n{'='*60}")
    print(f"  Collections ready. Use builder.search_docs/search_examples/search_migration")
    print(f"{'='*60}\n")
