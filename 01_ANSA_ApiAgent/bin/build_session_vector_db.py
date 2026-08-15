"""Session Command Vector DB Builder.

Builds a separate ChromaDB collection for META session commands,
enabling semantic search over session syntax for the agent.

Data source: knowledge-base/session_commands.json
- Each entry has: command, syntax, description, parameters, examples, category
- Embeddings use the same BGE model as the main API vector store

Usage:
    python build_session_vector_db.py --source knowledge-base/session_commands.json --rebuild
    python build_session_vector_db.py --source knowledge-base/session_commands.json
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from shared.config import settings


logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

DEFAULT_COLLECTION = "meta_session_commands"
DEFAULT_PERSIST_DIR = "vector_db"


# =============================================================================
# Session Vector Store
# =============================================================================

class SessionVectorStore:
    """ChromaDB collection for META session commands."""
    
    def __init__(
        self,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION,
    ):
        """Initialize session vector store.
        
        Args:
            persist_dir: ChromaDB persistence directory
            collection_name: Name of the session commands collection
        """
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self._model = None
        
        # Initialize ChromaDB client
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    
    @property
    def model(self):
        """Lazy-load the embedding model."""
        if self._model is None:
            if SentenceTransformer is None:
                raise ImportError("sentence-transformers required. pip install sentence-transformers")
            model_name = settings.embedding.model_name
            self._model = SentenceTransformer(model_name)
            logger.info(f"Loaded embedding model: {model_name}")
        return self._model
    
    def build(self, source_path: str, rebuild: bool = False) -> dict:
        """Build the session commands collection from JSON source.
        
        Args:
            source_path: Path to session_commands.json
            rebuild: If True, delete existing collection first
        
        Returns:
            Build statistics dict
        """
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source not found: {source}")
        
        # Load commands
        with open(source) as f:
            commands = json.load(f)
        
        if not isinstance(commands, list):
            raise ValueError("Expected JSON array of command objects")
        
        logger.info(f"Loaded {len(commands)} session commands from {source}")
        
        # Get or create collection
        if rebuild:
            try:
                self.client.delete_collection(self.collection_name)
                logger.info(f"Deleted existing collection: {self.collection_name}")
            except Exception:
                pass
        
        collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        
        # Skip existing documents (incremental)
        existing_ids = set(collection.get()["ids"]) if not rebuild else set()
        
        # Prepare documents for embedding
        docs_to_add = []
        for cmd in commands:
            doc_id = f"ses_{cmd.get('command', 'unknown')}_{cmd.get('category', '')}"
            
            if doc_id in existing_ids:
                continue
            
            # Build embedding text (combine relevant fields)
            embed_text = self._build_embed_text(cmd)
            
            docs_to_add.append({
                "id": doc_id,
                "text": embed_text,
                "metadata": {
                    "command": cmd.get("command", ""),
                    "category": cmd.get("category", ""),
                    "syntax": cmd.get("syntax", ""),
                    "description": cmd.get("description", ""),
                    "parameters": json.dumps(cmd.get("parameters", {})),
                },
            })
        
        if not docs_to_add:
            logger.info("No new documents to add.")
            return {"added": 0, "total": collection.count()}
        
        # Embed in batches
        batch_size = settings.embedding.batch_size
        total_added = 0
        start_time = time.time()
        
        for i in range(0, len(docs_to_add), batch_size):
            batch = docs_to_add[i:i + batch_size]
            
            texts = [d["text"] for d in batch]
            embeddings = self.model.encode(texts, show_progress_bar=False).tolist()
            
            collection.add(
                ids=[d["id"] for d in batch],
                documents=texts,
                embeddings=embeddings,
                metadatas=[d["metadata"] for d in batch],
            )
            
            total_added += len(batch)
        
        elapsed = time.time() - start_time
        
        stats = {
            "added": total_added,
            "total": collection.count(),
            "elapsed_seconds": round(elapsed, 2),
            "source": str(source),
        }
        
        logger.info(f"Session vector DB built: {stats}")
        return stats
    
    def search(self, query: str, top_k: int = 5, category: str = "") -> list[dict]:
        """Search session commands by semantic similarity.
        
        Args:
            query: Natural language query about session commands
            top_k: Number of results
            category: Optional category filter
        
        Returns:
            List of matching session command dicts
        """
        collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        
        if collection.count() == 0:
            return []
        
        # Embed query
        query_embedding = self.model.encode([query]).tolist()
        
        # Build where filter
        where = {"category": category} if category else None
        
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, 10),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        
        # Format results
        formatted = []
        for i in range(len(results["ids"][0])):
            metadata = results["metadatas"][0][i]
            formatted.append({
                "command": metadata.get("command", ""),
                "category": metadata.get("category", ""),
                "syntax": metadata.get("syntax", ""),
                "description": metadata.get("description", ""),
                "parameters": json.loads(metadata.get("parameters", "{}")),
                "content": results["documents"][0][i],
                "score": 1 - results["distances"][0][i],  # cosine similarity
            })
        
        return formatted
    
    def get_stats(self) -> dict:
        """Get collection statistics."""
        try:
            collection = self.client.get_collection(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "total_documents": collection.count(),
                "persist_dir": str(self.persist_dir),
            }
        except Exception:
            return {"status": "not_found", "collection_name": self.collection_name}
    
    def _build_embed_text(self, cmd: dict) -> str:
        """Build text for embedding from command fields."""
        parts = []
        
        if cmd.get("command"):
            parts.append(f"Command: {cmd['command']}")
        if cmd.get("description"):
            parts.append(f"Description: {cmd['description']}")
        if cmd.get("syntax"):
            parts.append(f"Syntax: {cmd['syntax']}")
        if cmd.get("examples"):
            examples = cmd["examples"]
            if isinstance(examples, list):
                parts.append(f"Examples: {'; '.join(examples[:3])}")
            else:
                parts.append(f"Example: {examples}")
        if cmd.get("category"):
            parts.append(f"Category: {cmd['category']}")
        
        return "\n".join(parts)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    
    parser = argparse.ArgumentParser(description="Build META session commands vector DB")
    parser.add_argument("--source", required=True, help="Path to session_commands.json")
    parser.add_argument("--persist-dir", default=DEFAULT_PERSIST_DIR, help="ChromaDB persist directory")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Collection name")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild from scratch")
    args = parser.parse_args()
    
    store = SessionVectorStore(
        persist_dir=args.persist_dir,
        collection_name=args.collection,
    )
    
    stats = store.build(source_path=args.source, rebuild=args.rebuild)
    
    print(f"\nBuild complete:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    print(f"\nTest search:")
    results = store.search("how to create animation of deformation")
    for r in results[:3]:
        print(f"  [{r['score']:.3f}] {r['command']}: {r['description'][:60]}")
