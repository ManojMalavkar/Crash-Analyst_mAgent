"""ChromaDB Vector Store Builder for ANSA/META API Knowledge Base.

Embeds ingested documents using sentence-transformers (BGE model) and stores
them in a persistent ChromaDB collection with metadata for filtered retrieval.

Features:
- Persistent ChromaDB storage (survives restarts)
- Batch embedding for efficiency
- Metadata-based filtering (module, type, software)
- Incremental updates (skip already-indexed documents)
- Collection management (create, rebuild, stats)

Usage:
    # Build from source docs directory
    python build_vector_db.py --source /path/to/docs --rebuild
    
    # Incremental update
    python build_vector_db.py --source /path/to/docs
    
    # Programmatic
    from bin.build_vector_db import VectorStoreBuilder
    builder = VectorStoreBuilder(persist_dir="vector_db")
    builder.build(documents)
"""

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
from bin.ingest import Document, IngestionPipeline


logger = logging.getLogger(__name__)


# =============================================================================
# Embedding Model Wrapper
# =============================================================================

class EmbeddingModel:
    """Wrapper around sentence-transformers for document embedding."""
    
    def __init__(self, model_name: Optional[str] = None):
        """Initialize embedding model.
        
        Args:
            model_name: HuggingFace model name (default: from settings)
        """
        self.model_name = model_name or settings.embedding.model_name
        self.batch_size = settings.embedding.batch_size
        self._model = None
    
    @property
    def model(self):
        """Lazy-load the embedding model."""
        if self._model is None:
            if SentenceTransformer is None:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install with: pip install sentence-transformers"
                )
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts in batches.
        
        Args:
            texts: List of document texts to embed
            
        Returns:
            List of embedding vectors
        """
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            embeddings = self.model.encode(
                batch,
                show_progress_bar=False,
                normalize_embeddings=True,  # For cosine similarity
            )
            all_embeddings.extend(embeddings.tolist())
            
            if i > 0 and i % (self.batch_size * 10) == 0:
                logger.info(f"  Embedded {i}/{len(texts)} documents...")
        
        return all_embeddings
    
    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )
        return embedding[0].tolist()


# =============================================================================
# Vector Store Builder
# =============================================================================

class VectorStoreBuilder:
    """Builds and manages ChromaDB vector collections."""
    
    DEFAULT_COLLECTION = "ansa_meta_api"
    
    def __init__(
        self,
        persist_dir: Optional[str | Path] = None,
        collection_name: str = DEFAULT_COLLECTION,
    ):
        """Initialize the vector store builder.
        
        Args:
            persist_dir: Directory for persistent ChromaDB storage
            collection_name: Name of the ChromaDB collection
        """
        self.persist_dir = Path(persist_dir or settings.paths.ansa_vector_db)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.collection_name = collection_name
        self._embedding_model = EmbeddingModel()
        
        # Initialize ChromaDB client (persistent)
        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(
                anonymized_telemetry=False,
            ),
        )
        
        logger.info(f"ChromaDB initialized at: {self.persist_dir}")
    
    def build(
        self,
        documents: list[Document],
        rebuild: bool = False,
    ) -> dict:
        """Build or update the vector collection from documents.
        
        Args:
            documents: List of Document objects to index
            rebuild: If True, delete and recreate the collection
            
        Returns:
            Build statistics dict
        """
        start_time = time.time()
        
        # Get or create collection
        if rebuild:
            try:
                self._client.delete_collection(self.collection_name)
                logger.info(f"Deleted existing collection: {self.collection_name}")
            except Exception:
                pass
        
        collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},  # Cosine similarity
        )
        
        # Filter out already-indexed documents (incremental update)
        existing_ids = set()
        if not rebuild and collection.count() > 0:
            existing = collection.get()
            existing_ids = set(existing["ids"])
        
        new_documents = [
            doc for doc in documents
            if doc.doc_id not in existing_ids
        ]
        
        if not new_documents:
            logger.info("No new documents to index.")
            return {
                "total_in_collection": collection.count(),
                "new_documents": 0,
                "skipped": len(documents),
                "duration_seconds": 0,
            }
        
        logger.info(
            f"Indexing {len(new_documents)} new documents "
            f"(skipping {len(documents) - len(new_documents)} existing)"
        )
        
        # Prepare data for ChromaDB
        ids = [doc.doc_id for doc in new_documents]
        texts = [doc.content for doc in new_documents]
        metadatas = [doc.to_metadata() for doc in new_documents]
        
        # Embed documents
        logger.info("Generating embeddings...")
        embeddings = self._embedding_model.embed_documents(texts)
        
        # Upsert in batches (ChromaDB limit: 41666 per batch)
        batch_size = 5000
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            collection.upsert(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                documents=texts[i:end],
                metadatas=metadatas[i:end],
            )
            logger.info(f"  Upserted batch {i//batch_size + 1} ({end}/{len(ids)})")
        
        duration = time.time() - start_time
        
        stats = {
            "total_in_collection": collection.count(),
            "new_documents": len(new_documents),
            "skipped": len(documents) - len(new_documents),
            "duration_seconds": round(duration, 1),
        }
        
        logger.info(
            f"Build complete: {stats['new_documents']} indexed, "
            f"{stats['total_in_collection']} total, "
            f"{stats['duration_seconds']}s"
        )
        
        return stats
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None,
    ) -> list[dict]:
        """Search the vector collection.
        
        Args:
            query: Natural language search query
            top_k: Number of results to return
            where: Metadata filter (e.g., {"doc_type": "function"})
            where_document: Document content filter
            
        Returns:
            List of result dicts with content, metadata, and score
        """
        collection = self._client.get_collection(self.collection_name)
        
        # Embed query
        query_embedding = self._embedding_model.embed_query(query)
        
        # Search
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        if where_document:
            kwargs["where_document"] = where_document
        
        results = collection.query(**kwargs)
        
        # Format results
        formatted = []
        for i in range(len(results["ids"][0])):
            formatted.append({
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i],  # Convert distance to similarity
            })
        
        return formatted
    
    def get_stats(self) -> dict:
        """Get collection statistics."""
        try:
            collection = self._client.get_collection(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "total_documents": collection.count(),
                "persist_dir": str(self.persist_dir),
            }
        except Exception:
            return {
                "collection_name": self.collection_name,
                "total_documents": 0,
                "persist_dir": str(self.persist_dir),
                "status": "not_found",
            }
    
    def delete_collection(self) -> None:
        """Delete the entire collection."""
        try:
            self._client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.warning(f"Could not delete collection: {e}")


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    
    parser = argparse.ArgumentParser(description="Build ANSA/META vector database")
    parser.add_argument("--source", default="knowledge-base/raw", help="Source docs directory")
    parser.add_argument("--persist-dir", default=None, help="ChromaDB persist directory")
    parser.add_argument("--collection", default="ansa_meta_api", help="Collection name")
    parser.add_argument("--software", default="ansa", choices=["ansa", "meta"], help="Target software")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild collection from scratch")
    args = parser.parse_args()
    
    # Step 1: Ingest documents
    print(f"\n{'='*50}")
    print(f"  Step 1: Ingesting from {args.source}")
    print(f"{'='*50}")
    
    pipeline = IngestionPipeline(source_dir=args.source, software=args.software)
    documents = pipeline.run()
    
    if not documents:
        print("No documents found. Check your source directory.")
        sys.exit(1)
    
    stats = pipeline.get_stats(documents)
    print(f"  Documents: {stats['total_documents']}")
    print(f"  By type: {stats['by_type']}")
    
    # Step 2: Build vector store
    print(f"\n{'='*50}")
    print(f"  Step 2: Building vector database")
    print(f"{'='*50}")
    
    builder = VectorStoreBuilder(
        persist_dir=args.persist_dir,
        collection_name=args.collection,
    )
    build_stats = builder.build(documents, rebuild=args.rebuild)
    
    print(f"\n  Results:")
    for k, v in build_stats.items():
        print(f"    {k}: {v}")
    
    # Step 3: Quick test search
    print(f"\n{'='*50}")
    print(f"  Step 3: Test search")
    print(f"{'='*50}")
    
    test_queries = ["mesh generation", "create material", "export model"]
    for query in test_queries:
        results = builder.search(query, top_k=2)
        print(f"\n  Query: '{query}'")
        for r in results:
            name = r['metadata'].get('function_name', 'unknown')
            score = r['score']
            print(f"    [{score:.3f}] {name} ({r['metadata'].get('doc_type', '?')})")
    
    print(f"\n{'='*50}")
    print(f"  Build complete! Collection at: {builder.persist_dir}")
    print(f"{'='*50}\n")
