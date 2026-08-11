"""Knowledge Graph Retriever for ANSA/META API Relationships.

Builds and queries a NetworkX-based knowledge graph capturing structural
relationships between API symbols: inheritance, method ownership, module
hierarchy, and cross-references.

Complementary to vector search:
- Vector search: "Find functions related to meshing" (semantic)
- Knowledge graph: "What methods does MeshParams class have?" (structural)

Features:
- Class hierarchy traversal (inheritance chains)
- Method-to-class mapping
- Module dependency graph
- Cross-reference lookup (related functions)
- Pickle-cached for fast loading
- Combinable with vector search for hybrid retrieval

Usage:
    from bin.kg_retriever import KnowledgeGraph
    
    kg = KnowledgeGraph()
    kg.build_from_documents(documents)
    kg.save("vector_db/knowledge_graph.pkl")
    
    # Query
    methods = kg.get_class_methods("ansa.base.Entity")
    parents = kg.get_inheritance_chain("ShellElement")
    related = kg.get_related_functions("CreateMesh")
"""

import pickle
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import networkx as nx

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from bin.ingest import Document


logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class GraphNode:
    """A node in the knowledge graph."""
    name: str               # Fully qualified name (e.g., ansa.base.Entity)
    node_type: str          # "class", "function", "method", "module"
    module: str = ""        # Parent module
    docstring: str = ""     # Brief description
    signature: str = ""     # Function signature
    software: str = ""      # "ansa" or "meta"


# =============================================================================
# Knowledge Graph
# =============================================================================

class KnowledgeGraph:
    """NetworkX-based knowledge graph for API structural relationships."""
    
    # Edge types
    EDGE_INHERITS = "inherits_from"      # Class -> Parent class
    EDGE_HAS_METHOD = "has_method"       # Class -> Method
    EDGE_BELONGS_TO = "belongs_to"       # Function/Class -> Module
    EDGE_RELATED = "related_to"          # Function <-> Function
    EDGE_IMPORTS = "imports"             # Module -> Module
    EDGE_RETURNS = "returns"             # Function -> Class (return type)
    
    def __init__(self):
        """Initialize an empty knowledge graph."""
        self.graph = nx.DiGraph()
        self._name_index: dict[str, str] = {}  # short_name -> full_name lookup
    
    # -------------------------------------------------------------------------
    # Graph Construction
    # -------------------------------------------------------------------------
    
    def build_from_documents(self, documents: list[Document]) -> dict:
        """Build the knowledge graph from ingested documents.
        
        Args:
            documents: List of Document objects from ingestion pipeline
            
        Returns:
            Build statistics
        """
        nodes_added = 0
        edges_added = 0
        
        for doc in documents:
            # Add node
            full_name = self._get_full_name(doc)
            if not full_name:
                continue
            
            self.graph.add_node(full_name, **{
                "type": doc.doc_type,
                "module": doc.module_name,
                "docstring": doc.content[:200],
                "signature": doc.signature,
                "software": doc.software,
            })
            nodes_added += 1
            
            # Index short name for fuzzy lookup
            short_name = full_name.split(".")[-1]
            self._name_index[short_name.lower()] = full_name
            self._name_index[full_name.lower()] = full_name
            
            # Add edges based on relationships
            
            # 1. Class inheritance
            if doc.parent_class:
                parent_full = self._resolve_name(doc.parent_class)
                self.graph.add_edge(full_name, parent_full, relation=self.EDGE_INHERITS)
                edges_added += 1
            
            # 2. Method -> Class
            if doc.doc_type == "method" and doc.class_name:
                class_full = f"{doc.module_name}.{doc.class_name}" if doc.module_name else doc.class_name
                self.graph.add_edge(class_full, full_name, relation=self.EDGE_HAS_METHOD)
                edges_added += 1
            
            # 3. Symbol -> Module
            if doc.module_name:
                self.graph.add_edge(full_name, doc.module_name, relation=self.EDGE_BELONGS_TO)
                edges_added += 1
            
            # 4. Related functions
            for related in doc.related_functions:
                related_full = self._resolve_name(related)
                self.graph.add_edge(full_name, related_full, relation=self.EDGE_RELATED)
                edges_added += 1
            
            # 5. Imports
            for imp in doc.imports:
                self.graph.add_edge(full_name, imp, relation=self.EDGE_IMPORTS)
                edges_added += 1
        
        stats = {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "nodes_added": nodes_added,
            "edges_added": edges_added,
        }
        
        logger.info(
            f"Knowledge graph built: {stats['nodes']} nodes, {stats['edges']} edges"
        )
        
        return stats
    
    # -------------------------------------------------------------------------
    # Query Methods
    # -------------------------------------------------------------------------
    
    def get_class_methods(self, class_name: str) -> list[dict]:
        """Get all methods of a class.
        
        Args:
            class_name: Full or short class name
            
        Returns:
            List of method info dicts
        """
        resolved = self._resolve_name(class_name)
        if resolved not in self.graph:
            return []
        
        methods = []
        for _, target, data in self.graph.out_edges(resolved, data=True):
            if data.get("relation") == self.EDGE_HAS_METHOD:
                node_data = self.graph.nodes.get(target, {})
                methods.append({
                    "name": target.split(".")[-1],
                    "full_name": target,
                    "signature": node_data.get("signature", ""),
                    "docstring": node_data.get("docstring", ""),
                })
        
        return methods
    
    def get_inheritance_chain(self, class_name: str) -> list[str]:
        """Get the full inheritance chain (class -> parent -> grandparent...)."""
        resolved = self._resolve_name(class_name)
        chain = [resolved]
        
        current = resolved
        visited = set()
        
        while current and current not in visited:
            visited.add(current)
            parents = [
                target for _, target, data in self.graph.out_edges(current, data=True)
                if data.get("relation") == self.EDGE_INHERITS
            ]
            if parents:
                current = parents[0]
                chain.append(current)
            else:
                break
        
        return chain
    
    def get_related_functions(self, function_name: str, max_results: int = 10) -> list[dict]:
        """Get functions related to the given function."""
        resolved = self._resolve_name(function_name)
        if resolved not in self.graph:
            return []
        
        related = []
        
        # Get directly related
        for _, target, data in self.graph.out_edges(resolved, data=True):
            if data.get("relation") == self.EDGE_RELATED:
                node_data = self.graph.nodes.get(target, {})
                related.append({
                    "name": target.split(".")[-1],
                    "full_name": target,
                    "relation": "related",
                    "docstring": node_data.get("docstring", ""),
                })
        
        # Also get reverse relations (functions that reference this one)
        for source, _, data in self.graph.in_edges(resolved, data=True):
            if data.get("relation") == self.EDGE_RELATED:
                node_data = self.graph.nodes.get(source, {})
                related.append({
                    "name": source.split(".")[-1],
                    "full_name": source,
                    "relation": "referenced_by",
                    "docstring": node_data.get("docstring", ""),
                })
        
        return related[:max_results]
    
    def get_module_contents(self, module_name: str) -> dict:
        """Get all classes and functions in a module."""
        classes = []
        functions = []
        
        for source, _, data in self.graph.in_edges(module_name, data=True):
            if data.get("relation") == self.EDGE_BELONGS_TO:
                node_data = self.graph.nodes.get(source, {})
                item = {
                    "name": source.split(".")[-1],
                    "full_name": source,
                    "type": node_data.get("type", "unknown"),
                }
                if node_data.get("type") == "class":
                    classes.append(item)
                else:
                    functions.append(item)
        
        return {"classes": classes, "functions": functions}
    
    def get_imports_for(self, function_name: str) -> list[str]:
        """Get required imports for a function/class."""
        resolved = self._resolve_name(function_name)
        imports = []
        
        for _, target, data in self.graph.out_edges(resolved, data=True):
            if data.get("relation") == self.EDGE_IMPORTS:
                imports.append(target)
        
        # Also include the module it belongs to
        for _, target, data in self.graph.out_edges(resolved, data=True):
            if data.get("relation") == self.EDGE_BELONGS_TO:
                imports.insert(0, f"from {target} import {resolved.split('.')[-1]}")
        
        return imports
    
    def search_by_name(self, query: str, max_results: int = 10) -> list[dict]:
        """Fuzzy search nodes by name."""
        query_lower = query.lower()
        results = []
        
        for short_name, full_name in self._name_index.items():
            if query_lower in short_name:
                node_data = self.graph.nodes.get(full_name, {})
                results.append({
                    "name": full_name.split(".")[-1],
                    "full_name": full_name,
                    "type": node_data.get("type", "unknown"),
                    "module": node_data.get("module", ""),
                    "docstring": node_data.get("docstring", "")[:100],
                })
        
        # Deduplicate by full_name
        seen = set()
        unique = []
        for r in results:
            if r["full_name"] not in seen:
                seen.add(r["full_name"])
                unique.append(r)
        
        return unique[:max_results]
    
    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------
    
    def save(self, filepath: str | Path) -> None:
        """Save the knowledge graph to a pickle file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "graph": self.graph,
            "name_index": self._name_index,
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        
        logger.info(f"Knowledge graph saved to: {filepath}")
    
    def load(self, filepath: str | Path) -> bool:
        """Load a knowledge graph from a pickle file.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        filepath = Path(filepath)
        if not filepath.exists():
            logger.warning(f"Graph file not found: {filepath}")
            return False
        
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        
        self.graph = data["graph"]
        self._name_index = data["name_index"]
        
        logger.info(
            f"Knowledge graph loaded: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges"
        )
        return True
    
    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------
    
    def get_stats(self) -> dict:
        """Get graph statistics."""
        node_types = {}
        for _, data in self.graph.nodes(data=True):
            t = data.get("type", "unknown")
            node_types[t] = node_types.get(t, 0) + 1
        
        edge_types = {}
        for _, _, data in self.graph.edges(data=True):
            r = data.get("relation", "unknown")
            edge_types[r] = edge_types.get(r, 0) + 1
        
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_types": node_types,
            "edge_types": edge_types,
            "indexed_names": len(self._name_index),
        }
    
    # -------------------------------------------------------------------------
    # Private Helpers
    # -------------------------------------------------------------------------
    
    def _get_full_name(self, doc: Document) -> str:
        """Get fully qualified name from a document."""
        if doc.module_name and doc.class_name and doc.function_name:
            return f"{doc.module_name}.{doc.class_name}.{doc.function_name}"
        elif doc.module_name and doc.class_name:
            return f"{doc.module_name}.{doc.class_name}"
        elif doc.module_name and doc.function_name:
            return f"{doc.module_name}.{doc.function_name}"
        elif doc.function_name:
            return doc.function_name
        return ""
    
    def _resolve_name(self, name: str) -> str:
        """Resolve a short or partial name to its full qualified name."""
        # Try exact match first
        if name in self.graph:
            return name
        # Try index lookup
        return self._name_index.get(name.lower(), name)


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="Build API knowledge graph")
    parser.add_argument("--source", default="knowledge-base/raw", help="Source directory")
    parser.add_argument("--output", default="vector_db/knowledge_graph.pkl", help="Output pickle file")
    parser.add_argument("--software", default="ansa", choices=["ansa", "meta"])
    args = parser.parse_args()
    
    from bin.ingest import IngestionPipeline
    
    # Ingest
    pipeline = IngestionPipeline(source_dir=args.source, software=args.software)
    documents = pipeline.run()
    
    # Build graph
    kg = KnowledgeGraph()
    stats = kg.build_from_documents(documents)
    
    print(f"\nGraph Statistics:")
    for k, v in kg.get_stats().items():
        print(f"  {k}: {v}")
    
    # Save
    kg.save(args.output)
    print(f"\nSaved to: {args.output}")
