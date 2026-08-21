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
    # Build from JSONL (CLI):
    python bin/kg_retriever.py
    
    # Programmatic:
    from bin.kg_retriever import KnowledgeGraph
    
    kg = KnowledgeGraph()
    kg.build_from_jsonl("knowledge-base/")
    kg.save("vector_db/knowledge_graph.pkl")
    
    # Or load existing:
    kg = KnowledgeGraph()
    kg.load("vector_db/knowledge_graph.pkl")
    
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

import json


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
    
    def build_from_jsonl(self, knowledge_base_dir: str | Path) -> dict:
        """Build the knowledge graph from JSONL files produced by ingest.py.
        
        Args:
            knowledge_base_dir: Directory containing kg_nodes.jsonl and kg_edges.jsonl
            
        Returns:
            Build statistics
        """
        kb_dir = Path(knowledge_base_dir)
        nodes_file = kb_dir / "kg_nodes.jsonl"
        edges_file = kb_dir / "kg_edges.jsonl"
        
        if not nodes_file.exists():
            raise FileNotFoundError(f"Nodes file not found: {nodes_file}")
        if not edges_file.exists():
            raise FileNotFoundError(f"Edges file not found: {edges_file}")
        
        # Load nodes
        nodes_added = 0
        with open(nodes_file, "r", encoding="utf-8") as fp:
            for line in fp:
                if not line.strip():
                    continue
                node = json.loads(line)
                node_id = node.get("id") or node.get("node_id", "")
                if not node_id:
                    continue
                
                self.graph.add_node(node_id, **{
                    "type": node.get("type", "unknown"),
                    "module": node.get("module", ""),
                    "docstring": node.get("description", "")[:200],
                    "signature": node.get("signature", ""),
                    "software": node.get("software", ""),
                })
                nodes_added += 1
                
                # Index short name for fuzzy lookup
                short_name = node_id.split(".")[-1]
                self._name_index[short_name.lower()] = node_id
                self._name_index[node_id.lower()] = node_id
        
        # Load edges
        edges_added = 0
        with open(edges_file, "r", encoding="utf-8") as fp:
            for line in fp:
                if not line.strip():
                    continue
                edge = json.loads(line)
                source = edge.get("source", "")
                target = edge.get("target", "")
                rel = edge.get("relationship", "related_to")
                if source and target:
                    self.graph.add_edge(source, target, relation=rel)
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
    
    default_kb = str(Path(__file__).resolve().parent.parent / "knowledge-base")
    default_output = str(Path(__file__).resolve().parent.parent / "vector_db" / "knowledge_graph.pkl")
    
    parser = argparse.ArgumentParser(
        description="Build API knowledge graph from JSONL files",
        epilog="""
Examples:
  # Default (no args needed):
  python bin/kg_retriever.py
  
  # Custom paths:
  python bin/kg_retriever.py --source /path/to/knowledge-base/ --output /path/to/kg.pkl
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source", default=default_kb,
        help=f"Directory containing kg_nodes.jsonl and kg_edges.jsonl (default: {default_kb})"
    )
    parser.add_argument(
        "--output", default=default_output,
        help=f"Output pickle file (default: {default_output})"
    )
    args = parser.parse_args()
    
    source_path = Path(args.source)
    if not (source_path / "kg_nodes.jsonl").exists():
        print(f"\n  ERROR: kg_nodes.jsonl not found in: {source_path}")
        print(f"  Run ingest.py first to generate JSONL files.")
        print(f"  Example: python bin/ingest.py /path/to/docs")
        exit(1)
    
    # Build graph from JSONL
    print(f"\n{'='*50}")
    print(f"  Building Knowledge Graph")
    print(f"  Source: {source_path}")
    print(f"{'='*50}")
    
    kg = KnowledgeGraph()
    stats = kg.build_from_jsonl(source_path)
    
    print(f"\n  Graph Statistics:")
    full_stats = kg.get_stats()
    print(f"    Nodes: {full_stats['total_nodes']:,}")
    print(f"    Edges: {full_stats['total_edges']:,}")
    print(f"    Indexed names: {full_stats['indexed_names']:,}")
    print(f"\n    Node types:")
    for t, count in sorted(full_stats['node_types'].items(), key=lambda x: -x[1]):
        print(f"      {t:15s} {count:>6,}")
    print(f"\n    Edge types:")
    for r, count in sorted(full_stats['edge_types'].items(), key=lambda x: -x[1]):
        print(f"      {r:20s} {count:>6,}")
    
    # Save
    kg.save(args.output)
    print(f"\n  Saved to: {args.output}")
    print(f"{'='*50}\n")
