"""RAG Tool Functions for ANSA/META CodeRAG Agent.

5 retrieval tools that the agent can call during the tool-calling loop:
1. search_api          - Semantic vector search over API documentation
2. search_code_examples - Filtered search for code examples and scripts
3. get_function_details - Exact lookup by function/class name
4. get_class_hierarchy  - Knowledge graph: inheritance chain + methods
5. search_knowledge_graph - KG name search + related functions

These functions are auto-registered as OpenAI tools via tools.py.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from bin.build_vector_db import VectorStoreBuilder
from bin.kg_retriever import KnowledgeGraph


logger = logging.getLogger(__name__)


# =============================================================================
# Singleton Instances (lazy-loaded)
# =============================================================================

_vector_store: Optional[VectorStoreBuilder] = None
_knowledge_graph: Optional[KnowledgeGraph] = None


def _get_vector_store() -> VectorStoreBuilder:
    """Get or create the vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStoreBuilder(persist_dir="vector_db")
    return _vector_store


def _get_knowledge_graph() -> KnowledgeGraph:
    """Get or load the knowledge graph instance."""
    global _knowledge_graph
    if _knowledge_graph is None:
        _knowledge_graph = KnowledgeGraph()
        kg_path = Path("vector_db/knowledge_graph.pkl")
        if kg_path.exists():
            _knowledge_graph.load(kg_path)
        else:
            logger.warning(f"Knowledge graph not found at {kg_path}. Run kg_retriever.py first.")
    return _knowledge_graph


# =============================================================================
# Tool 1: search_api
# =============================================================================

def search_api(query: str, top_k: int = 5, software: str = "") -> str:
    """Search the ANSA/META API documentation using semantic similarity.
    
    Use this for natural language queries about API functionality,
    e.g., "how to create shell mesh", "export to nastran format".
    
    Args:
        query: Natural language description of what you're looking for
        top_k: Number of results to return (1-10)
        software: Filter by software - "ansa", "meta", or "" for both
    
    Returns:
        JSON string with matching API documentation entries
    """
    store = _get_vector_store()
    
    where = {"software": software} if software else None
    results = store.search(query=query, top_k=min(top_k, 10), where=where)
    
    if not results:
        return json.dumps({"results": [], "message": "No results found. Try rephrasing your query."})
    
    formatted = []
    for r in results:
        formatted.append({
            "function": r["metadata"].get("function_name", ""),
            "module": r["metadata"].get("module_name", ""),
            "type": r["metadata"].get("doc_type", ""),
            "signature": r["metadata"].get("signature", ""),
            "content": r["content"][:500],
            "score": round(r["score"], 3),
        })
    
    return json.dumps({"results": formatted}, indent=2)


# =============================================================================
# Tool 2: search_code_examples
# =============================================================================

def search_code_examples(query: str, top_k: int = 3) -> str:
    """Search for code examples and usage patterns in the knowledge base.
    
    Use this when you need working code snippets, usage examples,
    or complete script patterns for a specific task.
    
    Args:
        query: Description of the code example needed, e.g., "mesh quality check script"
        top_k: Number of examples to return (1-5)
    
    Returns:
        JSON string with code examples and their context
    """
    store = _get_vector_store()
    
    # Search with document content filter for code-related terms
    results = store.search(
        query=query,
        top_k=min(top_k, 5),
        where_document={"$contains": "import"},
    )
    
    # Fallback: regular search if filtered returns nothing
    if not results:
        results = store.search(query=f"example {query}", top_k=min(top_k, 5))
    
    if not results:
        return json.dumps({"results": [], "message": "No code examples found."})
    
    formatted = []
    for r in results:
        formatted.append({
            "function": r["metadata"].get("function_name", ""),
            "module": r["metadata"].get("module_name", ""),
            "content": r["content"][:800],  # Longer content for code
            "source_file": r["metadata"].get("source_file", ""),
            "score": round(r["score"], 3),
        })
    
    return json.dumps({"results": formatted}, indent=2)


# =============================================================================
# Tool 3: get_function_details
# =============================================================================

def get_function_details(function_name: str) -> str:
    """Get detailed information about a specific function or class by exact name.
    
    Use this when you know the exact function/class name and need its
    full documentation: signature, parameters, return type, and description.
    
    Args:
        function_name: Exact function or class name, e.g., "CreateMesh", "Entity", "CollectEntities"
    
    Returns:
        JSON string with full function/class details including signature and docs
    """
    store = _get_vector_store()
    kg = _get_knowledge_graph()
    
    # Try vector store with metadata filter
    results = store.search(
        query=function_name,
        top_k=3,
        where={"function_name": function_name},
    )
    
    # Fallback: search by name in content
    if not results:
        results = store.search(query=function_name, top_k=3)
    
    # Enrich with KG data
    kg_info = {}
    if kg.graph.number_of_nodes() > 0:
        kg_results = kg.search_by_name(function_name, max_results=1)
        if kg_results:
            full_name = kg_results[0]["full_name"]
            kg_info = {
                "full_name": full_name,
                "imports": kg.get_imports_for(full_name),
                "related": [r["name"] for r in kg.get_related_functions(full_name)[:5]],
            }
    
    if not results:
        return json.dumps({
            "found": False,
            "message": f"Function '{function_name}' not found. Try search_api for semantic search.",
            "kg_info": kg_info,
        })
    
    # Return the best match with full content
    best = results[0]
    detail = {
        "found": True,
        "function": best["metadata"].get("function_name", ""),
        "module": best["metadata"].get("module_name", ""),
        "type": best["metadata"].get("doc_type", ""),
        "signature": best["metadata"].get("signature", ""),
        "return_type": best["metadata"].get("return_type", ""),
        "content": best["content"],
        "kg_info": kg_info,
    }
    
    return json.dumps(detail, indent=2)


# =============================================================================
# Tool 4: get_class_hierarchy
# =============================================================================

def get_class_hierarchy(class_name: str) -> str:
    """Get the inheritance hierarchy and all methods of a class.
    
    Use this to understand class relationships, what a class inherits from,
    and what methods are available on it.
    
    Args:
        class_name: Class name to inspect, e.g., "Entity", "ShellElement", "MeshParams"
    
    Returns:
        JSON string with inheritance chain and method list
    """
    kg = _get_knowledge_graph()
    
    if kg.graph.number_of_nodes() == 0:
        return json.dumps({
            "error": "Knowledge graph not loaded. Run kg_retriever.py to build it."
        })
    
    # Get inheritance chain
    chain = kg.get_inheritance_chain(class_name)
    
    # Get methods
    methods = kg.get_class_methods(class_name)
    
    # Get module contents for context
    kg_results = kg.search_by_name(class_name, max_results=1)
    module = ""
    if kg_results:
        module = kg_results[0].get("module", "")
    
    result = {
        "class_name": class_name,
        "module": module,
        "inheritance_chain": chain,
        "methods": methods,
        "method_count": len(methods),
    }
    
    return json.dumps(result, indent=2)


# =============================================================================
# Tool 5: search_knowledge_graph
# =============================================================================

def search_knowledge_graph(query: str, max_results: int = 10) -> str:
    """Search the knowledge graph by name and get structural relationships.
    
    Use this for structural queries: finding all functions in a module,
    related functions, or exploring the API structure.
    
    Args:
        query: Name or partial name to search for, e.g., "mesh", "Entity", "ansa.base"
        max_results: Maximum number of results (1-20)
    
    Returns:
        JSON string with matching nodes and their relationships
    """
    kg = _get_knowledge_graph()
    
    if kg.graph.number_of_nodes() == 0:
        return json.dumps({
            "error": "Knowledge graph not loaded. Run kg_retriever.py to build it."
        })
    
    # Search by name
    results = kg.search_by_name(query, max_results=min(max_results, 20))
    
    if not results:
        return json.dumps({"results": [], "message": f"No matches for '{query}' in knowledge graph."})
    
    # Enrich each result with relationships
    enriched = []
    for r in results:
        item = {
            "name": r["name"],
            "full_name": r["full_name"],
            "type": r["type"],
            "module": r["module"],
        }
        
        # Add related functions for the first few results
        if len(enriched) < 5:
            related = kg.get_related_functions(r["full_name"])[:3]
            item["related"] = [rel["name"] for rel in related]
        
        enriched.append(item)
    
    return json.dumps({"results": enriched}, indent=2)


# =============================================================================
# Tool 6: read_session_file
# =============================================================================

def read_session_file(file_path: str) -> str:
    """Read and parse a META session file (.ses) from the given path.
    
    Use this when the user provides a session file path and wants to:
    - Understand what a session file does
    - Convert session commands to Python
    - Modify or extend an existing session workflow
    - Debug a failing session script
    
    Args:
        file_path: Absolute or relative path to the .ses session file
    
    Returns:
        JSON string with file content, parsed commands, and metadata
    """
    from pathlib import Path
    
    filepath = Path(file_path).expanduser().resolve()
    
    # Validate file exists
    if not filepath.exists():
        return json.dumps({
            "error": f"File not found: {filepath}",
            "suggestion": "Check the path and try again. Use absolute path if relative fails.",
        })
    
    if not filepath.is_file():
        return json.dumps({"error": f"Path is not a file: {filepath}"})
    
    # Read file content
    try:
        content = filepath.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return json.dumps({"error": f"Cannot read file: {e}"})
    
    # Parse session commands
    lines = content.splitlines()
    commands = []
    current_command = None
    
    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()
        
        # Skip empty lines and comments
        if not stripped or stripped.startswith("#") or stripped.startswith("//"):
            continue
        
        commands.append({
            "line": line_num,
            "content": stripped,
        })
    
    # Identify command categories
    categories = _categorize_session_commands(commands)
    
    # Build summary
    result = {
        "file_path": str(filepath),
        "file_name": filepath.name,
        "file_size_bytes": filepath.stat().st_size,
        "total_lines": len(lines),
        "total_commands": len(commands),
        "categories": categories,
        "content": content if len(content) <= 5000 else content[:5000] + f"\n\n... [truncated, {len(content)} total chars]",
        "commands_preview": commands[:50],
    }
    
    return json.dumps(result, indent=2)


def _categorize_session_commands(commands: list[dict]) -> dict:
    """Categorize session commands by type."""
    categories = {
        "file_io": [],       # open, save, export
        "display": [],       # view, plot, animate, contour
        "data": [],          # extract, filter, math
        "annotation": [],    # text, arrow, legend
        "window": [],        # window, layout, page
        "other": [],
    }
    
    keywords = {
        "file_io": ["open", "save", "export", "import", "read", "write", "load", "close"],
        "display": ["plot", "contour", "animate", "view", "iso", "section", "fringe", "deform", "display"],
        "data": ["extract", "filter", "math", "curve", "result", "value", "measure", "cross"],
        "annotation": ["text", "arrow", "legend", "title", "label", "note", "annotation"],
        "window": ["window", "layout", "page", "resize", "position", "toolbox"],
    }
    
    for cmd in commands:
        content_lower = cmd["content"].lower()
        categorized = False
        
        for category, kw_list in keywords.items():
            if any(kw in content_lower for kw in kw_list):
                categories[category].append(cmd["line"])
                categorized = True
                break
        
        if not categorized:
            categories["other"].append(cmd["line"])
    
    # Return counts + sample lines
    summary = {}
    for cat, line_nums in categories.items():
        if line_nums:
            summary[cat] = {
                "count": len(line_nums),
                "lines": line_nums[:10],  # First 10 line numbers
            }
    
    return summary


# =============================================================================
# Tool Registry
# =============================================================================

# All tools available to the agent
ALL_TOOLS = [
    search_api,
    search_code_examples,
    get_function_details,
    get_class_hierarchy,
    search_knowledge_graph,
    read_session_file,
]
