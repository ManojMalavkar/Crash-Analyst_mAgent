#!/usr/bin/env python3
"""Test RAG tools independently (without LLM).

Usage:
    python test_tools.py                    # Run all tests
    python test_tools.py --tool search_api  # Test specific tool
    python test_tools.py --query "mesh"     # Custom query
"""

import sys
import json
import argparse
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def test_vector_db():
    """Test ChromaDB connection and collection."""
    print("\n" + "="*50)
    print("  TEST: Vector DB Connection")
    print("="*50)
    
    try:
        import chromadb
        from bin.build_vector_db import VectorStoreBuilder
        
        # Check what path is being used
        agent_dir = Path(__file__).resolve().parent
        vector_db_path = agent_dir / "vector_db"
        print(f"  Path: {vector_db_path}")
        print(f"  Exists: {vector_db_path.exists()}")
        
        if not vector_db_path.exists():
            print("  ERROR: vector_db/ directory not found!")
            print(f"  Expected at: {vector_db_path}")
            return False
        
        # List collections
        client = chromadb.PersistentClient(path=str(vector_db_path))
        collections = client.list_collections()
        print(f"  Collections: {[c.name for c in collections]}")
        
        for col in collections:
            print(f"    - {col.name}: {col.count()} documents")
        
        return True
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def test_search_api(query: str = "create shell mesh"):
    """Test search_api tool."""
    print("\n" + "="*50)
    print(f"  TEST: search_api(query='{query}')")
    print("="*50)
    
    try:
        from bin.tool_functions import search_api
        result = search_api(query=query, top_k=3)
        
        data = json.loads(result)
        if "error" in data:
            print(f"  ERROR: {data['error']}")
            return False
        
        results = data.get("results", [])
        print(f"  Found: {len(results)} results")
        for i, r in enumerate(results):
            print(f"\n  [{i+1}] {r.get('symbol', 'N/A')}")
            print(f"      Type: {r.get('type', 'N/A')}")
            print(f"      Score: {r.get('score', 'N/A'):.3f}")
            desc = r.get('description', '')[:80]
            print(f"      Desc: {desc}")
        
        return len(results) > 0
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def test_get_function_details(name: str = "CreateMesh"):
    """Test get_function_details tool."""
    print("\n" + "="*50)
    print(f"  TEST: get_function_details(name='{name}')")
    print("="*50)
    
    try:
        from bin.tool_functions import get_function_details
        result = get_function_details(function_name=name)
        
        data = json.loads(result)
        if "error" in data:
            print(f"  ERROR: {data['error']}")
            return False
        
        print(f"  Symbol: {data.get('symbol', 'N/A')}")
        print(f"  Module: {data.get('module', 'N/A')}")
        print(f"  Type: {data.get('type', 'N/A')}")
        sig = data.get('signature', '')[:100]
        print(f"  Signature: {sig}")
        desc = data.get('description', '')[:100]
        print(f"  Description: {desc}")
        
        return True
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def test_knowledge_graph(query: str = "Entity"):
    """Test knowledge graph search."""
    print("\n" + "="*50)
    print(f"  TEST: search_knowledge_graph(query='{query}')")
    print("="*50)
    
    try:
        from bin.tool_functions import search_knowledge_graph
        result = search_knowledge_graph(query=query)
        
        data = json.loads(result)
        if "error" in data:
            print(f"  ERROR: {data['error']}")
            return False
        
        nodes = data.get("nodes", [])
        print(f"  Found: {len(nodes)} matching nodes")
        for i, n in enumerate(nodes[:5]):
            print(f"  [{i+1}] {n.get('full_name', 'N/A')} ({n.get('type', 'N/A')})")
        
        return len(nodes) > 0
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def test_class_hierarchy(class_name: str = "Entity"):
    """Test get_class_hierarchy tool."""
    print("\n" + "="*50)
    print(f"  TEST: get_class_hierarchy(class_name='{class_name}')")
    print("="*50)
    
    try:
        from bin.tool_functions import get_class_hierarchy
        result = get_class_hierarchy(class_name=class_name)
        
        data = json.loads(result)
        if "error" in data:
            print(f"  ERROR: {data['error']}")
            return False
        
        methods = data.get("methods", [])
        chain = data.get("inheritance_chain", [])
        print(f"  Inheritance: {' -> '.join(chain) if chain else 'N/A'}")
        print(f"  Methods: {len(methods)}")
        for m in methods[:5]:
            print(f"    - {m.get('name', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test RAG tools independently")
    parser.add_argument("--tool", choices=["db", "search_api", "details", "kg", "hierarchy", "all"], default="all")
    parser.add_argument("--query", default=None, help="Custom query string")
    args = parser.parse_args()
    
    print("\n" + "#"*50)
    print("  RAG Tools Test Suite")
    print("#"*50)
    
    results = {}
    
    if args.tool in ("all", "db"):
        results["vector_db"] = test_vector_db()
    
    if args.tool in ("all", "search_api"):
        q = args.query or "create shell mesh"
        results["search_api"] = test_search_api(q)
    
    if args.tool in ("all", "details"):
        q = args.query or "CreateMesh"
        results["get_function_details"] = test_get_function_details(q)
    
    if args.tool in ("all", "kg"):
        q = args.query or "Entity"
        results["search_knowledge_graph"] = test_knowledge_graph(q)
    
    if args.tool in ("all", "hierarchy"):
        q = args.query or "Entity"
        results["get_class_hierarchy"] = test_class_hierarchy(q)
    
    # Summary
    print("\n" + "="*50)
    print("  RESULTS SUMMARY")
    print("="*50)
    for tool, passed in results.items():
        status = "PASS" if passed else "FAIL"
        icon = "+" if passed else "X"
        print(f"  [{icon}] {tool}: {status}")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\n  {passed}/{total} tests passed")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
