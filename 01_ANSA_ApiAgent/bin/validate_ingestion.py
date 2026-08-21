"""Validate ingestion output quality.

Run after ingest.py to check extraction quality:
    python bin/validate_ingestion.py
    python bin/validate_ingestion.py --source knowledge-base/coderag_documents.jsonl
"""

import json
import argparse
from pathlib import Path
from collections import Counter


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                records.append(json.loads(line))
    return records


def validate(source: Path):
    print(f"\n{'='*60}")
    print(f"  Ingestion Validation Report")
    print(f"  Source: {source}")
    print(f"{'='*60}\n")

    records = load_jsonl(source)
    print(f"  Total records: {len(records):,}\n")

    # --- 1. Type distribution ---
    types = Counter(r.get("type") or "NONE" for r in records)
    print("  [1] Type Distribution:")
    for t, count in types.most_common():
        print(f"      {t:15s} {count:>6,}")

    # --- 2. Software distribution ---
    software = Counter(r.get("software") or "NONE" for r in records)
    print(f"\n  [2] Software Distribution:")
    for sw, count in software.most_common():
        print(f"      {sw:15s} {count:>6,}")

    # --- 3. Field completeness ---
    fields = ["signature", "description", "docstring", "module"]
    print(f"\n  [3] Field Completeness:")
    for field in fields:
        filled = sum(1 for r in records if r.get(field))
        pct = (filled / len(records) * 100) if records else 0
        print(f"      {field:15s} {filled:>6,} / {len(records):,}  ({pct:.1f}%)")

    # --- 4. Deprecated records ---
    deprecated = [r for r in records if r.get("deprecated")]
    print(f"\n  [4] Deprecated: {len(deprecated):,}")

    # --- 5. Records with code examples (notes) ---
    with_notes = [r for r in records if r.get("notes")]
    print(f"  [5] With code examples: {len(with_notes):,}")

    # --- 6. Module distribution (top 15) ---
    modules = Counter()
    for r in records:
        sym = r.get("symbol", "")
        parts = sym.split(".")
        if len(parts) >= 2:
            modules[".".join(parts[:2])] += 1
    print(f"\n  [6] Top 15 Modules:")
    for mod, count in modules.most_common(15):
        print(f"      {mod:30s} {count:>5,}")

    # --- 7. Sample records (first 3) ---
    print(f"\n  [7] Sample Records:")
    for i, rec in enumerate(records[:3]):
        print(f"\n      --- Record {i+1} ---")
        print(f"      symbol:      {rec.get('symbol')}")
        print(f"      type:        {rec.get('type')}")
        print(f"      software:    {rec.get('software')}")
        sig = rec.get('signature') or ''
        print(f"      signature:   {sig[:80]}{'...' if len(sig)>80 else ''}")
        desc = rec.get('description') or ''
        print(f"      description: {desc[:100]}{'...' if len(desc)>100 else ''}")
        notes_count = len(rec.get('notes', []))
        print(f"      notes:       {notes_count} code block(s)")

    # --- 8. Quality checks ---
    print(f"\n  [8] Quality Checks:")
    
    # Check for empty signatures on functions
    funcs_no_sig = [r for r in records if r.get("type") == "function" and not r.get("signature")]
    print(f"      Functions without signature: {len(funcs_no_sig)}")
    if funcs_no_sig[:3]:
        for r in funcs_no_sig[:3]:
            print(f"        - {r['symbol']}")

    # Check for very short descriptions
    short_desc = [r for r in records if r.get("description") and len(r["description"]) < 10]
    print(f"      Very short descriptions (<10 chars): {len(short_desc)}")

    # Check for duplicate symbols
    symbols = [r.get("symbol") for r in records]
    dupes = [s for s, c in Counter(symbols).items() if c > 1]
    print(f"      Duplicate symbols: {len(dupes)}")
    if dupes[:3]:
        for d in dupes[:3]:
            print(f"        - {d}")

    # Check signature cleanliness (no HTML artifacts)
    html_in_sig = [r for r in records if r.get("signature") and ("<" in r["signature"] or "&" in r["signature"])]
    print(f"      Signatures with HTML artifacts: {len(html_in_sig)}")
    if html_in_sig[:3]:
        for r in html_in_sig[:3]:
            print(f"        - {r['symbol']}: {r['signature'][:60]}")

    # --- 9. Knowledge Graph files ---
    kg_nodes_path = source.parent / "kg_nodes.jsonl"
    kg_edges_path = source.parent / "kg_edges.jsonl"
    if kg_nodes_path.exists() and kg_edges_path.exists():
        nodes = load_jsonl(kg_nodes_path)
        edges = load_jsonl(kg_edges_path)
        print(f"\n  [9] Knowledge Graph:")
        print(f"      Nodes: {len(nodes):,}")
        print(f"      Edges: {len(edges):,}")
        edge_types = Counter(e.get("relationship") for e in edges)
        print(f"      Edge types:")
        for rel, count in edge_types.most_common():
            print(f"        {rel:20s} {count:>6,}")

    print(f"\n{'='*60}")
    print(f"  VALIDATION COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate ingestion output")
    default_source = str(Path(__file__).resolve().parent.parent / "knowledge-base" / "coderag_documents.jsonl")
    parser.add_argument("--source", default=default_source, help="Path to coderag_documents.jsonl")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        print(f"ERROR: File not found: {source}")
        print("Run ingest.py first.")
        exit(1)

    validate(source)
