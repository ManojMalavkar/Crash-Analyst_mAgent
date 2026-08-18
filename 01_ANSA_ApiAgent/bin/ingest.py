"""Knowledge Base Ingestion Pipeline for ANSA/META API Documentation.

Extracts structured API documentation from multiple source formats
and writes intermediate JSONL files for vector DB and KG building.

Pipeline (3 separate steps):
  Step 1: python bin/ingest.py <docs_path>           -> coderag_documents.jsonl + coderag_examples.jsonl
  Step 2: python bin/build_vector_db.py              -> ChromaDB (reads JSONL)
  Step 3: python bin/kg_retriever.py --from-jsonl    -> knowledge_graph.pkl (reads kg_nodes/edges.jsonl)

Supported source formats:
  - HTML: Sphinx API reference pages (dt[id] + dd structure)
  - JSON/JSONL: Pre-structured API documentation exports
  - Python (.py): Source code stubs with docstrings
  - Markdown (.md): Tutorial and guide documents
  - Archives: .tar.gz and .zip (auto-extracted before parsing)

Output files (written to knowledge-base/):
  - coderag_documents.jsonl   -> One record per API symbol (for vector DB)
  - coderag_examples.jsonl    -> Code example scripts (for RAG)
  - kg_nodes.jsonl            -> Knowledge graph nodes
  - kg_edges.jsonl            -> Knowledge graph edges
  - knowledge_manifest.json   -> Build stats and metadata
  - full_inventory.csv        -> File inventory of source directory

Usage:
    python bin/ingest.py /path/to/api_ref_ansa --software ansa
    python bin/ingest.py /path/to/api_ref_meta --software meta
    python bin/ingest.py /path/to/docs --software ansa --output knowledge-base/
"""

import re
import ast
import json
import csv
import tarfile
import zipfile
import logging
from pathlib import Path
from typing import Optional

try:
    from bs4 import BeautifulSoup, Tag
except ImportError:
    BeautifulSoup = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

_HTML_NOISE_DIRS = {
    "_static", "_images", "_downloads",
    "_sphinx_design_static", "_sources"
}
_HTML_NOISE_FILES = {
    "genindex.html", "search.html",
    "py-modindex.html", "searchindex.js"
}


# =============================================================================
# Signature Cleanup
# =============================================================================

def clean_signature(sig: str) -> str:
    """Remove Sphinx-HTML whitespace artefacts from a function signature."""
    if not sig:
        return sig
    # Space after dot between identifiers
    sig = re.sub(r'(?<=[A-Za-z0-9_])\.\s+(?=[A-Za-z0-9_])', '.', sig)
    # Space before opening paren
    sig = re.sub(r'(?<=[A-Za-z0-9_])\s+\(', '(', sig)
    # Space after opening paren
    sig = re.sub(r'\(\s+', '(', sig)
    # Space before closing paren
    sig = re.sub(r'\s+\)', ')', sig)
    # Space before colon in type annotations
    sig = re.sub(r'(?<=[A-Za-z0-9_\]\)])\s+:', ':', sig)
    # Space before comma
    sig = re.sub(r'\s+,', ',', sig)
    # Ensure space after comma
    sig = re.sub(r',(?!\s)', ', ', sig)
    # Collapse double spaces
    sig = re.sub(r'  +', ' ', sig)
    return sig.strip()


# =============================================================================
# Knowledge Extractor
# =============================================================================

class KnowledgeExtractor:
    """Extract API knowledge from documentation sources.

    Processes JSON, Python, and HTML files to build a unified
    knowledge base with records, examples, and a knowledge graph.
    """

    def __init__(self, root_dir: str, software: str = "ansa"):
        self.root = Path(root_dir)
        self.software = software
        self.records = {}
        self.examples = []
        self.stats = {
            "json_records": 0,
            "py_records": 0,
            "html_records": 0,
            "examples": 0,
        }
        self.kg_nodes = {}
        self.kg_edges = set()

    # ------------------------------------------------------------------
    # Record Management
    # ------------------------------------------------------------------

    def get_record(self, symbol: str) -> dict:
        """Get or create a record for an API symbol."""
        if symbol not in self.records:
            self.records[symbol] = {
                "symbol": symbol,
                "module": None,
                "type": None,
                "deprecated": False,
                "signature": None,
                "description": None,
                "docstring": None,
                "examples": [],
                "notes": [],
                "sources": [],
                "software": self.software,
            }
        return self.records[symbol]

    # ------------------------------------------------------------------
    # Step 1: Extract Archives
    # ------------------------------------------------------------------

    def extract_archives(self) -> Path:
        """Extract .tar.gz and .zip archives into _extracted/ folder."""
        extract_root = self.root / "_extracted"
        extract_root.mkdir(exist_ok=True)

        for archive in self.root.rglob("*.tar.gz"):
            if extract_root in archive.parents:
                continue
            target = extract_root / archive.stem.replace(".tar", "")
            target.mkdir(parents=True, exist_ok=True)
            try:
                with tarfile.open(archive, "r:gz") as tar:
                    tar.extractall(target)
                print(f"  [TAR] {archive.name}")
            except Exception as e:
                print(f"  [FAIL] {archive.name}: {e}")

        for archive in self.root.rglob("*.zip"):
            if extract_root in archive.parents:
                continue
            target = extract_root / archive.stem
            target.mkdir(parents=True, exist_ok=True)
            try:
                with zipfile.ZipFile(archive) as z:
                    z.extractall(target)
                print(f"  [ZIP] {archive.name}")
            except Exception as e:
                print(f"  [FAIL] {archive.name}: {e}")

        return extract_root

    # ------------------------------------------------------------------
    # Step 2: Build Inventory
    # ------------------------------------------------------------------

    def build_inventory(self, output_dir: Path) -> list:
        """Build file inventory and save as CSV."""
        inventory = []
        for f in self.root.rglob("*"):
            if f.is_file():
                inventory.append({
                    "name": f.name,
                    "extension": f.suffix.lower(),
                    "size_bytes": f.stat().st_size,
                    "path": str(f.relative_to(self.root)),
                })

        inventory_file = output_dir / "full_inventory.csv"
        with open(inventory_file, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=["name", "extension", "size_bytes", "path"])
            writer.writeheader()
            writer.writerows(inventory)

        print(f"  Inventory: {len(inventory)} files -> {inventory_file}")
        return inventory

    # ------------------------------------------------------------------
    # JSON Parser
    # ------------------------------------------------------------------

    def process_json(self, file: Path):
        """Process JSON files containing API documentation."""
        try:
            content = file.read_text(encoding="utf-8", errors="ignore")
            data = json.loads(content)
        except Exception:
            return

        if not isinstance(data, list):
            return

        for item in data:
            symbol = item.get("name")
            if not symbol:
                continue

            rec = self.get_record(symbol)
            item_type = item.get("type")

            if item_type == "deprecated":
                rec["deprecated"] = True
            else:
                rec["type"] = rec["type"] or item_type

            rec["description"] = rec["description"] or item.get("description")
            rec["docstring"] = rec["docstring"] or item.get("text")
            rec["sources"].append(str(file))
            self.stats["json_records"] += 1

    # ------------------------------------------------------------------
    # Python Parser
    # ------------------------------------------------------------------

    def process_python(self, file: Path):
        """Process Python source files (stubs with docstrings)."""
        try:
            content = file.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(content)
        except Exception:
            return

        module = file.stem
        functions_found = []
        api_calls = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                symbol = f"{module}.{node.name}"
                functions_found.append(node.name)

                rec = self.get_record(symbol)
                rec["module"] = module
                rec["type"] = rec["type"] or "function"

                try:
                    rec["signature"] = rec["signature"] or ast.unparse(node.args)
                except Exception:
                    pass

                doc = ast.get_docstring(node)
                if doc:
                    rec["docstring"] = rec["docstring"] or doc

                # Return type -> KG edge
                if node.returns:
                    try:
                        return_type = ast.unparse(node.returns)
                        self.add_node(f"TYPE::{return_type}", node_type="Type")
                        self.add_edge(symbol, f"TYPE::{return_type}", "RETURNS")
                    except Exception:
                        pass

                rec["sources"].append(str(file))
                self.stats["py_records"] += 1

            elif isinstance(node, ast.ClassDef):
                symbol = f"{module}.{node.name}"
                rec = self.get_record(symbol)
                rec["module"] = module
                rec["type"] = rec["type"] or "class"

                doc = ast.get_docstring(node)
                if doc:
                    rec["docstring"] = rec["docstring"] or doc
                rec["sources"].append(str(file))

            elif isinstance(node, ast.Call):
                try:
                    api_calls.add(ast.unparse(node.func))
                except Exception:
                    pass

        # Track example files
        if any(kw in file.name.lower() for kw in ["example", "canvas", "check", "script"]):
            self.examples.append({
                "file": str(file),
                "functions": functions_found,
                "api_calls": list(api_calls),
                "content": content,
            })
            self.stats["examples"] += 1

    # ------------------------------------------------------------------
    # HTML Parser (Sphinx-aware)
    # ------------------------------------------------------------------

    def _text_skip_code(self, tag) -> str:
        """Recursively extract text, skipping highlight code blocks."""
        parts = []
        for child in tag.children:
            if isinstance(child, Tag):
                cls = child.get("class") or []
                if any("highlight" in c for c in cls):
                    continue
                parts.append(self._text_skip_code(child))
            else:
                s = str(child).strip()
                if s:
                    parts.append(s)
        return " ".join(filter(None, parts))

    def process_html(self, file: Path):
        """Process Sphinx HTML API reference pages."""
        if BeautifulSoup is None:
            return

        try:
            html = file.read_text(encoding="utf-8", errors="ignore")
            soup = BeautifulSoup(html, "html.parser")
        except Exception:
            return

        for entry in soup.select("dt[id]"):
            symbol = entry.get("id")
            if not symbol:
                continue

            # Determine type from HTML structure
            rec_type = None
            prop = entry.find("em", class_="property")
            if prop:
                prop_text = prop.get_text(strip=True).lower()
                if "class" in prop_text:
                    rec_type = "class"
                elif "method" in prop_text or "function" in prop_text:
                    rec_type = "function"
                else:
                    rec_type = "attribute"
            if not rec_type and entry.find("span", class_="sig-paren"):
                rec_type = "function"

            # Extract signature
            anchor = entry.find("a", class_="headerlink")
            anchor_text = anchor.get_text() if anchor else "#"
            raw_sig = re.sub(r"\s+", " ", entry.get_text(" ", strip=True))
            signature = clean_signature(raw_sig.replace(anchor_text, ""))

            # Extract description and docstring from <dd>
            dd = entry.find_next_sibling("dd")
            description = None
            docstring = None
            notes = []

            if dd:
                first_p = dd.find("p")
                if first_p:
                    description = re.sub(r"\s+", " ", first_p.get_text(" ", strip=True))

                for cb in dd.find_all("div", class_=re.compile(r"highlight")):
                    code = cb.get_text().strip()
                    if code:
                        notes.append(code)

                full_text = re.sub(r"\s+", " ", self._text_skip_code(dd))
                if len(full_text) > 30:
                    docstring = full_text

            # Update record
            rec = self.get_record(symbol)
            if rec_type:
                rec["type"] = rec["type"] or rec_type
            if signature and not rec["signature"]:
                rec["signature"] = signature
            if description and not rec["description"]:
                rec["description"] = description
            if docstring and not rec["docstring"]:
                rec["docstring"] = docstring
            if notes:
                rec["notes"].extend(notes)

            # Deprecated detection
            if dd:
                depr_div = dd.find("div", class_=re.compile(r"deprecated"))
                if depr_div or (description and re.search(r"^Deprecated", description, re.I)):
                    rec["deprecated"] = True

            rec["sources"].append(str(file))
            self.stats["html_records"] += 1

    # ------------------------------------------------------------------
    # Signature Second Pass
    # ------------------------------------------------------------------

    def _fill_missing_signatures(self) -> int:
        """Extract signatures from python fences in docstrings."""
        filled = 0
        for rec in self.records.values():
            if rec.get("signature"):
                continue
            sig = self._sig_from_docstring(rec.get("docstring"))
            if sig:
                rec["signature"] = sig
                filled += 1
        return filled

    def _sig_from_docstring(self, docstring: str) -> Optional[str]:
        """Extract signature from first python fence in docstring."""
        if not docstring:
            return None
        m = re.search(r'```python\s*\n(.*?)```', docstring, re.DOTALL)
        if not m:
            return None
        raw = m.group(1).strip()
        first_line = raw.split("\n")[0]
        if re.match(r'^(#|import\s|from\s)', first_line):
            return None
        if first_line.startswith("(variable)"):
            sig = re.sub(r'^\(variable\)\s*', '', raw).strip()
            return clean_signature(re.sub(r'\s+', ' ', sig)) or None
        raw = re.sub(r'^\(\w+\)\s+', '', raw)
        if not re.match(r'^(?:async\s+)?(?:def|class)\s', raw):
            return None
        raw = re.sub(r'^(?:async\s+)?def\s+', '', raw)
        raw = re.sub(r'\s+', ' ', raw).strip()
        return clean_signature(raw) or None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> tuple:
        """Split records into valid and invalid."""
        valid, invalid = [], []
        for rec in self.records.values():
            if not rec["symbol"]:
                invalid.append(rec)
            elif not any([rec["description"], rec["docstring"], rec["signature"]]):
                invalid.append(rec)
            else:
                valid.append(rec)
        return valid, invalid

    # ------------------------------------------------------------------
    # Knowledge Graph
    # ------------------------------------------------------------------

    def add_node(self, node_id: str, node_type: str, **properties):
        if not node_id:
            return
        if node_id not in self.kg_nodes:
            self.kg_nodes[node_id] = {"id": node_id, "type": node_type}
        elif node_type != "Unknown":
            if self.kg_nodes[node_id].get("type") == "Unknown":
                self.kg_nodes[node_id]["type"] = node_type
        self.kg_nodes[node_id].update(properties)

    def add_edge(self, source: str, target: str, relationship: str):
        if source and target:
            self.kg_edges.add((str(source), str(target), relationship))

    def build_knowledge_graph(self):
        """Build KG from extracted records."""
        for symbol, rec in self.records.items():
            node_type = rec.get("type") or "Unknown"
            self.add_node(symbol, node_type=node_type, module=rec.get("module"),
                          deprecated=rec.get("deprecated", False))

            # BELONGS_TO
            module = rec.get("module")
            if module:
                self.add_node(module, node_type="Module")
                self.add_edge(symbol, module, "BELONGS_TO")

            # DEPRECATED_BY
            if rec.get("deprecated"):
                content_str = (rec.get("description") or "") + " " + (rec.get("docstring") or "")
                repl = re.search(r':py:\w+:`([^`]+)`', content_str)
                if not repl:
                    repl = re.search(r'[Uu]se\s+`?([A-Za-z][A-Za-z0-9_.]{4,})`?', content_str)
                if repl:
                    replacement = repl.group(1).strip("`").strip()
                    self.add_node(replacement, node_type="Unknown")
                    self.add_edge(symbol, replacement, "DEPRECATED_BY")

            # USES_TYPE from signature
            sig = rec.get("signature")
            if sig:
                for dtype in re.findall(r':\s*([A-Za-z0-9_\[\]\|\.]+)', sig):
                    self.add_node(f"TYPE::{dtype}", node_type="Type")
                    self.add_edge(symbol, f"TYPE::{dtype}", "USES_TYPE")

            # CLASS -> HAS_METHOD
            parts = symbol.split(".")
            if len(parts) >= 3:
                class_candidate = ".".join(parts[:-1])
                if class_candidate in self.records:
                    if (self.records[class_candidate].get("type") or "").lower() == "class":
                        self.add_edge(class_candidate, symbol, "HAS_METHOD")

            # SEE_ALSO
            content_str = rec.get("docstring") or ""
            for target in re.findall(r'ansa\.[A-Za-z0-9_.]+', content_str):
                if target != symbol:
                    self.add_node(target, node_type="Unknown")
                    self.add_edge(symbol, target, "SEE_ALSO")

        # Examples -> USES edges
        for ex in self.examples:
            ex_node = f"EXAMPLE::{Path(ex['file']).name}"
            self.add_node(ex_node, node_type="Example")
            for api in ex.get("api_calls", []):
                self.add_node(api, node_type="Unknown")
                self.add_edge(ex_node, api, "USES")

    # ------------------------------------------------------------------
    # Write Output Files
    # ------------------------------------------------------------------

    def write(self, output_dir: Path) -> dict:
        """Write all output files to output directory."""
        output_dir.mkdir(parents=True, exist_ok=True)

        valid, invalid = self.validate()

        # coderag_documents.jsonl
        docs_file = output_dir / "coderag_documents.jsonl"
        with open(docs_file, "w", encoding="utf-8") as fp:
            for record in valid:
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")

        # coderag_examples.jsonl
        examples_file = output_dir / "coderag_examples.jsonl"
        with open(examples_file, "w", encoding="utf-8") as fp:
            for ex in self.examples:
                fp.write(json.dumps(ex, ensure_ascii=False) + "\n")

        # Knowledge graph
        self.build_knowledge_graph()

        nodes_file = output_dir / "kg_nodes.jsonl"
        with open(nodes_file, "w", encoding="utf-8") as fp:
            for node in self.kg_nodes.values():
                fp.write(json.dumps(node, ensure_ascii=False) + "\n")

        edges_file = output_dir / "kg_edges.jsonl"
        with open(edges_file, "w", encoding="utf-8") as fp:
            for source, target, rel in self.kg_edges:
                fp.write(json.dumps({"source": source, "target": target, "relationship": rel}, ensure_ascii=False) + "\n")

        # Manifest
        manifest = {
            "total_records": len(self.records),
            "valid_records": len(valid),
            "invalid_records": len(invalid),
            "deprecated_records": sum(1 for r in self.records.values() if r.get("deprecated")),
            "examples_count": len(self.examples),
            "kg_nodes": len(self.kg_nodes),
            "kg_edges": len(self.kg_edges),
            "software": self.software,
            **self.stats,
            "output_files": {
                "documents": str(docs_file),
                "examples": str(examples_file),
                "kg_nodes": str(nodes_file),
                "kg_edges": str(edges_file),
            },
        }

        manifest_file = output_dir / "knowledge_manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as fp:
            json.dump(manifest, fp, indent=2)

        return manifest

    # ------------------------------------------------------------------
    # Main Pipeline
    # ------------------------------------------------------------------

    def run(self, output_dir: Optional[str] = None) -> dict:
        """Run the full extraction pipeline."""
        out = Path(output_dir) if output_dir else self.root

        print(f"\n{'='*60}")
        print(f"  Knowledge Extraction Pipeline")
        print(f"  Source: {self.root}")
        print(f"  Software: {self.software}")
        print(f"{'='*60}\n")

        # 1. Extract archives
        print("[1/6] Extracting archives...")
        self.extract_archives()

        # 2. Build inventory
        print("[2/6] Building file inventory...")
        self.build_inventory(out)

        # 3. Process JSON
        print("[3/6] Processing JSON files...")
        json_files = list(self.root.rglob("*.json"))
        for file in tqdm(json_files, desc="  JSON", unit="file"):
            self.process_json(file)

        # 4. Process Python
        print("[4/6] Processing Python files...")
        py_files = list(self.root.rglob("*.py"))
        for file in tqdm(py_files, desc="  Python", unit="file"):
            self.process_python(file)

        # 5. Process HTML
        print("[5/6] Processing HTML files...")
        html_files = [
            f for f in self.root.rglob("*.html")
            if not any(part in _HTML_NOISE_DIRS for part in f.parts)
            and f.name not in _HTML_NOISE_FILES
        ]
        for file in tqdm(html_files, desc="  HTML", unit="file"):
            self.process_html(file)

        # 6. Signature second pass + write
        print("[6/6] Finalizing...")
        filled = self._fill_missing_signatures()
        print(f"  Signature second pass: {filled} filled from docstrings")

        manifest = self.write(out)

        # Summary
        print(f"\n{'='*60}")
        print(f"  EXTRACTION COMPLETE")
        print(f"{'='*60}")
        print(f"  Valid records:      {manifest['valid_records']:,}")
        print(f"  Invalid (skipped):  {manifest['invalid_records']:,}")
        print(f"  Deprecated:         {manifest['deprecated_records']:,}")
        print(f"  Examples:           {manifest['examples_count']:,}")
        print(f"  KG nodes:           {manifest['kg_nodes']:,}")
        print(f"  KG edges:           {manifest['kg_edges']:,}")
        print(f"  JSON records:       {manifest['json_records']:,}")
        print(f"  Python records:     {manifest['py_records']:,}")
        print(f"  HTML records:       {manifest['html_records']:,}")
        print(f"\n  Output: {out}")
        print()

        return manifest


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Extract ANSA/META API knowledge from documentation sources"
    )
    parser.add_argument(
        "source_dir",
        help="Directory containing docs (.py, .html, .json, .md)",
    )
    parser.add_argument(
        "--software",
        default="ansa",
        choices=["ansa", "meta"],
        help="Software identifier (default: ansa)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for JSONL files (default: same as source_dir)",
    )
    args = parser.parse_args()

    extractor = KnowledgeExtractor(
        root_dir=args.source_dir,
        software=args.software,
    )
    extractor.run(output_dir=args.output)
