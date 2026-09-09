"""ANSA/META CodeRAG + GraphRAG Ingestion Pipeline.

Extracts structured API documentation from HTML (Sphinx), JSON, and Python
sources, producing a normalized JSON schema optimized for:
  - Qdrant vector search (semantic retrieval)
  - Neo4j graph database (structural reasoning)

Normalized JSON Schema per API entry:
{
  "module": "meta.annotations",
  "type": "function|method|class|attribute|constant",
  "name": "AnnotationById",
  "full_name": "meta.annotations.AnnotationById",
  "class_name": null | "Annotation",
  "parameters": [{"name": "...", "type": "...", "optional": false, "default": null}],
  "returns": "Annotation" | null,
  "description": "...",
  "docstring": "...",
  "signature": "...",
  "deprecated": true|false,
  "deprecated_version": "20.1.0" | null,
  "replacement": "meta.pages.Page.get_annotations" | null,
  "see_also": ["meta.annotations.Annotation"],
  "examples": ["a = AnnotationById(1)\nnode = a.get_node()"],
  "remarks": "...",
  "version": null,
  "software": "ansa|meta",
  "navigation_targets": ["Node", "Window", "Part"],
  "accepts_types": ["Curve", "int"],
  "source_file": "..."
}

Pipeline:
  Step 1: python bin/ingest.py <docs_path>           -> api_normalized.jsonl + examples.jsonl
  Step 2: python bin/build_vector_db.py              -> Qdrant (reads JSONL)
  Step 3: python bin/build_graph_db.py               -> Neo4j (reads JSONL)

Usage:
    python bin/ingest.py /path/to/html/reference/
    python bin/ingest.py /path/to/docs --output /custom/path/ --software meta
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
from collections import defaultdict

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
    "_sphinx_design_static", "_sources",
    ".doctrees"
}
_HTML_NOISE_FILES = {
    "genindex.html", "search.html",
    "py-modindex.html", "searchindex.js"
}

# Known ANSA/META return types that indicate navigation targets
_NAVIGABLE_TYPES = {
    "Node", "Element", "Part", "Material", "Group", "Curve",
    "Window", "Page", "Annotation", "Connection", "Boundary",
    "Set", "LoadCase", "Model", "Result", "Component",
}


# =============================================================================
# Signature Cleanup
# =============================================================================

def clean_signature(sig: str) -> str:
    """Remove Sphinx-HTML whitespace artefacts from a function signature."""
    if not sig:
        return sig
    sig = re.sub(r'(?<=[A-Za-z0-9_])\.\s+(?=[A-Za-z0-9_])', '.', sig)
    sig = re.sub(r'(?<=[A-Za-z0-9_])\s+\(', '(', sig)
    sig = re.sub(r'\(\s+', '(', sig)
    sig = re.sub(r'\s+\)', ')', sig)
    sig = re.sub(r'(?<=[A-Za-z0-9_\]\)])\s+:', ':', sig)
    sig = re.sub(r'\s+,', ',', sig)
    sig = re.sub(r',(?!\s)', ', ', sig)
    sig = re.sub(r'  +', ' ', sig)
    return sig.strip()


# =============================================================================
# Parameter Parser
# =============================================================================

def parse_parameters(signature: str) -> list:
    """Parse function signature into structured parameter list.

    Returns:
        List of dicts: [{"name": str, "type": str|None, "optional": bool, "default": str|None}]
    """
    if not signature:
        return []

    # Extract content between parentheses
    m = re.search(r'\((.*)\)', signature, re.DOTALL)
    if not m:
        return []

    params_str = m.group(1).strip()
    if not params_str:
        return []

    parameters = []
    # Split on commas, respecting nested brackets
    depth = 0
    current = ""
    for char in params_str:
        if char in "([{":
            depth += 1
            current += char
        elif char in ")]}":
            depth -= 1
            current += char
        elif char == "," and depth == 0:
            parameters.append(current.strip())
            current = ""
        else:
            current += char
    if current.strip():
        parameters.append(current.strip())

    result = []
    for param in parameters:
        if param in ("self", "cls"):
            continue

        param_info = {"name": None, "type": None, "optional": False, "default": None}

        # Check for default value
        if "=" in param:
            param_info["optional"] = True
            parts = param.split("=", 1)
            param = parts[0].strip()
            param_info["default"] = parts[1].strip()

        # Check for type annotation
        if ":" in param:
            parts = param.split(":", 1)
            param_info["name"] = parts[0].strip()
            param_info["type"] = parts[1].strip()
        else:
            param_info["name"] = param.strip()

        # Handle *args, **kwargs
        if param_info["name"]:
            if param_info["name"].startswith("**"):
                param_info["name"] = param_info["name"][2:]
                param_info["type"] = param_info["type"] or "dict"
                param_info["optional"] = True
            elif param_info["name"].startswith("*"):
                param_info["name"] = param_info["name"][1:]
                param_info["type"] = param_info["type"] or "tuple"
                param_info["optional"] = True

        if param_info["name"]:
            result.append(param_info)

    return result


def extract_return_type(signature: str, description: str = None) -> Optional[str]:
    """Extract return type from signature annotation or description text."""
    if not signature:
        return _return_from_description(description)

    # Check -> annotation in signature
    m = re.search(r'->\s*([A-Za-z0-9_\[\],. |]+)', signature)
    if m:
        return m.group(1).strip()

    # Try from description
    return _return_from_description(description)


def _return_from_description(description: str) -> Optional[str]:
    """Extract return type from description text patterns."""
    if not description:
        return None

    # "Returns: TypeName" or "Returns TypeName"
    m = re.search(r'[Rr]eturns?:?\s+(?:an?\s+)?([A-Z][A-Za-z0-9_]+)', description)
    if m:
        return m.group(1)

    # "Return type: TypeName"
    m = re.search(r'[Rr]eturn\s+type:?\s*([A-Z][A-Za-z0-9_]+)', description)
    if m:
        return m.group(1)

    return None


# =============================================================================
# Normalized Record Builder
# =============================================================================

def make_normalized_record(
    module: str,
    name: str,
    full_name: str,
    api_type: str = "function",
    class_name: str = None,
    parameters: list = None,
    returns: str = None,
    description: str = None,
    docstring: str = None,
    signature: str = None,
    deprecated: bool = False,
    deprecated_version: str = None,
    replacement: str = None,
    see_also: list = None,
    examples: list = None,
    remarks: str = None,
    version: str = None,
    software: str = "ansa",
    navigation_targets: list = None,
    accepts_types: list = None,
    source_file: str = None,
) -> dict:
    """Create a normalized API record matching the spec schema."""
    return {
        "module": module,
        "type": api_type,
        "name": name,
        "full_name": full_name,
        "class_name": class_name,
        "parameters": parameters or [],
        "returns": returns,
        "description": description,
        "docstring": docstring,
        "signature": signature,
        "deprecated": deprecated,
        "deprecated_version": deprecated_version,
        "replacement": replacement,
        "see_also": see_also or [],
        "examples": examples or [],
        "remarks": remarks,
        "version": version,
        "software": software,
        "navigation_targets": navigation_targets or [],
        "accepts_types": accepts_types or [],
        "source_file": source_file,
    }


# =============================================================================
# Knowledge Extractor (Enhanced for GraphRAG)
# =============================================================================

class KnowledgeExtractor:
    """Extract API knowledge from documentation sources.

    Produces normalized JSON records for Qdrant + Neo4j ingestion,
    with full relationship extraction for GraphRAG.
    """

    def __init__(self, root_dir: str, software: str = "ansa"):
        self.root = Path(root_dir)
        self.software = software
        self.records: dict[str, dict] = {}  # full_name -> normalized record
        self.examples: list[dict] = []       # standalone code examples
        self.workflows: list[list[str]] = [] # sequences of API calls from examples
        self.stats = {
            "json_records": 0,
            "py_records": 0,
            "html_records": 0,
            "examples": 0,
            "workflows_extracted": 0,
        }

    # ------------------------------------------------------------------
    # Record Management
    # ------------------------------------------------------------------

    def get_or_create(self, full_name: str) -> dict:
        """Get existing record or create a new normalized one."""
        if full_name not in self.records:
            parts = full_name.rsplit(".", 1)
            name = parts[-1] if len(parts) > 1 else full_name
            module = parts[0] if len(parts) > 1 else None

            # Detect class_name from dotted path
            class_name = None
            module_parts = full_name.split(".")
            if len(module_parts) >= 3:
                # e.g. meta.annotations.Annotation.get_node
                potential_class = module_parts[-2]
                if potential_class and potential_class[0].isupper():
                    class_name = potential_class
                    module = ".".join(module_parts[:-2])

            self.records[full_name] = make_normalized_record(
                module=module,
                name=name,
                full_name=full_name,
                software=self.software,
            )
            if class_name:
                self.records[full_name]["class_name"] = class_name

        return self.records[full_name]

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
        """Process JSON/JSONL files containing API documentation."""
        try:
            content = file.read_text(encoding="utf-8", errors="ignore")
            # Try as JSON array first
            try:
                data = json.loads(content)
                if not isinstance(data, list):
                    data = [data]
            except json.JSONDecodeError:
                # Try as JSONL
                data = []
                for line in content.splitlines():
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except Exception:
            return

        for item in data:
            symbol = item.get("full_name") or item.get("name")
            if not symbol:
                continue

            rec = self.get_or_create(symbol)

            # Map fields from various JSON formats
            rec["type"] = rec["type"] or item.get("type")
            rec["description"] = rec["description"] or item.get("description")
            rec["docstring"] = rec["docstring"] or item.get("text") or item.get("docstring")
            rec["signature"] = rec["signature"] or item.get("signature")
            rec["returns"] = rec["returns"] or item.get("returns")
            rec["module"] = rec["module"] or item.get("module")

            if item.get("deprecated"):
                rec["deprecated"] = True
            if item.get("deprecated_version"):
                rec["deprecated_version"] = item["deprecated_version"]
            if item.get("replacement"):
                rec["replacement"] = item["replacement"]
            if item.get("see_also"):
                rec["see_also"] = list(set(rec["see_also"] + item["see_also"]))
            if item.get("examples"):
                rec["examples"] = list(set(rec["examples"] + item["examples"]))
            if item.get("parameters"):
                rec["parameters"] = rec["parameters"] or item["parameters"]

            rec["source_file"] = rec["source_file"] or str(file)
            self.stats["json_records"] += 1

    # ------------------------------------------------------------------
    # Python Parser
    # ------------------------------------------------------------------

    def process_python(self, file: Path):
        """Process Python source files (stubs, examples, scripts)."""
        try:
            content = file.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(content)
        except Exception:
            return

        module = file.stem
        functions_found = []
        api_calls_ordered = []  # Preserve order for workflow extraction

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                full_name = f"{module}.{node.name}"
                functions_found.append(node.name)

                rec = self.get_or_create(full_name)
                rec["module"] = rec["module"] or module
                rec["type"] = rec["type"] or "function"

                # Signature
                try:
                    sig_str = ast.unparse(node.args)
                    rec["signature"] = rec["signature"] or f"{node.name}({sig_str})"
                    rec["parameters"] = rec["parameters"] or parse_parameters(rec["signature"])
                except Exception:
                    pass

                # Return type
                if node.returns:
                    try:
                        return_type = ast.unparse(node.returns)
                        rec["returns"] = rec["returns"] or return_type
                        if return_type in _NAVIGABLE_TYPES:
                            rec["navigation_targets"] = list(
                                set(rec["navigation_targets"] + [return_type])
                            )
                    except Exception:
                        pass

                # Docstring
                doc = ast.get_docstring(node)
                if doc:
                    rec["docstring"] = rec["docstring"] or doc

                rec["source_file"] = rec["source_file"] or str(file)
                self.stats["py_records"] += 1

            elif isinstance(node, ast.ClassDef):
                full_name = f"{module}.{node.name}"
                rec = self.get_or_create(full_name)
                rec["module"] = rec["module"] or module
                rec["type"] = rec["type"] or "class"
                doc = ast.get_docstring(node)
                if doc:
                    rec["docstring"] = rec["docstring"] or doc
                rec["source_file"] = rec["source_file"] or str(file)

            elif isinstance(node, ast.Call):
                try:
                    call_name = ast.unparse(node.func)
                    api_calls_ordered.append(call_name)
                except Exception:
                    pass

        # Track example files and extract workflows
        if any(kw in file.name.lower() for kw in ["example", "canvas", "check", "script", "demo", "tutorial"]):
            self.examples.append({
                "file": str(file),
                "functions": functions_found,
                "api_calls": api_calls_ordered,
                "content": content,
                "software": self.software,
            })
            self.stats["examples"] += 1

            # Extract workflow sequences (COMMONLY_FOLLOWED_BY)
            if len(api_calls_ordered) >= 2:
                self.workflows.append(api_calls_ordered)
                self.stats["workflows_extracted"] += 1

    # ------------------------------------------------------------------
    # HTML Parser (Enhanced Sphinx-aware for GraphRAG)
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

    def _extract_code_examples(self, dd_tag) -> list:
        """Extract code examples from <dd> element."""
        examples = []
        if not dd_tag:
            return examples

        for cb in dd_tag.find_all("div", class_=re.compile(r"highlight")):
            code = cb.get_text().strip()
            if code and len(code) > 10:
                # Filter out single-line non-code content
                if any(kw in code for kw in ["=", "(", "import", ".", "print"]):
                    examples.append(code)

        return examples

    def _extract_see_also(self, dd_tag) -> list:
        """Extract see_also references from <dd> element."""
        see_also = []
        if not dd_tag:
            return see_also

        # Look for "See also" section
        see_section = dd_tag.find("p", class_="rubric", string=re.compile(r"See\s+[Aa]lso", re.I))
        if see_section:
            # Get following list or paragraph
            next_el = see_section.find_next_sibling()
            if next_el:
                for ref in next_el.find_all("a"):
                    href = ref.get("href", "")
                    text = ref.get_text(strip=True)
                    if text and re.match(r'^[a-z][a-z0-9_.]+', text):
                        see_also.append(text)

        # Also extract from inline references
        for ref in dd_tag.find_all("a", class_="reference"):
            text = ref.get_text(strip=True)
            if text and re.match(r'^(?:ansa|meta)\.[a-z0-9_.]+', text):
                see_also.append(text)

        return list(set(see_also))

    def _extract_deprecation_info(self, dd_tag) -> tuple:
        """Extract deprecation version and replacement from <dd>.

        Returns:
            (deprecated_version, replacement) tuple
        """
        if not dd_tag:
            return None, None

        deprecated_version = None
        replacement = None

        # Look for deprecated div/admonition
        depr_div = dd_tag.find("div", class_=re.compile(r"deprecated"))
        if depr_div:
            text = depr_div.get_text(" ", strip=True)

            # Version: "Deprecated since version 20.1.0"
            m = re.search(r'[Dd]eprecated\s+(?:since\s+)?(?:version\s+)?([0-9]+\.[0-9]+(?:\.[0-9]+)?)', text)
            if m:
                deprecated_version = m.group(1)

            # Replacement: "Use X instead" or :py:func:`X`
            m = re.search(r'[Uu]se\s+:?(?:py:\w+:)?`?([A-Za-z][A-Za-z0-9_.]+)`?\s+instead', text)
            if not m:
                m = re.search(r'[Rr]eplaced?\s+by\s+:?(?:py:\w+:)?`?([A-Za-z][A-Za-z0-9_.]+)`?', text)
            if not m:
                m = re.search(r'[Uu]se\s+`?([A-Za-z][A-Za-z0-9_.]{4,})`?', text)
            if m:
                replacement = m.group(1).strip("`").strip()

        # Also check first paragraph for deprecation marker
        first_p = dd_tag.find("p")
        if first_p:
            text = first_p.get_text(" ", strip=True)
            if re.search(r'^Deprecated', text, re.I):
                if not deprecated_version:
                    m = re.search(r'([0-9]+\.[0-9]+(?:\.[0-9]+)?)', text)
                    if m:
                        deprecated_version = m.group(1)

        return deprecated_version, replacement

    def _extract_return_type_from_dd(self, dd_tag) -> Optional[str]:
        """Extract return type from docstring text in <dd>."""
        if not dd_tag:
            return None

        # Look for "Return type:" or "Returns:" field list
        for field in dd_tag.find_all("dt", class_="field-odd"):
            if "return" in field.get_text(strip=True).lower():
                dd_field = field.find_next_sibling("dd")
                if dd_field:
                    rtype = dd_field.get_text(strip=True)
                    # Clean up type references
                    rtype = re.sub(r'^[a-z]+\.', '', rtype)  # Remove module prefix for short name
                    if rtype and len(rtype) < 80:
                        return rtype

        # Look for rtype in field lists
        for item in dd_tag.find_all("li"):
            text = item.get_text(" ", strip=True)
            m = re.match(r'[Rr](?:eturn|type)[s]?\s*[-–:]\s*(.+)', text)
            if m:
                rtype = m.group(1).strip()
                if len(rtype) < 80:
                    return rtype

        return None

    def _infer_accepts_types(self, parameters: list) -> list:
        """Infer accepted types from parameter list for ACCEPTS relationships."""
        accepts = []
        for param in parameters:
            ptype = param.get("type")
            if ptype:
                # Extract class names (capitalized types)
                for t in re.findall(r'[A-Z][A-Za-z0-9_]+', ptype):
                    if t in _NAVIGABLE_TYPES or len(t) > 3:
                        accepts.append(t)
        return list(set(accepts))

    def process_html(self, file: Path):
        """Process Sphinx HTML API reference pages with full extraction."""
        if BeautifulSoup is None:
            if not getattr(self, "_bs4_warned", False):
                print("  WARNING: beautifulsoup4 not installed — skipping HTML files")
                print("  Fix: pip install beautifulsoup4")
                self._bs4_warned = True
            return

        try:
            html = file.read_text(encoding="utf-8", errors="ignore")
            soup = BeautifulSoup(html, "html.parser")
        except Exception:
            return

        # Detect module from page title or breadcrumb
        page_module = None
        title = soup.find("h1")
        if title:
            title_text = title.get_text(strip=True).replace("¶", "").strip()
            # "meta.annotations module" or "ansa.base module"
            m = re.match(r'^((?:ansa|meta)\.[a-z0-9_.]+)\s+module', title_text, re.I)
            if m:
                page_module = m.group(1)
            # Also try: "Module: meta.annotations"
            if not page_module:
                m = re.match(r'^(?:Module:?\s+)?((?:ansa|meta)\.[a-z0-9_.]+)', title_text, re.I)
                if m:
                    page_module = m.group(1)

        for entry in soup.select("dt[id]"):
            symbol = entry.get("id")
            if not symbol:
                continue

            # Determine API type from HTML structure
            rec_type = None
            prop = entry.find("em", class_="property")
            if prop:
                prop_text = prop.get_text(strip=True).lower()
                if "class" in prop_text:
                    rec_type = "class"
                elif "method" in prop_text:
                    rec_type = "method"
                elif "function" in prop_text:
                    rec_type = "function"
                elif "attribute" in prop_text or "property" in prop_text:
                    rec_type = "attribute"
                else:
                    rec_type = "attribute"
            if not rec_type and entry.find("span", class_="sig-paren"):
                rec_type = "function"

            # Distinguish method from function based on class context
            if rec_type == "function" and "." in symbol:
                parts = symbol.split(".")
                if len(parts) >= 3:
                    potential_class = parts[-2]
                    if potential_class and potential_class[0].isupper():
                        rec_type = "method"

            # Extract signature
            anchor = entry.find("a", class_="headerlink")
            anchor_text = anchor.get_text() if anchor else "#"
            raw_sig = re.sub(r"\s+", " ", entry.get_text(" ", strip=True))
            signature = clean_signature(raw_sig.replace(anchor_text, ""))

            # Extract from <dd>
            dd = entry.find_next_sibling("dd")
            description = None
            docstring = None
            remarks = None

            if dd:
                first_p = dd.find("p")
                if first_p:
                    description = re.sub(r"\s+", " ", first_p.get_text(" ", strip=True))

                full_text = re.sub(r"\s+", " ", self._text_skip_code(dd))
                if len(full_text) > 30:
                    docstring = full_text

                # Remarks section
                remarks_header = dd.find("p", class_="rubric", string=re.compile(r"[Rr]emarks?|[Nn]otes?"))
                if remarks_header:
                    next_el = remarks_header.find_next_sibling()
                    if next_el:
                        remarks = next_el.get_text(" ", strip=True)[:500]

            # Parse parameters from signature
            parameters = parse_parameters(signature)

            # Extract return type
            return_type = extract_return_type(signature, description)
            if not return_type and dd:
                return_type = self._extract_return_type_from_dd(dd)

            # Extract examples
            examples = self._extract_code_examples(dd)

            # Extract see_also
            see_also = self._extract_see_also(dd)

            # Extract deprecation info
            deprecated = False
            deprecated_version = None
            replacement = None
            if dd:
                depr_div = dd.find("div", class_=re.compile(r"deprecated"))
                if depr_div or (description and re.search(r"^Deprecated", description, re.I)):
                    deprecated = True
                    deprecated_version, replacement = self._extract_deprecation_info(dd)

            # Determine navigation targets from return type
            # Handles compound types: "Node", "list of Node", "Node or None",
            # "Optional[Node]", "List[Node]", "meta.nodes.Node",
            # "tuple(Node, Element)", "Annotation | None"
            navigation_targets = []
            if return_type:
                for candidate in re.findall(r'[A-Z][A-Za-z0-9_]+', return_type):
                    if candidate in _NAVIGABLE_TYPES:
                        navigation_targets.append(candidate)
                navigation_targets = list(set(navigation_targets))

            # Determine accepts_types from parameters
            accepts_types = self._infer_accepts_types(parameters)

            # Determine module
            module = page_module
            if not module:
                parts = symbol.split(".")
                if len(parts) >= 2:
                    # meta.annotations.Annotation.get_node -> meta.annotations
                    module_parts = []
                    for p in parts[:-1]:
                        if p[0].islower():
                            module_parts.append(p)
                        else:
                            break
                    module = ".".join(module_parts) if module_parts else None

            # Determine class_name
            class_name = None
            parts = symbol.split(".")
            if len(parts) >= 3:
                potential_class = parts[-2] if rec_type in ("method", "attribute") else None
                if potential_class and potential_class[0].isupper():
                    class_name = potential_class

            # Build/update record
            rec = self.get_or_create(symbol)
            rec["type"] = rec["type"] or rec_type
            rec["module"] = rec["module"] or module
            rec["class_name"] = rec["class_name"] or class_name
            rec["signature"] = rec["signature"] or signature
            rec["description"] = rec["description"] or description
            rec["docstring"] = rec["docstring"] or docstring
            rec["remarks"] = rec["remarks"] or remarks
            rec["returns"] = rec["returns"] or return_type
            rec["deprecated"] = rec["deprecated"] or deprecated
            rec["deprecated_version"] = rec["deprecated_version"] or deprecated_version
            rec["replacement"] = rec["replacement"] or replacement
            rec["parameters"] = rec["parameters"] or parameters
            rec["navigation_targets"] = list(
                set(rec["navigation_targets"] + navigation_targets)
            )
            rec["accepts_types"] = list(
                set(rec["accepts_types"] + accepts_types)
            )
            rec["see_also"] = list(set(rec["see_also"] + see_also))
            rec["examples"] = rec["examples"] + [e for e in examples if e not in rec["examples"]]
            rec["source_file"] = rec["source_file"] or str(file)

            # Promote inline examples to self.examples for examples.jsonl + workflows
            for ex_code in examples:
                api_calls = self._extract_api_calls_from_code(ex_code)
                self.examples.append({
                    "file": str(file),
                    "parent_api": symbol,
                    "functions": [],
                    "api_calls": api_calls,
                    "content": ex_code,
                    "software": self.software,
                })
                self.stats["examples"] += 1

                # Extract workflow sequences from inline examples
                if len(api_calls) >= 2:
                    self.workflows.append(api_calls)
                    self.stats["workflows_extracted"] += 1

            self.stats["html_records"] += 1

    # ------------------------------------------------------------------
    # API Call Extraction from Code Strings
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_api_calls_from_code(code: str) -> list:
        """Extract ordered API call names from a code snippet string.

        Handles patterns like:
          - ansa.base.CreateCurve(...)
          - meta.annotations.AnnotationById(...)
          - curve.AddData(...)
          - node = a.get_node()
        """
        calls = []

        # Pattern 1: module-qualified calls  e.g. ansa.base.CreateMesh(...)
        for m in re.finditer(r'((?:ansa|meta)\.[A-Za-z0-9_.]+)\s*\(', code):
            calls.append(m.group(1))

        # Pattern 2: method calls on objects  e.g. curve.AddData(...)  a.get_node()
        for m in re.finditer(r'\b([a-z_][a-z_0-9]*)\.([A-Za-z_][A-Za-z0-9_]*)\s*\(', code):
            calls.append(m.group(2))

        # Pattern 3: bare function calls  e.g. CreateCurve(...)
        for m in re.finditer(r'(?<![.\w])([A-Z][A-Za-z0-9_]+)\s*\(', code):
            name = m.group(1)
            if name not in ("True", "False", "None", "Exception", "TypeError",
                            "ValueError", "KeyError", "IndexError", "AttributeError",
                            "RuntimeError", "StopIteration", "NotImplementedError"):
                calls.append(name)

        return calls

    # ------------------------------------------------------------------
    # Workflow Extraction from Examples
    # ------------------------------------------------------------------

    def _extract_workflows_from_examples(self):
        """Extract additional COMMONLY_FOLLOWED_BY sequences from standalone example files.

        Note: inline HTML examples already extract workflows in process_html().
        This method handles standalone Python example files only.
        """
        for ex in self.examples:
            # Skip inline HTML examples (already processed)
            if ex.get("parent_api"):
                continue
            calls = ex.get("api_calls", [])
            if len(calls) >= 2:
                # Filter to known ANSA/META calls
                filtered = [
                    c for c in calls
                    if re.match(r'^(?:ansa|meta|[A-Z])', c)
                    or c in self.records
                ]
                if len(filtered) >= 2:
                    self.workflows.append(filtered)

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
                rec["parameters"] = rec["parameters"] or parse_parameters(sig)
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
    # Post-Processing: Enrich navigation and accepts
    # ------------------------------------------------------------------

    def _enrich_relationships(self):
        """Second pass: enrich navigation_targets and accepts from cross-references."""
        # Build class method registry
        class_methods = defaultdict(list)
        for full_name, rec in self.records.items():
            if rec["class_name"]:
                class_methods[rec["class_name"]].append(rec)

        # For each class, collect all navigation targets from its methods
        for class_name, methods in class_methods.items():
            nav_targets = set()
            for method_rec in methods:
                ret = method_rec.get("returns")
                if ret:
                    # Extract all navigable types from compound returns
                    for candidate in re.findall(r'[A-Z][A-Za-z0-9_]+', ret):
                        if candidate in _NAVIGABLE_TYPES and candidate != class_name:
                            nav_targets.add(candidate)
                            method_rec["navigation_targets"] = list(
                                set(method_rec["navigation_targets"] + [candidate])
                            )

            # Also tag the class record itself
            class_full_names = [
                fn for fn, r in self.records.items()
                if r.get("name") == class_name and r.get("type") == "class"
            ]
            for cfn in class_full_names:
                self.records[cfn]["navigation_targets"] = list(
                    set(self.records[cfn]["navigation_targets"] + list(nav_targets))
                )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> tuple:
        """Split records into valid and invalid."""
        valid, invalid = [], []
        for rec in self.records.values():
            if not rec["full_name"]:
                invalid.append(rec)
            elif not any([rec["description"], rec["docstring"], rec["signature"]]):
                invalid.append(rec)
            else:
                valid.append(rec)
        return valid, invalid

    # ------------------------------------------------------------------
    # Write Output Files
    # ------------------------------------------------------------------

    def write(self, output_dir: Path) -> dict:
        """Write normalized JSON and workflow files."""
        output_dir.mkdir(parents=True, exist_ok=True)

        valid, invalid = self.validate()

        # api_normalized.jsonl — main output for vector + graph
        docs_file = output_dir / "api_normalized.jsonl"
        with open(docs_file, "w", encoding="utf-8") as fp:
            for record in valid:
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")

        # examples.jsonl — standalone code examples
        examples_file = output_dir / "examples.jsonl"
        with open(examples_file, "w", encoding="utf-8") as fp:
            for ex in self.examples:
                fp.write(json.dumps(ex, ensure_ascii=False) + "\n")

        # workflows.jsonl — API call sequences for COMMONLY_FOLLOWED_BY
        workflows_file = output_dir / "workflows.jsonl"
        with open(workflows_file, "w", encoding="utf-8") as fp:
            for wf in self.workflows:
                fp.write(json.dumps({"sequence": wf, "software": self.software}, ensure_ascii=False) + "\n")

        # Legacy compatibility: also output kg_nodes/edges for NetworkX fallback
        self._write_legacy_kg(output_dir, valid)

        # Manifest
        manifest = {
            "total_records": len(self.records),
            "valid_records": len(valid),
            "invalid_records": len(invalid),
            "deprecated_records": sum(1 for r in self.records.values() if r.get("deprecated")),
            "records_with_parameters": sum(1 for r in valid if r.get("parameters")),
            "records_with_returns": sum(1 for r in valid if r.get("returns")),
            "records_with_examples": sum(1 for r in valid if r.get("examples")),
            "records_with_navigation": sum(1 for r in valid if r.get("navigation_targets")),
            "examples_count": len(self.examples),
            "workflows_count": len(self.workflows),
            "software": self.software,
            **self.stats,
            "output_files": {
                "api_normalized": str(docs_file),
                "examples": str(examples_file),
                "workflows": str(workflows_file),
            },
        }

        manifest_file = output_dir / "knowledge_manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as fp:
            json.dump(manifest, fp, indent=2)

        return manifest

    def _write_legacy_kg(self, output_dir: Path, valid_records: list):
        """Write legacy kg_nodes.jsonl and kg_edges.jsonl for NetworkX fallback."""
        kg_nodes = {}
        kg_edges = set()

        for rec in valid_records:
            full_name = rec["full_name"]
            node_type = rec.get("type") or "Unknown"

            # Node
            kg_nodes[full_name] = {
                "id": full_name,
                "type": node_type,
                "module": rec.get("module"),
                "deprecated": rec.get("deprecated", False),
            }

            # BELONGS_TO module
            module = rec.get("module")
            if module:
                kg_nodes.setdefault(module, {"id": module, "type": "Module"})
                kg_edges.add((full_name, module, "BELONGS_TO"))

            # HAS_METHOD
            if rec.get("class_name") and rec["type"] in ("method", "attribute"):
                class_full = f"{rec['module']}.{rec['class_name']}" if rec["module"] else rec["class_name"]
                kg_nodes.setdefault(class_full, {"id": class_full, "type": "class"})
                edge_type = "HAS_METHOD" if rec["type"] == "method" else "HAS_ATTRIBUTE"
                kg_edges.add((class_full, full_name, edge_type))

            # RETURNS
            if rec.get("returns"):
                ret_type = rec["returns"]
                kg_nodes.setdefault(f"TYPE::{ret_type}", {"id": f"TYPE::{ret_type}", "type": "ReturnType"})
                kg_edges.add((full_name, f"TYPE::{ret_type}", "RETURNS"))

            # ACCEPTS
            for atype in rec.get("accepts_types", []):
                kg_nodes.setdefault(f"TYPE::{atype}", {"id": f"TYPE::{atype}", "type": "Type"})
                kg_edges.add((full_name, f"TYPE::{atype}", "ACCEPTS"))

            # CAN_NAVIGATE_TO
            for nav in rec.get("navigation_targets", []):
                kg_nodes.setdefault(f"TYPE::{nav}", {"id": f"TYPE::{nav}", "type": "Class"})
                kg_edges.add((full_name, f"TYPE::{nav}", "CAN_NAVIGATE_TO"))

            # DEPRECATED_IN / REPLACED_BY
            if rec.get("deprecated"):
                if rec.get("deprecated_version"):
                    ver_id = f"VERSION::{rec['deprecated_version']}"
                    kg_nodes.setdefault(ver_id, {"id": ver_id, "type": "Version"})
                    kg_edges.add((full_name, ver_id, "DEPRECATED_IN"))
                if rec.get("replacement"):
                    kg_nodes.setdefault(rec["replacement"], {"id": rec["replacement"], "type": "Unknown"})
                    kg_edges.add((full_name, rec["replacement"], "REPLACED_BY"))

            # SEE_ALSO
            for ref in rec.get("see_also", []):
                kg_nodes.setdefault(ref, {"id": ref, "type": "Unknown"})
                kg_edges.add((full_name, ref, "SEE_ALSO"))

        # COMMONLY_FOLLOWED_BY from workflows
        for wf in self.workflows:
            for i in range(len(wf) - 1):
                src, tgt = wf[i], wf[i + 1]
                if src != tgt:
                    kg_edges.add((src, tgt, "COMMONLY_FOLLOWED_BY"))

        # Write files
        nodes_file = output_dir / "kg_nodes.jsonl"
        with open(nodes_file, "w", encoding="utf-8") as fp:
            for node in kg_nodes.values():
                fp.write(json.dumps(node, ensure_ascii=False) + "\n")

        edges_file = output_dir / "kg_edges.jsonl"
        with open(edges_file, "w", encoding="utf-8") as fp:
            for source, target, rel in kg_edges:
                fp.write(json.dumps({"source": source, "target": target, "relationship": rel}, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Main Pipeline
    # ------------------------------------------------------------------

    def run(self, output_dir: Optional[str] = None) -> dict:
        """Run the full extraction pipeline."""
        out = Path(output_dir) if output_dir else self.root

        print(f"\n{'='*60}")
        print(f"  ANSA/META CodeRAG+GraphRAG Ingestion Pipeline")
        print(f"  Source: {self.root}")
        print(f"  Software: {self.software}")
        print(f"{'='*60}\n")

        # 1. Extract archives
        print("[1/7] Extracting archives...")
        self.extract_archives()

        # 2. Build inventory
        print("[2/7] Building file inventory...")
        self.build_inventory(out)

        # 3. Process JSON
        print("[3/7] Processing JSON/JSONL files...")
        json_files = list(self.root.rglob("*.json")) + list(self.root.rglob("*.jsonl"))
        for file in tqdm(json_files, desc="  JSON", unit="file"):
            self.process_json(file)

        # 4. Process Python
        print("[4/7] Processing Python files...")
        py_files = list(self.root.rglob("*.py"))
        for file in tqdm(py_files, desc="  Python", unit="file"):
            self.process_python(file)

        # 5. Process HTML
        print("[5/7] Processing HTML files...")
        html_files = [
            f for f in self.root.rglob("*.html")
            if not any(part in _HTML_NOISE_DIRS for part in f.parts)
            and f.name not in _HTML_NOISE_FILES
        ]
        for file in tqdm(html_files, desc="  HTML", unit="file"):
            self.process_html(file)

        # 6. Post-processing
        print("[6/7] Post-processing (signatures, relationships, workflows)...")
        filled = self._fill_missing_signatures()
        print(f"  Signatures filled from docstrings: {filled}")
        self._extract_workflows_from_examples()
        print(f"  Workflows extracted: {len(self.workflows)}")
        self._enrich_relationships()
        print(f"  Relationships enriched")

        # 7. Write output
        print("[7/7] Writing output files...")
        manifest = self.write(out)

        # Summary
        print(f"\n{'='*60}")
        print(f"  EXTRACTION COMPLETE")
        print(f"{'='*60}")
        print(f"  Valid records:         {manifest['valid_records']:,}")
        print(f"  Invalid (skipped):     {manifest['invalid_records']:,}")
        print(f"  Deprecated:            {manifest['deprecated_records']:,}")
        print(f"  With parameters:       {manifest['records_with_parameters']:,}")
        print(f"  With return types:     {manifest['records_with_returns']:,}")
        print(f"  With examples:         {manifest['records_with_examples']:,}")
        print(f"  With navigation:       {manifest['records_with_navigation']:,}")
        print(f"  Examples:              {manifest['examples_count']:,}")
        print(f"  Workflows:             {manifest['workflows_count']:,}")
        print(f"  JSON records:          {manifest['json_records']:,}")
        print(f"  Python records:        {manifest['py_records']:,}")
        print(f"  HTML records:          {manifest['html_records']:,}")
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
        description="ANSA/META CodeRAG+GraphRAG Ingestion — HTML/JSON/Python -> Normalized JSON"
    )
    parser.add_argument(
        "source_dir",
        help="Directory containing docs (.py, .html, .json, .jsonl)",
    )
    default_output = str(Path(__file__).resolve().parent.parent / "knowledge-base")
    parser.add_argument(
        "--output",
        default=default_output,
        help=f"Output directory for JSONL files (default: {default_output})",
    )
    parser.add_argument(
        "--software",
        choices=["ansa", "meta"],
        default=None,
        help="Force software type (auto-detected from folder name if not set)",
    )
    args = parser.parse_args()

    # Auto-detect software from folder name
    if args.software:
        software = args.software
    else:
        source_lower = args.source_dir.lower()
        software = "meta" if "meta" in source_lower else "ansa"

    extractor = KnowledgeExtractor(
        root_dir=args.source_dir,
        software=software,
    )
    extractor.run(output_dir=args.output)
