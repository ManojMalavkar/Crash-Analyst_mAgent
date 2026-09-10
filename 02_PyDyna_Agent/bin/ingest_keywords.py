"""PyDyna Keyword Ingestion — parse PyDyna source files into JSONL.

Reads the auto-generated keyword class files from the PyDyna SDK
(ansys.dyna.core.keywords.keyword_classes.auto.*) and extracts:
  - Keyword name (*MAT_024, *CONTACT_AUTOMATIC_SINGLE_SURFACE, etc.)
  - Card fields with name, type, width, default
  - Property docstrings (parameter descriptions)
  - Link fields (references to DEFINE_CURVE, etc.)
  - Class hierarchy (keyword + subkeyword)

Outputs two JSONL files:
  - keyword_reference.jsonl  : LS-DYNA keyword cards with all parameters
  - pydyna_docs.jsonl        : PyDyna Python API docs (classes + methods)

Usage:
    # Parse from local PyDyna source clone
    python bin/ingest_keywords.py --source /path/to/pydyna/src/ansys/dyna/core/keywords/keyword_classes/auto

    # Default: looks for ../pydyna relative to this repo
    python bin/ingest_keywords.py

Pipeline:
    Step 1: python bin/ingest_keywords.py --source <pydyna_auto_dir>
    Step 2: python bin/build_vector_db.py --rebuild
"""

import ast
import json
import logging
import re
import sys
import argparse
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict


logger = logging.getLogger(__name__)

_AGENT_DIR = Path(__file__).resolve().parent.parent          # 02_PyDyna_Agent/
_KB_DIR = _AGENT_DIR / "knowledge-base"
_DEFAULT_PYDYNA_AUTO = (
    Path(__file__).resolve().parent.parent.parent.parent      # repo root/../
    / "pydyna" / "src" / "ansys" / "dyna" / "core"
    / "keywords" / "keyword_classes" / "auto"
)

# ── Crash-relevant keyword categories (prioritized for safety engineering) ──
_CRASH_CATEGORIES = {
    "mat":         "material",
    "contact":     "contact",
    "section":     "section",
    "control":     "control",
    "database":    "database",
    "boundary":    "boundary",
    "load":        "load",
    "define":      "define",
    "constrained": "constrained",
    "set":         "set",
    "part":        "part",
    "element":     "element",
    "node":        "node",
    "rigid":       "rigid",
    "rigidwall":   "rigidwall",
    "damping":     "damping",
    "hourglass":   "hourglass",
    "airbag":      "airbag",
    "sensor":      "sensor",
    "eos":         "eos",
    "include":     "include",
    "parameter":   "parameter",
    "delete":      "delete",
    "ale":         "ale",
    "other":       "other",
}


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class FieldInfo:
    """One field on a keyword card."""
    name: str
    type: str           # "int", "float", "str"
    offset: int
    width: int
    default: Optional[str]
    description: str = ""


@dataclass
class CardInfo:
    """One card (line) in a keyword."""
    card_index: int
    fields: list[FieldInfo] = field(default_factory=list)


@dataclass
class KeywordRecord:
    """Complete keyword parsed from a .py source file."""
    keyword: str            # e.g. "MAT"
    subkeyword: str         # e.g. "024"
    full_name: str          # e.g. "*MAT_024"
    class_name: str         # e.g. "Mat024"
    module_path: str        # e.g. "auto.mat.mat_024"
    category: str           # e.g. "material"
    docstring: str
    cards: list[CardInfo] = field(default_factory=list)
    link_fields: dict = field(default_factory=dict)
    option_specs: list[str] = field(default_factory=list)
    properties: dict = field(default_factory=dict)  # name -> description


@dataclass
class PyDynaAPIRecord:
    """PyDyna Python API record for pydyna_docs.jsonl."""
    symbol: str
    type: str               # "class", "method", "property"
    module: str
    description: str
    signature: str = ""
    parent_class: str = ""
    category: str = ""
    params: list[dict] = field(default_factory=list)
    keyword_ref: str = ""
    methods: list[str] = field(default_factory=list)
    docstring: str = ""


# =============================================================================
# AST-based Parser
# =============================================================================

class KeywordFileParser:
    """Parse a single PyDyna keyword .py file using the AST."""

    def __init__(self, filepath: Path, category: str):
        self.filepath = filepath
        self.category = category
        self._source = filepath.read_text(encoding="utf-8")
        self._tree = ast.parse(self._source)

    def parse(self) -> Optional[KeywordRecord]:
        """Extract keyword record from the source file."""
        # Find the main class (inherits from KeywordBase)
        cls_node = self._find_keyword_class()
        if cls_node is None:
            return None

        keyword = self._get_class_attr(cls_node, "keyword") or ""
        subkeyword = self._get_class_attr(cls_node, "subkeyword") or ""
        full_name = f"*{keyword}"
        if subkeyword:
            full_name += f"_{subkeyword}"

        docstring = ast.get_docstring(cls_node) or ""

        # Parse FieldSchema tuples from module-level assignments
        cards = self._parse_field_schemas()

        # Parse link_fields dict
        link_fields = self._parse_link_fields(cls_node)

        # Parse option specs
        option_specs = self._parse_option_specs(cls_node)

        # Parse property docstrings
        properties = self._parse_properties(cls_node)

        # Enrich field descriptions from properties
        for card in cards:
            for fld in card.fields:
                if fld.name in properties:
                    fld.description = properties[fld.name]

        module_path = f"auto.{self.category}.{self.filepath.stem}"

        return KeywordRecord(
            keyword=keyword,
            subkeyword=subkeyword,
            full_name=full_name,
            class_name=cls_node.name,
            module_path=module_path,
            category=_CRASH_CATEGORIES.get(self.category, self.category),
            docstring=docstring,
            cards=cards,
            link_fields=link_fields,
            option_specs=option_specs,
            properties=properties,
        )

    def _find_keyword_class(self) -> Optional[ast.ClassDef]:
        """Find the first class that inherits from KeywordBase."""
        for node in ast.walk(self._tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    base_name = ""
                    if isinstance(base, ast.Name):
                        base_name = base.id
                    elif isinstance(base, ast.Attribute):
                        base_name = base.attr
                    if base_name == "KeywordBase":
                        return node
        return None

    def _get_class_attr(self, cls_node: ast.ClassDef, attr_name: str) -> Optional[str]:
        """Get a simple string/constant class attribute."""
        for item in cls_node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name) and target.id == attr_name:
                        if isinstance(item.value, ast.Constant):
                            return str(item.value.value)
        return None

    def _parse_field_schemas(self) -> list[CardInfo]:
        """Parse all _KEYWORD_CARDn = (FieldSchema(...), ...) tuples."""
        cards = []
        card_pattern = re.compile(r"_\w+_CARD(\d+)$")

        for node in self._tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        match = card_pattern.match(target.id)
                        # Skip OPTION cards — they are metadata, not keyword data
                        if match and "OPTION" not in target.id:
                            card_idx = int(match.group(1))
                            fields = self._extract_fields_from_tuple(node.value)
                            if fields:
                                cards.append(CardInfo(
                                    card_index=card_idx,
                                    fields=fields,
                                ))
        return cards

    def _extract_fields_from_tuple(self, node: ast.AST) -> list[FieldInfo]:
        """Extract FieldSchema calls from a tuple assignment."""
        fields = []
        if isinstance(node, ast.Tuple):
            for elt in node.elts:
                fld = self._parse_field_schema_call(elt)
                if fld and fld.name != "unused":
                    fields.append(fld)
        return fields

    def _parse_field_schema_call(self, node: ast.AST) -> Optional[FieldInfo]:
        """Parse a single FieldSchema(...) call."""
        if not isinstance(node, ast.Call):
            return None

        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        if func_name != "FieldSchema":
            return None

        args = node.args
        if len(args) < 4:
            return None

        name = self._get_constant(args[0]) or ""
        type_name = self._get_type_name(args[1])
        offset = self._get_constant(args[2]) or 0
        width = self._get_constant(args[3]) or 10
        default = self._get_constant(args[4]) if len(args) > 4 else None

        return FieldInfo(
            name=str(name),
            type=type_name,
            offset=int(offset),
            width=int(width),
            default=str(default) if default is not None else None,
        )

    @staticmethod
    def _get_constant(node: ast.AST):
        """Extract a constant value from an AST node."""
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            if isinstance(node.operand, ast.Constant):
                return -node.operand.value
        return None

    @staticmethod
    def _get_type_name(node: ast.AST) -> str:
        """Extract type name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return "unknown"

    def _parse_link_fields(self, cls_node: ast.ClassDef) -> dict:
        """Parse _link_fields = {...} from class body."""
        for item in cls_node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name) and target.id == "_link_fields":
                        if isinstance(item.value, ast.Dict):
                            result = {}
                            for k, v in zip(item.value.keys, item.value.values):
                                key = self._get_constant(k)
                                val = ""
                                if isinstance(v, ast.Attribute):
                                    val = v.attr
                                elif isinstance(v, ast.Name):
                                    val = v.id
                                if key:
                                    result[str(key)] = str(val)
                            return result
        return {}

    def _parse_option_specs(self, cls_node: ast.ClassDef) -> list[str]:
        """Parse _option_spec_list for option names (e.g., TITLE)."""
        for item in cls_node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name) and target.id == "_option_spec_list":
                        if isinstance(item.value, ast.List):
                            specs = []
                            for elt in item.value.elts:
                                if isinstance(elt, ast.Call) and elt.args:
                                    name = self._get_constant(elt.args[0])
                                    if name:
                                        specs.append(str(name))
                            return specs
        return []

    def _parse_properties(self, cls_node: ast.ClassDef) -> dict:
        """Parse @property docstrings for field descriptions."""
        props = {}
        for item in cls_node.body:
            if isinstance(item, ast.FunctionDef):
                # Check if it's a property getter (has @property decorator)
                is_prop = any(
                    (isinstance(d, ast.Name) and d.id == "property")
                    or (isinstance(d, ast.Attribute) and d.attr == "property")
                    for d in item.decorator_list
                )
                if is_prop:
                    doc = ast.get_docstring(item)
                    if doc:
                        # Clean up the docstring
                        doc = doc.strip().replace("\n", " ")
                        doc = re.sub(r"\s+", " ", doc)
                        # Remove "Get or set the" prefix
                        doc = re.sub(r"^Get or set the\s+", "", doc)
                        props[item.name] = doc
        return props


# =============================================================================
# JSONL Builders
# =============================================================================

def build_keyword_record(kw: KeywordRecord) -> dict:
    """Convert KeywordRecord to keyword_reference.jsonl format."""
    cards_data = []
    all_params = []

    for card in kw.cards:
        card_fields = []
        for f in card.fields:
            param = {
                "name": f.name,
                "type": f.type,
                "width": f.width,
                "description": f.description,
            }
            if f.default is not None:
                param["default"] = f.default
            card_fields.append(param)
            all_params.append(param)
        cards_data.append({
            "card": card.card_index,
            "fields": card_fields,
        })

    return {
        "keyword": kw.full_name,
        "class_name": kw.class_name,
        "module": f"ansys.dyna.core.keywords.keyword_classes.{kw.module_path}",
        "category": kw.category,
        "description": kw.docstring,
        "cards": cards_data,
        "params": all_params,
        "link_fields": kw.link_fields,
        "options": kw.option_specs,
        "param_count": len(all_params),
        "card_count": len(cards_data),
    }


def build_pydyna_api_record(kw: KeywordRecord) -> dict:
    """Convert KeywordRecord to pydyna_docs.jsonl format."""
    # Build constructor params from first card
    params = []
    for card in kw.cards:
        for f in card.fields:
            params.append({
                "name": f.name,
                "type": f.type,
                "description": f.description,
            })

    # Build property list as methods
    prop_names = list(kw.properties.keys())

    return {
        "symbol": kw.class_name,
        "type": "class",
        "module": f"ansys.dyna.core.keywords.keyword_classes.{kw.module_path}",
        "description": kw.docstring,
        "signature": f"{kw.class_name}(**kwargs)",
        "category": kw.category,
        "params": params,
        "keyword_ref": kw.full_name,
        "methods": prop_names,
        "docstring": f"{kw.class_name} represents the {kw.full_name} keyword. "
                     f"Properties: {', '.join(prop_names[:8])}."
                     f" Category: {kw.category}.",
    }


# =============================================================================
# Directory Walker
# =============================================================================

def discover_keyword_files(auto_dir: Path) -> list[tuple[Path, str]]:
    """Walk the auto/ directory and discover all keyword .py files.

    Returns:
        List of (filepath, category) tuples.
    """
    results = []

    if not auto_dir.exists():
        logger.error(f"PyDyna auto directory not found: {auto_dir}")
        return results

    for category_dir in sorted(auto_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        if category.startswith("_") or category.startswith("."):
            continue

        for py_file in sorted(category_dir.glob("*.py")):
            if py_file.name.startswith("__"):
                continue
            results.append((py_file, category))

    logger.info(f"Discovered {len(results)} keyword files across "
                f"{len(set(r[1] for r in results))} categories")
    return results


def ingest_all(
    auto_dir: Path,
    output_keywords: Path,
    output_pydyna: Path,
    categories: Optional[set[str]] = None,
) -> tuple[int, int]:
    """Parse all keyword files and write JSONL outputs.

    Args:
        auto_dir:        Path to .../keyword_classes/auto/
        output_keywords: Path to keyword_reference.jsonl
        output_pydyna:   Path to pydyna_docs.jsonl (appended, not overwritten)
        categories:      Optional filter — only process these categories.

    Returns:
        (keyword_count, api_count) tuple.
    """
    files = discover_keyword_files(auto_dir)
    if not files:
        return 0, 0

    kw_count = 0
    api_count = 0
    errors = 0

    with open(output_keywords, "w", encoding="utf-8") as kw_f, \
         open(output_pydyna, "w", encoding="utf-8") as api_f:

        for filepath, category in files:
            if categories and category not in categories:
                continue

            try:
                parser = KeywordFileParser(filepath, category)
                record = parser.parse()
                if record is None:
                    continue

                # Write keyword reference
                kw_json = build_keyword_record(record)
                kw_f.write(json.dumps(kw_json, ensure_ascii=False) + "\n")
                kw_count += 1

                # Write API doc
                api_json = build_pydyna_api_record(record)
                api_f.write(json.dumps(api_json, ensure_ascii=False) + "\n")
                api_count += 1

            except Exception as exc:
                errors += 1
                logger.warning(f"Failed to parse {filepath.name}: {exc}")

    logger.info(
        f"Ingestion complete: {kw_count} keywords, {api_count} API records, "
        f"{errors} errors"
    )
    return kw_count, api_count


# =============================================================================
# CLI
# =============================================================================

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Ingest PyDyna keyword source files into JSONL knowledge base"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=_DEFAULT_PYDYNA_AUTO,
        help="Path to pydyna/src/ansys/dyna/core/keywords/keyword_classes/auto/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_KB_DIR,
        help="Output directory for JSONL files (default: knowledge-base/)",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        help="Only process these categories (e.g., mat contact control)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print per-category statistics after ingestion",
    )
    args = parser.parse_args()

    output_keywords = args.output_dir / "keyword_reference.jsonl"
    output_pydyna = args.output_dir / "pydyna_docs.jsonl"
    categories = set(args.categories) if args.categories else None

    logger.info(f"Source:  {args.source}")
    logger.info(f"Output:  {args.output_dir}")
    if categories:
        logger.info(f"Filter:  {categories}")

    kw_count, api_count = ingest_all(
        auto_dir=args.source,
        output_keywords=output_keywords,
        output_pydyna=output_pydyna,
        categories=categories,
    )

    print(f"\n{'='*60}")
    print(f"  Ingestion Complete")
    print(f"{'='*60}")
    print(f"  keyword_reference.jsonl : {kw_count:>6} records")
    print(f"  pydyna_docs.jsonl       : {api_count:>6} records")
    print(f"{'='*60}")

    if args.stats:
        # Read back and count per category
        print(f"\nPer-category breakdown:")
        cats = {}
        with open(output_keywords, "r") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    cat = rec.get("category", "unknown")
                    cats[cat] = cats.get(cat, 0) + 1
        for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
            print(f"  {cat:<20} {count:>5}")


if __name__ == "__main__":
    main()
