"""PyDyna Keyword Retriever — hybrid retrieval for LS-DYNA keywords.

Three retrieval modes:
  1. Exact match   : lookup_keyword("*MAT_024") → full card definition
  2. Semantic search: search_keywords("elastic material for steel")
  3. Parameter validation: validate_keyword(keyword, params)

Backed by:
  - ChromaDB vector search (semantic)
  - In-memory JSONL index (exact match + param validation)

Usage:
    from bin.keyword_retriever import KeywordRetriever

    retriever = KeywordRetriever()
    retriever.load()  # loads JSONL index + connects to ChromaDB

    # Exact match
    result = retriever.lookup_keyword("*MAT_024")

    # Semantic search
    results = retriever.search_keywords("contact between deformable parts")

    # Get material by use case
    results = retriever.search_by_category("material", "crash steel")

    # Validate parameters
    issues = retriever.validate_keyword("*MAT_024", {"ro": 7.85e-9, "e": 210000})
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

from bin.build_vector_db import (
    VectorStoreBuilder,
    KEYWORD_COLLECTION,
    API_COLLECTION,
    _AGENT_DIR,
    _KB_DIR,
    _VECTOR_DB_DIR,
)


logger = logging.getLogger(__name__)


# =============================================================================
# In-memory Keyword Index (for exact match + validation)
# =============================================================================

class KeywordIndex:
    """In-memory index over keyword_reference.jsonl for O(1) exact lookups."""

    def __init__(self):
        self._by_name: dict[str, dict] = {}      # "*MAT_024" -> record
        self._by_class: dict[str, dict] = {}      # "Mat024" -> record
        self._by_category: dict[str, list] = {}   # "material" -> [records]
        self._loaded = False

    def load(self, source: Optional[Path] = None) -> int:
        """Load keyword_reference.jsonl into memory."""
        source = source or (_KB_DIR / "keyword_reference.jsonl")
        if not source.exists():
            logger.warning(f"Keyword reference not found: {source}")
            return 0

        count = 0
        with open(source, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                name = rec.get("keyword", "")
                cls = rec.get("class_name", "")
                cat = rec.get("category", "")

                self._by_name[name.upper()] = rec
                self._by_name[name] = rec
                if cls:
                    self._by_class[cls] = rec
                    self._by_class[cls.lower()] = rec

                if cat:
                    self._by_category.setdefault(cat, []).append(rec)

                count += 1

        self._loaded = True
        logger.info(f"KeywordIndex loaded: {count} keywords, "
                    f"{len(self._by_category)} categories")
        return count

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def get(self, name: str) -> Optional[dict]:
        """Exact lookup by keyword name or class name.

        Accepts: "*MAT_024", "MAT_024", "Mat024", "mat024"
        """
        # Try as-is
        rec = self._by_name.get(name) or self._by_class.get(name)
        if rec:
            return rec

        # Try with * prefix
        if not name.startswith("*"):
            rec = self._by_name.get(f"*{name.upper()}")
            if rec:
                return rec

        # Try uppercase
        rec = self._by_name.get(name.upper())
        if rec:
            return rec

        # Try lowercase class lookup
        rec = self._by_class.get(name.lower())
        return rec

    def get_by_category(self, category: str) -> list[dict]:
        """Get all keywords in a category."""
        return self._by_category.get(category, [])

    def list_categories(self) -> dict[str, int]:
        """Return category -> count mapping."""
        return {cat: len(recs) for cat, recs in self._by_category.items()}

    def search_name(self, pattern: str) -> list[dict]:
        """Regex search over keyword names."""
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error:
            return []
        return [
            rec for name, rec in self._by_name.items()
            if name.startswith("*") and regex.search(name)
        ]


# =============================================================================
# Parameter Validator
# =============================================================================

class ParameterValidator:
    """Validate keyword parameters against the keyword spec."""

    @staticmethod
    def validate(keyword_record: dict, params: dict) -> list[dict]:
        """Check provided params against the keyword's field definitions.

        Returns list of issues: [{"field", "severity", "message"}]
        """
        issues = []
        all_fields = {}

        # Build field map from cards
        for card in keyword_record.get("cards", []):
            for field_def in card.get("fields", []):
                all_fields[field_def["name"]] = field_def

        if not all_fields:
            return [{"field": "_meta", "severity": "warning",
                     "message": "No field definitions found for this keyword"}]

        # Check for unknown params
        for param_name in params:
            if param_name.lower() not in {f.lower() for f in all_fields}:
                issues.append({
                    "field": param_name,
                    "severity": "warning",
                    "message": f"Unknown parameter '{param_name}'. "
                               f"Valid fields: {', '.join(sorted(all_fields.keys())[:15])}",
                })

        # Check required fields (those without defaults)
        for fname, fdef in all_fields.items():
            if fdef.get("default") is None:
                # Field has no default → likely required
                if fname not in params and fname.lower() not in {
                    p.lower() for p in params
                }:
                    issues.append({
                        "field": fname,
                        "severity": "info",
                        "message": f"Parameter '{fname}' has no default — "
                                   f"consider providing a value. "
                                   f"Type: {fdef.get('type', 'unknown')}",
                    })

        # Type checking
        for param_name, param_value in params.items():
            field_def = all_fields.get(param_name)
            if field_def is None:
                # Try case-insensitive match
                for fn, fd in all_fields.items():
                    if fn.lower() == param_name.lower():
                        field_def = fd
                        break
            if field_def is None:
                continue

            expected_type = field_def.get("type", "")
            if expected_type == "int" and isinstance(param_value, float):
                if param_value != int(param_value):
                    issues.append({
                        "field": param_name,
                        "severity": "warning",
                        "message": f"'{param_name}' expects int but got float {param_value}",
                    })
            elif expected_type == "float" and isinstance(param_value, str):
                issues.append({
                    "field": param_name,
                    "severity": "error",
                    "message": f"'{param_name}' expects float but got string '{param_value}'",
                })

        return issues


# =============================================================================
# Keyword Retriever (main public API)
# =============================================================================

class KeywordRetriever:
    """Unified retrieval interface for LS-DYNA keywords and PyDyna API.

    Combines:
    - In-memory exact match (KeywordIndex)
    - ChromaDB semantic search (VectorStoreBuilder)
    - Parameter validation (ParameterValidator)
    """

    def __init__(self, persist_dir: Optional[str] = None):
        self._index = KeywordIndex()
        self._store = VectorStoreBuilder(persist_dir=persist_dir)
        self._validator = ParameterValidator()
        self._loaded = False

    def load(self) -> dict:
        """Load both the in-memory index and verify ChromaDB."""
        kw_count = self._index.load()
        stats = self._store.get_stats()
        self._loaded = True
        return {
            "keyword_index": kw_count,
            "vector_db": stats,
        }

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ----- 1. Exact Match -----

    def lookup_keyword(self, name: str) -> str:
        """Exact lookup for an LS-DYNA keyword by name or class.

        Use for known keywords like *MAT_024, *CONTACT_AUTOMATIC_SINGLE_SURFACE.

        Args:
            name: Keyword name (*MAT_024) or class name (Mat024)

        Returns:
            JSON string with full keyword definition (cards, params, links)
        """
        if not self._index.is_loaded:
            self.load()

        rec = self._index.get(name)
        if rec is None:
            # Try fuzzy: search name patterns
            candidates = self._index.search_name(name.replace("*", ".*"))
            if candidates:
                return json.dumps({
                    "found": False,
                    "message": f"'{name}' not found. Did you mean:",
                    "suggestions": [c["keyword"] for c in candidates[:5]],
                }, indent=2)
            return json.dumps({
                "found": False,
                "message": f"Keyword '{name}' not found in reference.",
            })

        return json.dumps({
            "found": True,
            "keyword": rec["keyword"],
            "class_name": rec.get("class_name", ""),
            "category": rec.get("category", ""),
            "description": rec.get("description", ""),
            "card_count": rec.get("card_count", 0),
            "param_count": rec.get("param_count", 0),
            "cards": rec.get("cards", []),
            "link_fields": rec.get("link_fields", {}),
            "options": rec.get("options", []),
        }, indent=2)

    # ----- 2. Semantic Search -----

    def search_keywords(
        self, query: str, top_k: int = 5, category: str = "",
    ) -> str:
        """Semantic search for LS-DYNA keywords.

        Use for natural-language queries like "material for steel crash",
        "contact between deformable and rigid parts", "output energy balance".

        Args:
            query: Natural language description
            top_k: Number of results (1-10)
            category: Filter by category (material, contact, control, etc.)

        Returns:
            JSON string with matching keywords
        """
        where = {"category": category} if category else None

        results = self._store.search(
            query=query,
            collection_name=KEYWORD_COLLECTION,
            top_k=min(top_k, 10),
            where=where,
        )

        if not results:
            return json.dumps({
                "results": [],
                "message": "No keywords found. Try rephrasing.",
            })

        formatted = []
        for r in results:
            meta = r["metadata"]
            formatted.append({
                "keyword": meta.get("keyword", ""),
                "class_name": meta.get("class_name", ""),
                "category": meta.get("category", ""),
                "param_count": meta.get("param_count", 0),
                "score": round(r["score"], 3),
            })

        return json.dumps({"results": formatted}, indent=2)

    def search_pydyna_api(
        self, query: str, top_k: int = 5, category: str = "",
    ) -> str:
        """Semantic search for PyDyna Python API classes.

        Use for queries about PyDyna SDK usage, e.g.,
        "how to set timestep in PyDyna", "material class for plasticity".

        Args:
            query: Natural language description
            top_k: Number of results (1-10)
            category: Filter by category

        Returns:
            JSON string with matching API classes
        """
        where = {"category": category} if category else None

        results = self._store.search(
            query=query,
            collection_name=API_COLLECTION,
            top_k=min(top_k, 10),
            where=where,
        )

        if not results:
            return json.dumps({
                "results": [],
                "message": "No API docs found.",
            })

        formatted = []
        for r in results:
            meta = r["metadata"]
            formatted.append({
                "symbol": meta.get("symbol", ""),
                "module": meta.get("module", ""),
                "keyword_ref": meta.get("keyword_ref", ""),
                "category": meta.get("category", ""),
                "signature": meta.get("signature", ""),
                "content": r["content"][:400],
                "score": round(r["score"], 3),
            })

        return json.dumps({"results": formatted}, indent=2)

    # ----- 3. Category-based -----

    def search_by_category(
        self, category: str, query: str = "", top_k: int = 10,
    ) -> str:
        """Get keywords by category with optional semantic filtering.

        Args:
            category: material, contact, control, section, database, etc.
            query: Optional semantic filter within category
            top_k: Max results

        Returns:
            JSON string with keywords in that category
        """
        if query:
            return self.search_keywords(query, top_k=top_k, category=category)

        # No query → return all in category from index
        if not self._index.is_loaded:
            self.load()

        records = self._index.get_by_category(category)
        if not records:
            return json.dumps({
                "results": [],
                "categories": self._index.list_categories(),
                "message": f"Category '{category}' not found.",
            })

        formatted = [{
            "keyword": r["keyword"],
            "class_name": r.get("class_name", ""),
            "param_count": r.get("param_count", 0),
        } for r in records[:top_k]]

        return json.dumps({
            "category": category,
            "total": len(records),
            "returned": len(formatted),
            "results": formatted,
        }, indent=2)

    # ----- 4. Parameter Validation -----

    def validate_keyword(
        self, keyword_name: str, params: dict,
    ) -> str:
        """Validate parameters against a keyword's specification.

        Args:
            keyword_name: Keyword name (e.g., "*MAT_024")
            params: Dict of parameter name -> value

        Returns:
            JSON string with validation results
        """
        if not self._index.is_loaded:
            self.load()

        rec = self._index.get(keyword_name)
        if rec is None:
            return json.dumps({
                "valid": False,
                "message": f"Keyword '{keyword_name}' not found.",
            })

        issues = self._validator.validate(rec, params)

        errors = [i for i in issues if i["severity"] == "error"]
        warnings = [i for i in issues if i["severity"] == "warning"]
        infos = [i for i in issues if i["severity"] == "info"]

        return json.dumps({
            "keyword": keyword_name,
            "valid": len(errors) == 0,
            "errors": len(errors),
            "warnings": len(warnings),
            "info": len(infos),
            "issues": issues,
        }, indent=2)

    # ----- 5. Model Helper -----

    def get_material_model(self, description: str) -> str:
        """Find the best LS-DYNA material model for a description.

        Use for questions like "what material for steel crash?",
        "rigid wall material", "foam for seat cushion".

        Args:
            description: Natural language description of the material need

        Returns:
            JSON with recommended material keywords
        """
        return self.search_keywords(
            query=f"material {description}",
            top_k=5,
            category="material",
        )

    def get_contact_type(self, description: str) -> str:
        """Find the best LS-DYNA contact type for a scenario.

        Use for questions like "contact between bumper and barrier",
        "tied contact for spot welds", "self-contact for sheet metal".

        Args:
            description: Natural language description of the contact scenario

        Returns:
            JSON with recommended contact keywords
        """
        return self.search_keywords(
            query=f"contact {description}",
            top_k=5,
            category="contact",
        )

    # ----- Stats -----

    def get_stats(self) -> str:
        """Return statistics about loaded data."""
        if not self._index.is_loaded:
            self.load()

        cats = self._index.list_categories()
        db_stats = self._store.get_stats()

        return json.dumps({
            "keyword_index": {
                "total": sum(cats.values()),
                "categories": cats,
            },
            "vector_db": db_stats,
        }, indent=2)


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="PyDyna Keyword Retriever — search and validate LS-DYNA keywords"
    )
    parser.add_argument(
        "--lookup", type=str,
        help="Exact keyword lookup (e.g., *MAT_024)",
    )
    parser.add_argument(
        "--search", type=str,
        help="Semantic search query",
    )
    parser.add_argument(
        "--category", type=str, default="",
        help="Filter by category",
    )
    parser.add_argument(
        "--api", type=str,
        help="Search PyDyna API docs",
    )
    parser.add_argument(
        "--validate", type=str,
        help="Validate keyword (requires --params)",
    )
    parser.add_argument(
        "--params", type=str,
        help='JSON dict of params (e.g., \'{"ro": 7.85e-9, "e": 210000}\')',
    )
    parser.add_argument(
        "--material", type=str,
        help="Find material model by description",
    )
    parser.add_argument(
        "--contact", type=str,
        help="Find contact type by description",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Show retriever statistics",
    )
    args = parser.parse_args()

    retriever = KeywordRetriever()
    retriever.load()

    if args.stats:
        print(retriever.get_stats())
    elif args.lookup:
        print(retriever.lookup_keyword(args.lookup))
    elif args.search:
        print(retriever.search_keywords(args.search, category=args.category))
    elif args.api:
        print(retriever.search_pydyna_api(args.api, category=args.category))
    elif args.validate and args.params:
        params = json.loads(args.params)
        print(retriever.validate_keyword(args.validate, params))
    elif args.material:
        print(retriever.get_material_model(args.material))
    elif args.contact:
        print(retriever.get_contact_type(args.contact))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
