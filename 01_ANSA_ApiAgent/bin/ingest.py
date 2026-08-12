"""Knowledge Base Ingestion Pipeline for ANSA/META API Documentation.

Parses ANSA and META API documentation from multiple source formats
(HTML, JSON, Python source files) into structured chunks suitable
for embedding and vector storage.

Supported source formats:
- HTML: Parsed API reference pages (class/function docs)
- JSON/JSONL: Pre-structured API documentation exports
- Python (.py): Source code with docstrings
- Markdown (.md): Tutorial and guide documents

Output: List of Document objects with content, metadata, and relationships.

Usage:
    from bin.ingest import IngestionPipeline
    
    pipeline = IngestionPipeline(source_dir="knowledge-base/raw")
    documents = pipeline.run()
    print(f"Ingested {len(documents)} documents")
"""

import re
import ast
import json
import hashlib
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from html.parser import HTMLParser


logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class Document:
    """A single document chunk ready for embedding."""
    
    content: str                    # Text content to embed
    doc_id: str = ""               # Unique ID (generated from content hash)
    
    # Metadata for filtering and display
    source_file: str = ""          # Original file path
    doc_type: str = ""             # "class", "function", "method", "module", "guide"
    module_name: str = ""          # e.g., "ansa.base", "meta.post"
    class_name: str = ""           # Parent class (if method)
    function_name: str = ""        # Function/method name
    signature: str = ""            # Full function signature
    return_type: str = ""          # Return type annotation
    
    # Relationships (for knowledge graph)
    parent_class: str = ""         # Inheritance parent
    imports: list = field(default_factory=list)  # Required imports
    related_functions: list = field(default_factory=list)  # Cross-references
    
    # Classification
    api_category: str = ""         # "mesh", "geometry", "material", "contact", etc.
    software: str = ""             # "ansa" or "meta"
    
    def __post_init__(self):
        if not self.doc_id:
            self.doc_id = hashlib.md5(self.content.encode()).hexdigest()[:12]
    
    def to_metadata(self) -> dict:
        """Convert to flat metadata dict for ChromaDB storage."""
        return {
            k: v for k, v in {
                "source_file": self.source_file,
                "doc_type": self.doc_type,
                "module_name": self.module_name,
                "class_name": self.class_name,
                "function_name": self.function_name,
                "signature": self.signature,
                "return_type": self.return_type,
                "parent_class": self.parent_class,
                "api_category": self.api_category,
                "software": self.software,
            }.items() if v  # Only include non-empty fields
        }


# =============================================================================
# Source Parsers
# =============================================================================

class PythonSourceParser:
    """Parse Python source files to extract classes, functions, and docstrings."""
    
    def __init__(self, software: str = "ansa"):
        self.software = software
    
    def parse_file(self, filepath: Path) -> list[Document]:
        """Parse a Python file and extract documented symbols."""
        documents = []
        source = filepath.read_text(encoding="utf-8", errors="ignore")
        
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            logger.warning(f"Syntax error in {filepath}: {e}")
            return documents
        
        module_name = self._infer_module_name(filepath)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                doc = self._extract_class(node, module_name, filepath)
                if doc:
                    documents.append(doc)
                
                # Extract methods
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        doc = self._extract_function(
                            item, module_name, filepath,
                            class_name=node.name
                        )
                        if doc:
                            documents.append(doc)
            
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not self._is_nested(node, tree):
                    doc = self._extract_function(node, module_name, filepath)
                    if doc:
                        documents.append(doc)
        
        return documents
    
    def _extract_class(self, node: ast.ClassDef, module: str, filepath: Path) -> Optional[Document]:
        """Extract class documentation."""
        docstring = ast.get_docstring(node) or ""
        bases = [self._get_name(b) for b in node.bases]
        
        methods = [
            item.name for item in node.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            and not item.name.startswith("_")
        ]
        
        content = f"""Class: {module}.{node.name}
Inherits from: {', '.join(bases) if bases else 'object'}

{docstring}

Public methods: {', '.join(methods[:20])}"""
        
        return Document(
            content=content.strip(),
            source_file=str(filepath),
            doc_type="class",
            module_name=module,
            class_name=node.name,
            parent_class=bases[0] if bases else "",
            software=self.software,
            related_functions=methods[:20],
        )
    
    def _extract_function(
        self, node, module: str, filepath: Path, class_name: str = ""
    ) -> Optional[Document]:
        """Extract function/method documentation."""
        docstring = ast.get_docstring(node) or ""
        if not docstring and node.name.startswith("_"):
            return None  # Skip undocumented private methods
        
        signature = self._get_signature(node)
        full_name = f"{module}.{class_name}.{node.name}" if class_name else f"{module}.{node.name}"
        
        content = f"""Function: {full_name}
Signature: {signature}

{docstring}"""
        
        return Document(
            content=content.strip(),
            source_file=str(filepath),
            doc_type="method" if class_name else "function",
            module_name=module,
            class_name=class_name,
            function_name=node.name,
            signature=signature,
            software=self.software,
        )
    
    def _get_signature(self, node) -> str:
        """Reconstruct function signature from AST."""
        args = []
        for arg in node.args.args:
            name = arg.arg
            if arg.annotation:
                name += f": {ast.unparse(arg.annotation)}"
            args.append(name)
        
        sig = f"{node.name}({', '.join(args)})"
        if node.returns:
            sig += f" -> {ast.unparse(node.returns)}"
        return sig
    
    def _get_name(self, node) -> str:
        """Get name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return "unknown"
    
    def _infer_module_name(self, filepath: Path) -> str:
        """Infer module name from file path."""
        parts = filepath.stem.split("_")
        if self.software in str(filepath).lower():
            return f"{self.software}.{filepath.stem}"
        return filepath.stem
    
    def _is_nested(self, node, tree) -> bool:
        """Check if function is nested inside a class."""
        for parent in ast.walk(tree):
            if isinstance(parent, ast.ClassDef):
                if node in parent.body:
                    return True
        return False


class JSONParser:
    """Parse JSON/JSONL documentation files."""
    
    def __init__(self, software: str = "ansa"):
        self.software = software
    
    def parse_file(self, filepath: Path) -> list[Document]:
        """Parse a JSON or JSONL file."""
        documents = []
        
        if filepath.suffix == ".jsonl":
            documents = self._parse_jsonl(filepath)
        else:
            documents = self._parse_json(filepath)
        
        return documents
    
    def _parse_jsonl(self, filepath: Path) -> list[Document]:
        """Parse JSONL (one JSON object per line)."""
        documents = []
        for line_num, line in enumerate(filepath.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                doc = self._json_to_document(data, filepath)
                if doc:
                    documents.append(doc)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON error in {filepath}:{line_num}: {e}")
        return documents
    
    def _parse_json(self, filepath: Path) -> list[Document]:
        """Parse a single JSON file (array or object)."""
        data = json.loads(filepath.read_text(encoding="utf-8"))
        
        if isinstance(data, list):
            return [
                doc for item in data
                if (doc := self._json_to_document(item, filepath))
            ]
        elif isinstance(data, dict):
            doc = self._json_to_document(data, filepath)
            return [doc] if doc else []
        return []
    
    def _json_to_document(self, data: dict, filepath: Path) -> Optional[Document]:
        """Convert a JSON object to a Document."""
        # Support multiple JSON formats
        content = data.get("content") or data.get("description") or data.get("docstring", "")
        name = data.get("name") or data.get("function_name") or data.get("title", "")
        
        if not content and not name:
            return None
        
        # Build rich content
        parts = []
        if name:
            parts.append(f"Name: {name}")
        if data.get("signature"):
            parts.append(f"Signature: {data['signature']}")
        if data.get("module"):
            parts.append(f"Module: {data['module']}")
        if content:
            parts.append(f"\n{content}")
        if data.get("example"):
            parts.append(f"\nExample:\n{data['example']}")
        if data.get("parameters"):
            params = data["parameters"]
            if isinstance(params, list):
                param_str = "\n".join(f"  - {p}" for p in params)
            else:
                param_str = str(params)
            parts.append(f"\nParameters:\n{param_str}")
        
        return Document(
            content="\n".join(parts),
            source_file=str(filepath),
            doc_type=data.get("type", "function"),
            module_name=data.get("module", ""),
            class_name=data.get("class_name", ""),
            function_name=data.get("name", name),
            signature=data.get("signature", ""),
            return_type=data.get("return_type", ""),
            api_category=data.get("category", ""),
            software=self.software,
            imports=data.get("imports", []),
            related_functions=data.get("related", []),
        )


class HTMLParser_Custom:
    """Parse HTML API documentation pages."""
    
    def __init__(self, software: str = "ansa"):
        self.software = software
    
    def parse_file(self, filepath: Path) -> list[Document]:
        """Parse HTML documentation file."""
        html_content = filepath.read_text(encoding="utf-8", errors="ignore")
        
        # Strip HTML tags, keep text structure
        text = self._strip_html(html_content)
        
        # Split into sections (by headers or function definitions)
        sections = self._split_sections(text)
        
        documents = []
        for section in sections:
            if len(section.strip()) < 50:  # Skip very short sections
                continue
            
            doc = Document(
                content=section.strip(),
                source_file=str(filepath),
                doc_type=self._classify_section(section),
                software=self.software,
                module_name=self._extract_module(section),
                function_name=self._extract_function_name(section),
            )
            documents.append(doc)
        
        return documents
    
    def _strip_html(self, html: str) -> str:
        """Remove HTML tags preserving text content."""
        # Remove script/style blocks
        html = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL)
        # Replace block elements with newlines
        html = re.sub(r"<(br|p|div|h[1-6]|li|tr)[^>]*>", "\n", html, flags=re.IGNORECASE)
        # Remove remaining tags
        html = re.sub(r"<[^>]+>", "", html)
        # Decode entities
        html = html.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
        html = html.replace("&nbsp;", " ").replace("&quot;", '"')
        # Collapse whitespace
        html = re.sub(r"\n{3,}", "\n\n", html)
        return html.strip()
    
    def _split_sections(self, text: str) -> list[str]:
        """Split text into logical sections."""
        # Split on function/class definitions or double newlines
        patterns = [
            r"(?=\n(?:class|def|function)\s+\w+)",
            r"\n{2,}(?=[A-Z])",
        ]
        
        sections = [text]
        for pattern in patterns:
            new_sections = []
            for section in sections:
                parts = re.split(pattern, section)
                new_sections.extend(parts)
            sections = new_sections
        
        # Limit section size (max 1000 chars each)
        final_sections = []
        for section in sections:
            if len(section) > 1500:
                # Split long sections at paragraph breaks
                chunks = section.split("\n\n")
                current = ""
                for chunk in chunks:
                    if len(current) + len(chunk) > 1200:
                        if current:
                            final_sections.append(current)
                        current = chunk
                    else:
                        current += "\n\n" + chunk if current else chunk
                if current:
                    final_sections.append(current)
            else:
                final_sections.append(section)
        
        return final_sections
    
    def _classify_section(self, text: str) -> str:
        """Classify section type based on content."""
        if re.search(r"^class\s+", text, re.MULTILINE):
            return "class"
        if re.search(r"^def\s+|^function\s+", text, re.MULTILINE):
            return "function"
        if re.search(r"import|from\s+\w+\s+import", text):
            return "module"
        return "guide"
    
    def _extract_module(self, text: str) -> str:
        """Extract module name from text."""
        match = re.search(r"(?:ansa|meta)\.\w+(?:\.\w+)*", text)
        return match.group(0) if match else ""
    
    def _extract_function_name(self, text: str) -> str:
        """Extract function name from text."""
        match = re.search(r"(?:def|function)\s+(\w+)", text)
        return match.group(1) if match else ""


class MarkdownParser:
    """Parse Markdown documentation files."""
    
    def __init__(self, software: str = "ansa"):
        self.software = software
    
    def parse_file(self, filepath: Path) -> list[Document]:
        """Parse Markdown file into sections."""
        content = filepath.read_text(encoding="utf-8", errors="ignore")
        sections = re.split(r"\n#{1,3}\s+", content)
        
        documents = []
        for section in sections:
            if len(section.strip()) < 50:
                continue
            
            # Extract title from first line
            lines = section.strip().split("\n")
            title = lines[0].strip("# ").strip()
            body = "\n".join(lines[1:]).strip()
            
            documents.append(Document(
                content=f"{title}\n\n{body}" if body else title,
                source_file=str(filepath),
                doc_type="guide",
                software=self.software,
                function_name=title,
            ))
        
        return documents


# =============================================================================
# Ingestion Pipeline
# =============================================================================

class IngestionPipeline:
    """Main ingestion pipeline that orchestrates all parsers.
    
    Scans a source directory, selects the appropriate parser for each file,
    and produces a unified list of Document objects ready for embedding.
    """
    
    # Supported file extensions and their parsers
    PARSER_MAP = {
        ".py": "python",
        ".json": "json",
        ".jsonl": "json",
        ".html": "html",
        ".htm": "html",
        ".md": "markdown",
    }
    
    def __init__(
        self,
        source_dir: str | Path,
        software: str = "ansa",
        exclude_patterns: list[str] = None,
    ):
        """Initialize the ingestion pipeline.
        
        Args:
            source_dir: Directory containing source documentation
            software: Target software ("ansa" or "meta")
            exclude_patterns: File patterns to exclude
        """
        self.source_dir = Path(source_dir)
        self.software = software
        self.exclude_patterns = exclude_patterns or ["__pycache__", ".git", "node_modules"]
        
        # Initialize parsers
        self._parsers = {
            "python": PythonSourceParser(software=software),
            "json": JSONParser(software=software),
            "html": HTMLParser_Custom(software=software),
            "markdown": MarkdownParser(software=software),
        }
    
    def run(self) -> list[Document]:
        """Execute the full ingestion pipeline.
        
        Returns:
            List of Document objects ready for embedding
        """
        if not self.source_dir.exists():
            logger.error(f"Source directory not found: {self.source_dir}")
            return []
        
        documents = []
        files_processed = 0
        files_skipped = 0
        
        for filepath in self._find_files():
            parser_type = self.PARSER_MAP.get(filepath.suffix.lower())
            if not parser_type:
                files_skipped += 1
                continue
            
            try:
                parser = self._parsers[parser_type]
                docs = parser.parse_file(filepath)
                documents.extend(docs)
                files_processed += 1
            except Exception as e:
                logger.error(f"Error processing {filepath}: {e}")
                files_skipped += 1
        
        # Deduplicate by content hash
        seen_ids = set()
        unique_docs = []
        for doc in documents:
            if doc.doc_id not in seen_ids:
                seen_ids.add(doc.doc_id)
                unique_docs.append(doc)
        
        logger.info(
            f"Ingestion complete: {files_processed} files processed, "
            f"{files_skipped} skipped, {len(unique_docs)} documents extracted "
            f"(from {len(documents)} total, {len(documents) - len(unique_docs)} duplicates removed)"
        )
        
        return unique_docs
    
    def _find_files(self):
        """Find all parseable files in source directory."""
        for filepath in self.source_dir.rglob("*"):
            if not filepath.is_file():
                continue
            if any(excl in str(filepath) for excl in self.exclude_patterns):
                continue
            if filepath.suffix.lower() in self.PARSER_MAP:
                yield filepath
    
    def get_stats(self, documents: list[Document]) -> dict:
        """Get ingestion statistics."""
        return {
            "total_documents": len(documents),
            "by_type": self._count_by(documents, "doc_type"),
            "by_module": self._count_by(documents, "module_name"),
            "by_software": self._count_by(documents, "software"),
            "avg_content_length": (
                sum(len(d.content) for d in documents) / max(len(documents), 1)
            ),
        }
    
    def _count_by(self, documents: list[Document], field: str) -> dict:
        """Count documents by a metadata field."""
        counts = {}
        for doc in documents:
            val = getattr(doc, field, "") or "unknown"
            counts[val] = counts.get(val, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import sys
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    
    parser = argparse.ArgumentParser(
        description="Ingest ANSA/META documentation into structured Document objects",
        epilog="""
Examples:
  # Point to your ANSA python docs
  python ingest.py "C:\\BETA_CAE_Systems\\ANSA_v2025.2.2\\python"
  
  # Linux
  python ingest.py /opt/BETA_CAE_Systems/ansa_v2025.2.2/python
  
  # Custom directory
  python ingest.py /path/to/my/docs --software meta

Note: This only PARSES files. To build the searchable vector database,
      use build_vector_db.py instead (it calls ingest internally).
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "source_dir",
        help="Directory containing ANSA/META documentation (.py, .html, .json, .jsonl, .md)"
    )
    parser.add_argument(
        "--software", default="ansa", choices=["ansa", "meta"],
        help="Target software (default: ansa)"
    )
    args = parser.parse_args()
    
    pipeline = IngestionPipeline(source_dir=args.source_dir, software=args.software)
    documents = pipeline.run()
    
    if not documents:
        print("\n  ERROR: No documents found!")
        print(f"  Directory: {args.source_dir}")
        print("  Check that the path contains .py, .html, .json, .jsonl, or .md files.")
        sys.exit(1)
    
    stats = pipeline.get_stats(documents)
    print(f"\n{'='*60}")
    print(f"  Ingestion Complete")
    print(f"{'='*60}")
    print(f"  Source          : {args.source_dir}")
    print(f"  Software        : {args.software}")
    print(f"  Total documents : {stats['total_documents']}")
    print(f"  Avg content len : {stats['avg_content_length']:.0f} chars")
    print(f"  By type         : {stats['by_type']}")
    print(f"  By module       : {dict(list(stats['by_module'].items())[:10])}")
    print(f"{'='*60}")
    print(f"\n  Next step: Build the vector database with:")
    print(f"    python bin/build_vector_db.py --source \"{args.source_dir}\"")
    print()
