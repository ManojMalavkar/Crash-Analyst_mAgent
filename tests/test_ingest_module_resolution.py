import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = PROJECT_ROOT / "01_ANSA_ApiAgent"
for path in (PROJECT_ROOT, AGENT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from bin.ingest import KnowledgeExtractor


def test_module_from_path_flat_module(tmp_path):
    extractor = KnowledgeExtractor(str(tmp_path), software="meta")
    file = tmp_path / "autocomplete" / "py_dev" / "pydev_meta" / "meta" / "windows.py"
    assert extractor._module_from_path(file) == "meta.windows"


def test_module_from_path_nested_package(tmp_path):
    extractor = KnowledgeExtractor(str(tmp_path), software="meta")
    file = tmp_path / "pydev_meta" / "meta" / "spdrm" / "process.py"
    assert extractor._module_from_path(file) == "meta.spdrm.process"


def test_module_from_path_init_file(tmp_path):
    extractor = KnowledgeExtractor(str(tmp_path), software="meta")
    file = tmp_path / "pydev_meta" / "meta" / "boundaries" / "__init__.py"
    assert extractor._module_from_path(file) == "meta.boundaries"


def test_python_class_uses_fully_qualified_symbol(tmp_path):
    extractor = KnowledgeExtractor(str(tmp_path), software="meta")
    py_dir = tmp_path / "pydev_meta" / "meta"
    py_dir.mkdir(parents=True)
    py_file = py_dir / "windows.py"
    py_file.write_text(
        'class ParametricPointPath:\n'
        '    """Class for Parametric Point Paths."""\n'
        '    pass\n',
        encoding="utf-8",
    )
    extractor.process_python(py_file)

    assert "meta.windows.ParametricPointPath" in extractor.records
    assert "windows.ParametricPointPath" not in extractor.records


def test_derive_missing_class_members_from_sibling_records(tmp_path):
    extractor = KnowledgeExtractor(str(tmp_path), software="meta")

    class_rec = extractor.get_record("meta.windows.Window")
    class_rec["type"] = "class"
    class_rec["description"] = "Class for windows."

    method_rec = extractor.get_record("meta.windows.Window.activate")
    method_rec["type"] = "function"
    method_rec["signature"] = "activate() -> bool"

    attr_rec = extractor.get_record("meta.windows.Window.name")
    attr_rec["type"] = "attribute"
    attr_rec["description"] = "Name of the window."

    extractor._derive_missing_class_members()

    assert class_rec["methods"] == ["meta.windows.Window.activate"]
    assert class_rec["attributes"] == [
        {"name": "name", "type": "", "description": "Name of the window."}
    ]


def test_build_hierarchy_nests_html_sourced_class_members(tmp_path):
    extractor = KnowledgeExtractor(str(tmp_path), software="meta")

    class_rec = extractor.get_record("meta.windows.Window")
    class_rec["type"] = "class"

    method_rec = extractor.get_record("meta.windows.Window.activate")
    method_rec["type"] = "function"
    method_rec["signature"] = "activate() -> bool"

    hierarchy = extractor.build_hierarchy()

    assert len(hierarchy) == 1
    class_entry = hierarchy[0]
    assert class_entry["NAME"] == "meta.windows.Window"
    assert len(class_entry["METHODS"]) == 1
    assert class_entry["METHODS"][0]["NAME"] == "meta.windows.Window.activate"
