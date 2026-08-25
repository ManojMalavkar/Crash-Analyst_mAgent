import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = PROJECT_ROOT / "01_ANSA_ApiAgent"
for path in (PROJECT_ROOT, AGENT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from bin.ingest import KnowledgeExtractor

CLASS_BLOCK = """Window, (class)    BETA PYTHON development Manual
NAME:
Window

SYNOPSIS:
meta.windows.Window()

DESCRIPTION:
Class for windows.

ARGUMENTS:
* name : str

The name of the window.

* page_id : int

The id of the Page the window belongs to.


ATTRIBUTES:
* name : str

Id of the page of the window.

* active : int

1 if window is active, 0 if window is not active.


METHODS:
    meta.windows.Window.__init__
    meta.windows.Window.activate
    meta.windows.Window.activate_fringe

EXAMPLE:
# PYTHON script
import meta
from meta import windows


def main():
    win = windows.Window(name="MetaPost", page_id=0)
    print(win)


if __name__ == "__main__":
    main()
"""

METHOD_BLOCK = """Window.activate,     BETA PYTHON development Manual
NAME:
meta.windows.Window.activate

SYNOPSIS:
activate() -> bool

DESCRIPTION:
This method makes active the window. It works only in the active page.

ARGUMENTS:

EXCEPTIONS:

RETURN TYPE:
bool

RETURN VALUE:
Upon success, it returns True. Upon failure, it returns False.

EXAMPLE:
# PYTHON script
import meta
from meta import windows


def main():
    win = windows.Window(name="MetaPost", page_id=0)
    ret = win.activate()
    print(ret)


if __name__ == "__main__":
    main()
"""

SEPARATOR = "\n" + ("-" * 67) + "\n"
SAMPLE_MANUAL = CLASS_BLOCK + SEPARATOR + METHOD_BLOCK + SEPARATOR


def _build_extractor(tmp_path: Path) -> tuple[KnowledgeExtractor, Path]:
    extractor = KnowledgeExtractor(str(tmp_path), software="ansa")
    txt_file = tmp_path / "meta_windows_manual.txt"
    txt_file.write_text(SAMPLE_MANUAL, encoding="utf-8")
    return extractor, txt_file


def test_class_block_resolves_fully_qualified_symbol(tmp_path):
    extractor, txt_file = _build_extractor(tmp_path)
    extractor.process_manual_text(txt_file)

    rec = extractor.records["meta.windows.Window"]
    assert rec["type"] == "class"
    assert rec["module"] == "meta.windows"
    assert rec["software"] == "meta"
    assert rec["signature"] == "meta.windows.Window()"


def test_class_block_extracts_arguments_attributes_methods(tmp_path):
    extractor, txt_file = _build_extractor(tmp_path)
    extractor.process_manual_text(txt_file)

    rec = extractor.records["meta.windows.Window"]
    assert rec["parameters"] == [
        {"name": "name", "type": "str", "description": "The name of the window."},
        {"name": "page_id", "type": "int", "description": "The id of the Page the window belongs to."},
    ]
    assert rec["attributes"][0] == {
        "name": "name", "type": "str", "description": "Id of the page of the window.",
    }
    assert rec["methods"] == [
        "meta.windows.Window.__init__",
        "meta.windows.Window.activate",
        "meta.windows.Window.activate_fringe",
    ]


def test_method_block_extracts_return_type_and_value(tmp_path):
    extractor, txt_file = _build_extractor(tmp_path)
    extractor.process_manual_text(txt_file)

    rec = extractor.records["meta.windows.Window.activate"]
    assert rec["type"] == "function"
    assert rec["module"] == "meta.windows.Window"
    assert rec["signature"] == "activate() -> bool"
    assert rec["parameters"] == []
    assert rec["returns"] == {
        "type": "bool",
        "description": "Upon success, it returns True. Upon failure, it returns False.",
    }


def test_manual_records_are_counted(tmp_path):
    extractor, txt_file = _build_extractor(tmp_path)
    extractor.process_manual_text(txt_file)

    assert extractor.stats["manual_records"] == 2


def test_class_methods_produce_has_method_edges(tmp_path):
    extractor, txt_file = _build_extractor(tmp_path)
    extractor.process_manual_text(txt_file)
    extractor.build_knowledge_graph()

    assert (
        "meta.windows.Window",
        "meta.windows.Window.activate",
        "HAS_METHOD",
    ) in extractor.kg_edges
