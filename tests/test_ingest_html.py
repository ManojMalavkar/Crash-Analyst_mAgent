import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = PROJECT_ROOT / "01_ANSA_ApiAgent"
for path in (PROJECT_ROOT, AGENT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from bin.ingest import KnowledgeExtractor

SAMPLE_HTML = """
<html><body><main>
<article>
<section id="module-meta.windows">
<h1>Module meta.windows<a href="#">#</a></h1>
<dl class="py function">
<dt class="sig sig-object py" id="meta.windows.ActivateFringe">
<span class="sig-name descname">ActivateFringe</span>
<span class="sig-paren">(</span>window_name<span class="sig-paren">)</span>
<a class="headerlink" href="#meta.windows.ActivateFringe">#</a>
</dt>
<dd>
<div class="deprecated">
<p>Deprecated since version 20.1.0: Use
<a class="reference internal" href="#meta.windows.Window.activate_fringe">
<code>meta.windows.Window.activate_fringe()</code></a> instead.</p>
</div>
<p>This function makes active a fringe bar with a given name on a window.</p>
<dl class="field-list">
<dt class="field-odd">Parameters:</dt>
<dd class="field-odd"><dl>
<dt><strong>window_name</strong><span class="classifier">str</span></dt>
<dd><p>Name of the window.</p></dd>
</dl></dd>
<dt class="field-even">Returns:</dt>
<dd class="field-even"><dl>
<dt>int</dt>
<dd><p>It returns 1 upon success and 0 upon failure.</p></dd>
</dl></dd>
</dl>
</dd>
</dl>
</section>
</article>
</main></body></html>
"""


def _build_extractor(tmp_path: Path) -> tuple[KnowledgeExtractor, Path]:
    extractor = KnowledgeExtractor(str(tmp_path), software="ansa")
    html_file = tmp_path / "meta.windows.html"
    html_file.write_text(SAMPLE_HTML, encoding="utf-8")
    return extractor, html_file


def test_process_html_extracts_module_and_software(tmp_path):
    extractor, html_file = _build_extractor(tmp_path)
    extractor.process_html(html_file)

    rec = extractor.records["meta.windows.ActivateFringe"]
    assert rec["module"] == "meta.windows"
    assert rec["software"] == "meta"


def test_process_html_extracts_structured_parameters_and_returns(tmp_path):
    extractor, html_file = _build_extractor(tmp_path)
    extractor.process_html(html_file)

    rec = extractor.records["meta.windows.ActivateFringe"]
    assert rec["parameters"] == [
        {"name": "window_name", "type": "str", "description": "Name of the window."}
    ]
    assert rec["returns"] == {
        "type": "int",
        "description": "It returns 1 upon success and 0 upon failure.",
    }


def test_process_html_extracts_deprecated_target(tmp_path):
    extractor, html_file = _build_extractor(tmp_path)
    extractor.process_html(html_file)

    rec = extractor.records["meta.windows.ActivateFringe"]
    assert rec["deprecated"] is True
    assert rec["deprecated_target"] == "meta.windows.Window.activate_fringe"


def test_html_records_are_counted(tmp_path):
    extractor, html_file = _build_extractor(tmp_path)
    extractor.process_html(html_file)

    assert extractor.stats["html_records"] == 1


def test_see_also_and_deprecated_by_use_meta_prefix(tmp_path):
    extractor, html_file = _build_extractor(tmp_path)
    extractor.process_html(html_file)
    extractor.build_knowledge_graph()

    assert (
        "meta.windows.ActivateFringe",
        "meta.windows.Window.activate_fringe",
        "DEPRECATED_BY",
    ) in extractor.kg_edges
