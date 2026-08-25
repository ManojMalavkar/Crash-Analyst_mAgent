import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = PROJECT_ROOT / "01_ANSA_ApiAgent"
for path in (PROJECT_ROOT, AGENT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from bin.ingest import KnowledgeExtractor

CLASS_HTML = """
<html><body><main><article>
<section id="module-meta.windows">
<h1>Module meta.windows<a href="#">#</a></h1>
<dl class="py class">
<dt class="sig sig-object py" id="meta.windows.Window">
<em class="property">class </em>
<span class="sig-name descname">Window</span>
<span class="sig-paren">(</span>name<span class="sig-paren">)</span>
<a class="headerlink" href="#meta.windows.Window">#</a>
</dt>
<dd><p>Class for windows.</p></dd>
</dl>
<dl class="py method">
<dt class="sig sig-object py" id="meta.windows.Window.activate">
<span class="sig-name descname">activate</span>
<span class="sig-paren">(</span><span class="sig-paren">)</span>
<span class="sig-return"><span class="sig-return-icon">&#8594;</span> <span class="sig-return-typehint">bool</span></span>
<a class="headerlink" href="#meta.windows.Window.activate">#</a>
</dt>
<dd>
<div class="deprecated">
<p>Deprecated since version 20.1.0: Use
<a class="reference internal" href="#meta.windows.Window.activate_fringe">
<code>meta.windows.Window.activate_fringe()</code></a> instead.</p>
</div>
<p>This method makes active the window.</p>
<dl class="field-list">
<dt class="field-odd">Returns:</dt>
<dd class="field-odd"><dl>
<dt>bool</dt>
<dd><p>Upon success, it returns True. Upon failure, it returns False.</p></dd>
</dl></dd>
</dl>
</dd>
</dl>
</section>
</article></main></body></html>
"""


def test_build_hierarchy_nests_methods_under_class(tmp_path):
    extractor = KnowledgeExtractor(str(tmp_path), software="ansa")
    html_file = tmp_path / "meta.windows.html"
    html_file.write_text(CLASS_HTML, encoding="utf-8")
    extractor.process_html(html_file)

    # Wire the class -> method relationship the way the manual-text parser would
    extractor.records["meta.windows.Window"]["methods"] = ["meta.windows.Window.activate"]

    hierarchy = extractor.build_hierarchy()
    assert len(hierarchy) == 1  # method is nested, not a separate top-level entry

    class_entry = hierarchy[0]
    assert class_entry["NAME"] == "meta.windows.Window"
    assert set(class_entry.keys()) == {
        "NAME", "SYNOPSIS", "DESCRIPTION", "ARGUMENTS", "ATTRIBUTES", "METHODS", "EXAMPLE",
    }

    method_entry = class_entry["METHODS"][0]
    assert method_entry["NAME"] == "meta.windows.Window.activate"
    assert set(method_entry.keys()) == {
        "NAME", "SYNOPSIS", "DESCRIPTION", "DEPRECATED SINCE", "REPLACED BY",
        "ARGUMENTS", "EXCEPTIONS", "RETURN TYPE", "RETURN VALUE", "EXAMPLE",
    }
    assert method_entry["DEPRECATED SINCE"] == "20.1.0"
    assert method_entry["REPLACED BY"] == "meta.windows.Window.activate_fringe"
    assert method_entry["RETURN TYPE"] == "bool"


def test_standalone_function_uses_method_shape(tmp_path):
    extractor = KnowledgeExtractor(str(tmp_path), software="ansa")
    extractor.get_record("ansa.base.CollectEntities")
    extractor.records["ansa.base.CollectEntities"]["type"] = "function"

    hierarchy = extractor.build_hierarchy()
    assert len(hierarchy) == 1
    assert hierarchy[0]["NAME"] == "ansa.base.CollectEntities"
    assert "RETURN TYPE" in hierarchy[0]
