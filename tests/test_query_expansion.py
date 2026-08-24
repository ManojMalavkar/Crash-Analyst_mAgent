import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = PROJECT_ROOT / "01_ANSA_ApiAgent"
for path in (PROJECT_ROOT, AGENT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from bin.tool_functions import _expand_query_terms


def test_expand_mesh_query():
    expanded = _expand_query_terms("set mesh parameters for element size")
    assert "SetMeshParams" in expanded
    assert "MeshParams" in expanded
    assert "element_size" in expanded


def test_expand_merge_query():
    expanded = _expand_query_terms("delete duplicate nodes with tolerance")
    assert "MergeNodes" in expanded
    assert "duplicate" in expanded
    assert "tolerance" in expanded


def test_expand_timestep_query():
    expanded = _expand_query_terms("get number of timesteps in result")
    assert "Timestep" in expanded
    assert "timestep" in expanded or "time_step" in expanded
