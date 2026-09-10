"""PyDyna RAG Tool Functions for the LS-DYNA Agent.

6 retrieval tools that the agent can call during the tool-calling loop:
1. search_keyword       - Semantic search for LS-DYNA keywords
2. lookup_keyword       - Exact keyword lookup with full card definition
3. search_pydyna_api    - Semantic search for PyDyna Python API classes
4. get_material_model   - Find material model by description
5. get_contact_type     - Find contact type by scenario
6. validate_keyword     - Validate parameters against keyword spec

These functions are auto-registered as OpenAI tools via tools.py.
All backed by KeywordRetriever (keyword_retriever.py).
"""

import json
import logging
from pathlib import Path
from typing import Optional

from bin.keyword_retriever import KeywordRetriever


logger = logging.getLogger(__name__)


# =============================================================================
# Singleton Retriever (lazy-loaded)
# =============================================================================

_retriever: Optional[KeywordRetriever] = None


def _get_retriever() -> KeywordRetriever:
    """Get or create the keyword retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = KeywordRetriever()
        stats = _retriever.load()
        logger.info(f"KeywordRetriever loaded: {stats}")
    return _retriever


# =============================================================================
# Tool 1: search_keyword
# =============================================================================

def search_keyword(query: str, top_k: int = 5, category: str = "") -> str:
    """Search for LS-DYNA keywords by natural language description.

    Use for open-ended questions like "material for steel crash",
    "contact between parts", "output energy balance", "hourglass control".

    Args:
        query: Natural language description of what you're looking for
        top_k: Number of results to return (1-10)
        category: Filter by category - material, contact, control, section, etc.

    Returns:
        JSON string with matching LS-DYNA keywords and relevance scores
    """
    retriever = _get_retriever()
    return retriever.search_keywords(query, top_k=min(top_k, 10), category=category)


# =============================================================================
# Tool 2: lookup_keyword
# =============================================================================

def lookup_keyword(name: str) -> str:
    """Exact lookup for an LS-DYNA keyword by name or class.

    Use when you know the exact keyword name, e.g., *MAT_024,
    *CONTACT_AUTOMATIC_SINGLE_SURFACE, *CONTROL_TIMESTEP.
    Returns full card definition with all parameters, types, and defaults.

    Args:
        name: Keyword name (*MAT_024), partial name (MAT_024), or class name (Mat024)

    Returns:
        JSON string with full keyword definition (cards, params, links, options)
    """
    retriever = _get_retriever()
    return retriever.lookup_keyword(name)


# =============================================================================
# Tool 3: search_pydyna_api
# =============================================================================

def search_pydyna_api(query: str, top_k: int = 5, category: str = "") -> str:
    """Search for PyDyna Python API classes and methods.

    Use for queries about PyDyna SDK usage, e.g.,
    "how to set timestep in PyDyna", "material class for plasticity",
    "create section shell", "boundary conditions".

    Args:
        query: Natural language description of the API feature needed
        top_k: Number of results to return (1-10)
        category: Filter by category - material, contact, control, etc.

    Returns:
        JSON string with matching PyDyna API classes and their details
    """
    retriever = _get_retriever()
    return retriever.search_pydyna_api(query, top_k=min(top_k, 10), category=category)


# =============================================================================
# Tool 4: get_material_model
# =============================================================================

def get_material_model(description: str) -> str:
    """Find the best LS-DYNA material model for a given application.

    Use for questions like "what material for steel crash?",
    "rigid wall material", "foam for seat cushion", "spotweld material",
    "rubber for tire", "honeycomb for barrier".

    Args:
        description: Natural language description of the material need

    Returns:
        JSON string with recommended material keywords and their parameters
    """
    retriever = _get_retriever()
    return retriever.get_material_model(description)


# =============================================================================
# Tool 5: get_contact_type
# =============================================================================

def get_contact_type(description: str) -> str:
    """Find the best LS-DYNA contact type for a given scenario.

    Use for questions like "contact between bumper and barrier",
    "self-contact for sheet metal folding", "tied contact for spot welds",
    "eroding contact for penetration", "forming contact".

    Args:
        description: Natural language description of the contact scenario

    Returns:
        JSON string with recommended contact keywords
    """
    retriever = _get_retriever()
    return retriever.get_contact_type(description)


# =============================================================================
# Tool 6: validate_keyword
# =============================================================================

def validate_keyword(keyword_name: str, params: str) -> str:
    """Validate parameters against an LS-DYNA keyword specification.

    Checks for unknown fields, missing required parameters, and type mismatches.
    Use after generating keyword parameters to verify correctness.

    Args:
        keyword_name: Keyword name (e.g., *MAT_024)
        params: JSON string of parameter name-value pairs, e.g. '{"ro": 7.85e-9, "e": 210000}'

    Returns:
        JSON string with validation result (valid/invalid, errors, warnings)
    """
    retriever = _get_retriever()

    # Parse params from JSON string (LLM sends strings)
    try:
        param_dict = json.loads(params) if isinstance(params, str) else params
    except json.JSONDecodeError as e:
        return json.dumps({
            "valid": False,
            "message": f"Invalid JSON in params: {e}",
        })

    return retriever.validate_keyword(keyword_name, param_dict)


# =============================================================================
# Tool Registry (imported by agent.py)
# =============================================================================

ALL_TOOLS = [
    search_keyword,
    lookup_keyword,
    search_pydyna_api,
    get_material_model,
    get_contact_type,
    validate_keyword,
]
