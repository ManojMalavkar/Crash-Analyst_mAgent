"""Auto Tool Spec Generator.

Converts Python functions into OpenAI-compatible tool specifications.
Inspects function signatures, type hints, and docstrings to produce
JSON schemas automatically.

Usage:
    from bin.tools import generate_tool_spec, generate_all_tool_specs
    
    # Single function
    spec = generate_tool_spec(search_api)
    
    # All functions from a module
    specs = generate_all_tool_specs([search_api, get_function_details, ...])
"""

import inspect
from typing import get_type_hints, Any, Callable


# =============================================================================
# Type Mapping
# =============================================================================

# Python type -> JSON Schema type
TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    type(None): "null",
}


def _python_type_to_json_schema(py_type) -> dict:
    """Convert a Python type hint to JSON schema."""
    # Handle basic types
    if py_type in TYPE_MAP:
        return {"type": TYPE_MAP[py_type]}
    
    # Handle Optional (Union[X, None])
    origin = getattr(py_type, "__origin__", None)
    args = getattr(py_type, "__args__", ())
    
    if origin is list:
        item_type = args[0] if args else Any
        return {
            "type": "array",
            "items": _python_type_to_json_schema(item_type),
        }
    
    if origin is dict:
        return {"type": "object"}
    
    # Union types (Optional[X] is Union[X, None])
    import typing
    if origin is typing.Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            # Optional[X] -> schema for X
            return _python_type_to_json_schema(non_none[0])
    
    # Default to string for unknown types
    return {"type": "string"}


# =============================================================================
# Docstring Parser
# =============================================================================

def _parse_docstring(func: Callable) -> dict:
    """Parse Google-style docstring into structured parts.
    
    Extracts:
    - description: First line or paragraph
    - params: Dict of param_name -> description
    - returns: Return description
    """
    doc = inspect.getdoc(func) or ""
    
    result = {
        "description": "",
        "params": {},
        "returns": "",
    }
    
    if not doc:
        return result
    
    lines = doc.split("\n")
    
    # State machine for parsing
    section = "description"  # "description", "args", "returns"
    desc_lines = []
    current_param = None
    
    for line in lines:
        stripped = line.strip()
        
        # Detect section headers
        if stripped.lower() in ("args:", "arguments:", "parameters:", "params:"):
            section = "args"
            continue
        elif stripped.lower() in ("returns:", "return:"):
            section = "returns"
            continue
        elif stripped.lower() in ("raises:", "examples:", "example:", "note:", "notes:"):
            section = "other"
            continue
        
        if section == "description":
            if stripped:
                desc_lines.append(stripped)
            elif desc_lines:
                # End of description paragraph
                pass
        
        elif section == "args":
            # Detect param: "param_name: description" or "param_name (type): description"
            if stripped and not stripped.startswith(" ") and ":" in stripped:
                # New param
                parts = stripped.split(":", 1)
                param_name = parts[0].strip().split("(")[0].strip()
                param_desc = parts[1].strip() if len(parts) > 1 else ""
                result["params"][param_name] = param_desc
                current_param = param_name
            elif stripped and current_param:
                # Continuation of previous param description
                result["params"][current_param] += " " + stripped
        
        elif section == "returns":
            if stripped:
                result["returns"] += stripped + " "
    
    result["description"] = " ".join(desc_lines).strip()
    result["returns"] = result["returns"].strip()
    
    return result


# =============================================================================
# Tool Spec Generator
# =============================================================================

def generate_tool_spec(func: Callable) -> dict:
    """Generate OpenAI-compatible tool specification from a Python function.
    
    Args:
        func: The function to generate a spec for.
              Must have type hints and a docstring.
    
    Returns:
        Dict in OpenAI tool format:
        {
            "type": "function",
            "function": {
                "name": "...",
                "description": "...",
                "parameters": { JSON Schema }
            }
        }
    """
    # Get function metadata
    sig = inspect.signature(func)
    hints = get_type_hints(func)
    doc = _parse_docstring(func)
    
    # Build parameters schema
    properties = {}
    required = []
    
    for param_name, param in sig.parameters.items():
        # Skip self, cls, **kwargs
        if param_name in ("self", "cls"):
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        
        # Get type
        param_type = hints.get(param_name, str)
        schema = _python_type_to_json_schema(param_type)
        
        # Add description from docstring
        if param_name in doc["params"]:
            schema["description"] = doc["params"][param_name]
        
        # Add default value
        if param.default is not inspect.Parameter.empty:
            schema["default"] = param.default
        else:
            required.append(param_name)
        
        properties[param_name] = schema
    
    # Build tool spec
    tool_spec = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": doc["description"] or f"Call {func.__name__}",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }
    
    return tool_spec


def generate_all_tool_specs(functions: list[Callable]) -> list[dict]:
    """Generate tool specs for a list of functions.
    
    Args:
        functions: List of functions to generate specs for
    
    Returns:
        List of OpenAI tool spec dicts
    """
    return [generate_tool_spec(f) for f in functions]


def create_tool_registry(functions: list[Callable]) -> dict:
    """Create a registry mapping tool names to their functions and specs.
    
    Args:
        functions: List of functions to register
    
    Returns:
        Dict with 'specs' (list for API) and 'dispatch' (name -> function)
    """
    specs = []
    dispatch = {}
    
    for func in functions:
        spec = generate_tool_spec(func)
        specs.append(spec)
        dispatch[func.__name__] = func
    
    return {
        "specs": specs,
        "dispatch": dispatch,
    }
