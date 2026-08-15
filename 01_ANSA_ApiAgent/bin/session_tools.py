"""META Session Command Tools.

Provides utilities for working with META post-processor session files (.ses):
- Parse and structure session commands
- Convert session commands to equivalent Python code
- Generate new session files from templates
- Validate session syntax

META session files automate post-processing workflows: plotting contours,
creating animations, extracting time-history data, generating reports.

Usage:
    from bin.session_tools import SessionParser, SessionConverter
    
    parser = SessionParser()
    commands = parser.parse_file("/path/to/post.ses")
    
    converter = SessionConverter()
    python_code = converter.to_python(commands)
"""

import re
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class SessionCommand:
    """A single parsed META session command."""
    line_number: int
    raw: str                        # Original line content
    command: str = ""               # Command name (e.g., "contour", "animate")
    category: str = ""              # Category (display, data, file_io, etc.)
    parameters: dict = field(default_factory=dict)  # Parsed parameters
    comment: str = ""               # Inline comment if any


@dataclass
class SessionScript:
    """A parsed session file with metadata."""
    file_path: str
    commands: list[SessionCommand] = field(default_factory=list)
    total_lines: int = 0
    comment_lines: int = 0
    blank_lines: int = 0
    
    @property
    def command_count(self) -> int:
        return len(self.commands)
    
    @property
    def categories(self) -> dict[str, int]:
        counts = {}
        for cmd in self.commands:
            counts[cmd.category] = counts.get(cmd.category, 0) + 1
        return counts


# =============================================================================
# Command Classification
# =============================================================================

# META session command categories and their keyword patterns
COMMAND_PATTERNS = {
    "display": {
        "keywords": [
            "contour", "fringe", "iso", "section", "deform", "view",
            "display", "show", "hide", "transparency", "render",
            "color", "visible", "invisible", "rotate", "zoom", "pan",
        ],
        "description": "Visualization and display commands",
    },
    "animation": {
        "keywords": [
            "animate", "animation", "frame", "play", "record",
            "avi", "gif", "movie", "sequence", "step",
        ],
        "description": "Animation creation and playback",
    },
    "data_extraction": {
        "keywords": [
            "curve", "plot", "graph", "extract", "history",
            "measure", "value", "max", "min", "time",
            "cross_section", "section_force", "energy",
        ],
        "description": "Data extraction and curve plotting",
    },
    "file_io": {
        "keywords": [
            "open", "read", "load", "save", "write", "export",
            "import", "close", "d3plot", "binout", "keyword",
            "image", "png", "jpg", "bmp", "pdf",
        ],
        "description": "File operations (open, save, export)",
    },
    "annotation": {
        "keywords": [
            "text", "label", "title", "legend", "annotation",
            "arrow", "note", "header", "footer",
        ],
        "description": "Annotations, labels, and text",
    },
    "window": {
        "keywords": [
            "window", "page", "layout", "resize", "position",
            "split", "toolbox", "toolbar", "panel",
        ],
        "description": "Window and layout management",
    },
    "selection": {
        "keywords": [
            "select", "pick", "part", "entity", "group",
            "filter", "set", "all", "none", "invert",
        ],
        "description": "Entity selection and filtering",
    },
    "model": {
        "keywords": [
            "state", "timestep", "model", "include",
            "component", "material", "property",
        ],
        "description": "Model and state management",
    },
}


def classify_command(line: str) -> str:
    """Classify a session command line into a category."""
    line_lower = line.lower()
    
    for category, info in COMMAND_PATTERNS.items():
        if any(kw in line_lower for kw in info["keywords"]):
            return category
    
    return "other"


# =============================================================================
# Session Parser
# =============================================================================

class SessionParser:
    """Parse META session files into structured command objects."""
    
    def parse_file(self, file_path: str) -> SessionScript:
        """Parse a session file from disk.
        
        Args:
            file_path: Path to the .ses file
        
        Returns:
            SessionScript with parsed commands
        """
        filepath = Path(file_path).expanduser().resolve()
        
        if not filepath.exists():
            raise FileNotFoundError(f"Session file not found: {filepath}")
        
        content = filepath.read_text(encoding="utf-8", errors="replace")
        script = self.parse_content(content)
        script.file_path = str(filepath)
        return script
    
    def parse_content(self, content: str) -> SessionScript:
        """Parse session content string.
        
        Args:
            content: Raw session file content
        
        Returns:
            SessionScript with parsed commands
        """
        script = SessionScript(file_path="<string>")
        lines = content.splitlines()
        script.total_lines = len(lines)
        
        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Count blanks and comments
            if not stripped:
                script.blank_lines += 1
                continue
            if stripped.startswith("#") or stripped.startswith("//"):
                script.comment_lines += 1
                continue
            
            # Parse command
            cmd = self._parse_line(line_num, stripped)
            script.commands.append(cmd)
        
        return script
    
    def _parse_line(self, line_num: int, line: str) -> SessionCommand:
        """Parse a single command line."""
        # Extract inline comment
        comment = ""
        if "#" in line:
            parts = line.split("#", 1)
            line = parts[0].strip()
            comment = parts[1].strip()
        
        # Extract command name (first token)
        tokens = line.split()
        command_name = tokens[0] if tokens else ""
        
        # Extract parameters (key=value pairs or positional)
        parameters = self._extract_parameters(tokens[1:] if len(tokens) > 1 else [])
        
        # Classify
        category = classify_command(line)
        
        return SessionCommand(
            line_number=line_num,
            raw=line,
            command=command_name,
            category=category,
            parameters=parameters,
            comment=comment,
        )
    
    def _extract_parameters(self, tokens: list[str]) -> dict:
        """Extract parameters from command tokens."""
        params = {}
        positional_idx = 0
        
        for token in tokens:
            if "=" in token:
                key, value = token.split("=", 1)
                params[key.strip()] = value.strip().strip('"').strip("'")
            else:
                params[f"arg{positional_idx}"] = token
                positional_idx += 1
        
        return params


# =============================================================================
# Session Converter (SES -> Python)
# =============================================================================

# Mapping of common session commands to META Python API equivalents
PYTHON_TEMPLATES = {
    "open": "meta.post.open_file('{arg0}')",
    "read": "meta.post.read_result('{arg0}')",
    "load": "meta.post.read_result('{arg0}')",
    "close": "meta.post.close_file()",
    "contour": "meta.post.plot_contour(component='{arg0}', state={arg1})",
    "fringe": "meta.post.plot_fringe(component='{arg0}')",
    "deform": "meta.post.set_deformation(scale={arg0})",
    "animate": "meta.post.create_animation(start={arg0}, end={arg1}, step={arg2})",
    "export": "meta.post.export_image('{arg0}', format='{arg1}')",
    "save": "meta.post.save_image('{arg0}')",
    "view": "meta.post.set_view('{arg0}')",
    "rotate": "meta.post.rotate(axis='{arg0}', angle={arg1})",
    "zoom": "meta.post.zoom(factor={arg0})",
    "select": "meta.post.select_entities('{arg0}')",
    "section": "meta.post.create_section(origin=({arg0},{arg1},{arg2}), normal=({arg3},{arg4},{arg5}))",
    "curve": "meta.post.extract_curve(entity={arg0}, component='{arg1}')",
    "text": "meta.post.add_annotation(text='{arg0}', position=({arg1},{arg2}))",
    "title": "meta.post.set_title('{arg0}')",
    "window": "meta.post.set_window(layout='{arg0}')",
    "state": "meta.post.set_state({arg0})",
    "timestep": "meta.post.set_timestep({arg0})",
}


class SessionConverter:
    """Convert META session commands to Python code."""
    
    def to_python(self, script: SessionScript) -> str:
        """Convert a parsed session script to Python code.
        
        Args:
            script: Parsed SessionScript object
        
        Returns:
            Python code string
        """
        lines = [
            '"""Auto-converted from META session file.',
            f'Source: {script.file_path}',
            f'Commands: {script.command_count}',
            '"""',
            '',
            'from meta import post as meta_post',
            '',
            '',
        ]
        
        current_category = ""
        
        for cmd in script.commands:
            # Add category header comment
            if cmd.category != current_category:
                current_category = cmd.category
                lines.append(f"# --- {current_category.upper()} ---")
            
            # Convert to Python
            python_line = self._convert_command(cmd)
            
            # Add original as comment if conversion is approximate
            if python_line.startswith("# TODO"):
                lines.append(python_line)
            else:
                if cmd.comment:
                    lines.append(f"{python_line}  # {cmd.comment}")
                else:
                    lines.append(python_line)
        
        return "\n".join(lines)
    
    def _convert_command(self, cmd: SessionCommand) -> str:
        """Convert a single command to Python."""
        command_lower = cmd.command.lower()
        
        # Check if we have a template
        if command_lower in PYTHON_TEMPLATES:
            template = PYTHON_TEMPLATES[command_lower]
            try:
                return template.format(**cmd.parameters)
            except (KeyError, IndexError):
                # Template params don't match — return as comment
                return f"# TODO: {cmd.raw}  (params: {cmd.parameters})"
        
        # No template — return as TODO comment
        return f"# TODO: Convert manually: {cmd.raw}"
    
    def to_python_file(self, script: SessionScript, output_path: str) -> Path:
        """Convert session to Python and save to file.
        
        Args:
            script: Parsed session script
            output_path: Output .py file path
        
        Returns:
            Path to the created file
        """
        python_code = self.to_python(script)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(python_code, encoding="utf-8")
        logger.info(f"Converted session saved to: {out}")
        return out


# =============================================================================
# Session Generator
# =============================================================================

class SessionGenerator:
    """Generate META session files from templates and parameters."""
    
    # Common session templates
    TEMPLATES = {
        "contour_plot": [
            "open {d3plot_path}",
            "state {state}",
            "contour {component}",
            "view {view_angle}",
            "deform scale={scale}",
            "export {output_path} format=png",
        ],
        "animation": [
            "open {d3plot_path}",
            "contour {component}",
            "deform scale={scale}",
            "animate start=1 end={end_state} step=1",
            "export {output_path} format=avi",
        ],
        "section_force": [
            "open {d3plot_path}",
            "section origin=({ox},{oy},{oz}) normal=({nx},{ny},{nz})",
            "curve entity=section component=force_x",
            "export {output_path} format=csv",
        ],
        "multi_view": [
            "open {d3plot_path}",
            "window layout=4",
            "state {state}",
            "# Window 1: Top view",
            "view top",
            "contour {component}",
            "# Window 2: Front view",
            "view front",
            "contour {component}",
            "# Window 3: Iso view",
            "view iso",
            "contour {component}",
            "# Window 4: Side view",
            "view right",
            "contour {component}",
            "export {output_path} format=png",
        ],
    }
    
    def generate(self, template_name: str, params: dict) -> str:
        """Generate a session file from a named template.
        
        Args:
            template_name: Template key (contour_plot, animation, section_force, multi_view)
            params: Template parameters dict
        
        Returns:
            Session file content string
        """
        if template_name not in self.TEMPLATES:
            available = ", ".join(self.TEMPLATES.keys())
            raise ValueError(f"Unknown template: {template_name}. Available: {available}")
        
        template_lines = self.TEMPLATES[template_name]
        
        output_lines = [
            f"# META Session File",
            f"# Template: {template_name}",
            f"# Generated by SafetyAgent",
            "",
        ]
        
        for line in template_lines:
            try:
                output_lines.append(line.format(**params))
            except KeyError as e:
                output_lines.append(f"# MISSING PARAM {e}: {line}")
        
        return "\n".join(output_lines)
    
    def generate_file(self, template_name: str, params: dict, output_path: str) -> Path:
        """Generate session file and save to disk.
        
        Args:
            template_name: Template name
            params: Template parameters
            output_path: Output .ses file path
        
        Returns:
            Path to created file
        """
        content = self.generate(template_name, params)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(content, encoding="utf-8")
        logger.info(f"Generated session file: {out}")
        return out
    
    def list_templates(self) -> dict:
        """List available templates with their required parameters."""
        result = {}
        for name, lines in self.TEMPLATES.items():
            # Extract {param} placeholders
            params = set()
            for line in lines:
                params.update(re.findall(r"\{(\w+)\}", line))
            result[name] = {
                "description": COMMAND_PATTERNS.get(name, {}).get("description", ""),
                "required_params": sorted(params),
                "line_count": len([l for l in lines if not l.startswith("#")]),
            }
        return result


# =============================================================================
# Session Validator
# =============================================================================

class SessionValidator:
    """Validate META session file syntax."""
    
    # Known valid command prefixes
    VALID_COMMANDS = set()
    for category_info in COMMAND_PATTERNS.values():
        VALID_COMMANDS.update(category_info["keywords"])
    
    def validate(self, script: SessionScript) -> dict:
        """Validate a parsed session script.
        
        Args:
            script: Parsed SessionScript
        
        Returns:
            Dict with {valid, warnings, errors}
        """
        warnings = []
        errors = []
        
        for cmd in script.commands:
            # Check for known commands
            cmd_lower = cmd.command.lower()
            if cmd_lower not in self.VALID_COMMANDS and cmd.category == "other":
                warnings.append({
                    "line": cmd.line_number,
                    "message": f"Unknown command: '{cmd.command}'",
                    "raw": cmd.raw,
                })
            
            # Check for common issues
            if "=" in cmd.raw and not cmd.parameters:
                warnings.append({
                    "line": cmd.line_number,
                    "message": "Parameter syntax detected but not parsed",
                    "raw": cmd.raw,
                })
            
            # Check file paths exist (if file_io command)
            if cmd.category == "file_io" and "arg0" in cmd.parameters:
                fpath = cmd.parameters["arg0"]
                if fpath and not fpath.startswith("{"):  # Skip templates
                    p = Path(fpath).expanduser()
                    if not p.exists() and not any(
                        c in fpath for c in ["*", "?", "%", "$"]
                    ):
                        warnings.append({
                            "line": cmd.line_number,
                            "message": f"Referenced file may not exist: {fpath}",
                            "raw": cmd.raw,
                        })
        
        return {
            "valid": len(errors) == 0,
            "total_commands": script.command_count,
            "warnings": warnings,
            "warning_count": len(warnings),
            "errors": errors,
            "error_count": len(errors),
            "categories": script.categories,
        }


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="META session file tools")
    subparsers = parser.add_subparsers(dest="action")
    
    # Parse sub-command
    parse_parser = subparsers.add_parser("parse", help="Parse a session file")
    parse_parser.add_argument("file", help="Path to .ses file")
    
    # Convert sub-command
    convert_parser = subparsers.add_parser("convert", help="Convert .ses to .py")
    convert_parser.add_argument("file", help="Path to .ses file")
    convert_parser.add_argument("--output", "-o", help="Output .py path")
    
    # Generate sub-command
    gen_parser = subparsers.add_parser("generate", help="Generate session from template")
    gen_parser.add_argument("template", help="Template name")
    gen_parser.add_argument("--list", action="store_true", help="List available templates")
    
    # Validate sub-command
    val_parser = subparsers.add_parser("validate", help="Validate session syntax")
    val_parser.add_argument("file", help="Path to .ses file")
    
    args = parser.parse_args()
    
    if args.action == "parse":
        sp = SessionParser()
        script = sp.parse_file(args.file)
        print(f"\nFile: {script.file_path}")
        print(f"Lines: {script.total_lines} (commands: {script.command_count}, comments: {script.comment_lines})")
        print(f"Categories: {script.categories}")
        print(f"\nFirst 10 commands:")
        for cmd in script.commands[:10]:
            print(f"  L{cmd.line_number}: [{cmd.category}] {cmd.raw}")
    
    elif args.action == "convert":
        sp = SessionParser()
        sc = SessionConverter()
        script = sp.parse_file(args.file)
        python_code = sc.to_python(script)
        
        if args.output:
            sc.to_python_file(script, args.output)
            print(f"Saved to: {args.output}")
        else:
            print(python_code)
    
    elif args.action == "generate":
        sg = SessionGenerator()
        if hasattr(args, "list") and args.list:
            templates = sg.list_templates()
            for name, info in templates.items():
                print(f"  {name}: params={info['required_params']}")
        else:
            print(f"Usage: session_tools.py generate <template> (use --list to see available)")
    
    elif args.action == "validate":
        sp = SessionParser()
        sv = SessionValidator()
        script = sp.parse_file(args.file)
        result = sv.validate(script)
        print(f"\nValid: {result['valid']}")
        print(f"Commands: {result['total_commands']}")
        print(f"Warnings: {result['warning_count']}")
        for w in result["warnings"][:10]:
            print(f"  L{w['line']}: {w['message']}")
    
    else:
        parser.print_help()
