"""LS-DYNA Model File Read/Write Tools for the PyDyna Agent.

6 model manipulation tools that the agent can call to interact with
LS-DYNA keyword files (.k, .key, .dyn):
  1. read_keyword_file   - Parse an LS-DYNA keyword file into structured data
  2. modify_material     - Change material properties in the loaded model
  3. modify_contact      - Change contact settings in the loaded model
  4. add_keyword         - Insert a new keyword block into the model
  5. list_parts          - List all parts (PID, name, section, material)
  6. get_model_summary   - High-level overview of the model content

These functions are auto-registered as OpenAI tools via tools.py.
Model state is held in a singleton ModelState (lazy-loaded on first read).
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class KeywordBlock:
    """A single keyword block in an LS-DYNA file."""
    keyword: str               # e.g. "*MAT_024"
    lines: list[str]           # raw data lines (excluding keyword line)
    line_start: int            # line number where this block starts
    options: list[str] = field(default_factory=list)  # e.g. ["TITLE"]
    comment_lines: list[str] = field(default_factory=list)

    @property
    def full_keyword(self) -> str:
        """Full keyword with options, e.g. *MAT_024_TITLE."""
        if self.options:
            return f"{self.keyword}_{'_'.join(self.options)}"
        return self.keyword


@dataclass
class PartInfo:
    """Parsed *PART data."""
    pid: int
    name: str
    secid: int
    mid: int
    eosid: int = 0
    hgid: int = 0
    grav: int = 0
    adpopt: int = 0
    tmid: int = 0


# =============================================================================
# LS-DYNA Keyword File Parser
# =============================================================================

class KeywordFileParser:
    """Parse LS-DYNA keyword files (.k, .key, .dyn) into structured blocks.

    LS-DYNA keyword files consist of:
    - Lines starting with '$' are comments
    - Lines starting with '*' introduce keyword blocks
    - Data lines follow in fixed-width columns (8 or 10 chars)
    - *END terminates the file
    """

    # Keywords that carry a TITLE option (title line before data)
    TITLE_KEYWORDS = {
        "*PART", "*SECTION", "*MAT", "*DEFINE_CURVE",
        "*SET_NODE", "*SET_PART", "*SET_SHELL", "*SET_SOLID",
    }

    @staticmethod
    def parse(filepath: str) -> list[KeywordBlock]:
        """Parse an LS-DYNA keyword file into a list of KeywordBlock objects.

        Args:
            filepath: Path to the .k/.key/.dyn file.

        Returns:
            List of KeywordBlock objects.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        blocks: list[KeywordBlock] = []
        current_keyword: Optional[str] = None
        current_lines: list[str] = []
        current_comments: list[str] = []
        current_options: list[str] = []
        current_start: int = 0

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line_num, raw_line in enumerate(f, start=1):
                line = raw_line.rstrip("\n").rstrip("\r")

                # Comment line
                if line.startswith("$"):
                    if current_keyword:
                        current_comments.append(line)
                    continue

                # Keyword line
                if line.startswith("*"):
                    # Save previous block
                    if current_keyword:
                        blocks.append(KeywordBlock(
                            keyword=current_keyword,
                            lines=current_lines,
                            line_start=current_start,
                            options=current_options,
                            comment_lines=current_comments,
                        ))

                    # Parse new keyword and options
                    token = line.strip().split()[0].upper()
                    current_keyword, current_options = (
                        KeywordFileParser._parse_keyword_token(token)
                    )
                    current_lines = []
                    current_comments = []
                    current_start = line_num
                    continue

                # Data line (belongs to current keyword)
                if current_keyword:
                    current_lines.append(line)

        # Save last block
        if current_keyword:
            blocks.append(KeywordBlock(
                keyword=current_keyword,
                lines=current_lines,
                line_start=current_start,
                options=current_options,
                comment_lines=current_comments,
            ))

        return blocks

    @staticmethod
    def _parse_keyword_token(token: str) -> tuple[str, list[str]]:
        """Split keyword token into base keyword and options.

        e.g. '*MAT_024_TITLE' -> ('*MAT_024', ['TITLE'])
             '*PART' -> ('*PART', [])
        """
        known_options = {"TITLE", "ID"}
        parts = token.split("_")
        options = []

        # Walk backwards to peel off known option suffixes
        while len(parts) > 1 and parts[-1] in known_options:
            options.insert(0, parts.pop())

        base = "_".join(parts)
        return base, options

    @staticmethod
    def parse_fixed_fields(line: str, width: int = 10) -> list[str]:
        """Parse a fixed-width data line into fields.

        Args:
            line: Data line from the keyword file.
            width: Column width (8 for old format, 10 for standard, 20 for long).

        Returns:
            List of field values as stripped strings.
        """
        fields = []
        for i in range(0, len(line), width):
            fields.append(line[i:i + width].strip())
        return fields


# =============================================================================
# Model State (singleton — holds the loaded model in memory)
# =============================================================================

class ModelState:
    """In-memory representation of a loaded LS-DYNA model.

    Holds parsed keyword blocks and provides query/modification methods.
    """

    def __init__(self):
        self.filepath: Optional[str] = None
        self.blocks: list[KeywordBlock] = []
        self._raw_lines: list[str] = []
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self, filepath: str) -> dict:
        """Load and parse an LS-DYNA keyword file.

        Returns:
            Summary dict with counts by keyword category.
        """
        self.filepath = filepath
        self.blocks = KeywordFileParser.parse(filepath)
        self._loaded = True

        # Keep raw lines for write-back
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            self._raw_lines = f.readlines()

        summary = self._categorize_blocks()
        logger.info(f"Model loaded: {filepath} — {len(self.blocks)} blocks")
        return summary

    def _categorize_blocks(self) -> dict:
        """Categorize blocks into high-level groups."""
        categories: dict[str, int] = {}
        for block in self.blocks:
            # Extract category from keyword: *MAT_024 -> MAT
            base = block.keyword.lstrip("*").split("_")[0]
            categories[base] = categories.get(base, 0) + 1
        return categories

    def find_blocks(self, prefix: str) -> list[KeywordBlock]:
        """Find all blocks whose keyword starts with a prefix.

        Args:
            prefix: Keyword prefix, e.g. '*MAT', '*CONTACT', '*PART'.
        """
        prefix_upper = prefix.upper().lstrip("*")
        return [
            b for b in self.blocks
            if b.keyword.lstrip("*").startswith(prefix_upper)
        ]

    def find_block_by_id(self, prefix: str, block_id: int) -> Optional[KeywordBlock]:
        """Find a specific block by keyword prefix and its first-field ID.

        For *MAT keywords the first data field is MID,
        for *CONTACT it's SSID/MSID, etc.
        """
        for block in self.find_blocks(prefix):
            if not block.lines:
                continue
            fields = KeywordFileParser.parse_fixed_fields(block.lines[0])
            if fields and fields[0]:
                try:
                    if int(float(fields[0])) == block_id:
                        return block
                except (ValueError, IndexError):
                    continue
        return None

    def get_parts(self) -> list[PartInfo]:
        """Extract all *PART definitions."""
        parts = []
        for block in self.find_blocks("PART"):
            if block.keyword != "*PART":
                continue
            data_lines = [l for l in block.lines if l.strip()]
            if not data_lines:
                continue

            # PART has: title line (if TITLE option), then data card(s)
            title = ""
            data_idx = 0
            if "TITLE" in block.options or len(data_lines) >= 2:
                title = data_lines[0].strip()
                data_idx = 1

            if data_idx >= len(data_lines):
                continue

            fields = KeywordFileParser.parse_fixed_fields(data_lines[data_idx])
            # Pad fields to at least 8
            while len(fields) < 8:
                fields.append("")

            try:
                parts.append(PartInfo(
                    pid=int(float(fields[0])) if fields[0] else 0,
                    name=title,
                    secid=int(float(fields[1])) if fields[1] else 0,
                    mid=int(float(fields[2])) if fields[2] else 0,
                    eosid=int(float(fields[3])) if fields[3] else 0,
                    hgid=int(float(fields[4])) if fields[4] else 0,
                    grav=int(float(fields[5])) if fields[5] else 0,
                    adpopt=int(float(fields[6])) if fields[6] else 0,
                    tmid=int(float(fields[7])) if fields[7] else 0,
                ))
            except (ValueError, IndexError) as e:
                logger.warning(f"Failed to parse PART block at line {block.line_start}: {e}")

        return parts

    def modify_block_fields(
        self,
        block: KeywordBlock,
        card_index: int,
        field_updates: dict[int, str],
    ) -> None:
        """Modify specific fields in a keyword block's data card.

        Args:
            block: The KeywordBlock to modify.
            card_index: Which data line (0-based) to modify.
            field_updates: {field_position: new_value} (0-based positions).
        """
        if card_index >= len(block.lines):
            logger.warning(f"Card index {card_index} out of range for {block.keyword}")
            return

        line = block.lines[card_index]
        width = 10
        fields = KeywordFileParser.parse_fixed_fields(line, width)

        for pos, value in field_updates.items():
            while len(fields) <= pos:
                fields.append("")
            fields[pos] = str(value)

        # Rebuild fixed-width line
        new_line = "".join(f"{f:>{width}}" for f in fields)
        block.lines[card_index] = new_line

    def add_block(self, keyword: str, data_lines: list[str],
                  options: Optional[list[str]] = None) -> KeywordBlock:
        """Add a new keyword block to the model.

        The block is inserted before *END if present, otherwise appended.
        """
        new_block = KeywordBlock(
            keyword=keyword.upper(),
            lines=data_lines,
            line_start=-1,  # not yet written
            options=options or [],
        )
        # Insert before *END
        end_idx = None
        for i, block in enumerate(self.blocks):
            if block.keyword == "*END":
                end_idx = i
                break

        if end_idx is not None:
            self.blocks.insert(end_idx, new_block)
        else:
            self.blocks.append(new_block)

        return new_block

    def write(self, output_path: Optional[str] = None) -> str:
        """Write the model back to a keyword file.

        Args:
            output_path: Destination path. If None, overwrites original.

        Returns:
            Path to the written file.
        """
        dest = output_path or self.filepath
        if not dest:
            raise ValueError("No output path specified and no model loaded")

        lines: list[str] = []
        for block in self.blocks:
            # Write keyword line
            lines.append(block.full_keyword)
            # Write comments
            for comment in block.comment_lines:
                lines.append(comment)
            # Write data lines
            for data_line in block.lines:
                lines.append(data_line)

        with open(dest, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        logger.info(f"Model written to {dest}")
        return dest


# =============================================================================
# Singleton Model State (lazy-loaded)
# =============================================================================

_model_state: Optional[ModelState] = None


def _get_model_state() -> ModelState:
    """Get or create the model state instance."""
    global _model_state
    if _model_state is None:
        _model_state = ModelState()
    return _model_state


# =============================================================================
# Tool 1: read_keyword_file
# =============================================================================

def read_keyword_file(filepath: str) -> str:
    """Parse an LS-DYNA keyword file (.k, .key, .dyn) into structured data.

    Loads the model into memory for subsequent modification operations.
    Must be called before using modify_material, modify_contact, etc.

    Args:
        filepath: Path to the LS-DYNA keyword file

    Returns:
        JSON string with model summary (keyword counts, parts, materials)
    """
    state = _get_model_state()
    try:
        categories = state.load(filepath)
        parts = state.get_parts()

        result = {
            "status": "loaded",
            "filepath": filepath,
            "total_blocks": len(state.blocks),
            "keyword_categories": categories,
            "parts_count": len(parts),
            "materials_count": len(state.find_blocks("MAT")),
            "contacts_count": len(state.find_blocks("CONTACT")),
            "sections_count": len(state.find_blocks("SECTION")),
            "controls_count": len(state.find_blocks("CONTROL")),
        }
        return json.dumps(result, indent=2)

    except FileNotFoundError as e:
        return json.dumps({"status": "error", "message": str(e)})
    except Exception as e:
        logger.error(f"Failed to parse keyword file: {e}")
        return json.dumps({"status": "error", "message": f"Parse error: {e}"})


# =============================================================================
# Tool 2: modify_material
# =============================================================================

def modify_material(mat_id: int, params: str) -> str:
    """Modify material properties in the currently loaded LS-DYNA model.

    Changes parameter values on an existing *MAT keyword block.
    Requires read_keyword_file to be called first.

    Args:
        mat_id: Material ID (MID) to modify
        params: JSON string of field_position:value pairs, e.g. '{"2": "7.85e-9", "3": "210000"}'
               Field positions are 0-based (0=MID, 1=RO, 2=E, etc.)

    Returns:
        JSON string with modification result (success/failure, old vs new values)
    """
    state = _get_model_state()
    if not state.is_loaded:
        return json.dumps({"status": "error", "message": "No model loaded. Call read_keyword_file first."})

    # Parse params
    try:
        field_updates = json.loads(params) if isinstance(params, str) else params
        field_updates = {int(k): str(v) for k, v in field_updates.items()}
    except (json.JSONDecodeError, ValueError) as e:
        return json.dumps({"status": "error", "message": f"Invalid params JSON: {e}"})

    # Find the material block
    block = state.find_block_by_id("MAT", mat_id)
    if block is None:
        available = [
            KeywordFileParser.parse_fixed_fields(b.lines[0])[0]
            for b in state.find_blocks("MAT")
            if b.lines
        ]
        return json.dumps({
            "status": "error",
            "message": f"Material MID={mat_id} not found.",
            "available_mids": available[:20],
        })

    # Capture old values
    old_fields = KeywordFileParser.parse_fixed_fields(block.lines[0]) if block.lines else []

    # Apply modifications (card_index=0 is the primary data card for MAT)
    state.modify_block_fields(block, card_index=0, field_updates=field_updates)

    new_fields = KeywordFileParser.parse_fixed_fields(block.lines[0]) if block.lines else []

    return json.dumps({
        "status": "modified",
        "keyword": block.keyword,
        "mat_id": mat_id,
        "changes": {
            str(pos): {"old": old_fields[pos] if pos < len(old_fields) else "",
                       "new": val}
            for pos, val in field_updates.items()
        },
    }, indent=2)


# =============================================================================
# Tool 3: modify_contact
# =============================================================================

def modify_contact(contact_id: int, params: str) -> str:
    """Modify contact settings in the currently loaded LS-DYNA model.

    Changes parameter values on an existing *CONTACT keyword block.
    Requires read_keyword_file to be called first.

    Args:
        contact_id: Contact ID (first field — SSID) to modify
        params: JSON string of card_index.field_position:value pairs,
                e.g. '{"0.2": "0.15", "0.3": "0.15"}' for card 0, fields 2 and 3
                Or simple '{"2": "0.15"}' for card 0 (default)

    Returns:
        JSON string with modification result (success/failure, changes applied)
    """
    state = _get_model_state()
    if not state.is_loaded:
        return json.dumps({"status": "error", "message": "No model loaded. Call read_keyword_file first."})

    # Parse params
    try:
        raw_params = json.loads(params) if isinstance(params, str) else params
    except (json.JSONDecodeError, ValueError) as e:
        return json.dumps({"status": "error", "message": f"Invalid params JSON: {e}"})

    # Find the contact block
    block = state.find_block_by_id("CONTACT", contact_id)
    if block is None:
        available = [
            KeywordFileParser.parse_fixed_fields(b.lines[0])[0]
            for b in state.find_blocks("CONTACT")
            if b.lines
        ]
        return json.dumps({
            "status": "error",
            "message": f"Contact SSID={contact_id} not found.",
            "available_contact_ids": available[:20],
        })

    changes = {}
    for key, value in raw_params.items():
        # Support "card.field" notation or plain "field" (defaults to card 0)
        if "." in key:
            card_idx, field_pos = key.split(".", 1)
            card_idx, field_pos = int(card_idx), int(field_pos)
        else:
            card_idx, field_pos = 0, int(key)

        old_fields = (
            KeywordFileParser.parse_fixed_fields(block.lines[card_idx])
            if card_idx < len(block.lines) else []
        )
        old_val = old_fields[field_pos] if field_pos < len(old_fields) else ""

        state.modify_block_fields(block, card_idx, {field_pos: str(value)})
        changes[key] = {"old": old_val, "new": str(value)}

    return json.dumps({
        "status": "modified",
        "keyword": block.keyword,
        "contact_id": contact_id,
        "changes": changes,
    }, indent=2)


# =============================================================================
# Tool 4: add_keyword
# =============================================================================

def add_keyword(keyword_type: str, params: str) -> str:
    """Insert a new keyword block into the currently loaded model.

    Adds the keyword before *END. Use for adding new materials, contacts,
    boundary conditions, control cards, etc.
    Requires read_keyword_file to be called first.

    Args:
        keyword_type: Keyword name, e.g. '*MAT_024', '*CONTROL_TIMESTEP',
                      '*BOUNDARY_SPC_SET'
        params: JSON string with 'cards' (list of lists), e.g.
                '{"cards": [["1", "7.85e-9", "210000", "0.3", "250"]]}'
                Each inner list is one data card's field values.
                Optional 'title' key for TITLE-option keywords.

    Returns:
        JSON string confirming the keyword was added
    """
    state = _get_model_state()
    if not state.is_loaded:
        return json.dumps({"status": "error", "message": "No model loaded. Call read_keyword_file first."})

    # Parse params
    try:
        data = json.loads(params) if isinstance(params, str) else params
    except (json.JSONDecodeError, ValueError) as e:
        return json.dumps({"status": "error", "message": f"Invalid params JSON: {e}"})

    cards = data.get("cards", [])
    title = data.get("title", "")
    if not cards:
        return json.dumps({"status": "error", "message": "'cards' list is required in params"})

    # Build data lines (fixed-width 10-char columns)
    data_lines = []
    options = []

    if title:
        data_lines.append(title)
        options.append("TITLE")

    for card_fields in cards:
        line = "".join(f"{str(v):>10}" for v in card_fields)
        data_lines.append(line)

    keyword = keyword_type.upper()
    if not keyword.startswith("*"):
        keyword = f"*{keyword}"

    new_block = state.add_block(keyword, data_lines, options)

    return json.dumps({
        "status": "added",
        "keyword": new_block.full_keyword,
        "cards_count": len(cards),
        "total_blocks": len(state.blocks),
    }, indent=2)


# =============================================================================
# Tool 5: list_parts
# =============================================================================

def list_parts() -> str:
    """List all parts in the currently loaded LS-DYNA model.

    Returns PID, name, section ID, and material ID for every *PART block.
    Requires read_keyword_file to be called first.

    Returns:
        JSON string with parts list (pid, name, secid, mid for each)
    """
    state = _get_model_state()
    if not state.is_loaded:
        return json.dumps({"status": "error", "message": "No model loaded. Call read_keyword_file first."})

    parts = state.get_parts()
    if not parts:
        return json.dumps({"status": "ok", "parts": [], "message": "No *PART blocks found"})

    return json.dumps({
        "status": "ok",
        "parts_count": len(parts),
        "parts": [
            {
                "pid": p.pid,
                "name": p.name,
                "section_id": p.secid,
                "material_id": p.mid,
                "eos_id": p.eosid,
                "hourglass_id": p.hgid,
            }
            for p in parts
        ],
    }, indent=2)


# =============================================================================
# Tool 6: get_model_summary
# =============================================================================

def get_model_summary() -> str:
    """Get a high-level overview of the currently loaded LS-DYNA model.

    Returns keyword counts by category, element types, material types,
    contact definitions, control settings, and output requests.
    Requires read_keyword_file to be called first.

    Returns:
        JSON string with comprehensive model summary
    """
    state = _get_model_state()
    if not state.is_loaded:
        return json.dumps({"status": "error", "message": "No model loaded. Call read_keyword_file first."})

    categories = state._categorize_blocks()
    parts = state.get_parts()

    # Gather material types
    materials = []
    for block in state.find_blocks("MAT"):
        mid = ""
        if block.lines:
            fields = KeywordFileParser.parse_fixed_fields(block.lines[0])
            mid = fields[0] if fields else ""
        materials.append({"keyword": block.keyword, "mid": mid})

    # Gather contact types
    contacts = []
    for block in state.find_blocks("CONTACT"):
        contacts.append({"keyword": block.keyword})

    # Gather control cards
    controls = [block.keyword for block in state.find_blocks("CONTROL")]

    # Gather element sections
    sections = []
    for block in state.find_blocks("SECTION"):
        sections.append({"keyword": block.keyword})

    # Gather output / database keywords
    db_keywords = [block.keyword for block in state.find_blocks("DATABASE")]

    # Gather boundary conditions
    boundaries = [block.keyword for block in state.find_blocks("BOUNDARY")]

    # Gather load definitions
    loads = [block.keyword for block in state.find_blocks("LOAD")]

    return json.dumps({
        "status": "ok",
        "filepath": state.filepath,
        "total_keyword_blocks": len(state.blocks),
        "keyword_categories": categories,
        "parts": {
            "count": len(parts),
            "pids": [p.pid for p in parts[:50]],
        },
        "materials": {
            "count": len(materials),
            "types": materials[:50],
        },
        "contacts": {
            "count": len(contacts),
            "types": contacts[:30],
        },
        "sections": {
            "count": len(sections),
            "types": sections[:30],
        },
        "control_cards": controls,
        "database_output": db_keywords,
        "boundary_conditions": boundaries[:20],
        "loads": loads[:20],
    }, indent=2)


# =============================================================================
# Tool Registry (imported by agent.py)
# =============================================================================

ALL_TOOLS = [
    read_keyword_file,
    modify_material,
    modify_contact,
    add_keyword,
    list_parts,
    get_model_summary,
]
