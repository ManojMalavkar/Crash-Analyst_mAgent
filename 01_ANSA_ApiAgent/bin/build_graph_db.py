"""Neo4j Graph Database Builder for ANSA/META GraphRAG.

Reads normalized API records from api_normalized.jsonl + workflows.jsonl
(produced by ingest.py) and builds the full Neo4j knowledge graph.

Graph Design:

  NODE TYPES:
    Module, Class, Function, Method, Attribute, Parameter,
    ReturnType, Example, Version, Constant, EnumValue

  RELATIONSHIPS:
    Documentation:  CONTAINS, HAS_CLASS, HAS_FUNCTION, HAS_METHOD, HAS_ATTRIBUTE, HAS_PARAMETER
    Type:           RETURNS, ACCEPTS, CAN_NAVIGATE_TO
    Knowledge:      SEE_ALSO, HAS_EXAMPLE, USES_CONSTANT, ALLOWS_VALUE
    Migration:      DEPRECATED_IN, REPLACED_BY, MIGRATION_PATH
    Workflow:       COMMONLY_FOLLOWED_BY, USED_BY

Pipeline:
  Step 1: python bin/ingest.py <docs_path>       -> api_normalized.jsonl + workflows.jsonl
  Step 2: python bin/build_graph_db.py           -> Neo4j graph

Usage:
    # Build graph (default: bolt://localhost:7687)
    python bin/build_graph_db.py

    # Custom Neo4j connection
    python bin/build_graph_db.py --uri bolt://neo4j:7687 --user neo4j --password secret

    # Rebuild (clear existing graph)
    python bin/build_graph_db.py --rebuild

    # Programmatic
    from bin.build_graph_db import GraphBuilder
    builder = GraphBuilder(uri="bolt://localhost:7687")
    builder.build_from_jsonl("knowledge-base/api_normalized.jsonl")
"""

import json
import logging
import time
import os
import sys
from pathlib import Path
from typing import Optional
from collections import defaultdict

try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None


logger = logging.getLogger(__name__)


# =============================================================================
# Neo4j Graph Builder
# =============================================================================

class GraphBuilder:
    """Builds Neo4j knowledge graph for ANSA/META API documentation.

    Implements the full graph design from the CodeRAG/GraphRAG spec:
    - Module hierarchy (Module -> Class -> Method/Attribute)
    - Type relationships (RETURNS, ACCEPTS, CAN_NAVIGATE_TO)
    - Deprecation paths (DEPRECATED_IN, REPLACED_BY)
    - Workflow sequences (COMMONLY_FOLLOWED_BY)
    - Cross-references (SEE_ALSO, HAS_EXAMPLE)
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
    ):
        """Initialize Neo4j connection.

        Args:
            uri: Neo4j bolt URI
            user: Neo4j username
            password: Neo4j password
        """
        if GraphDatabase is None:
            raise ImportError(
                "neo4j driver required. Install: pip install neo4j"
            )

        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        # Verify connectivity
        self._driver.verify_connectivity()
        logger.info(f"Neo4j connected: {uri}")

    def close(self):
        """Close Neo4j driver."""
        self._driver.close()

    # ------------------------------------------------------------------
    # Schema Setup
    # ------------------------------------------------------------------

    def _create_constraints_and_indexes(self, session):
        """Create uniqueness constraints and indexes for performance."""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Module) REQUIRE m.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Class) REQUIRE c.full_name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (f:Function) REQUIRE f.full_name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Method) REQUIRE m.full_name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Attribute) REQUIRE a.full_name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (r:ReturnType) REQUIRE r.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (v:Version) REQUIRE v.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Example) REQUIRE e.id IS UNIQUE",
        ]

        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (n:Function) ON (n.name)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Method) ON (n.name)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Class) ON (n.name)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Function) ON (n.software)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Method) ON (n.software)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Function) ON (n.deprecated)",
        ]

        for stmt in constraints + indexes:
            try:
                session.run(stmt)
            except Exception as e:
                logger.debug(f"Constraint/index skip: {e}")

    def _clear_graph(self, session):
        """Delete all nodes and relationships."""
        session.run("MATCH (n) DETACH DELETE n")
        logger.info("Graph cleared")

    # ------------------------------------------------------------------
    # Node Creation
    # ------------------------------------------------------------------

    def _create_modules(self, session, records: list[dict]) -> set:
        """Create Module nodes."""
        modules = set()
        for rec in records:
            module = rec.get("module")
            if module:
                modules.add(module)

        if modules:
            session.run(
                "UNWIND $modules AS mod "
                "MERGE (m:Module {name: mod}) "
                "SET m.software = CASE WHEN mod STARTS WITH 'meta' THEN 'meta' ELSE 'ansa' END",
                modules=list(modules),
            )
        logger.info(f"  Modules: {len(modules)}")
        return modules

    def _create_classes(self, session, records: list[dict]) -> int:
        """Create Class nodes."""
        classes = [
            rec for rec in records if rec.get("type") == "class"
        ]

        batch = []
        for rec in classes:
            batch.append({
                "full_name": rec["full_name"],
                "name": rec["name"],
                "module": rec.get("module", ""),
                "description": rec.get("description", "")[:500],
                "software": rec.get("software", ""),
                "deprecated": rec.get("deprecated", False),
                "navigation_targets": rec.get("navigation_targets", []),
            })

        if batch:
            session.run(
                "UNWIND $batch AS rec "
                "MERGE (c:Class {full_name: rec.full_name}) "
                "SET c.name = rec.name, "
                "    c.module = rec.module, "
                "    c.description = rec.description, "
                "    c.software = rec.software, "
                "    c.deprecated = rec.deprecated",
                batch=batch,
            )
        logger.info(f"  Classes: {len(batch)}")
        return len(batch)

    def _create_functions(self, session, records: list[dict]) -> int:
        """Create Function nodes (standalone functions, not methods)."""
        functions = [
            rec for rec in records if rec.get("type") == "function"
        ]

        batch = []
        for rec in functions:
            batch.append({
                "full_name": rec["full_name"],
                "name": rec["name"],
                "module": rec.get("module", ""),
                "signature": rec.get("signature", ""),
                "description": rec.get("description", "")[:500],
                "returns": rec.get("returns", ""),
                "software": rec.get("software", ""),
                "deprecated": rec.get("deprecated", False),
                "deprecated_version": rec.get("deprecated_version", ""),
                "replacement": rec.get("replacement", ""),
            })

        if batch:
            session.run(
                "UNWIND $batch AS rec "
                "MERGE (f:Function {full_name: rec.full_name}) "
                "SET f.name = rec.name, "
                "    f.module = rec.module, "
                "    f.signature = rec.signature, "
                "    f.description = rec.description, "
                "    f.returns = rec.returns, "
                "    f.software = rec.software, "
                "    f.deprecated = rec.deprecated, "
                "    f.deprecated_version = rec.deprecated_version, "
                "    f.replacement = rec.replacement",
                batch=batch,
            )
        logger.info(f"  Functions: {len(batch)}")
        return len(batch)

    def _create_methods(self, session, records: list[dict]) -> int:
        """Create Method nodes."""
        methods = [
            rec for rec in records if rec.get("type") == "method"
        ]

        batch = []
        for rec in methods:
            batch.append({
                "full_name": rec["full_name"],
                "name": rec["name"],
                "module": rec.get("module", ""),
                "class_name": rec.get("class_name", ""),
                "signature": rec.get("signature", ""),
                "description": rec.get("description", "")[:500],
                "returns": rec.get("returns", ""),
                "software": rec.get("software", ""),
                "deprecated": rec.get("deprecated", False),
            })

        if batch:
            session.run(
                "UNWIND $batch AS rec "
                "MERGE (m:Method {full_name: rec.full_name}) "
                "SET m.name = rec.name, "
                "    m.module = rec.module, "
                "    m.class_name = rec.class_name, "
                "    m.signature = rec.signature, "
                "    m.description = rec.description, "
                "    m.returns = rec.returns, "
                "    m.software = rec.software, "
                "    m.deprecated = rec.deprecated",
                batch=batch,
            )
        logger.info(f"  Methods: {len(batch)}")
        return len(batch)

    def _create_attributes(self, session, records: list[dict]) -> int:
        """Create Attribute nodes."""
        attributes = [
            rec for rec in records if rec.get("type") == "attribute"
        ]

        batch = []
        for rec in attributes:
            batch.append({
                "full_name": rec["full_name"],
                "name": rec["name"],
                "module": rec.get("module", ""),
                "class_name": rec.get("class_name", ""),
                "description": rec.get("description", "")[:500],
                "software": rec.get("software", ""),
            })

        if batch:
            session.run(
                "UNWIND $batch AS rec "
                "MERGE (a:Attribute {full_name: rec.full_name}) "
                "SET a.name = rec.name, "
                "    a.module = rec.module, "
                "    a.class_name = rec.class_name, "
                "    a.description = rec.description, "
                "    a.software = rec.software",
                batch=batch,
            )
        logger.info(f"  Attributes: {len(batch)}")
        return len(batch)

    def _create_return_types(self, session, records: list[dict]) -> int:
        """Create ReturnType nodes."""
        return_types = set()
        for rec in records:
            ret = rec.get("returns")
            if ret:
                # Use the short type name (last part after dot)
                clean = ret.rsplit(".", 1)[-1]
                return_types.add(clean)

        if return_types:
            session.run(
                "UNWIND $types AS t "
                "MERGE (r:ReturnType {name: t})",
                types=list(return_types),
            )
        logger.info(f"  ReturnTypes: {len(return_types)}")
        return len(return_types)

    def _create_versions(self, session, records: list[dict]) -> int:
        """Create Version nodes from deprecated_version fields."""
        versions = set()
        for rec in records:
            ver = rec.get("deprecated_version")
            if ver:
                versions.add(ver)

        if versions:
            session.run(
                "UNWIND $versions AS v "
                "MERGE (ver:Version {name: v})",
                versions=list(versions),
            )
        logger.info(f"  Versions: {len(versions)}")
        return len(versions)

    def _create_parameters(self, session, records: list[dict]) -> int:
        """Create Parameter nodes and HAS_PARAMETER relationships."""
        count = 0
        batch = []

        for rec in records:
            for param in rec.get("parameters", []):
                if not param.get("name"):
                    continue
                param_id = f"{rec['full_name']}::{param['name']}"
                batch.append({
                    "param_id": param_id,
                    "name": param["name"],
                    "type": param.get("type", ""),
                    "optional": param.get("optional", False),
                    "default": param.get("default", ""),
                    "parent_full_name": rec["full_name"],
                    "parent_type": rec.get("type", "function"),
                })
                count += 1

        # Batch create params
        if batch:
            # Create in chunks to avoid transaction size limits
            chunk_size = 500
            for i in range(0, len(batch), chunk_size):
                chunk = batch[i:i + chunk_size]
                session.run(
                    "UNWIND $batch AS rec "
                    "MERGE (p:Parameter {id: rec.param_id}) "
                    "SET p.name = rec.name, "
                    "    p.type = rec.type, "
                    "    p.optional = rec.optional, "
                    "    p.default = rec.default",
                    batch=chunk,
                )

        logger.info(f"  Parameters: {count}")
        return count

    # ------------------------------------------------------------------
    # Relationship Creation
    # ------------------------------------------------------------------

    def _create_contains_relationships(self, session, records: list[dict]):
        """Module CONTAINS Class/Function."""
        batch = []
        for rec in records:
            module = rec.get("module")
            if not module:
                continue
            if rec["type"] == "class":
                batch.append({"module": module, "child": rec["full_name"], "rel": "HAS_CLASS"})
            elif rec["type"] == "function":
                batch.append({"module": module, "child": rec["full_name"], "rel": "HAS_FUNCTION"})

        if batch:
            # HAS_CLASS
            class_batch = [b for b in batch if b["rel"] == "HAS_CLASS"]
            if class_batch:
                session.run(
                    "UNWIND $batch AS rec "
                    "MATCH (m:Module {name: rec.module}) "
                    "MATCH (c:Class {full_name: rec.child}) "
                    "MERGE (m)-[:HAS_CLASS]->(c)",
                    batch=class_batch,
                )
            # HAS_FUNCTION
            func_batch = [b for b in batch if b["rel"] == "HAS_FUNCTION"]
            if func_batch:
                session.run(
                    "UNWIND $batch AS rec "
                    "MATCH (m:Module {name: rec.module}) "
                    "MATCH (f:Function {full_name: rec.child}) "
                    "MERGE (m)-[:HAS_FUNCTION]->(f)",
                    batch=func_batch,
                )
        logger.info(f"  CONTAINS/HAS_CLASS/HAS_FUNCTION: {len(batch)}")

    def _create_method_relationships(self, session, records: list[dict]):
        """Class HAS_METHOD and HAS_ATTRIBUTE."""
        method_batch = []
        attr_batch = []

        for rec in records:
            if rec.get("type") == "method" and rec.get("class_name") and rec.get("module"):
                class_full = f"{rec['module']}.{rec['class_name']}"
                method_batch.append({"class_fn": class_full, "method_fn": rec["full_name"]})
            elif rec.get("type") == "attribute" and rec.get("class_name") and rec.get("module"):
                class_full = f"{rec['module']}.{rec['class_name']}"
                attr_batch.append({"class_fn": class_full, "attr_fn": rec["full_name"]})

        if method_batch:
            session.run(
                "UNWIND $batch AS rec "
                "MATCH (c:Class {full_name: rec.class_fn}) "
                "MATCH (m:Method {full_name: rec.method_fn}) "
                "MERGE (c)-[:HAS_METHOD]->(m)",
                batch=method_batch,
            )
        if attr_batch:
            session.run(
                "UNWIND $batch AS rec "
                "MATCH (c:Class {full_name: rec.class_fn}) "
                "MATCH (a:Attribute {full_name: rec.attr_fn}) "
                "MERGE (c)-[:HAS_ATTRIBUTE]->(a)",
                batch=attr_batch,
            )
        logger.info(f"  HAS_METHOD: {len(method_batch)}, HAS_ATTRIBUTE: {len(attr_batch)}")

    def _create_parameter_relationships(self, session, records: list[dict]):
        """Function/Method HAS_PARAMETER."""
        batch = []
        for rec in records:
            for param in rec.get("parameters", []):
                if not param.get("name"):
                    continue
                param_id = f"{rec['full_name']}::{param['name']}"
                batch.append({
                    "parent_fn": rec["full_name"],
                    "parent_label": "Function" if rec["type"] == "function" else "Method",
                    "param_id": param_id,
                })

        # Split by label for matching
        func_params = [b for b in batch if b["parent_label"] == "Function"]
        method_params = [b for b in batch if b["parent_label"] == "Method"]

        chunk_size = 500
        if func_params:
            for i in range(0, len(func_params), chunk_size):
                session.run(
                    "UNWIND $batch AS rec "
                    "MATCH (f:Function {full_name: rec.parent_fn}) "
                    "MATCH (p:Parameter {id: rec.param_id}) "
                    "MERGE (f)-[:HAS_PARAMETER]->(p)",
                    batch=func_params[i:i + chunk_size],
                )
        if method_params:
            for i in range(0, len(method_params), chunk_size):
                session.run(
                    "UNWIND $batch AS rec "
                    "MATCH (m:Method {full_name: rec.parent_fn}) "
                    "MATCH (p:Parameter {id: rec.param_id}) "
                    "MERGE (m)-[:HAS_PARAMETER]->(p)",
                    batch=method_params[i:i + chunk_size],
                )
        logger.info(f"  HAS_PARAMETER: {len(batch)}")

    def _create_returns_relationships(self, session, records: list[dict]):
        """Function/Method RETURNS ReturnType."""
        func_batch = []
        method_batch = []

        for rec in records:
            ret = rec.get("returns")
            if not ret:
                continue
            clean_ret = ret.rsplit(".", 1)[-1]
            entry = {"fn": rec["full_name"], "ret_type": clean_ret}

            if rec["type"] == "function":
                func_batch.append(entry)
            elif rec["type"] == "method":
                method_batch.append(entry)

        if func_batch:
            session.run(
                "UNWIND $batch AS rec "
                "MATCH (f:Function {full_name: rec.fn}) "
                "MATCH (r:ReturnType {name: rec.ret_type}) "
                "MERGE (f)-[:RETURNS]->(r)",
                batch=func_batch,
            )
        if method_batch:
            session.run(
                "UNWIND $batch AS rec "
                "MATCH (m:Method {full_name: rec.fn}) "
                "MATCH (r:ReturnType {name: rec.ret_type}) "
                "MERGE (m)-[:RETURNS]->(r)",
                batch=method_batch,
            )
        logger.info(f"  RETURNS: {len(func_batch) + len(method_batch)}")

    def _create_accepts_relationships(self, session, records: list[dict]):
        """Function/Method ACCEPTS type."""
        func_batch = []
        method_batch = []

        for rec in records:
            for atype in rec.get("accepts_types", []):
                entry = {"fn": rec["full_name"], "accepts_type": atype}
                if rec["type"] == "function":
                    func_batch.append(entry)
                elif rec["type"] == "method":
                    method_batch.append(entry)

        if func_batch:
            session.run(
                "UNWIND $batch AS rec "
                "MATCH (f:Function {full_name: rec.fn}) "
                "MERGE (r:ReturnType {name: rec.accepts_type}) "
                "MERGE (f)-[:ACCEPTS]->(r)",
                batch=func_batch,
            )
        if method_batch:
            session.run(
                "UNWIND $batch AS rec "
                "MATCH (m:Method {full_name: rec.fn}) "
                "MERGE (r:ReturnType {name: rec.accepts_type}) "
                "MERGE (m)-[:ACCEPTS]->(r)",
                batch=method_batch,
            )
        logger.info(f"  ACCEPTS: {len(func_batch) + len(method_batch)}")

    def _create_navigation_relationships(self, session, records: list[dict]):
        """CAN_NAVIGATE_TO between classes/methods and target types."""
        batch = []
        for rec in records:
            for nav_target in rec.get("navigation_targets", []):
                batch.append({
                    "source_fn": rec["full_name"],
                    "source_label": rec.get("type", "function").capitalize(),
                    "target_type": nav_target,
                })

        if batch:
            # Use generic MERGE with labels
            session.run(
                "UNWIND $batch AS rec "
                "MERGE (r:ReturnType {name: rec.target_type}) "
                "WITH rec, r "
                "OPTIONAL MATCH (f:Function {full_name: rec.source_fn}) "
                "OPTIONAL MATCH (m:Method {full_name: rec.source_fn}) "
                "OPTIONAL MATCH (c:Class {full_name: rec.source_fn}) "
                "WITH rec, r, coalesce(f, m, c) AS source "
                "WHERE source IS NOT NULL "
                "MERGE (source)-[:CAN_NAVIGATE_TO]->(r)",
                batch=batch,
            )
        logger.info(f"  CAN_NAVIGATE_TO: {len(batch)}")

    def _create_deprecation_relationships(self, session, records: list[dict]):
        """DEPRECATED_IN version and REPLACED_BY function."""
        deprecated_version_batch = []
        replaced_by_batch = []

        for rec in records:
            if not rec.get("deprecated"):
                continue

            fn = rec["full_name"]
            label = rec.get("type", "function").capitalize()

            if rec.get("deprecated_version"):
                deprecated_version_batch.append({
                    "fn": fn, "version": rec["deprecated_version"], "label": label
                })

            if rec.get("replacement"):
                replaced_by_batch.append({
                    "fn": fn, "replacement": rec["replacement"], "label": label
                })

        if deprecated_version_batch:
            session.run(
                "UNWIND $batch AS rec "
                "MATCH (v:Version {name: rec.version}) "
                "WITH rec, v "
                "OPTIONAL MATCH (f:Function {full_name: rec.fn}) "
                "OPTIONAL MATCH (m:Method {full_name: rec.fn}) "
                "WITH rec, v, coalesce(f, m) AS source "
                "WHERE source IS NOT NULL "
                "MERGE (source)-[:DEPRECATED_IN]->(v)",
                batch=deprecated_version_batch,
            )

        if replaced_by_batch:
            session.run(
                "UNWIND $batch AS rec "
                "OPTIONAL MATCH (f:Function {full_name: rec.fn}) "
                "OPTIONAL MATCH (m:Method {full_name: rec.fn}) "
                "WITH rec, coalesce(f, m) AS source "
                "WHERE source IS NOT NULL "
                "MERGE (target:Function {full_name: rec.replacement}) "
                "MERGE (source)-[:REPLACED_BY]->(target)",
                batch=replaced_by_batch,
            )

        logger.info(
            f"  DEPRECATED_IN: {len(deprecated_version_batch)}, "
            f"REPLACED_BY: {len(replaced_by_batch)}"
        )

    def _create_see_also_relationships(self, session, records: list[dict]):
        """SEE_ALSO cross-references."""
        batch = []
        for rec in records:
            for ref in rec.get("see_also", []):
                if ref != rec["full_name"]:
                    batch.append({"source": rec["full_name"], "target": ref})

        if batch:
            session.run(
                "UNWIND $batch AS rec "
                "OPTIONAL MATCH (s:Function {full_name: rec.source}) "
                "OPTIONAL MATCH (sm:Method {full_name: rec.source}) "
                "OPTIONAL MATCH (sc:Class {full_name: rec.source}) "
                "WITH rec, coalesce(s, sm, sc) AS source "
                "WHERE source IS NOT NULL "
                "MERGE (target:Function {full_name: rec.target}) "
                "MERGE (source)-[:SEE_ALSO]->(target)",
                batch=batch,
            )
        logger.info(f"  SEE_ALSO: {len(batch)}")

    def _create_workflow_relationships(self, session, workflows: list[dict]):
        """COMMONLY_FOLLOWED_BY from workflow sequences."""
        # Count co-occurrence to weight edges
        edge_counts = defaultdict(int)
        for wf in workflows:
            sequence = wf.get("sequence", [])
            for i in range(len(sequence) - 1):
                src, tgt = sequence[i], sequence[i + 1]
                if src != tgt:
                    edge_counts[(src, tgt)] += 1

        batch = [
            {"source": src, "target": tgt, "weight": count}
            for (src, tgt), count in edge_counts.items()
        ]

        if batch:
            session.run(
                "UNWIND $batch AS rec "
                "MERGE (s:Function {full_name: rec.source}) "
                "MERGE (t:Function {full_name: rec.target}) "
                "MERGE (s)-[r:COMMONLY_FOLLOWED_BY]->(t) "
                "SET r.weight = rec.weight",
                batch=batch,
            )
        logger.info(f"  COMMONLY_FOLLOWED_BY: {len(batch)} (from {len(workflows)} workflows)")

    def _create_example_nodes_and_relationships(self, session, records: list[dict]):
        """Create Example nodes and HAS_EXAMPLE relationships."""
        batch = []
        example_id = 0

        for rec in records:
            for i, example_code in enumerate(rec.get("examples", [])):
                if len(example_code) > 15:
                    eid = f"{rec['full_name']}::ex_{i}"
                    batch.append({
                        "id": eid,
                        "code": example_code[:2000],
                        "parent_fn": rec["full_name"],
                        "parent_type": rec.get("type", "function"),
                    })
                    example_id += 1

        if batch:
            # Create Example nodes
            session.run(
                "UNWIND $batch AS rec "
                "MERGE (e:Example {id: rec.id}) "
                "SET e.code = rec.code",
                batch=batch,
            )
            # Create HAS_EXAMPLE relationships
            session.run(
                "UNWIND $batch AS rec "
                "MATCH (e:Example {id: rec.id}) "
                "WITH rec, e "
                "OPTIONAL MATCH (f:Function {full_name: rec.parent_fn}) "
                "OPTIONAL MATCH (m:Method {full_name: rec.parent_fn}) "
                "OPTIONAL MATCH (c:Class {full_name: rec.parent_fn}) "
                "WITH rec, e, coalesce(f, m, c) AS parent "
                "WHERE parent IS NOT NULL "
                "MERGE (parent)-[:HAS_EXAMPLE]->(e)",
                batch=batch,
            )
        logger.info(f"  Examples + HAS_EXAMPLE: {len(batch)}")

    # ------------------------------------------------------------------
    # Main Build
    # ------------------------------------------------------------------

    def build_from_jsonl(
        self,
        api_jsonl: str | Path,
        workflows_jsonl: Optional[str | Path] = None,
        rebuild: bool = False,
    ) -> dict:
        """Build the full Neo4j graph from JSONL files.

        Args:
            api_jsonl: Path to api_normalized.jsonl
            workflows_jsonl: Path to workflows.jsonl (optional)
            rebuild: If True, clear existing graph first

        Returns:
            Build statistics dict
        """
        api_jsonl = Path(api_jsonl)
        if not api_jsonl.exists():
            raise FileNotFoundError(f"API JSONL not found: {api_jsonl}")

        start_time = time.time()

        # Load records
        records = []
        with open(api_jsonl, "r", encoding="utf-8") as fp:
            for line in fp:
                if line.strip():
                    records.append(json.loads(line))
        logger.info(f"Loaded {len(records)} API records")

        # Load workflows
        workflows = []
        if workflows_jsonl:
            wf_path = Path(workflows_jsonl)
            if wf_path.exists():
                with open(wf_path, "r", encoding="utf-8") as fp:
                    for line in fp:
                        if line.strip():
                            workflows.append(json.loads(line))
                logger.info(f"Loaded {len(workflows)} workflows")

        with self._driver.session() as session:
            # Setup
            if rebuild:
                self._clear_graph(session)
            self._create_constraints_and_indexes(session)

            # Create nodes
            print("  Creating nodes...")
            self._create_modules(session, records)
            n_classes = self._create_classes(session, records)
            n_functions = self._create_functions(session, records)
            n_methods = self._create_methods(session, records)
            n_attributes = self._create_attributes(session, records)
            n_return_types = self._create_return_types(session, records)
            n_versions = self._create_versions(session, records)
            n_params = self._create_parameters(session, records)

            # Create relationships
            print("  Creating relationships...")
            self._create_contains_relationships(session, records)
            self._create_method_relationships(session, records)
            self._create_parameter_relationships(session, records)
            self._create_returns_relationships(session, records)
            self._create_accepts_relationships(session, records)
            self._create_navigation_relationships(session, records)
            self._create_deprecation_relationships(session, records)
            self._create_see_also_relationships(session, records)
            self._create_example_nodes_and_relationships(session, records)

            # Workflows
            if workflows:
                print("  Creating workflow relationships...")
                self._create_workflow_relationships(session, workflows)

            # Get final counts
            result = session.run(
                "MATCH (n) RETURN count(n) AS nodes"
            ).single()
            total_nodes = result["nodes"]

            result = session.run(
                "MATCH ()-[r]->() RETURN count(r) AS rels"
            ).single()
            total_rels = result["rels"]

        duration = time.time() - start_time

        stats = {
            "total_nodes": total_nodes,
            "total_relationships": total_rels,
            "classes": n_classes,
            "functions": n_functions,
            "methods": n_methods,
            "attributes": n_attributes,
            "return_types": n_return_types,
            "parameters": n_params,
            "versions": n_versions,
            "workflows": len(workflows),
            "duration_seconds": round(duration, 1),
        }

        logger.info(f"Graph build complete: {total_nodes} nodes, {total_rels} relationships")
        return stats

    # ------------------------------------------------------------------
    # Query Helpers
    # ------------------------------------------------------------------

    def get_class_methods(self, class_name: str) -> list[dict]:
        """Get all methods of a class."""
        with self._driver.session() as session:
            result = session.run(
                "MATCH (c:Class)-[:HAS_METHOD]->(m:Method) "
                "WHERE c.full_name = $name OR c.name = $name "
                "RETURN m.full_name AS full_name, m.name AS name, "
                "       m.signature AS signature, m.returns AS returns",
                name=class_name,
            )
            return [dict(r) for r in result]

    def get_navigation_targets(self, class_name: str) -> list[str]:
        """Get types reachable from a class via CAN_NAVIGATE_TO."""
        with self._driver.session() as session:
            result = session.run(
                "MATCH (c:Class)-[:HAS_METHOD]->(m:Method)-[:CAN_NAVIGATE_TO]->(t:ReturnType) "
                "WHERE c.full_name = $name OR c.name = $name "
                "RETURN DISTINCT t.name AS target_type",
                name=class_name,
            )
            return [r["target_type"] for r in result]

    def get_workflow_next(self, function_name: str, top_k: int = 5) -> list[dict]:
        """Get commonly following functions (workflow recommendation)."""
        with self._driver.session() as session:
            result = session.run(
                "MATCH (f:Function)-[r:COMMONLY_FOLLOWED_BY]->(next:Function) "
                "WHERE f.full_name = $name OR f.name = $name "
                "RETURN next.full_name AS full_name, next.name AS name, "
                "       r.weight AS weight "
                "ORDER BY r.weight DESC LIMIT $limit",
                name=function_name,
                limit=top_k,
            )
            return [dict(r) for r in result]

    def get_deprecation_path(self, function_name: str) -> dict:
        """Get deprecation info and replacement for a function."""
        with self._driver.session() as session:
            result = session.run(
                "MATCH (f)-[:DEPRECATED_IN]->(v:Version) "
                "WHERE f.full_name = $name OR f.name = $name "
                "OPTIONAL MATCH (f)-[:REPLACED_BY]->(r) "
                "RETURN f.full_name AS deprecated_function, "
                "       v.name AS deprecated_version, "
                "       r.full_name AS replacement",
                name=function_name,
            )
            record = result.single()
            return dict(record) if record else {}

    def get_functions_accepting(self, type_name: str) -> list[dict]:
        """Get all functions that accept a given type (for ANSA workflow discovery)."""
        with self._driver.session() as session:
            result = session.run(
                "MATCH (f)-[:ACCEPTS]->(r:ReturnType {name: $type_name}) "
                "RETURN f.full_name AS full_name, f.name AS name, "
                "       f.signature AS signature, labels(f)[0] AS label",
                type_name=type_name,
            )
            return [dict(r) for r in result]

    def get_stats(self) -> dict:
        """Get graph statistics."""
        with self._driver.session() as session:
            nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            rels = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
            labels = session.run(
                "CALL db.labels() YIELD label "
                "RETURN label, count{(n) WHERE label IN labels(n)} AS count"
            )
            label_counts = {r["label"]: r["count"] for r in labels}

            return {
                "total_nodes": nodes,
                "total_relationships": rels,
                "labels": label_counts,
            }


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Build ANSA/META Neo4j knowledge graph from normalized JSONL",
        epilog="""
Examples:
  python bin/build_graph_db.py --rebuild
  python bin/build_graph_db.py --uri bolt://neo4j:7687 --password mysecret
  python bin/build_graph_db.py --source /path/to/api_normalized.jsonl
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    kb_dir = Path(__file__).resolve().parent.parent / "knowledge-base"
    default_source = str(kb_dir / "api_normalized.jsonl")
    default_workflows = str(kb_dir / "workflows.jsonl")

    parser.add_argument("--source", default=default_source,
                        help=f"Path to api_normalized.jsonl")
    parser.add_argument("--workflows", default=default_workflows,
                        help=f"Path to workflows.jsonl")
    parser.add_argument("--uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                        help="Neo4j bolt URI (env: NEO4J_URI)")
    parser.add_argument("--user", default=os.getenv("NEO4J_USER", "neo4j"),
                        help="Neo4j username (env: NEO4J_USER)")
    parser.add_argument("--password", default=os.getenv("NEO4J_PASSWORD", "password"),
                        help="Neo4j password (env: NEO4J_PASSWORD)")
    parser.add_argument("--rebuild", action="store_true",
                        help="Clear and rebuild the entire graph")
    args = parser.parse_args()

    source_path = Path(args.source)
    if not source_path.exists():
        print(f"\n  ERROR: Source file not found: {source_path}")
        print(f"  Run ingest.py first: python bin/ingest.py /path/to/docs")
        sys.exit(1)

    workflows_path = Path(args.workflows) if Path(args.workflows).exists() else None

    print(f"\n{'='*60}")
    print(f"  Building Neo4j Knowledge Graph")
    print(f"  Source:    {source_path}")
    print(f"  Workflows: {workflows_path or 'N/A'}")
    print(f"  Neo4j:     {args.uri}")
    print(f"  Rebuild:   {args.rebuild}")
    print(f"{'='*60}\n")

    builder = GraphBuilder(
        uri=args.uri,
        user=args.user,
        password=args.password,
    )

    try:
        build_stats = builder.build_from_jsonl(
            api_jsonl=source_path,
            workflows_jsonl=workflows_path,
            rebuild=args.rebuild,
        )

        print(f"\n{'='*60}")
        print(f"  GRAPH BUILD COMPLETE")
        print(f"{'='*60}")
        for k, v in build_stats.items():
            print(f"  {k:25s}: {v}")

        # Quick validation queries
        if build_stats["total_nodes"] > 0:
            print(f"\n  --- Validation Queries ---")

            # Test navigation
            nav = builder.get_navigation_targets("Annotation")
            if nav:
                print(f"  Annotation can navigate to: {', '.join(nav[:5])}")

            # Test workflow
            wf = builder.get_workflow_next("CreateCurve")
            if wf:
                print(f"  After CreateCurve: {', '.join(w['name'] for w in wf[:3])}")

        print(f"\n{'='*60}")
        print(f"  Graph ready for querying!")
        print(f"{'='*60}\n")

    finally:
        builder.close()
