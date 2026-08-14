"""Embedding Model Evaluation for ANSA/META API Retrieval.

Compares multiple embedding models on domain-specific retrieval quality
using a test set of (query, expected_result) pairs.

Metrics:
- Hit Rate @K: Does the correct document appear in top-K results?
- MRR (Mean Reciprocal Rank): How high does the correct result rank?
- Average Latency: Embedding speed per query

Usage:
    python evaluation/embedding_eval.py
    python evaluation/embedding_eval.py --models bge-small bge-base jina-code
"""

import time
import numpy as np
from dataclasses import dataclass
from typing import Optional

try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import cos_sim
except ImportError:
    print("Install: pip install sentence-transformers")
    exit(1)


# =============================================================================
# Evaluation Test Set
# =============================================================================
# (query, expected_keyword, category)
# Expand this with YOUR real user queries + correct API function matches

EVAL_TEST_SET = [
    # --- Semantic: Natural language -> API function ---
    {"query": "how to create shell mesh on a part", "expected": "CreateMesh", "category": "semantic"},
    {"query": "set element size for meshing", "expected": "element_size", "category": "semantic"},
    {"query": "delete duplicate nodes", "expected": "MergeNodes", "category": "semantic"},
    {"query": "export model to nastran format", "expected": "OutputNastran", "category": "semantic"},
    {"query": "check element quality jacobian", "expected": "QualityCriteria", "category": "semantic"},
    {"query": "apply boundary condition to nodes", "expected": "SPC", "category": "semantic"},
    {"query": "create contact between two parts", "expected": "CONTACT", "category": "semantic"},
    {"query": "read LS-DYNA keyword file", "expected": "InputDyna", "category": "semantic"},

    # --- Code: Technical/API queries ---
    {"query": "ansa.base.Entity get property", "expected": "Entity", "category": "code"},
    {"query": "base.CollectEntities(deck, collection)", "expected": "CollectEntities", "category": "code"},
    {"query": "mesh.SetMeshParams shell_size=5", "expected": "SetMeshParams", "category": "code"},
    {"query": "from ansa import base, mesh, connections", "expected": "import", "category": "code"},

    # --- Keyword: LS-DYNA specific ---
    {"query": "*MAT_024 piecewise linear plasticity", "expected": "MAT_024", "category": "keyword"},
    {"query": "*CONTACT_AUTOMATIC_SURFACE_TO_SURFACE", "expected": "CONTACT_AUTOMATIC", "category": "keyword"},
    {"query": "*CONTROL_TIMESTEP", "expected": "CONTROL_TIMESTEP", "category": "keyword"},
]


# =============================================================================
# Sample Document Corpus (replace with your actual ANSA/META docs)
# =============================================================================

SAMPLE_CORPUS = [
    "Function: ansa.base.CreateMesh(entities, element_size=5.0) - Creates shell mesh on given entities with specified element size parameters.",
    "Function: ansa.base.MergeNodes(tolerance=0.01) - Merges duplicate nodes within the given tolerance distance.",
    "Function: ansa.base.CollectEntities(deck, entity_type, search_set=None) - Collects all entities of a given type from the current database.",
    "Function: ansa.mesh.SetMeshParams(shell_size=5, quality_criteria='jacobian>0.3') - Sets mesh parameters for the current meshing session.",
    "Function: ansa.base.OutputNastran(filepath, entities) - Exports selected entities to Nastran bulk data format.",
    "Function: ansa.base.InputDyna(filepath) - Reads an LS-DYNA keyword file and imports all cards into the database.",
    "Class: ansa.base.Entity - Base class for all ANSA database entities. Methods: get_property, set_property, delete.",
    "Function: ansa.base.SetQualityCriteria(jacobian=0.3, aspect_ratio=5.0, warpage=15.0) - Sets quality criteria thresholds for element checking.",
    "Keyword: *MAT_024 (MAT_PIECEWISE_LINEAR_PLASTICITY) - Elasto-plastic material with piecewise linear stress-strain curve. Parameters: RO, E, PR, SIGY, ETAN.",
    "Keyword: *CONTACT_AUTOMATIC_SURFACE_TO_SURFACE - Automatic contact algorithm for surface-to-surface contact detection. Parameters: SSID, MSID, SSTYP, MSTYP.",
    "Keyword: *CONTROL_TIMESTEP - Controls the time step size for explicit analysis. Parameters: DTINIT, TSSFAC, ISDO, TSLIMT.",
    "Function: ansa.connections.CreateContact(master, slave, type='surface_to_surface') - Creates a CONTACT card between two part sets.",
    "Function: ansa.base.ApplySPC(nodes, dof='123456') - Applies Single Point Constraint (boundary condition) to selected nodes.",
    "Module: ansa - Main ANSA Python module. Submodules: base, mesh, connections, materials, loads. Import with: from ansa import base, mesh, connections",
    "Function: ansa.mesh.CheckQuality(entities, criteria='jacobian') - Checks element quality against defined QualityCriteria thresholds.",
]


# =============================================================================
# Model Candidates
# =============================================================================

MODEL_CANDIDATES = {
    "bge-small": "BAAI/bge-small-en-v1.5",
    "bge-base": "BAAI/bge-base-en-v1.5",
    "bge-large": "BAAI/bge-large-en-v1.5",
    "jina-code": "jinaai/jina-embeddings-v2-base-code",
    "nomic": "nomic-ai/nomic-embed-text-v1.5",
    "all-minilm": "sentence-transformers/all-MiniLM-L6-v2",
}


# =============================================================================
# Evaluation Engine
# =============================================================================

@dataclass
class EvalResult:
    """Evaluation result for a single model."""
    model_name: str
    model_id: str
    hit_rate_at_1: float
    hit_rate_at_3: float
    hit_rate_at_5: float
    mrr: float
    avg_latency_ms: float
    dimension: int
    hit_rate_semantic: float = 0.0
    hit_rate_code: float = 0.0
    hit_rate_keyword: float = 0.0


def evaluate_model(
    model_name: str,
    model_id: str,
    corpus: list[str],
    test_set: list[dict],
    top_k: int = 5,
) -> Optional["EvalResult"]:
    """Evaluate a single embedding model on the test set."""

    print(f"\n  Loading {model_name} ({model_id})...")

    try:
        model = SentenceTransformer(model_id, trust_remote_code=True)
    except Exception as e:
        print(f"    ERROR: {e}")
        return None

    # Embed corpus once
    start = time.time()
    corpus_embeddings = model.encode(corpus, normalize_embeddings=True)
    embed_time = (time.time() - start) * 1000
    print(f"    Corpus embedded: {len(corpus)} docs in {embed_time:.0f}ms")

    dimension = corpus_embeddings.shape[1]

    # Evaluate each query
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    reciprocal_ranks = []
    latencies = []
    category_hits = {"semantic": [], "code": [], "keyword": []}

    for test in test_set:
        query = test["query"]
        expected = test["expected"].lower()
        category = test["category"]

        # Embed query and measure latency
        t0 = time.time()
        query_embedding = model.encode([query], normalize_embeddings=True)
        latencies.append((time.time() - t0) * 1000)

        # Cosine similarity
        similarities = cos_sim(query_embedding, corpus_embeddings)[0].numpy()
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Find rank of expected result
        found_rank = None
        for rank, idx in enumerate(top_indices, 1):
            if expected in corpus[idx].lower():
                found_rank = rank
                break

        if found_rank:
            if found_rank <= 1: hits_at_1 += 1
            if found_rank <= 3: hits_at_3 += 1
            if found_rank <= 5: hits_at_5 += 1
            reciprocal_ranks.append(1.0 / found_rank)
            category_hits[category].append(1)
        else:
            reciprocal_ranks.append(0.0)
            category_hits[category].append(0)

    n = len(test_set)

    return EvalResult(
        model_name=model_name,
        model_id=model_id,
        hit_rate_at_1=hits_at_1 / n,
        hit_rate_at_3=hits_at_3 / n,
        hit_rate_at_5=hits_at_5 / n,
        mrr=float(np.mean(reciprocal_ranks)),
        avg_latency_ms=float(np.mean(latencies)),
        dimension=dimension,
        hit_rate_semantic=float(np.mean(category_hits["semantic"])) if category_hits["semantic"] else 0,
        hit_rate_code=float(np.mean(category_hits["code"])) if category_hits["code"] else 0,
        hit_rate_keyword=float(np.mean(category_hits["keyword"])) if category_hits["keyword"] else 0,
    )


def run_evaluation(model_names: Optional[list[str]] = None):
    """Run full evaluation and print comparison table."""

    models_to_test = model_names or ["bge-small", "bge-base", "all-minilm"]

    print("=" * 70)
    print("  Embedding Model Evaluation for ANSA/META API Retrieval")
    print("=" * 70)
    print(f"  Corpus   : {len(SAMPLE_CORPUS)} documents")
    print(f"  Queries  : {len(EVAL_TEST_SET)} test queries")
    print(f"  Models   : {models_to_test}")

    results = []
    for name in models_to_test:
        if name not in MODEL_CANDIDATES:
            print(f"  Unknown model: {name}. Options: {list(MODEL_CANDIDATES.keys())}")
            continue
        result = evaluate_model(name, MODEL_CANDIDATES[name], SAMPLE_CORPUS, EVAL_TEST_SET)
        if result:
            results.append(result)

    if not results:
        print("\n  No models evaluated successfully.")
        return []

    # === Results Table ===
    print("\n" + "=" * 70)
    print("  RESULTS (sorted by MRR)")
    print("=" * 70)
    print(f"  {'Model':<12} {'Hit@1':>7} {'Hit@3':>7} {'Hit@5':>7} {'MRR':>7} {'Latency':>9} {'Dim':>5}")
    print(f"  {'-'*12} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*9} {'-'*5}")

    for r in sorted(results, key=lambda x: x.mrr, reverse=True):
        print(
            f"  {r.model_name:<12} "
            f"{r.hit_rate_at_1:>6.1%} "
            f"{r.hit_rate_at_3:>6.1%} "
            f"{r.hit_rate_at_5:>6.1%} "
            f"{r.mrr:>6.3f} "
            f"{r.avg_latency_ms:>7.1f}ms "
            f"{r.dimension:>5}"
        )

    # === Per-Category Breakdown ===
    print(f"\n  {'Per-Category Hit@5':^70}")
    print(f"  {'Model':<12} {'Semantic':>10} {'Code':>10} {'Keyword':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10}")

    for r in sorted(results, key=lambda x: x.mrr, reverse=True):
        print(
            f"  {r.model_name:<12} "
            f"{r.hit_rate_semantic:>9.1%} "
            f"{r.hit_rate_code:>9.1%} "
            f"{r.hit_rate_keyword:>9.1%}"
        )

    # === Recommendation ===
    best = max(results, key=lambda x: x.mrr)
    fastest = min(results, key=lambda x: x.avg_latency_ms)

    print(f"\n  {'Recommendation':^70}")
    print(f"  {'-'*70}")
    print(f"  Best accuracy   : {best.model_name} (MRR = {best.mrr:.3f})")
    print(f"  Fastest          : {fastest.model_name} ({fastest.avg_latency_ms:.1f}ms/query)")

    # Balanced score: 70% accuracy, 30% speed
    max_lat = max(r.avg_latency_ms for r in results)
    best_balanced = max(
        results,
        key=lambda r: r.mrr * 0.7 + (1 - r.avg_latency_ms / max_lat) * 0.3
    )
    print(f"  Best trade-off   : {best_balanced.model_name} (balanced accuracy + speed)")

    print(f"\n{'='*70}")
    print("  NEXT STEPS:")
    print("  1. Replace SAMPLE_CORPUS with your actual ANSA/META API docs")
    print("  2. Add 30-50 real user queries to EVAL_TEST_SET")
    print("  3. Re-run to get production-accurate numbers")
    print(f"{'='*70}\n")

    return results


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate embedding models for API retrieval")
    parser.add_argument(
        "--models", nargs="+", default=None,
        help=f"Models to test. Available: {list(MODEL_CANDIDATES.keys())}"
    )
    args = parser.parse_args()

    run_evaluation(args.models)
