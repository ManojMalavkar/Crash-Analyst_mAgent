#!/usr/bin/env python3
"""RAG Evaluation — Retrieval + Answer Quality (LLM-as-Judge).

Follows the pattern from llm_engineering/week5/evaluation/eval.py:
- TestQuestion with keywords + reference_answer
- Retrieval eval: MRR, nDCG, keyword coverage
- Answer eval: LLM-as-judge (accuracy, completeness, relevance)

Two test files, two databases:
- tests_api.jsonl     → collection: "api" (ANSA/META Python API)
- tests_session.jsonl → collection: "meta_session_commands" (GUI commands)

Usage:
    # Evaluate API retrieval
    python evaluation/rag_eval.py --eval-file evaluation/tests_api.jsonl --collection api

    # Evaluate Session commands retrieval
    python evaluation/rag_eval.py --eval-file evaluation/tests_session.jsonl --collection meta_session_commands

    # Run single test (like reference eval.py)
    python evaluation/rag_eval.py --eval-file evaluation/tests_api.jsonl --test 0

    # Full evaluation (retrieval + LLM answer quality)
    python evaluation/rag_eval.py --eval-file evaluation/tests_api.jsonl --collection api --full
"""

import json
import math
import time
import sys
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel, Field

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = PROJECT_ROOT / "01_ANSA_ApiAgent"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(AGENT_DIR))

import chromadb
from sentence_transformers import SentenceTransformer

try:
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    HAS_LLM = True
except ImportError:
    HAS_LLM = False

from shared.config import settings


# =============================================================================
# Data Models (same pattern as reference eval.py)
# =============================================================================


class TestQuestion(BaseModel):
    """A single evaluation test question."""
    question: str
    keywords: list[str]
    reference_answer: str = ""
    category: str = "general"


class RetrievalEval(BaseModel):
    """Retrieval performance metrics."""
    mrr: float = Field(description="Mean Reciprocal Rank - average across all keywords")
    ndcg: float = Field(description="Normalized Discounted Cumulative Gain (binary relevance)")
    keywords_found: int = Field(description="Number of keywords found in top-k results")
    total_keywords: int = Field(description="Total number of keywords to find")
    keyword_coverage: float = Field(description="Percentage of keywords found")


class AnswerEval(BaseModel):
    """LLM-as-a-judge evaluation of answer quality."""
    feedback: str = Field(
        description="Concise feedback on answer quality, comparing to reference answer"
    )
    accuracy: float = Field(
        description="Factual correctness vs reference. 1 (wrong) to 5 (perfect). Wrong answers must score 1."
    )
    completeness: float = Field(
        description="How complete is the answer? 1 (missing key info) to 5 (all info from reference covered). Only 5 if ALL info included."
    )
    relevance: float = Field(
        description="How relevant to the question? 1 (off-topic) to 5 (directly answers, no extra). Only 5 if completely on-point."
    )



# =============================================================================
# Test Loader
# =============================================================================


def load_tests(eval_file: str | Path) -> list[TestQuestion]:
    """Load test questions from a JSONL file."""
    tests = []
    with open(eval_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                tests.append(TestQuestion(**data))
    return tests


# =============================================================================
# Retrieval Metrics (from reference eval.py)
# =============================================================================


def calculate_mrr(keyword: str, retrieved_docs: list[dict]) -> float:
    """Calculate reciprocal rank for a single keyword (case-insensitive)."""
    keyword_lower = keyword.lower()
    for rank, doc in enumerate(retrieved_docs, start=1):
        text = (doc.get("document", "") + " " + doc.get("symbol", "")).lower()
        if keyword_lower in text:
            return 1.0 / rank
    return 0.0


def calculate_dcg(relevances: list[int], k: int) -> float:
    """Calculate Discounted Cumulative Gain."""
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        dcg += relevances[i] / math.log2(i + 2)  # i+2 because rank starts at 1
    return dcg


def calculate_ndcg(keyword: str, retrieved_docs: list[dict], k: int = 10) -> float:
    """Calculate nDCG for a single keyword (binary relevance)."""
    keyword_lower = keyword.lower()
    relevances = [
        1 if keyword_lower in (doc.get("document", "") + " " + doc.get("symbol", "")).lower()
        else 0
        for doc in retrieved_docs[:k]
    ]
    dcg = calculate_dcg(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = calculate_dcg(ideal_relevances, k)
    return dcg / idcg if idcg > 0 else 0.0


# =============================================================================
# Retrieval Evaluation
# =============================================================================


def fetch_context(question: str, collection, model: SentenceTransformer, k: int = 10) -> list[dict]:
    """Retrieve documents from ChromaDB for a question."""
    query_embedding = model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(k, collection.count()),
        include=["metadatas", "documents", "distances"],
    )
    docs = []
    for i in range(len(results["documents"][0])):
        docs.append({
            "document": results["documents"][0][i] or "",
            "symbol": (results["metadatas"][0][i] or {}).get("symbol", ""),
            "module": (results["metadatas"][0][i] or {}).get("module", ""),
            "distance": results["distances"][0][i] if results.get("distances") else 0,
        })
    return docs


def evaluate_retrieval(test: TestQuestion, collection, model: SentenceTransformer, k: int = 10) -> RetrievalEval:
    """Evaluate retrieval performance for a single test question."""
    retrieved_docs = fetch_context(test.question, collection, model, k)

    # MRR (average across all keywords)
    mrr_scores = [calculate_mrr(kw, retrieved_docs) for kw in test.keywords]
    avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0

    # nDCG (average across all keywords)
    ndcg_scores = [calculate_ndcg(kw, retrieved_docs, k) for kw in test.keywords]
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

    # Keyword coverage
    keywords_found = sum(1 for score in mrr_scores if score > 0)
    total_keywords = len(test.keywords)
    keyword_coverage = (keywords_found / total_keywords * 100) if total_keywords > 0 else 0.0

    return RetrievalEval(
        mrr=avg_mrr,
        ndcg=avg_ndcg,
        keywords_found=keywords_found,
        total_keywords=total_keywords,
        keyword_coverage=keyword_coverage,
    )


# =============================================================================
# Answer Evaluation (LLM-as-Judge)
# =============================================================================

JUDGE_MODEL = os.environ.get("LLM_PRIMARY_MODEL", "gpt-4.1-nano")


def evaluate_answer(test: TestQuestion, collection, model: SentenceTransformer) -> tuple[AnswerEval, str, list]:
    """Evaluate answer quality using LLM-as-a-judge."""
    if not HAS_LLM:
        raise RuntimeError("openai + python-dotenv required for answer evaluation. pip install openai python-dotenv")

    # Get RAG context
    retrieved_docs = fetch_context(test.question, collection, model, k=5)
    context = "\n\n".join([
        f"[{d['symbol']}] {d['document'][:300]}" for d in retrieved_docs
    ])

    # Generate answer using the agent's pattern
    client = OpenAI(
        api_key=os.environ.get("LLM_API_KEY"),
        base_url=os.environ.get("LLM_API_BASE_URL"),
    )

    # Generate code answer
    gen_response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": "You are an ANSA/META Python API expert. Generate code to answer the user's question using the provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {test.question}\n\nProvide Python code using ANSA/META API."},
        ],
        max_tokens=500,
    )
    generated_answer = gen_response.choices[0].message.content

    # LLM Judge
    judge_messages = [
        {
            "role": "system",
            "content": "You are an expert evaluator assessing code answer quality. Evaluate the generated answer by comparing to the reference. Only give 5/5 for perfect answers. Respond in JSON with keys: feedback, accuracy, completeness, relevance.",
        },
        {
            "role": "user",
            "content": f"""Question:
{test.question}

Generated Answer:
{generated_answer}

Reference Answer:
{test.reference_answer}

Evaluate on:
1. Accuracy: Factually correct vs reference? 1 (wrong) to 5 (perfect). Wrong = 1.
2. Completeness: All aspects addressed? 1 (missing key info) to 5 (all covered).
3. Relevance: Directly answers the question? 1 (off-topic) to 5 (on-point, no extra).

Respond as JSON: {{"feedback": "...", "accuracy": N, "completeness": N, "relevance": N}}""",
        },
    ]

    judge_response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=judge_messages,
        max_tokens=300,
    )
    judge_text = judge_response.choices[0].message.content

    # Parse judge response
    try:
        # Strip markdown code block if present
        if "```json" in judge_text:
            judge_text = judge_text.split("```json")[1].split("```")[0]
        elif "```" in judge_text:
            judge_text = judge_text.split("```")[1].split("```")[0]
        judge_data = json.loads(judge_text.strip())
        answer_eval = AnswerEval(**judge_data)
    except (json.JSONDecodeError, Exception) as e:
        answer_eval = AnswerEval(
            feedback=f"Parse error: {e}. Raw: {judge_text[:200]}",
            accuracy=0, completeness=0, relevance=0,
        )

    return answer_eval, generated_answer, retrieved_docs


# =============================================================================
# Evaluation Runners
# =============================================================================


def evaluate_all_retrieval(tests: list[TestQuestion], collection, model: SentenceTransformer):
    """Evaluate retrieval for all tests (generator with progress)."""
    total = len(tests)
    for i, test in enumerate(tests):
        result = evaluate_retrieval(test, collection, model)
        progress = (i + 1) / total
        yield test, result, progress


def evaluate_all_answers(tests: list[TestQuestion], collection, model: SentenceTransformer):
    """Evaluate answers for all tests (generator with progress)."""
    total = len(tests)
    for i, test in enumerate(tests):
        result, answer, docs = evaluate_answer(test, collection, model)
        progress = (i + 1) / total
        yield test, result, answer, progress


# =============================================================================
# CLI: Run Evaluation for a Single Test
# =============================================================================


def run_single_test(test_number: int, tests: list[TestQuestion], collection, model: SentenceTransformer, full: bool = False):
    """Run evaluation for a single test (like reference eval.py CLI)."""
    if test_number < 0 or test_number >= len(tests):
        print(f"  Error: test number must be 0..{len(tests) - 1}")
        return

    test = tests[test_number]

    print(f"\n{'='*80}")
    print(f"  Test #{test_number}")
    print(f"{'='*80}")
    print(f"  Question : {test.question}")
    print(f"  Keywords : {test.keywords}")
    print(f"  Category : {test.category}")
    print(f"  Reference: {test.reference_answer[:100]}")

    # Retrieval evaluation
    print(f"\n{'='*80}")
    print("  Retrieval Evaluation")
    print(f"{'='*80}")

    result = evaluate_retrieval(test, collection, model)
    print(f"  MRR             : {result.mrr:.4f}")
    print(f"  nDCG            : {result.ndcg:.4f}")
    print(f"  Keywords Found  : {result.keywords_found}/{result.total_keywords}")
    print(f"  Coverage        : {result.keyword_coverage:.1f}%")

    # Show retrieved docs
    docs = fetch_context(test.question, collection, model, k=5)
    print(f"\n  Top-5 Retrieved:")
    for i, d in enumerate(docs[:5]):
        print(f"    [{i+1}] {d['symbol']:<40} (dist={d['distance']:.3f})")

    # Answer evaluation (optional)
    if full and HAS_LLM:
        print(f"\n{'='*80}")
        print("  Answer Evaluation (LLM-as-Judge)")
        print(f"{'='*80}")

        answer_eval, generated_answer, _ = evaluate_answer(test, collection, model)

        print(f"\n  Generated Answer:\n{generated_answer[:300]}")
        print(f"\n  Feedback: {answer_eval.feedback}")
        print(f"\n  Scores:")
        print(f"    Accuracy     : {answer_eval.accuracy:.1f}/5")
        print(f"    Completeness : {answer_eval.completeness:.1f}/5")
        print(f"    Relevance    : {answer_eval.relevance:.1f}/5")

    print(f"\n{'='*80}\n")


def run_all_tests(tests: list[TestQuestion], collection, model: SentenceTransformer, full: bool = False):
    """Run evaluation for all tests and show summary."""

    print(f"\n{'='*80}")
    print(f"  Running evaluation: {len(tests)} tests")
    print(f"{'='*80}")

    retrieval_results = []
    answer_results = []

    for test, result, progress in evaluate_all_retrieval(tests, collection, model):
        retrieval_results.append((test, result))
        pct = int(progress * 100)
        print(f"\r  Retrieval: [{pct:>3}%] {test.question[:50]:<50}", end="", flush=True)

    print("\n")

    # Summary Table
    print(f"  {'#':<4} {'Category':<12} {'MRR':>6} {'nDCG':>6} {'Cov%':>6}  Question")
    print(f"  {'-'*4} {'-'*12} {'-'*6} {'-'*6} {'-'*6}  {'-'*30}")

    for i, (test, result) in enumerate(retrieval_results):
        print(
            f"  {i:<4} {test.category:<12} "
            f"{result.mrr:>5.3f} {result.ndcg:>5.3f} "
            f"{result.keyword_coverage:>5.1f}  "
            f"{test.question[:40]}"
        )

    # Aggregate stats
    avg_mrr = sum(r.mrr for _, r in retrieval_results) / len(retrieval_results)
    avg_ndcg = sum(r.ndcg for _, r in retrieval_results) / len(retrieval_results)
    avg_coverage = sum(r.keyword_coverage for _, r in retrieval_results) / len(retrieval_results)

    print(f"\n  {'AVERAGE':<17} {avg_mrr:>5.3f} {avg_ndcg:>5.3f} {avg_coverage:>5.1f}")

    # Per-category breakdown
    categories = {}
    for test, result in retrieval_results:
        if test.category not in categories:
            categories[test.category] = []
        categories[test.category].append(result)

    print(f"\n  Per-Category Summary:")
    print(f"  {'Category':<15} {'Count':>5} {'Avg MRR':>8} {'Avg nDCG':>9} {'Avg Cov%':>9}")
    print(f"  {'-'*15} {'-'*5} {'-'*8} {'-'*9} {'-'*9}")
    for cat, results in sorted(categories.items()):
        n = len(results)
        m = sum(r.mrr for r in results) / n
        nd = sum(r.ndcg for r in results) / n
        cov = sum(r.keyword_coverage for r in results) / n
        print(f"  {cat:<15} {n:>5} {m:>7.3f} {nd:>8.3f} {cov:>8.1f}")

    # LLM Answer evaluation (optional)
    if full and HAS_LLM:
        print(f"\n{'='*80}")
        print("  Answer Evaluation (LLM-as-Judge)")
        print(f"{'='*80}")

        for test, answer_eval, answer, progress in evaluate_all_answers(tests, collection, model):
            answer_results.append((test, answer_eval))
            pct = int(progress * 100)
            print(f"\r  Judging: [{pct:>3}%] {test.question[:50]:<50}", end="", flush=True)

        print("\n")

        avg_acc = sum(r.accuracy for _, r in answer_results) / len(answer_results)
        avg_comp = sum(r.completeness for _, r in answer_results) / len(answer_results)
        avg_rel = sum(r.relevance for _, r in answer_results) / len(answer_results)

        print(f"  Average Scores:")
        print(f"    Accuracy     : {avg_acc:.2f}/5")
        print(f"    Completeness : {avg_comp:.2f}/5")
        print(f"    Relevance    : {avg_rel:.2f}/5")

    print(f"\n{'='*80}\n")


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="RAG Evaluation — Retrieval + Answer Quality")
    parser.add_argument(
        "--eval-file", required=True,
        help="Path to test JSONL file (tests_api.jsonl or tests_session.jsonl)"
    )
    parser.add_argument(
        "--collection", default=None,
        help="ChromaDB collection name (auto-detected if not specified)"
    )
    parser.add_argument(
        "--test", type=int, default=None,
        help="Run a specific test by number (0-indexed). Omit to run all."
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Include LLM-as-Judge answer evaluation (requires API key)"
    )
    parser.add_argument(
        "--compare-models", action="store_true",
        help="Compare embedding models (slow — rebuilds for each)"
    )
    args = parser.parse_args()

    # Load tests
    tests = load_tests(args.eval_file)
    print(f"\n  Loaded {len(tests)} tests from {args.eval_file}")

    # Connect to vector DB
    vector_db_path = AGENT_DIR / "vector_db"
    if not vector_db_path.exists():
        print(f"  ERROR: {vector_db_path} not found. Build the vector DB first.")
        sys.exit(1)

    client = chromadb.PersistentClient(path=str(vector_db_path))
    collections = client.list_collections()
    print(f"  Available collections: {[c.name for c in collections]}")

    # Select collection
    if args.collection:
        try:
            collection = client.get_collection(args.collection)
        except Exception:
            print(f"  ERROR: Collection '{args.collection}' not found.")
            print(f"  Available: {[c.name for c in collections]}")
            sys.exit(1)
    elif collections:
        collection = collections[0]
    else:
        print("  ERROR: No collections found.")
        sys.exit(1)

    print(f"  Using collection: {collection.name} ({collection.count()} docs)")

    # Load embedding model
    model_name = settings.embedding.model_name
    print(f"  Embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    # Run evaluation
    if args.test is not None:
        run_single_test(args.test, tests, collection, model, full=args.full)
    else:
        run_all_tests(tests, collection, model, full=args.full)


if __name__ == "__main__":
    main()
