# Evaluation Improvement Iteration Metrics

This document is the standard scorecard for tracking retrieval improvements across model, index, and embedding changes.

## 1. Core metrics

Track these for every evaluation run:

- MRR (Mean Reciprocal Rank)
- nDCG (Normalized Discounted Cumulative Gain)
- Keyword coverage (%)
- Missed queries
- Accuracy (LLM-as-judge, optional)
- Completeness (LLM-as-judge, optional)
- Relevance (LLM-as-judge, optional)

Target values:

- MRR: > 0.70
- nDCG: > 0.70
- Keyword coverage: > 80%
- Missed queries: < 5
- Accuracy: > 3.5
- Completeness: > 3.5
- Relevance: > 4.0

---

## 2. Baseline snapshot

Current baseline recorded for API retrieval:

| Metric | Value |
|---|---:|
| MRR | 0.4549 |
| nDCG | 0.4888 |
| Keyword coverage | 61.4% |
| Missed queries | 4 |

---

## 3. Iteration scorecard

Use one row per experiment.

| Iteration | Change made | Model | top_k | Collection | MRR | nDCG | Coverage % | Missed | Accuracy | Completeness | Relevance | Decision |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | Baseline | BAAI/bge-small-en-v1.5 | 5 | api | 0.4549 | 0.4888 | 61.4 | 4 | - | - | - | baseline |
| 1 | Enriched embedding text: module + symbol + signature + description | BAAI/bge-small-en-v1.5 | 5 | api |  |  |  |  |  |  |  |  |
| 2 | Added docstring/context fields | BAAI/bge-small-en-v1.5 | 5 | api |  |  |  |  |  |  |  |  |
| 3 | Switched to stronger embedding model | BAAI/bge-base-en-v1.5 | 5 | api |  |  |  |  |  |  |  |  |
| 4 | Increased retrieval window | same as iteration 3 | 10 | api |  |  |  |  |  |  |  |  |
| 5 | Added software metadata filter | same as iteration 3 | 5 | api |  |  |  |  |  |  |  |  |

---

## 4. Improvement decision rules

Use the following logic for every iteration:

- If MRR and nDCG improve and missed queries drop, keep the change.
- If MRR improves but coverage drops, check whether retrieval is more precise but less complete.
- If coverage improves but MRR stays flat, the system may be broader but not more accurate.
- If all metrics are flat or worse, revert and try the next change.
- Only move from retrieval tuning to answer quality evaluation after MRR > 0.70 and coverage > 80%.

---

## 5. Suggested improvement order

1. Enrich embedding text
2. Add exact metadata fields and module/symbol context
3. Rebuild vector DB
4. Re-run evaluation
5. Compare against baseline
6. Move to model upgrade only if retrieval remains weak
7. Tune top_k and metadata filters after the model is stable

---

## 6. Recommended evaluation command

```bash
.\.venv\Scripts\python.exe .\evaluation\rag_eval.py --eval-file .\evaluation\tests_api.jsonl --collection api --notes "iteration-1: enriched embed text"
```

---

## 7. Example summary note for each iteration

> Iteration 1: Enriched the vector embedding text to include module, symbol, signature, and description. Goal: improve exact API symbol retrieval without broadening noise. Result: MRR improved from 0.4549 to X, nDCG from 0.4888 to Y, coverage from 61.4% to Z, missed queries from 4 to N.

---

## 8. Recommended tracking format for final reporting

For each iteration, record:

- what changed
- why it was changed
- before/after metrics
- whether it was accepted or reverted
- next planned experiment

This keeps the improvement process explicit and prevents untracked tuning.
