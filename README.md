# Enterprise AI Copilot

> **Project status: in progress.** This repository is an actively developed
> RAG and enterprise AI experimentation project. Interfaces, evaluation data,
> and implementation details may change while the benchmark is expanded.

Enterprise AI Copilot is a Python/FastAPI project for building and evaluating
retrieval-augmented generation workflows over enterprise-style documents. It
combines Markdown-aware document processing, semantic retrieval with Qdrant,
local language-model integration, evidence validation, and reproducible
retrieval experiments.

The repository includes a synthetic AcmeOps dataset for early development and
a public Microsoft VS Code documentation benchmark for more realistic retrieval
evaluation. The VS Code benchmark currently uses 30 documents from commit
`6cf6db1eef7a0e2cbc887d2d69a9b4237f8a3d1c`.

## What is implemented

- FastAPI application with health and RAG endpoints.
- Local configuration for Qdrant and Ollama.
- Sentence Transformer embeddings with
  `sentence-transformers/all-MiniLM-L6-v2`.
- Qdrant storage and semantic retrieval.
- Markdown processing with YAML front-matter support.
- Documentation template-variable normalization.
- Fixed-size, section-aware, and hybrid chunking.
- Token-aware windows capped at 220 tokens for fixed and hybrid chunks.
- Small heading-only section merging without losing source text or metadata.
- Deterministic chunk UUIDs and source line/character tracking.
- Retrieval candidate preparation and manual relevance-annotation workflows.
- Ground-truth verification against original source documentation.
- Offline Hit@K, MRR@K, DCG, and nDCG evaluation.
- Cross-encoder reranking with
  `cross-encoder/ms-marco-MiniLM-L6-v2`.
- Automated tests for chunk integrity, evaluation data, metrics, and reranking.

## Current VS Code benchmark

Two read-only benchmark collections are used:

| Strategy | Collection | Points |
|---|---|---:|
| Fixed-size | `vscode_fixed_minilm_v1` | 428 |
| Hybrid | `vscode_hybrid_minilm_v1` | 599 |

The verified retrieval benchmark contains eight questions and 80 manually
reviewed top-five judgments: seven questions are fully answerable from the
indexed sample and one is partially answerable.

### Baseline retrieval results

These metrics describe the judged top-five candidate pool. They are not
exhaustive collection-wide recall measurements.

| Metric | Fixed | Hybrid |
|---|---:|---:|
| Hit@1 | 0.3750 | 0.1250 |
| Hit@3 | 0.7500 | 0.7500 |
| Hit@5 | 0.8750 | 0.7500 |
| MRR@5 | 0.5667 | 0.4167 |
| nDCG@5 | 0.8344 | 0.8548 |

Fixed chunking currently finds answer-bearing passages earlier and more often,
while hybrid chunking has slightly better graded nDCG ordering. With only eight
questions, these results are useful directional evidence rather than a strong
general conclusion.

## Cross-encoder reranking status

The reranking pipeline retrieves and reranks the top 20 candidates from each
strategy for every benchmark question. The latest experiment produced:

- 160 fixed candidates and 160 hybrid candidates.
- 80 relevance labels reused from the verified baseline.
- 240 new candidates awaiting manual relevance review.
- Exact top-five point-ID agreement with the original baseline.

Final reranking quality metrics are intentionally blocked until all 240 new
candidates are reviewed. Cross-encoder scores, cosine similarity, filenames,
and neighboring chunks are not used as automatic relevance labels.

## Repository layout

```text
backend/
  app/                    FastAPI, RAG, chunking, retrieval, and evaluation code
  scripts/                Indexing, search, benchmark, and evaluation commands
  tests/                  Automated unit and regression tests
data/
  evaluation/             Questions, annotations, metrics, and reports
  external/vscode-docs/   Local public VS Code documentation checkout
  documents/              Synthetic AcmeOps documents
infrastructure/
  compose.yaml            Local infrastructure services
```

## Local setup

### 1. Create the Python environment

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 2. Configure services

Copy `backend/.env.example` to `backend/.env` and adjust values if necessary:

```env
QDRANT_URL=http://localhost:6333
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
```

Start the infrastructure required by your workflow. The provided Compose file
is located at `infrastructure/compose.yaml`.

```powershell
docker compose -f infrastructure/compose.yaml up -d
```

### 3. Run the API

From `backend/`:

```powershell
uvicorn app.main:app --reload
```

### 4. Run tests

```powershell
python -m pytest tests -v
```

Some integration scripts require running Qdrant, Ollama, local model caches, or
the public VS Code documentation checkout. Unit tests mock external model and
Qdrant access where appropriate.

## Evaluation workflow

The important reproducible artifacts are under `data/evaluation/`:

- `vscode_retrieval_questions.json`: benchmark questions.
- `vscode_retrieval_annotations_verified.json`: verified top-five judgments and
  reference evidence.
- `vscode_retrieval_metrics.json`: baseline quantitative results.
- `vscode_retrieval_report.md`: human-readable baseline analysis.
- `vscode_reranking_candidates.json`: top-20 retrieval and reranking output.
- `vscode_reranking_annotations.json`: reranking annotation workspace.
- `vscode_reranking_annotation_readiness.json`: current labeling readiness.

After completing all null relevance labels in the reranking annotation file,
run the following from `backend/`:

```powershell
python -m scripts.evaluate_vscode_reranking
```

The evaluator refuses to produce final quality metrics while any required
candidate label is null.

## What remains to do

Immediate work:

1. Manually review the 240 new reranking candidates against the verified source
   documentation.
2. Run the four-way comparison: fixed baseline, fixed reranked, hybrid baseline,
   and hybrid reranked.
3. Analyze whether failures come from first-stage candidate retrieval or
   reranker ordering.

Planned follow-up work:

- Expand beyond eight questions and balance technical topics and answer types.
- Judge deeper and independently pooled candidates before claiming Recall@K.
- Add confidence intervals and query variants.
- Evaluate answer generation separately from retrieval quality.
- Improve citations, evidence validation, and unsupported-answer handling.
- Harden service configuration, observability, and production deployment.
- Add a user-facing application experience and enterprise access controls.

## Important limitations

- The benchmark is small and should not be treated as production evidence.
- Current relevance judgments are pooled from the evaluated retrieval systems.
- Hit@K is reported as success within the judged ranking, not exhaustive recall.
- The AcmeOps starter data is synthetic and contains no real customer data.
- The project is still under active development and is not production-ready.

## License and source data

Review the license of each external dataset or model before redistribution or
production use. External VS Code documentation is maintained as a separate
local checkout and should not be committed unintentionally.
