# RAG Strategy Lab – Project Notebook

Central place for notes, design decisions, checklists, and tasks for building the interactive RAG strategy playground (Streamlit/Chainlit frontend + evaluation & fine-tuning workflows).

> Keep this file updated as you work. Treat it as a living design + task tracker.

---

## 1. Vision & Goals

- Build a **RAG strategy lab** where users can:
  - Select existing sample documents or upload their own.
  - Trigger ingestion and preprocessing in the background.
  - Interactively **try multiple RAG strategies side by side** (multi-query, rerank, hybrid, self-reflective, etc.).
  - Combine strategies (e.g., multi-query + rerank) and compare outputs.
  - Generate a **Ragas-style silver dataset** (e.g., 100 QA pairs) for chosen strategies.
  - Run **Ragas evaluation** and view metrics dashboards.
  - Optionally trigger **fine-tuning** (embeddings + generation) and see new strategy variants.
  - (Later) Explore **Graph RAG** and visualize the resulting graph.

---

## 2. High-level Phases

- **Phase 1 – Strategy playground UI**
  - Streamlit/Chainlit frontend.
  - Strategy registry + side-by-side comparison.
  - Use existing ingested sample data.

- **Phase 2 – Document selection & ingestion orchestration**
  - Doc picker + file upload.
  - Ingestion jobs, status, and parallel workflows.

- **Phase 3 – Ragas-style eval & silver data**
  - Silver dataset generation.
  - Evaluation runs + metrics dashboard.

- **Phase 4 – Fine-tuned RAG strategies**
  - Embedder and LLM fine-tune flows.
  - Strategy variants and ETA/cost estimates.

- **Phase 5 – Graph RAG & visualization**
  - Graph construction pipeline.
  - Graph-based strategies and graph views.

---

## 2.5 Users & Environment Assumptions

- **Primary users**
  - ML / data engineers exploring RAG patterns.
  - Applied researchers and architect-level engineers comparing strategies.
  - Product / domain experts pairing with an engineer to understand tradeoffs.
- **Usage pattern**
  - Single-tenant demo / lab environment (not multi-tenant SaaS in v1).
  - Assumes users are comfortable with basic RAG concepts and reading metrics.
- **Runtime environment**
  - Local dev: Python 3.9+ with `uv` and PostgreSQL/pgvector (from `implementation/README.md`).
  - Optional: Docker/Compose for DB + app later.
  - One shared DB instance per environment; multiple frontend sessions reuse the same vector store.

---

## 3. Checklists (per Phase)

### Phase 1 – Strategy Playground UI

- [ ] Decide primary frontend framework (Streamlit vs Chainlit) for v1.
- [ ] Define a **StrategyResult** schema (inputs/outputs/metadata).
- [ ] Implement a `strategies` registry module that wraps:
  - [ ] Baseline semantic search (`rag_agent.py`).
  - [ ] Multi-query, rerank, hybrid, self-reflective (`rag_agent_advanced.py`).
- [ ] Add timing + rough cost estimation per strategy.
- [ ] Create minimal UI:
  - [ ] Strategy multiselect dropdown.
  - [ ] Single question input.
  - [ ] Side-by-side display for answers + retrieved chunks.
- [ ] Verify works against existing `implementation/documents/` with prior ingestion.

### Phase 1 Detail – Strategy Models & Registry

#### StrategyConfig (input / registry entry)

- Fields (initial):
  - `id: str` – strategy identifier (e.g., `"baseline"`, `"multi_query"`, `"rerank"`, `"hybrid"`, `"self_reflective"`).
  - `name: str` – human-readable name for the UI.
  - `description: str` – short description / tooltip text.
  - `max_results: int` – default top-k chunks to retrieve (e.g., 5).
  - `estimated_cost_class: Literal["fast", "medium", "slow"]` – rough latency/cost bucket.
  - `params: dict` – optional strategy-specific knobs (e.g., `candidate_limit`, `num_queries`).

#### StrategyResult (output of a strategy run)

- Core fields:
  - `strategy_id: str`
  - `question: str`
  - `answer: str`
  - `retrieved_chunks: list[RetrievedChunk]`
  - `latency_ms: float`
  - `cost_class: Literal["fast", "medium", "slow"]`
  - `meta: dict` – free-form metadata (e.g., model used, num_llm_calls, notes).

#### RetrievedChunk

- `document_title: str`
- `document_source: str`
- `content: str`
- `similarity: float | None`
- `chunk_id: str | None`

#### Strategy registry

- Python module (e.g., `implementation/app_backend/strategies.py`) providing:
  - `STRATEGIES: dict[str, StrategyConfig]`
  - `async def run_strategy(strategy_id: str, question: str, max_results: int | None = None) -> StrategyResult`
- Internal mapping (v1):
  - `"baseline"` → wrapper around baseline search in `rag_agent.py`.
  - `"multi_query"` → `search_with_multi_query` in `rag_agent_advanced.py`.
  - `"rerank"` → `search_with_reranking` in `rag_agent_advanced.py`.
  - `"hybrid"` → hybrid retrieval tool in `rag_agent_advanced.py`.
  - `"self_reflective"` → self-reflective search tool in `rag_agent_advanced.py`.
- Each wrapper:
  - Measures latency.
  - Calls the underlying async function.
  - Normalizes the raw string / rows into a `StrategyResult` with `retrieved_chunks` and `latency_ms` populated.

### Phase 2 – Document Selection & Ingestion

- [ ] UI: existing docs selector (from `implementation/documents/`).
- [ ] UI: file uploader for user docs (temp dir per session).
- [ ] Backend: ingestion job API/wrapper around `DocumentIngestionPipeline`.
- [ ] Job tracking: status, progress, errors.
- [ ] UI: show ingestion progress + enable playground once done.

### Phase 1 Detail – Minimal Streamlit UI (v1)

- **Layout**
  - Sidebar:
    - Multiselect: available strategies (from `STRATEGIES`).
    - Number input: `max_results` (top-k) with a reasonable default (e.g., 5).
  - Main area:
    - Text input: single question.
    - Button: "Run comparison".
    - Results section:
      - Either one column per strategy or a table with:
        - Strategy name.
        - Answer text.
        - Collapsible list of top-k retrieved chunks (title + snippet + source).
        - Latency and cost class label (`fast` / `medium` / `slow`).

- **Behavior**
  - For v1, assume documents are already ingested into PostgreSQL using `implementation/documents/`.
  - On "Run comparison":
    - For each selected strategy, call `run_strategy` (sequentially at first; parallelization later).
    - Collect `StrategyResult` objects and render answers + chunks side by side.
  - Error handling:
    - If DB/schema is not ready (e.g., `match_chunks` missing), surface a clear error message and point to `implementation/README.md` ingestion steps.

### Phase 3 – Ragas & Silver Data

- [ ] Choose Ragas integration vs custom metric implementation.
- [ ] Define `EvalExample` and `EvalRun` schemas.
- [ ] Implement silver dataset generation (100 examples):
  - [ ] Use LLM to create questions + ground-truth answers from docs.
  - [ ] Persist dataset (JSON/DB).
- [ ] Implement evaluation runner for selected strategies.
- [ ] UI: evaluation page with metrics table + charts.

### Phase 4 – Fine-tuning

- [ ] Embedding fine-tune pipeline design (data, model choice, infra).
- [ ] LLM fine-tune pipeline design.
- [ ] Background job handling for long-running training.
- [ ] Register new strategy variants using fine-tuned components.
- [ ] UI: controls to trigger fine-tuning and show ETA.

### Phase 5 – Graph RAG

- [ ] Decide on graph store (Neo4j, Postgres tables, or Graphiti).
- [ ] Implement graph ingestion from documents (entity & relation extraction).
- [ ] Implement graph-based strategy wrapper.
- [ ] UI: graph visualization (subgraph for a query).

---

## 4. Open Questions & Design Decisions

Track key choices and their rationale here.

- Frontend choice for v1:
  - Candidates: Streamlit, Chainlit, or both (chat vs dashboard views).
  - Decision: **Streamlit for v1** (comparison- and metrics-first UI; Chainlit optional later for chat-focused UX).
- Strategy combination semantics:
  - How to compose (multi-query → rerank, or parallel strategies)?
  - Decision: _TBD_.
- Ragas integration vs home-grown metrics:
  - Decision: _TBD_.
- Fine-tuning provider (OpenAI vs local models):
  - Decision: _TBD_.

---

## 5. Short-term Tasks (Next Up)

Use this as a lightweight sprint board; update as priorities change.

- [ ] Implement `StrategyResult` model and strategy registry.
- [ ] Build a minimal Streamlit app that:
  - [ ] Lists available strategies.
  - [ ] Lets user ask a question.
  - [ ] Shows side-by-side answers.
- [ ] Confirm ingestion + DB/schema are ready for playground usage.

---

## 6. Data, Storage & Non-functional Requirements

- **Data & storage**
  - Sample documents live in `implementation/documents/` and are safe for demos.
  - Uploaded docs (for now) are assumed to be **non-sensitive demo content**; store in a dedicated folder (e.g., `implementation/user_documents/`) with easy cleanup.
  - Silver evaluation datasets and Ragas results should be persisted (JSON or DB) with:
    - Strategy IDs, configuration snapshot, timestamps.
    - Clear labels that they are **silver / auto-generated**, not human-verified.
  - Fine-tuned artifacts (models, config) should be tracked by ID + metadata (provider, training data source, date).

- **Non-functional expectations**
  - Interactive playground:
    - Most strategies should respond in **< ~5 seconds** for typical queries on the sample corpus.
    - Expensive strategies (self-reflective, multi-hop, fine-tuned models) may be slower but should surface an ETA label in the UI.
  - Scale target (initial):
    - Corpus size: up to a few hundred documents / tens of thousands of chunks.
    - Single-node Postgres/pgvector; no sharding planned for v1.
  - Observability:
    - Use existing `logging` patterns for strategy calls, ingestion, and evaluation runs.
    - Log strategy ID, latency, and key errors; avoid logging raw PII from documents.

---

## 7. Safety, Privacy & Guardrails

- **Data assumptions**
  - This lab is intended for **internal / demo** use; do not ingest regulated or highly sensitive data without additional controls.
  - Clearly mark any UI that allows uploads as "for non-sensitive content only" in v1.

- **Logging & monitoring**
  - Avoid dumping full document contents or user queries in logs; log hashes/IDs where possible.
  - When storing silver datasets and evaluation results, include only what’s needed to reproduce metrics.

- **Evaluation clarity**
  - Ragas-style metrics and silver datasets should be labeled as **approximate**; they are useful for comparison, not formal guarantees.
  - When showing dashboards, always include which strategies, configs, and dataset version were used.

---

## 8. References within Repo

- High-level project overview: `README.md`, `GEMINI.md`.
- Implementation details: `implementation/README.md`, `implementation/IMPLEMENTATION_GUIDE.md`.
- RAG strategies implementation: `implementation/rag_agent.py`, `implementation/rag_agent_advanced.py`.
- Ingestion pipeline: `implementation/ingestion/ingest.py`, `implementation/ingestion/chunker.py`, `implementation/ingestion/contextual_enrichment.py`.
- DB & models: `implementation/sql/schema.sql`, `implementation/utils/models.py`, `implementation/utils/db_utils.py`.
- Theoretical docs: `docs/*.md`.
- Pseudocode examples: `examples/*.py`, `examples/README.md`.
