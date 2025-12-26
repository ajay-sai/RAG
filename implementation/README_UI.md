# RAG Strategy Lab - UI Guide

## 🚀 Running the Application

1. **Navigate to the implementation directory:**

```bash
cd implementation
```

2. **Ensure dependencies are installed:**

```bash
pip install -r requirements-advanced.txt
```

3. **Run the Streamlit app:**

```bash
streamlit run app.py
```

## 🧪 Experiment Playground Features

The interface is designed for side-by-side comparison of RAG strategies (A/B/C testing).


### 1. Strategy Configuration

You can configure up to **3 distinct strategies** simultaneously. Each strategy column allows you to customize:

* **Chunking:** (Note: Changing this requires re-ingestion, but it's documented for experiment tracking).
* **Embedding Model:** Select model version (informational unless multi-index support is added).
* **Retrieval Method:**
  * *Vector Search (Baseline):* Standard semantic similarity.
  * *Multi-Query:* Expands your query into 3 variations for broader recall.
  * *Hybrid:* Combines Vector search with BM25 keyword search.
* **Reranking:** Toggle Cross-Encoder reranking on/off for higher precision.
* **LLM Model:** Choose between models (e.g., `gpt-4o-mini` vs `gpt-4o`) to see impact on answer quality.
* **Generation Style:**
  * *Standard:* Direct answer generation.
  * *Fact Verification:* Generates an answer and then cross-checks claims against source text.
  * *Multi-Hop:* Performs iterative retrieval for complex questions.
  * *Uncertainty:* Generates multiple answers to estimate confidence.


### 2. Side-by-Side Results

When you click **🚀 Run Comparison**:
* All selected strategies run in parallel.
* Results are displayed in side-by-side columns.
* **Metrics:** Each result shows:
  * ⏱️ **Latency:** Execution time in milliseconds (end-to-end).
  * 🔎 **Retrieval Time:** Time spent fetching and (optionally) re-ranking results.
  * 🧠 **Generation Time:** Time spent in LLM generation (when applicable).
  * 📄 **Returned Results:** Number of result chunks returned by retrieval.
  * 📚 **Top Sources:** Up to 3 top contributing document titles shown inline.
  * 🔢 **Total Tokens:** Exact total tokens consumed by LLM calls when available (shown in metadata). Do not rely on estimates—this shows API-provided counts.
  * 🏷️ **Cost Class:** Estimated cost (Fast/Medium/Slow).

* **Details:** For power users, click **Show detailed metadata and traces** on a result to view the full metadata JSON with retrieval details, rerank scores, generation style, timing break-down, and precise token counts (field: `total_tokens`, plus per-call details when available).

### 3. Metrics Documentation

Expand the "📊 Understanding RAG Metrics" section at the bottom for detailed definitions.

## 🖼️ Screenshots

Please attach screenshots to PRs to help reviewers. Recommended screenshot:

* Strategy Lab showing at least one strategy result with the metadata expander open (include the `total_tokens` and timing tags).

To include a screenshot in the PR, drag & drop the image into the PR or add it to `implementation/docs/screenshots/` and reference it in the PR body.

## 🛠️ Developer Notes

* Dev/test dependency: `pytest-asyncio` has been added to the `dev` optional dependencies in `pyproject.toml`. Tests were migrated to async style using `@pytest.mark.asyncio` and `await` where appropriate.
* Unit tests rely on test stubs for optional external libs; see `rag_agent_advanced.py` for `# type: ignore[import]` annotations used to silence type diagnostics in dev environments.
* If you run tests locally, ensure dev dependencies are installed with:

```bash
pip install -e .[dev]
# or
pip install pytest-asyncio
```
