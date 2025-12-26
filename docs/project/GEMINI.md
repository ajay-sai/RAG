# Gemini Context for RAG Strategies Repository

This `GEMINI.md` file provides essential context and instructions for working with this repository, which focuses on Advanced Retrieval-Augmented Generation (RAG) strategies.

## Project Overview

This repository is a comprehensive guide and implementation of advanced RAG techniques. It serves two primary purposes:
1.  **Educational Resource:** Explaining complex RAG concepts through documentation (`docs/`) and simplified pseudocode (`examples/`).
2.  **Reference Implementation:** Providing a working, albeit educational, implementation of these strategies in a Python-based CLI application (`implementation/`).

**Key Technologies:**
*   **Language:** Python (>= 3.9)
*   **Agent Framework:** Pydantic AI
*   **Vector Database:** PostgreSQL with `pgvector` extension
*   **Document Processing:** Docling (supports PDF, Docx, Audio via Whisper, etc.)
*   **Embeddings & LLM:** OpenAI (GPT-4o, text-embedding-3-small)
*   **Package Manager:** `uv` (recommended) or `pip`

## Directory Structure & Purpose

*   **`docs/`**: Markdown files containing detailed theory and research for each of the 16+ RAG strategies.
*   **`examples/`**: Simplified, single-file Python scripts (< 50 lines) demonstrating the core logic of each strategy. These are often pseudocode or minimal working examples using `pydantic-ai`.
*   **`implementation/`**: The main codebase. A functional CLI application that implements the strategies.
    *   `rag_agent.py`: Basic RAG agent.
    *   `rag_agent_advanced.py`: Advanced agent integrating multiple strategies (Re-ranking, Multi-query, Self-reflection, etc.).
    *   `ingestion/`: Modules for document processing (Docling), chunking, and embedding.
    *   `documents/`: Directory for placing source documents for ingestion.

## Building and Running (Implementation)

The `implementation/` directory contains the runnable code.

### 1. Prerequisites
*   **PostgreSQL**: Must be running with the `pgvector` extension enabled.
*   **System Dependencies**:
    *   `ffmpeg` - Audio file transcription (MP3, WAV)
    *   `build-essential` & `gcc` - Compiling Python packages
    *   `postgresql-client` - Database CLI tools (psql)
    *   `libpq-dev` - PostgreSQL headers for psycopg2
    *   Ubuntu/Debian: `sudo apt-get install -y ffmpeg build-essential gcc postgresql-client libpq-dev`
    *   macOS: `brew install ffmpeg postgresql && xcode-select --install`
    *   All included in Dockerfile for containerized deployments
*   **Environment Variables**: A `.env` file is required.

### 2. Setup
Navigate to the implementation directory:
```bash
cd implementation
```

Install dependencies (using `uv` is recommended per the docs):
```bash
uv sync
# OR with pip
pip install -r requirements-advanced.txt
```

Configure environment:
```bash
cp .env.example .env
# Edit .env to add DATABASE_URL and OPENAI_API_KEY
```

Initialize Database:
```bash
# Run the schema script against your database
psql $DATABASE_URL < sql/schema.sql
```

### 3. Ingestion
Before running the agent, documents must be ingested into the vector database.
```bash
# Basic ingestion
uv run python -m ingestion.ingest --documents documents/

# Advanced ingestion with specific chunking strategy
uv run python -m ingestion.ingest --documents documents/ --chunker adaptive
```

### 4. Running the Agent
Run the interactive CLI agent:
```bash
# Basic Agent
uv run python rag_agent.py

# Advanced Agent (with all strategies)
uv run python rag_agent_advanced.py
```

### 5. Testing
Run the test suite to validate the ingestion pipeline:
```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run LoRA-SHIFT ingestion tests (17 tests)
pytest test_lora_shift_ingestion.py -v

# Expected: All tests pass, validating:
# - Paper structure and content
# - Chunking logic
# - Embedding generation
# - Retrieval queries
# - Error handling
```

## Development Conventions

*   **Async/Await:** The codebase relies heavily on Python's `asyncio`. Ensure all database and API calls are awaited properly.
*   **Type Safety:** `pydantic` and `pydantic-ai` are used for structured data handling and agent definitions. Maintain strict type hinting.
*   **Dependency Management:** The project uses `pyproject.toml`. Prefer `uv` for lockfile management.
*   **Testing:** `pytest` is configured. Run tests via `uv run pytest` or `pytest`.
*   **Code Style:** Adhere to standard Python conventions (PEP 8). The `pyproject.toml` configures `ruff` and `black` (via ruff's formatter) for linting and formatting.

## Key Strategies Implemented

When asked to modify or explain specific strategies, refer to these definitions:

1.  **Re-ranking:** Two-stage retrieval (Vector Search -> Cross-Encoder Reranking).
2.  **Agentic RAG:** Agent dynamically selects tools (Vector Search vs. Full Doc Retrieval).
3.  **Multi-Query:** Generating multiple query variations and deduplicating results.
4.  **Contextual Retrieval:** enriching chunks with document-level context before embedding (Anthropic's method).
5.  **Self-Reflective RAG:** Loop of Search -> Grade -> Refine Query -> Search.
6.  **Adaptive Chunking:** Intelligent splitting based on document structure (using Docling).

## Common Tasks

*   **Adding a new Strategy:**
    1.  Create documentation in `docs/`.
    2.  Create a simplified example in `examples/`.
    3.  Implement the logic in `implementation/rag_agent_advanced.py` (or a new module if complex).
*   **Debugging Ingestion:** Check `ingestion/ingest.py` and `ingestion/chunker.py`. Docling integration is central here.
*   **Modifying Database Schema:** Update `sql/schema.sql`. Note that schema changes may require clearing existing data.
*   **Testing Changes:** Run `pytest test_lora_shift_ingestion.py -v` to validate ingestion pipeline.
*   **Troubleshooting:** Refer to `../guides/TROUBLESHOOTING.md` for common issues and solutions.
*   **Learning RAG:** Follow the structured path in `../guides/STUDENT_GUIDE.md` (9-week curriculum).
