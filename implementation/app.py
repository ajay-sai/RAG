try:
    import streamlit as st  # type: ignore[import]
except Exception:
    # Minimal stub for test environments where streamlit isn't installed
    import types
    class _DummyCtx:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False
    st = types.SimpleNamespace()
    st.set_page_config = lambda *a, **k: None
    st.markdown = lambda *a, **k: None
    st.rerun = lambda *a, **k: None
    # session_state should support attribute access
    st.session_state = types.SimpleNamespace()
    st.session_state.theme = 'light'
    st.columns = lambda n: [_DummyCtx() for _ in range(n)]
    st.container = lambda *a, **k: _DummyCtx()
    st.sidebar = _DummyCtx()
    st.info = lambda *a, **k: None
    st.header = lambda *a, **k: None
    st.text_area = lambda *a, **k: ''
    st.selectbox = lambda *a, **k: a[1][0] if len(a) > 1 and a[1] else None
    st.checkbox = lambda *a, **k: False
    st.button = lambda *a, **k: False
    st.file_uploader = lambda *a, **k: []
    st.progress = lambda *a, **k: types.SimpleNamespace(progress=lambda v: None)
    st.empty = lambda *a, **k: types.SimpleNamespace(text=lambda v: None)
    st.multiselect = lambda *a, **k: a[1] if len(a) > 1 else []
    st.expander = lambda *a, **k: _DummyCtx()
    st.warning = lambda *a, **k: None
    st.error = lambda *a, **k: None
    st.success = lambda *a, **k: None
    st.subheader = lambda *a, **k: None
    st.caption = lambda *a, **k: None
    st.slider = lambda *a, **k: a[3] if len(a) >= 4 else None
    st.spinner = lambda *a, **k: types.SimpleNamespace(__enter__=lambda s: s, __exit__=lambda *args: False)
    st.divider = lambda *a, **k: None
    st.title = lambda *a, **k: None
    st.write = lambda *a, **k: None
    st.run = lambda *a, **k: None
    st.radio = lambda *a, **k: a[1][0] if len(a) > 1 and a[1] else None
    st.code = lambda *a, **k: None
    st.number_input = lambda *a, **k: k.get('value', 0)
    st.tabs = lambda labels: [_DummyCtx() for _ in labels]
    st.metric = lambda *a, **k: None
    st.json = lambda *a, **k: None
    st.dataframe = lambda *a, **k: None
    st.plotly_chart = lambda *a, **k: None
import asyncio
import logging
import time
import os
import sys
import re
import html
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Add implementation directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Check for required environment variables
from dotenv import load_dotenv
load_dotenv()

def check_environment():
    """Check if required environment variables are set."""
    errors = []
    warnings = []
    
    if not os.getenv('DATABASE_URL'):
        errors.append("DATABASE_URL is not set in .env file")
    
    if not os.getenv('OPENAI_API_KEY'):
        warnings.append("OPENAI_API_KEY is not set - RAG functionality will be limited")
    
    return errors, warnings

env_errors, env_warnings = check_environment()

# Auto-start PostgreSQL if not running
def ensure_postgres_running():
    """Ensure PostgreSQL container is running."""
    import subprocess
    try:
        # Check if Docker is available and container exists (running or stopped)
        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", "name=rag_postgres", "--format", "{{.Names}}:{{.Status}}"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if "rag_postgres" in result.stdout:
            # Container exists, check if it's running
            if "Up" not in result.stdout:
                # Container exists but is stopped, start it
                subprocess.run(
                    ["docker", "start", "rag_postgres"],
                    capture_output=True,
                    timeout=15
                )
                # Wait for PostgreSQL to be ready
                time.sleep(3)
                return True
            # Already running
            return True
        else:
            # Container doesn't exist, create and start it with docker-compose
            compose_path = os.path.join(os.path.dirname(__file__), "docker-compose.yml")
            if os.path.exists(compose_path):
                subprocess.run(
                    ["docker-compose", "up", "-d", "postgres"],
                    cwd=os.path.dirname(__file__),
                    capture_output=True,
                    timeout=30
                )
                # Wait for PostgreSQL to be ready
                time.sleep(5)
                return True
        return True
    except Exception as e:
        # Silently fail if Docker not available or error occurs
        return False

# Start PostgreSQL automatically
ensure_postgres_running()

# Import backend logic
try:
    from rag_agent_advanced import (
        initialize_db,
        close_db,
        search_knowledge_base,
        search_knowledge_base_meta,
        search_with_multi_query,
        search_with_multi_query_meta,
        search_with_hybrid_retrieval,
        search_with_hybrid_retrieval_meta,
        search_with_reranking,
        search_with_reranking_meta,
        search_with_self_reflection,
        search_with_self_reflection_meta,
    )
    from ingestion.ingest import DocumentIngestionPipeline
    from ingestion.resource_monitor import ResourceMonitor, IngestionMode
    from utils.models import IngestionConfig
    from utils.config_manager import save_active_config
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    IMPORTS_SUCCESSFUL = False
    IMPORT_ERROR = str(e)

# Evaluation module (always available – pure Python, no DB needed at import)
try:
    from utils.evaluation import RAGEvaluator, extract_contexts_from_formatted
    EVAL_AVAILABLE = True
except ImportError:
    EVAL_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="RAG Strategy Lab - Learn Advanced RAG Techniques",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': """### RAG Strategy Lab
An educational tool for learning and testing advanced RAG strategies.

Built for AI/ML and Data Science students."""
    }
)

# --- Theme & Styling ---

if not hasattr(st.session_state, 'theme'):
    st.session_state.theme = 'light'

def toggle_theme():
    st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
    st.rerun()

# Dynamic CSS variables based on theme
theme_colors = {
    'light': {
        '--bg-color': '#ffffff',
        '--text-color': '#262730',
        '--card-bg': '#ffffff',
        '--card-border': '#e0e0e0',
        '--metric-bg': '#f0f2f6',
        '--metric-text': '#31333F',
        '--result-bg': '#f8f9fa',
        '--result-border-left': '#4CAF50',
        '--shadow': '0 2px 4px rgba(0,0,0,0.05)'
    },
    'dark': {
        '--bg-color': '#0E1117',
        '--text-color': '#FAFAFA',
        '--card-bg': '#262730',
        '--card-border': '#41444C',
        '--metric-bg': '#363945',
        '--metric-text': '#FAFAFA',
        '--result-bg': '#1E2129',
        '--result-border-left': '#81C784',
        '--shadow': '0 2px 4px rgba(0,0,0,0.4)'
    }
}

current_theme = theme_colors[st.session_state.theme]

theme_css = f"""
<style>
    :root {{
        --bg-color: {current_theme['--bg-color']};
        --text-color: {current_theme['--text-color']};
        --card-bg: {current_theme['--card-bg']};
        --card-border: {current_theme['--card-border']};
        --metric-bg: {current_theme['--metric-bg']};
        --metric-text: {current_theme['--metric-text']};
        --result-bg: {current_theme['--result-bg']};
        --result-border-left: {current_theme['--result-border-left']};
        --shadow: {current_theme['--shadow']};
    }}

    /* Main strategy container with better shadows and borders */
    .strategy-container {{
        position: relative;
        border: 2px solid var(--card-border);
        border-radius: 12px;
        padding: 24px;
        background-color: var(--card-bg);
        margin-bottom: 20px;
        box-shadow: var(--shadow);
        color: var(--text-color);
        transition: all 0.3s ease;
    }}
    
    .strategy-container:hover {{
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }}
    
    /* Result box with gradient border */
    .result-box {{
        background-color: var(--result-bg);
        padding: 20px;
        border-radius: 10px;
        margin-top: 15px;
        border-left: 5px solid var(--result-border-left);
        color: var(--text-color);
        line-height: 1.6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }}
    
    /* Metric tags with better styling */
    .metric-tag {{
        background-color: var(--metric-bg);
        color: var(--metric-text);
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85em;
        margin-right: 10px;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    
    /* Header styling */
    h1, h2, h3 {{
        color: var(--text-color);
    }}
    
    /* Better spacing for containers */
    .stContainer {{
        padding: 1rem;
    }}
    
    /* Improved tabs styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        font-weight: 500;
    }}

    /* Accessibility: Focus states */
    button:focus, input:focus, textarea:focus, select:focus {{
        outline: 3px solid #4CAF50;
        outline-offset: 2px;
    }}
    
    /* Info boxes */
    .stAlert {{
        border-radius: 8px;
        padding: 1rem;
    }}
    
    /* Better button styling */
    .stButton > button {{
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }}

    /* Evaluation scorecard styling */
    .eval-scorecard {{
        background-color: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 10px;
        padding: 16px;
        margin-top: 12px;
        box-shadow: var(--shadow);
    }}

    .eval-metric-row {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 6px 0;
        border-bottom: 1px solid var(--card-border);
    }}

    .eval-metric-row:last-child {{
        border-bottom: none;
    }}

    .eval-metric-name {{
        font-size: 0.88em;
        color: var(--metric-text);
        font-weight: 500;
    }}

    .eval-score-pill {{
        padding: 3px 12px;
        border-radius: 12px;
        font-size: 0.85em;
        font-weight: 700;
        color: #fff;
    }}

    .eval-score-high   {{ background: #43a047; }}
    .eval-score-med    {{ background: #fb8c00; }}
    .eval-score-low    {{ background: #e53935; }}
    .eval-score-na     {{ background: #9e9e9e; }}

    .eval-overall-badge {{
        font-size: 1.4em;
        font-weight: 800;
        padding: 8px 20px;
        border-radius: 20px;
        color: #fff;
        display: inline-block;
        margin-top: 10px;
    }}
</style>
"""

st.markdown(theme_css, unsafe_allow_html=True)
st.markdown(f'<div data-theme="{st.session_state.theme}"></div>', unsafe_allow_html=True)

# --- Data Models & Logic ---

@dataclass
class StrategyConfig:
    name: str
    retrieval_method: str
    reranking: bool
    llm_model: str
    generation_style: str
    chunking_strategy: str

async def execute_pipeline(config: StrategyConfig, query: str) -> Dict[str, Any]:
    """Executes a RAG pipeline based on the configuration and returns rich metadata."""
    overall_start = time.time()
    meta: Dict[str, Any] = {}
    try:
        # 1. Retrieval Phase (use metadata-enabled wrappers when available)
        retrieval_start = time.time()
        retrieval_result = None

        if config.retrieval_method == "Vector Search (Baseline)":
            if config.reranking:
                retrieval_result = await search_with_reranking_meta(None, query, limit=5)
            else:
                retrieval_result = await search_knowledge_base_meta(None, query, limit=5)
        elif config.retrieval_method == "Multi-Query":
            retrieval_result = await search_with_multi_query_meta(None, query, limit=5)
        elif config.retrieval_method == "Hybrid (Vector + BM25)":
            retrieval_result = await search_with_hybrid_retrieval_meta(None, query, limit=5)
        elif config.retrieval_method == "Self-Reflective RAG":
            retrieval_result = await search_with_self_reflection_meta(None, query, limit=5)
        else:
            retrieval_result = await search_knowledge_base_meta(None, query, limit=5)

        retrieval_end = time.time()

        # Normalize retrieval result to formatted text and meta
        if isinstance(retrieval_result, dict):
            formatted = retrieval_result.get('formatted')
            retrieval_meta = retrieval_result.get('meta', {})
        else:
            formatted = retrieval_result
            retrieval_meta = {}

        # 2. Generation phase - always generate an answer
        generation_start = time.time()
        final_output = formatted
        raw_chunks = formatted  # Store raw chunks for toggle

        if config.generation_style == "Fact Verification":
            from rag_agent_advanced import answer_with_fact_verification
            final_output = await answer_with_fact_verification(None, query)
            generation_meta = {"generation_style": "fact_verification"}
        elif config.generation_style == "Multi-Hop Reasoning":
            from rag_agent_advanced import answer_with_multi_hop
            final_output = await answer_with_multi_hop(None, query)
            generation_meta = {"generation_style": "multi_hop"}
        elif config.generation_style == "Uncertainty Estimation":
            from rag_agent_advanced import answer_with_uncertainty
            final_output = await answer_with_uncertainty(None, query)
            generation_meta = {"generation_style": "uncertainty_estimation"}
        else:
            # Standard generation - always generate an answer from retrieved chunks
            final_output = await generate_answer_from_context(formatted, query, config.llm_model)
            generation_meta = {"generation_style": "standard"}

        generation_end = time.time()

        duration = (time.time() - overall_start) * 1000

        # Build full meta
        meta = {
            "retrieval_time_ms": (retrieval_end - retrieval_start) * 1000,
            "generation_time_ms": (generation_end - generation_start) * 1000,
            "retrieval_meta": retrieval_meta,
            "generation_meta": generation_meta,
            "strategy_config": {
                "retrieval_method": config.retrieval_method,
                "reranking": config.reranking,
                "llm_model": config.llm_model,
                "generation_style": config.generation_style,
                "chunking_strategy": config.chunking_strategy
            }
        }

        # Attach exact total tokens if present in retrieval_meta
        if isinstance(retrieval_meta, dict) and retrieval_meta.get('total_tokens') is not None:
            meta['total_tokens'] = retrieval_meta.get('total_tokens')
            meta['tokens_breakdown'] = retrieval_meta.get('tokens_breakdown', {})

        # Clean output from common prefixes like 'Answer:'
        cleaned_output = clean_output(final_output) if isinstance(final_output, str) else final_output
        cleaned_raw_chunks = clean_output(raw_chunks) if isinstance(raw_chunks, str) else raw_chunks

        return {
            "status": "Success",
            "output": cleaned_output,
            "raw_chunks": cleaned_raw_chunks,
            "duration": duration,
            "cost_label": estimate_cost(config),
            "name": config.name,
            "meta": meta
        }
    except Exception as e:
        return {
            "status": "Error",
            "error": str(e),
            "duration": (time.time() - overall_start) * 1000,
            "name": config.name,
            "meta": meta
        }

def estimate_cost(config: StrategyConfig) -> str:
    cost = "$"
    if config.retrieval_method == "Multi-Query": cost += "$"
    if config.retrieval_method == "Self-Reflective RAG": cost += "$$"
    if config.reranking: cost += "$"
    if config.generation_style != "Standard": cost += "$"
    if len(cost) == 1: return "⚡ Fast ($)"
    if len(cost) == 2: return "⚖️ Medium ($$)"
    return "🐌 Slow ($$$)"

# Helper: clean common answer markers
def clean_output(text: str) -> str:
    if not text:
        return text
    # Remove leading 'Answer:' and similar wrappers
    text = re.sub(r"^\s*Answer\s*:\s*\n?", "", text, flags=re.IGNORECASE)
    return text.strip()


async def generate_answer_from_context(context: str, query: str, llm_model: str) -> str:
    """Generate a natural language answer from retrieved context."""
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        prompt = f"""Based on the following context, provide a comprehensive and accurate answer to the question.

Context:
{context}

Question: {query}

Provide a clear, well-structured answer based solely on the information in the context above:"""
        
        response = await client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Answer generation failed: {e}", exc_info=True)
        return f"Error generating answer: {str(e)}"


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _score_color_class(score: Optional[float]) -> str:
    """Return CSS class for a 0-1 score."""
    if score is None:
        return "eval-score-na"
    if score >= 0.70:
        return "eval-score-high"
    if score >= 0.40:
        return "eval-score-med"
    return "eval-score-low"


def _score_label(score: Optional[float]) -> str:
    """Return formatted label for a 0-1 score."""
    if score is None:
        return "N/A"
    return f"{score:.0%}"


def _render_eval_scorecard(eval_report: Dict[str, Any], title: str = "🔬 Evaluation Metrics") -> None:
    """Render evaluation metrics as an HTML scorecard inside an st.expander."""
    DISPLAY_METRICS = [
        ("overall_score",        "⭐ Overall RAG Score"),
        ("faithfulness",         "✅ Faithfulness"),
        ("answer_relevance",     "🎯 Answer Relevance"),
        ("context_precision",    "📌 Context Precision"),
        ("context_recall",       "🔁 Context Recall"),
        ("groundedness",         "🏗️ Groundedness"),
        ("coherence",            "📖 Coherence"),
        ("conciseness",          "✂️ Conciseness"),
        ("avg_similarity",       "🔍 Avg Retrieval Similarity"),
        ("ndcg",                 "📊 NDCG@5"),
        ("token_efficiency",     "⚡ Token Efficiency"),
        ("answer_correctness",   "🏆 Answer Correctness"),
    ]

    rows_html = ""
    for key, display_name in DISPLAY_METRICS:
        if key == "overall_score":
            raw_score = eval_report.get("overall_score")
            score_html = (
                f'<span class="eval-score-pill {_score_color_class(raw_score)}">'
                f'{_score_label(raw_score)}</span>'
            )
            rows_html += (
                f'<div class="eval-metric-row" style="font-weight:700">'
                f'<span class="eval-metric-name">{display_name}</span>'
                f'{score_html}</div>'
            )
            continue

        metric = eval_report.get(key)
        if metric is None:
            continue

        score = metric.get("score")
        detail = metric.get("detail", "")

        score_html = (
            f'<span class="eval-score-pill {_score_color_class(score)}">'
            f'{_score_label(score)}</span>'
        )
        detail_html = f'<span style="font-size:0.78em;opacity:0.7;margin-left:6px">{html.escape(str(detail)[:80])}</span>' if detail else ""

        rows_html += (
            f'<div class="eval-metric-row">'
            f'<span class="eval-metric-name">{display_name}{detail_html}</span>'
            f'{score_html}</div>'
        )

    html_block = f'<div class="eval-scorecard">{rows_html}</div>'

    with st.expander(title, expanded=False):
        st.markdown(html_block, unsafe_allow_html=True)


async def _run_evaluation(
    question: str,
    answer: str,
    contexts: List[str],
    reference_answer: Optional[str] = None,
    similarity_scores: Optional[List[float]] = None,
    total_tokens: Optional[int] = None,
    include_groundedness: bool = True,
    include_coherence: bool = True,
    include_conciseness: bool = True,
) -> Dict[str, Any]:
    """Run full evaluation using the RAGEvaluator."""
    evaluator = RAGEvaluator(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
    )
    return await evaluator.run_full_evaluation(
        question=question,
        answer=answer,
        contexts=contexts,
        reference_answer=reference_answer,
        similarity_scores=similarity_scores,
        total_tokens=total_tokens,
        include_groundedness=include_groundedness,
        include_coherence=include_coherence,
        include_conciseness=include_conciseness,
    )


# --- Page: Learning Center ---


def render_learning_page():
    st.header("📚 Strategies")
    st.markdown("""
    Welcome to the **RAG Strategy Lab**! This platform is designed to help you understand and implement advanced Retrieval-Augmented Generation strategies.
    
    ### 🚀 Quick Start Guide
    1. **Ingest Documents:** Go to the **Ingestion Lab** to upload and process your documents.
    2. **Experiment:** Go to the **Strategy Lab** to compare different RAG strategies side-by-side.
    
    ---
    
    ### 🧠 All 16 RAG Strategies
    
    #### 1. Ingestion & Chunking
    - **✂️ Context-Aware Chunking:** Splits documents based on structure (headings, sections) rather than just token count.
    - **📏 Adaptive Chunking:** Dynamically adjusts chunk size based on content density and semantic coherence.
    - **⏳ Late Chunking:** Embeds the full document first, then chunks the embeddings to preserve global context.
    - **📝 Contextual Retrieval:** Adds document-level context (summary/title) to each chunk before embedding.
    - **🎯 Fine-tuned Embeddings:** Trains embedding models on domain-specific data for better representation.
    - **🕸️ Knowledge Graphs:** Maps entities and relationships to capture structured knowledge alongside vectors.
    
    #### 2. Retrieval & Querying
    - **🔍 Re-ranking:** Two-stage process: fast vector search followed by high-precision cross-encoder scoring.
    - **➕ Query Expansion:** Enriches a short query with related terms and context to improve recall.
    - **🔀 Multi-Query RAG:** Generates multiple diverse query variations to capture different perspectives.
    - **⚖️ Hybrid Retrieval:** Combines dense vector search (semantic) with sparse BM25 search (keyword).
    - **🌳 Hierarchical RAG:** Searches summaries or parent chunks first, then retrieves detailed child chunks.
    - **🤖 Agentic RAG:** Uses an autonomous agent to select the best retrieval tool (search, full doc, etc.) for the query.
    - **🤔 Self-Reflective RAG:** Iteratively critiques and refines search results until they meet a quality threshold.
    
    #### 3. Generation & Reasoning
    - **✅ Fact Verification:** Generates an answer and then cross-checks every claim against source text.
    - **🔗 Multi-Hop Reasoning:** Breaks down complex questions into sub-questions and retrieves information for each step.
    - **📊 Uncertainty Estimation:** Generates multiple answers to estimate confidence and identify ambiguity.
    """)

# --- Page: Ingestion Lab ---

def render_ingestion_page():
    if not IMPORTS_SUCCESSFUL:
        st.error("❌ Cannot load ingestion modules. Please ensure all dependencies are installed.")
        st.code("pip install -r requirements-advanced.txt", language="bash")
        return
    
    st.header("📥 Ingestion Lab")
    st.markdown("""
    Transform your documents into searchable knowledge. This lab processes documents through:
    - **Document Loading** (PDF, DOCX, Markdown, Audio)
    - **Intelligent Chunking** (Semantic, Fixed, Adaptive)
    - **Vector Embedding** (OpenAI models)
    - **Database Storage** (PostgreSQL with pgvector)
    """)
    
    st.info("💡 **Tip for Learners:** Start by uploading a few small documents to see how different chunking strategies affect retrieval quality.", icon="💡")
    
    # File Upload Section
    with st.container(border=True):
        st.subheader("📁 Documents")
        
        tab1, tab2 = st.tabs(["Upload Files", "Available Files"])
        
        with tab1:
            uploaded_files = st.file_uploader(
                "Upload documents to process",
                type=["pdf", "docx", "md", "txt", "mp3", "wav"],
                accept_multiple_files=True,
                help="Supported formats: PDF, DOCX, Markdown, Text, Audio (MP3, WAV)"
            )
            
            if uploaded_files:
                st.write(f"**{len(uploaded_files)} file(s) selected for upload:**")
                for file in uploaded_files:
                    st.write(f"- {file.name} ({file.size / 1024:.2f} KB)")
                
                if st.button("💾 Save Uploaded Files", type="secondary"):
                    try:
                        docs_dir = os.path.join(os.path.dirname(__file__), "documents")
                        os.makedirs(docs_dir, exist_ok=True)
                        saved_files = []
                        
                        for file in uploaded_files:
                            # Sanitize filename to prevent path traversal
                            safe_filename = os.path.basename(file.name)
                            file_path = os.path.join(docs_dir, safe_filename)
                            with open(file_path, "wb") as f:
                                f.write(file.getbuffer())
                            saved_files.append(safe_filename)
                        
                        st.success(f"✅ Successfully saved {len(saved_files)} file(s) to documents folder!")
                        st.info("You can now configure and run the ingestion pipeline below.")
                    except Exception as e:
                        st.error(f"Error saving files: {e}")
        
        with tab2:
            docs_dir = os.path.join(os.path.dirname(__file__), "documents")
            if os.path.exists(docs_dir):
                files = [f for f in os.listdir(docs_dir) if os.path.isfile(os.path.join(docs_dir, f))]
                if files:
                    st.write(f"**{len(files)} file(s) available in documents folder:**")
                    
                    # File selection
                    if 'selected_files' not in st.session_state:
                        st.session_state.selected_files = files  # All selected by default
                    
                    select_all = st.checkbox("Select All Files", value=True, key="select_all_docs")
                    
                    # Determine default selection based on checkbox and session state
                    default_selection = files if select_all else st.session_state.selected_files
                    
                    selected_files = st.multiselect(
                        "Select files to ingest:",
                        options=files,
                        default=default_selection,
                        help="Choose which documents to process"
                    )
                    st.session_state.selected_files = selected_files
                    
                    # Show file details
                    with st.expander("View File Details"):
                        for file in files:
                            file_path = os.path.join(docs_dir, file)
                            file_size = os.path.getsize(file_path) / 1024  # KB
                            selected = "✅" if file in selected_files else "⬜"
                            st.write(f"{selected} **{file}** - {file_size:.2f} KB")
                else:
                    st.warning("No files found in documents folder. Please upload some files first.")
            else:
                st.warning("Documents folder does not exist. Please upload files to create it.")
    
    with st.container(border=True):
        st.subheader("⚙️ Configuration")
        st.caption("Configure how your documents will be processed and embedded")
        
        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.slider(
                "Chunk Size (tokens)", 
                min_value=100, max_value=2000, value=1000, step=100,
                help="📏 Target size for each document chunk. Larger chunks provide more context but may be less precise. Recommended: 500-1000 for most use cases."
            )
            chunk_overlap = st.slider(
                "Chunk Overlap", 
                min_value=0, max_value=500, value=200, step=50,
                help="🔗 Number of overlapping tokens between adjacent chunks. Overlap helps maintain context across chunk boundaries. Recommended: 10-20% of chunk size."
            )
            
        with col2:
            chunker_type = st.selectbox(
                "Chunker Type",
                ["semantic", "fixed", "adaptive"],
                index=0,
                help="✂️ **Semantic:** Splits at natural boundaries (sentences, paragraphs). **Fixed:** Equal-sized chunks. **Adaptive:** Document-structure aware splitting."
            )
            embedding_model = st.selectbox(
                "Embedding Model",
                ["text-embedding-3-small", "text-embedding-3-large"],
                index=0,
                help="🧠 **small:** Faster and cheaper, good for most tasks. **large:** Higher quality embeddings, better for complex domains."
            )
            
        contextual = st.checkbox(
            "Use Contextual Enrichment", 
            value=False,
            help="🎯 Uses an LLM to add document-level context to each chunk before embedding. Improves retrieval accuracy but increases cost and processing time. (Anthropic's Contextual Retrieval technique)"
        )        
        # Ingestion mode selection
        st.markdown("### 🎯 Ingestion Mode")
        st.markdown("""
        Choose how documents are processed based on your system resources:
        - **Auto-Detect**: Automatically select best mode based on available RAM
        - **Full**: All features (Whisper Turbo, OCR, enrichment) - needs 8GB+ RAM
        - **Standard**: Whisper Base, OCR, no enrichment - needs 4GB+ RAM  
        - **Light**: Whisper Tiny, no OCR/enrichment - needs 2GB+ RAM
        - **Minimal**: Skip audio/images, text only - works with any RAM
        """)
        
        ingestion_mode_str = st.selectbox(
            "Select Ingestion Mode",
            options=["auto", "full", "standard", "light", "minimal"],
            index=0,
            help="Auto-detect will check your system resources and choose the best mode"
        )
        
        # Show resource summary if not auto
        if st.button("📊 Check System Resources"):
            with st.spinner("Checking system resources..."):
                resources = ResourceMonitor.get_system_resources()
                recommended_mode = ResourceMonitor.recommend_ingestion_mode()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Available Memory", f"{resources.get('memory_available_gb', 0):.1f} GB")
                with col2:
                    st.metric("Free Disk", f"{resources.get('disk_free_gb', 0):.1f} GB")
                with col3:
                    st.metric("CPU Usage", f"{resources.get('cpu_percent', 0):.1f}%")
                
                st.info(f"✅ Recommended mode: **{recommended_mode.value.upper()}**")
        if st.button("🔄 Run Ingestion Pipeline", type="primary"):
            # Check if files are selected
            selected_files = st.session_state.get('selected_files', [])
            
            if not selected_files:
                st.warning("⚠️ No files selected. Please select files to ingest in the 'Available Files' tab.")
            else:
                # Save config
                config_data = {
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "chunker_type": chunker_type,
                    "embedding_model": embedding_model,
                    "use_contextual_enrichment": contextual
                }
                save_active_config(config_data)
                
                # Run Pipeline
                st.info(f"Starting ingestion of {len(selected_files)} file(s)... This will clear existing data.")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Parse ingestion mode
                    ingestion_mode = None
                    auto_detect = False
                    if ingestion_mode_str == "auto":
                        auto_detect = True
                        st.info("🔍 Auto-detecting best ingestion mode based on system resources...")
                    else:
                        mode_map = {
                            "full": IngestionMode.FULL,
                            "standard": IngestionMode.STANDARD,
                            "light": IngestionMode.LIGHT,
                            "minimal": IngestionMode.MINIMAL,
                        }
                        ingestion_mode = mode_map[ingestion_mode_str]
                        st.info(f"🎯 Using **{ingestion_mode_str.upper()}** mode")
                    
                    # Initialize Pipeline
                    ingest_config = IngestionConfig(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        use_semantic_chunking=(chunker_type=="semantic"),
                        chunker_type=chunker_type,
                        use_contextual_enrichment=contextual
                    )
                    
                    # We need to run async code in sync context
                    async def run_pipeline():
                        pipeline = DocumentIngestionPipeline(
                            config=ingest_config,
                            documents_folder=os.path.join(os.path.dirname(__file__), "documents"),
                            clean_before_ingest=True,
                            ingestion_mode=ingestion_mode,
                            auto_detect_mode=auto_detect
                        )
                        
                        # Show detected mode if auto
                        if auto_detect:
                            detected_mode_name = pipeline.ingestion_mode.value.upper()
                            status_text.text(f"🎯 Detected mode: {detected_mode_name}")
                        
                        def update_progress(current, total):
                            progress_bar.progress(current / total)
                            status_text.text(f"Processing document {current}/{total}...")
                        
                        return await pipeline.ingest_documents(
                            progress_callback=update_progress,
                            specific_files=selected_files
                        )

                    # Close existing database connection if any
                    import rag_agent_advanced
                    if rag_agent_advanced.db_pool:
                        asyncio.run(close_db())
                        rag_agent_advanced.db_pool = None
                    
                    results = asyncio.run(run_pipeline())
                    
                    st.success(f"✅ Ingestion Complete! Processed {len(results)} documents.")
                    
                    # Show detailed results
                    st.subheader("📊 Ingestion Results")
                    for r in results:
                        if r.errors:
                            st.error(f"❌ **{r.title}**: {r.chunks_created} chunks created, but encountered errors: {', '.join(r.errors)}")
                        else:
                            st.success(f"✅ **{r.title}**: {r.chunks_created} chunks created in {r.processing_time_ms:.0f}ms")
                    
                except Exception as e:
                    st.error(f"❌ Ingestion Failed: {e}")
                    import traceback
                    with st.expander("View Error Details"):
                        st.code(traceback.format_exc())

# --- Page: Retrieval Lab ---

def render_retrieval_page():
    if not IMPORTS_SUCCESSFUL:
        st.error("❌ Cannot load RAG modules. Please ensure all dependencies are installed.")
        st.code("pip install -r requirements-advanced.txt", language="bash")
        return
    
    st.header("🧪 Strategy Lab")
    st.markdown("""
    Compare up to 3 RAG strategies side-by-side to understand their tradeoffs.
    
    **What you can test:**
    - 🔍 **Retrieval Methods:** Vector search, multi-query, hybrid, self-reflective
    - 🎯 **Reranking:** Cross-encoder reranking for better relevance
    - 🤖 **LLM Models:** Compare different model sizes (GPT-4o vs GPT-4o-mini)
    - 📝 **Generation Styles:** Standard, fact verification, multi-hop reasoning, uncertainty estimation
    """)
    
    st.info("💡 **Tip for Learners:** Try comparing baseline vector search vs. multi-query to see how query expansion improves results!", icon="💡")

    # Global Query
    user_query = st.text_area("Test Query:", height=80, placeholder="Enter a complex question about your documents...")

    # Strategy Columns
    cols = st.columns(3)
    configs = []

    for i, col in enumerate(cols):
        with col:
            with st.container(border=True):
                st.subheader(f"Strategy {i + 1}")
                
                c1, c2 = st.columns(2)
                with c1:
                    # Allow selecting chunking strategy
                    chunking = st.selectbox(
                        "Chunking", 
                        ["semantic", "fixed", "adaptive"],
                        index=0,
                        key=f"chunk_{i}",
                        help="✂️ How text was split into chunks"
                    )
                    st.selectbox(
                        "Embedding",
                        ["text-embedding-3-small", "text-embedding-3-large"],
                        index=0,
                        key=f"embed_{i}",
                        help="🧠 Embedding model used"
                    )
                with c2:
                    retrieval = st.selectbox(
                        "Retrieval",
                        ["Vector Search (Baseline)", "Multi-Query", "Hybrid (Vector + BM25)", "Self-Reflective RAG"],
                        key=f"retrieval_{i}",
                        help="🔍 **Baseline:** Simple vector similarity. **Multi-Query:** Expands query into variations. **Hybrid:** Combines vector + keyword search. **Self-Reflective:** Iteratively refines search based on relevance."
                    )
                    rerank = st.checkbox("Reranking", key=f"rerank_{i}", help="🎯 Use Cross-Encoder to rerank results for better relevance (adds latency)")

                llm = st.selectbox(
                    "LLM", 
                    ["gpt-4o-mini", "gpt-4o"], 
                    key=f"llm_{i}",
                    help="🤖 **gpt-4o-mini:** Faster and cheaper. **gpt-4o:** More capable for complex reasoning."
                )
                gen_style = st.selectbox(
                    "Generation",
                    ["Standard", "Fact Verification", "Multi-Hop Reasoning", "Uncertainty Estimation"],
                    key=f"gen_{i}",
                    help="📝 **Standard:** Direct answer. **Fact Verification:** Validates claims against sources. **Multi-Hop:** Breaks down complex questions. **Uncertainty:** Provides confidence scores."
                )
                
                configs.append(StrategyConfig(
                    name=f"Strategy {i + 1}",
                    retrieval_method=retrieval,
                    reranking=rerank,
                    llm_model=llm,
                    generation_style=gen_style,
                    chunking_strategy=chunking
                ))

    # Display toggle for raw chunks vs generated answer
    show_raw_chunks = st.checkbox(
        "📄 Show Raw Chunks (instead of generated answer)",
        value=False,
        help="Toggle between viewing the raw retrieved chunks and the generated natural language answer"
    )

    # Run Button
    if st.button("🚀 Run Comparison", type="primary", use_container_width=True):
        if not user_query:
            st.warning("Please enter a query first.")
        else:
            # Close existing database connection if any
            import rag_agent_advanced
            if rag_agent_advanced.db_pool:
                asyncio.run(close_db())
                rag_agent_advanced.db_pool = None
            
            async def run_all():
                await initialize_db()
                try:
                    tasks = [execute_pipeline(cfg, user_query) for cfg in configs]
                    return await asyncio.gather(*tasks)
                finally:
                    await close_db()
            
            with st.spinner("Running strategies..."):
                results = asyncio.run(run_all())
                
                st.markdown("### Results Comparison")
                r_cols = st.columns(3)
                
                for i, res in enumerate(results):
                    with r_cols[i]:
                        if res["status"] == "Success":
                            # Extract and format metadata
                            meta = res.get('meta', {}) or {}
                            retrieval_meta = meta.get('retrieval_meta', {}) if isinstance(meta, dict) else {}
                            retrieval_time = meta.get('retrieval_time_ms')
                            generation_time = meta.get('generation_time_ms')
                            returned = retrieval_meta.get('returned') if isinstance(retrieval_meta, dict) else None
                            candidates = retrieval_meta.get('candidates_considered') if isinstance(retrieval_meta, dict) else None
                            top_sources = retrieval_meta.get('top_sources') if isinstance(retrieval_meta, dict) else None
                            total_tokens = meta.get('total_tokens')

                            # Build metric tags
                            metrics_html = ''
                            if res.get('duration') is not None:
                                metrics_html += f"<span class=\"metric-tag\" title=\"Execution Time\">⏱️ {res['duration']:.0f} ms</span>"
                            if retrieval_time:
                                metrics_html += f"<span class=\"metric-tag\" title=\"Retrieval Time\">🔎 {retrieval_time:.0f} ms</span>"
                            if generation_time:
                                metrics_html += f"<span class=\"metric-tag\" title=\"Generation Time\">🧠 {generation_time:.0f} ms</span>"
                            if total_tokens is not None:
                                metrics_html += f"<span class=\"metric-tag\" title=\"Total Tokens\">🔢 {int(total_tokens)}</span>"
                            if res.get('cost_label'):
                                metrics_html += f"<span class=\"metric-tag\" title=\"Estimated Cost\">{res['cost_label']}</span>"
                            if returned is not None:
                                metrics_html += f"<span class=\"metric-tag\" title=\"Returned Results\">📄 {returned}</span>"
                            if candidates is not None:
                                metrics_html += f"<span class=\"metric-tag\" title=\"Candidates Considered\">🧾 {candidates}</span>"

                            top_sources_html = ''
                            if top_sources:
                                sample = ', '.join(top_sources[:3])
                                top_sources_html = f"<div style=\"margin-top:8px;color:var(--metric-text)\"><strong>Top Sources:</strong> {html.escape(sample)}</div>"

                            # Choose content based on toggle
                            display_content = res.get('raw_chunks') if show_raw_chunks else res['output']
                            content_type_label = "Raw Retrieved Chunks" if show_raw_chunks else "Generated Answer"

                            # Use Streamlit container with custom styling via CSS class
                            with st.container():
                                # Apply styling and display header
                                st.markdown(f"""
                                <div style="border: 2px solid var(--card-border); border-radius: 12px; padding: 24px; background-color: var(--card-bg); box-shadow: var(--shadow); margin-bottom: 10px;">
                                    <h3 style="margin: 0; font-size: 1.2em; color: var(--text-color);">{html.escape(res['name'])}</h3>
                                    <div style="margin-top: 8px;">{metrics_html}</div>
                                    {top_sources_html}
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Display content label and text (pure Streamlit, no HTML)
                                st.markdown(f"**{content_type_label}**")
                                st.markdown("")  # Spacing
                                # Display content with proper line breaks
                                st.markdown(display_content)

                            # Detailed metadata expander
                            with st.expander("Show detailed metadata and traces"):
                                import json
                                st.subheader("Metadata")
                                st.code(json.dumps(meta, indent=2))

                            # Inline evaluation scorecard
                            if EVAL_AVAILABLE and res.get('output'):
                                eval_key = f"eval_result_{i}"
                                if eval_key in st.session_state:
                                    _render_eval_scorecard(
                                        st.session_state[eval_key],
                                        title="🔬 Evaluation Metrics (cached)"
                                    )
                        else:
                            st.error(f"Error: {res.get('error')}")

            # Evaluate Results button (shows after comparison)
            if EVAL_AVAILABLE:
                st.markdown("---")
                ref_answer = st.text_area(
                    "📝 Optional: Reference / Ground-Truth Answer (for Answer Correctness metric)",
                    key="ref_answer_strategy_lab",
                    height=60,
                    placeholder="Leave blank to skip Answer Correctness evaluation",
                    help="If you provide a reference answer, the evaluation will also compute Answer Correctness metric."
                )
                if st.button("🔬 Evaluate All Strategies", type="secondary", use_container_width=True,
                             key="eval_all_btn",
                             help="Run evaluation metrics (Faithfulness, Answer Relevance, etc.) on the results above"):
                    if 'results' not in dir():
                        st.warning("Please run a comparison first.")
                    else:
                        async def run_evals(results_data):
                            eval_tasks = []
                            for res in results_data:
                                if res.get("status") == "Success" and res.get("output"):
                                    ctxs = extract_contexts_from_formatted(res.get("raw_chunks", ""))
                                    tokens = res.get("meta", {}).get("total_tokens") if isinstance(res.get("meta"), dict) else None
                                    eval_tasks.append(_run_evaluation(
                                        question=user_query,
                                        answer=res["output"],
                                        contexts=ctxs,
                                        reference_answer=ref_answer if ref_answer else None,
                                        total_tokens=tokens,
                                    ))
                                else:
                                    async def _empty():
                                        return {}
                                    eval_tasks.append(_empty())
                            return await asyncio.gather(*eval_tasks, return_exceptions=True)

                        with st.spinner("🔬 Running evaluation metrics (this may take 15-30 seconds)..."):
                            try:
                                import asyncio as _asyncio
                                eval_reports = _asyncio.run(run_evals(results))
                                e_cols = st.columns(3)
                                for idx, (res, report) in enumerate(zip(results, eval_reports)):
                                    with e_cols[idx]:
                                        if isinstance(report, dict) and report:
                                            st.session_state[f"eval_result_{idx}"] = report
                                            _render_eval_scorecard(report, title=f"🔬 {res.get('name', f'Strategy {idx+1}')} Metrics")
                            except Exception as exc:
                                st.error(f"Evaluation error: {exc}")


# --- Page: Evaluation Lab ---

def render_evaluation_page():
    """Full dedicated Evaluation Lab for systematic RAG evaluation."""
    if not EVAL_AVAILABLE:
        st.error("❌ Evaluation module not available. Check `utils/evaluation.py`.")
        return

    if not IMPORTS_SUCCESSFUL:
        st.error("❌ Cannot load RAG modules. Please ensure all dependencies are installed.")
        st.code("pip install -r requirements-advanced.txt", language="bash")
        return

    st.header("🔬 Evaluation Lab")
    st.markdown("""
    Systematically evaluate your RAG pipeline with industry-standard metrics.

    **Metric categories:**
    - 🎯 **Retrieval Quality** — Context Precision, Context Recall, Average Similarity, NDCG
    - ✅ **Generation Quality** — Faithfulness, Answer Relevance, Groundedness, Hallucination Rate
    - 📖 **Answer Quality** — Coherence, Conciseness, Answer Correctness (optional)
    - ⚡ **Efficiency** — Token Efficiency, Latency

    *LLM-as-Judge metrics use GPT-4o-mini. Ensure `OPENAI_API_KEY` is set.*
    """)

    st.info(
        "💡 **Tip:** Use the Strategy Lab first to get answers, then come here to evaluate them. "
        "Or enter answers manually below for batch evaluation.",
        icon="💡"
    )

    tab_single, tab_batch, tab_compare = st.tabs(["Single Evaluation", "Batch Evaluation", "Strategy Comparison"])

    # --- Single Evaluation Tab ---
    with tab_single:
        st.subheader("Single Query Evaluation")
        st.caption("Evaluate one question–answer pair with full metric breakdown")

        with st.container(border=True):
            col1, col2 = st.columns(2)
            with col1:
                eval_question = st.text_area(
                    "❓ Question",
                    key="eval_q_single",
                    height=80,
                    placeholder="What is RAG and why is it useful?",
                )
                eval_answer = st.text_area(
                    "🤖 Generated Answer",
                    key="eval_a_single",
                    height=120,
                    placeholder="Paste the generated answer here...",
                )
                eval_context = st.text_area(
                    "📄 Retrieved Context (paste full context or chunk text)",
                    key="eval_ctx_single",
                    height=120,
                    placeholder="Paste retrieved context chunks here (one per line or all together)...",
                )
            with col2:
                eval_reference = st.text_area(
                    "✅ Reference Answer (optional — for Correctness metric)",
                    key="eval_ref_single",
                    height=80,
                    placeholder="Leave blank to skip Answer Correctness evaluation",
                )
                eval_sim_scores = st.text_input(
                    "📐 Retrieval Similarity Scores (optional, comma-separated 0-1 values)",
                    key="eval_sim_single",
                    placeholder="e.g. 0.92, 0.87, 0.81, 0.76, 0.71",
                    help="Cosine similarity scores from the DB query — used for NDCG and Avg Similarity."
                )

                st.markdown("**⚙️ Metric Options**")
                inc_groundedness = st.checkbox("Groundedness / Hallucination Rate", value=True, key="inc_ground_single")
                inc_coherence = st.checkbox("Coherence", value=True, key="inc_coh_single")
                inc_conciseness = st.checkbox("Conciseness", value=True, key="inc_conc_single")

        if st.button("🚀 Run Evaluation", type="primary", key="run_eval_single"):
            if not eval_question or not eval_answer:
                st.warning("Please provide at least a Question and a Generated Answer.")
            else:
                contexts = [c.strip() for c in eval_context.split("\n\n") if c.strip()] if eval_context else []
                if not contexts and eval_context:
                    contexts = [eval_context.strip()]

                sim_scores = None
                if eval_sim_scores:
                    try:
                        sim_scores = [float(s.strip()) for s in eval_sim_scores.split(",") if s.strip()]
                    except ValueError:
                        st.warning("Could not parse similarity scores — ignoring.")

                with st.spinner("🔬 Running evaluation metrics (15-30 seconds)..."):
                    try:
                        report = asyncio.run(_run_evaluation(
                            question=eval_question,
                            answer=eval_answer,
                            contexts=contexts,
                            reference_answer=eval_reference if eval_reference else None,
                            similarity_scores=sim_scores,
                            include_groundedness=inc_groundedness,
                            include_coherence=inc_coherence,
                            include_conciseness=inc_conciseness,
                        ))
                        st.session_state["last_single_eval"] = report

                        # Overall score banner
                        overall = report.get("overall_score")
                        if overall is not None:
                            color = "#43a047" if overall >= 0.70 else ("#fb8c00" if overall >= 0.40 else "#e53935")
                            st.markdown(
                                f'<div style="text-align:center;margin:12px 0">'
                                f'<span class="eval-overall-badge" style="background:{color}">'
                                f'Overall RAG Score: {overall:.0%}</span></div>',
                                unsafe_allow_html=True
                            )

                        # Metric grid
                        _render_eval_scorecard(report, title="📊 Full Metric Breakdown")

                        # Expanded details
                        with st.expander("🔎 Raw Evaluation Data (JSON)"):
                            import json
                            st.code(json.dumps(report, indent=2, default=str))

                    except Exception as exc:
                        st.error(f"Evaluation failed: {exc}")
                        import traceback
                        with st.expander("Error details"):
                            st.code(traceback.format_exc())

    # --- Batch Evaluation Tab ---
    with tab_batch:
        st.subheader("Batch Evaluation")
        st.caption("Evaluate multiple question-answer pairs and compare aggregate metrics")

        st.markdown("""
        Enter one pair per row in the table below. Use `|` to separate columns:
        `Question | Answer | Context (optional) | Reference Answer (optional)`
        """)

        default_batch = (
            "What is RAG? | RAG stands for Retrieval-Augmented Generation... | RAG combines retrieval with generation... | \n"
            "How does vector search work? | Vector search finds semantically similar content... | Embeddings represent text as vectors... | "
        )
        batch_input = st.text_area(
            "📋 Batch Input (Question | Answer | Context | Reference)",
            value=default_batch,
            height=200,
            key="batch_eval_input",
        )

        eval_model = st.selectbox(
            "LLM Judge Model",
            ["gpt-4o-mini", "gpt-4o"],
            key="batch_eval_model",
            help="gpt-4o-mini is faster/cheaper; gpt-4o gives higher quality evaluations"
        )

        if st.button("🚀 Run Batch Evaluation", type="primary", key="run_eval_batch"):
            rows = [r.strip() for r in batch_input.strip().split("\n") if r.strip()]
            if not rows:
                st.warning("Please enter at least one row.")
            else:
                parsed = []
                errors = []
                for idx, row in enumerate(rows):
                    parts = [p.strip() for p in row.split("|")]
                    if len(parts) < 2:
                        errors.append(f"Row {idx+1}: need at least Question|Answer")
                        continue
                    parsed.append({
                        "question": parts[0],
                        "answer": parts[1],
                        "context": parts[2] if len(parts) > 2 else "",
                        "reference": parts[3] if len(parts) > 3 else "",
                    })

                if errors:
                    for err in errors:
                        st.warning(err)

                if parsed:
                    async def run_batch(items):
                        evaluator = RAGEvaluator(
                            openai_api_key=os.getenv("OPENAI_API_KEY"),
                            model=eval_model,
                        )
                        tasks = []
                        for item in items:
                            ctxs = [item["context"]] if item["context"] else []
                            tasks.append(evaluator.run_full_evaluation(
                                question=item["question"],
                                answer=item["answer"],
                                contexts=ctxs,
                                reference_answer=item["reference"] if item["reference"] else None,
                                include_groundedness=True,
                                include_coherence=True,
                                include_conciseness=True,
                            ))
                        return await asyncio.gather(*tasks, return_exceptions=True)

                    with st.spinner(f"🔬 Evaluating {len(parsed)} items..."):
                        try:
                            reports = asyncio.run(run_batch(parsed))

                            # Summary table
                            METRIC_KEYS = [
                                "faithfulness", "answer_relevance", "context_precision",
                                "context_recall", "groundedness", "coherence", "conciseness",
                            ]
                            table_rows = []
                            for item, report in zip(parsed, reports):
                                if isinstance(report, Exception):
                                    row = {"Question": item["question"][:50], "Error": str(report)}
                                else:
                                    row = {"Question": item["question"][:50]}
                                    for mk in METRIC_KEYS:
                                        m = report.get(mk, {})
                                        s = m.get("score") if isinstance(m, dict) else None
                                        row[mk.replace("_", " ").title()] = f"{s:.0%}" if s is not None else "N/A"
                                    row["Overall"] = f"{report.get('overall_score', 0):.0%}" if report.get('overall_score') else "N/A"
                                table_rows.append(row)

                            try:
                                import pandas as pd
                                df = pd.DataFrame(table_rows)
                                st.dataframe(df, use_container_width=True)
                            except ImportError:
                                st.json(table_rows)

                            # Aggregate averages
                            st.markdown("**📈 Aggregate Averages**")
                            agg_cols = st.columns(len(METRIC_KEYS) + 1)
                            for col_widget, mk in zip(agg_cols, METRIC_KEYS):
                                scores = []
                                for report in reports:
                                    if isinstance(report, dict):
                                        m = report.get(mk, {})
                                        s = m.get("score") if isinstance(m, dict) else None
                                        if s is not None:
                                            scores.append(s)
                                avg = sum(scores) / len(scores) if scores else None
                                with col_widget:
                                    st.metric(
                                        label=mk.replace("_", " ").title(),
                                        value=f"{avg:.0%}" if avg is not None else "N/A"
                                    )

                            overall_scores = [
                                r.get("overall_score") for r in reports
                                if isinstance(r, dict) and r.get("overall_score") is not None
                            ]
                            avg_overall = sum(overall_scores) / len(overall_scores) if overall_scores else None
                            with agg_cols[-1]:
                                st.metric(
                                    label="Overall",
                                    value=f"{avg_overall:.0%}" if avg_overall is not None else "N/A"
                                )

                            st.session_state["last_batch_reports"] = list(zip(parsed, reports))

                        except Exception as exc:
                            st.error(f"Batch evaluation failed: {exc}")
                            import traceback
                            with st.expander("Error details"):
                                st.code(traceback.format_exc())

    # --- Strategy Comparison Tab ---
    with tab_compare:
        st.subheader("Strategy Evaluation Comparison")
        st.caption("Compare evaluation scores across different RAG strategy configurations")

        st.markdown("""
        Run a **Strategy Lab** comparison first, then click **Evaluate All Strategies** there.
        Cached results from your last evaluation will appear here for detailed comparison.
        """)

        # Check if we have cached strategy eval results
        cached_evals = {
            k: v for k, v in st.session_state.items()
            if k.startswith("eval_result_") and isinstance(v, dict)
        }

        if not cached_evals:
            st.info("No cached evaluation results found. Run a comparison in the **Strategy Lab** and click **🔬 Evaluate All Strategies**.")
        else:
            st.success(f"✅ Found {len(cached_evals)} cached strategy evaluation(s).")

            DISPLAY_KEYS = [
                ("faithfulness",       "Faithfulness"),
                ("answer_relevance",   "Answer Relevance"),
                ("context_precision",  "Context Precision"),
                ("context_recall",     "Context Recall"),
                ("groundedness",       "Groundedness"),
                ("coherence",          "Coherence"),
                ("conciseness",        "Conciseness"),
                ("overall_score",      "Overall Score"),
            ]

            # Build comparison table
            table_data: Dict[str, List] = {"Metric": [d for _, d in DISPLAY_KEYS]}
            sorted_keys = sorted(cached_evals.keys())
            for sk in sorted_keys:
                report = cached_evals[sk]
                strategy_name = sk.replace("eval_result_", "Strategy ")
                col_vals = []
                for key, _ in DISPLAY_KEYS:
                    if key == "overall_score":
                        s = report.get("overall_score")
                    else:
                        m = report.get(key, {})
                        s = m.get("score") if isinstance(m, dict) else None
                    col_vals.append(f"{s:.0%}" if s is not None else "N/A")
                table_data[strategy_name] = col_vals

            try:
                import pandas as pd
                df = pd.DataFrame(table_data)
                st.dataframe(df.set_index("Metric"), use_container_width=True)
            except ImportError:
                st.json(table_data)

            # Visual bar chart comparison
            try:
                import pandas as pd
                import plotly.graph_objects as go

                metric_names = [d for _, d in DISPLAY_KEYS[:-1]]  # exclude Overall from chart
                metric_keys = [k for k, _ in DISPLAY_KEYS[:-1]]

                fig = go.Figure()
                for sk in sorted_keys:
                    report = cached_evals[sk]
                    strategy_name = sk.replace("eval_result_", "Strategy ")
                    scores = []
                    for mk in metric_keys:
                        m = report.get(mk, {})
                        s = m.get("score") if isinstance(m, dict) else None
                        scores.append(s * 100 if s is not None else 0)
                    fig.add_trace(go.Bar(name=strategy_name, x=metric_names, y=scores))

                fig.update_layout(
                    barmode="group",
                    title="Strategy Evaluation Comparison",
                    yaxis_title="Score (%)",
                    yaxis=dict(range=[0, 100]),
                    legend_title="Strategy",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.info("Install `plotly` and `pandas` for visual comparison charts.")

            # Radar chart
            try:
                import plotly.graph_objects as go

                radar_keys = [k for k, _ in DISPLAY_KEYS[:-1]]
                radar_labels = [d for _, d in DISPLAY_KEYS[:-1]]
                radar_labels_closed = radar_labels + [radar_labels[0]]

                fig_radar = go.Figure()
                for sk in sorted_keys:
                    report = cached_evals[sk]
                    strategy_name = sk.replace("eval_result_", "Strategy ")
                    scores = []
                    for mk in radar_keys:
                        m = report.get(mk, {})
                        s = m.get("score") if isinstance(m, dict) else None
                        scores.append(s if s is not None else 0)
                    scores_closed = scores + [scores[0]]
                    fig_radar.add_trace(go.Scatterpolar(
                        r=scores_closed,
                        theta=radar_labels_closed,
                        fill="toself",
                        name=strategy_name,
                    ))

                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=True,
                    title="Radar: Strategy Evaluation Profile",
                    height=420,
                )
                st.plotly_chart(fig_radar, use_container_width=True)
            except ImportError:
                pass

            if st.button("🗑️ Clear Cached Evaluations", key="clear_evals"):
                for k in list(st.session_state.keys()):
                    if k.startswith("eval_result_"):
                        del st.session_state[k]
                st.rerun()


# --- Main Navigation ---


with st.sidebar:
    st.title("🧪 RAG Strategy Lab")
    st.markdown("---")
    
    # Show environment warnings/errors
    if env_errors:
        for error in env_errors:
            st.error(f"⚠️ {error}")
    
    if env_warnings:
        for warning in env_warnings:
            st.warning(f"ℹ️ {warning}")
    
    if not IMPORTS_SUCCESSFUL:
        st.error(f"❌ Import Error: {IMPORT_ERROR}")
        st.info("Some dependencies may be missing. Run: `pip install -r requirements-advanced.txt`")
    
    st.markdown("""
    ### 👋 Welcome!
    This tool helps you learn and compare advanced RAG strategies.
    
    **Quick Start:**
    1. Upload or select documents
    2. Configure ingestion settings
    3. Test different RAG strategies
    4. Evaluate with Evaluation Lab
    """)
    
    st.markdown("---")
    
    page = st.radio("Navigation", ["Strategies", "Ingestion Lab", "Strategy Lab", "Evaluation Lab"], index=0)
    st.divider()
    st.button("🎨 Toggle Theme", on_click=toggle_theme, use_container_width=True)
    st.caption(f"Current Theme: **{st.session_state.theme.title()}**")

if page == "Strategies":
    render_learning_page()
elif page == "Ingestion Lab":
    render_ingestion_page()
elif page == "Strategy Lab":
    render_retrieval_page()
else:
    render_evaluation_page()
