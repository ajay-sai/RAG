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
import asyncio
import time
import os
import sys
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

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
        # Check if Docker is available
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=rag_postgres", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if "rag_postgres" not in result.stdout:
            # PostgreSQL not running, start it
            compose_path = os.path.join(os.path.dirname(__file__), "docker-compose.yml")
            if os.path.exists(compose_path):
                subprocess.run(
                    ["docker-compose", "up", "-d", "postgres"],
                    cwd=os.path.dirname(__file__),
                    capture_output=True,
                    timeout=30
                )
                # Wait for PostgreSQL to be ready
                time.sleep(3)
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

        # 2. Generation phase (if any)
        generation_start = time.time()
        final_output = formatted

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

        return {
            "status": "Success",
            "output": cleaned_output,
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
                                top_sources_html = f"<div style=\"margin-top:8px;color:var(--metric-text)\"><strong>Top Sources:</strong> {sample}</div>"

                            # Display card
                            st.markdown(f"""
                            <article class="strategy-container" aria-label="Results for {res['name']}">
                                <header style="margin-bottom: 15px;">
                                    <h3 style="margin: 0; font-size: 1.2em;">{res['name']}</h3>
                                    <div class="metric-container" style="margin-top: 8px;">{metrics_html}</div>
                                    {top_sources_html}
                                </header>
                                <section class="result-box" role="region" aria-label="Output content">
                                    {res['output']}
                                </section>
                            </article>
                            """, unsafe_allow_html=True)

                            # Detailed metadata expander
                            with st.expander("Show detailed metadata and traces"):
                                import json
                                st.subheader("Metadata")
                                st.code(json.dumps(meta, indent=2))

                        else:
                            st.error(f"Error: {res.get('error')}")

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
    """)
    
    st.markdown("---")
    
    page = st.radio("Navigation", ["Strategies", "Ingestion Lab", "Strategy Lab"], index=0)
    st.divider()
    st.button("🎨 Toggle Theme", on_click=toggle_theme, use_container_width=True)
    st.caption(f"Current Theme: **{st.session_state.theme.title()}**")

if page == "Strategies":
    render_learning_page()
elif page == "Ingestion Lab":
    render_ingestion_page()
else:
    render_retrieval_page()
