import streamlit as st
import asyncio
import time
import os
import sys
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

# Import backend logic
try:
    from rag_agent_advanced import (
        initialize_db,
        close_db,
        search_knowledge_base,
        search_with_multi_query,
        search_with_hybrid_retrieval,
        search_with_reranking,
        search_with_self_reflection,
    )
    from ingestion.ingest import DocumentIngestionPipeline
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

if 'theme' not in st.session_state:
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
    """Executes a RAG pipeline based on the configuration."""
    start_time = time.time()
    try:
        # 1. Retrieval Phase
        if config.retrieval_method == "Vector Search (Baseline)":
            retrieval_result = await search_knowledge_base(None, query, limit=5)
        elif config.retrieval_method == "Multi-Query":
            retrieval_result = await search_with_multi_query(None, query, limit=5)
        elif config.retrieval_method == "Hybrid (Vector + BM25)":
            retrieval_result = await search_with_hybrid_retrieval(None, query, limit=5)
        elif config.retrieval_method == "Self-Reflective RAG":
            retrieval_result = await search_with_self_reflection(None, query, limit=5)
        else:
            retrieval_result = await search_knowledge_base(None, query, limit=5)

        # 2. Reranking (only applies to Vector Search baseline)
        if config.reranking and config.retrieval_method == "Vector Search (Baseline)":
            retrieval_result = await search_with_reranking(None, query, limit=5)
        
        # 3. Generation
        final_output = retrieval_result
        if config.generation_style == "Fact Verification":
            from rag_agent_advanced import answer_with_fact_verification
            final_output = await answer_with_fact_verification(None, query)
        elif config.generation_style == "Multi-Hop Reasoning":
            from rag_agent_advanced import answer_with_multi_hop
            final_output = await answer_with_multi_hop(None, query)
        elif config.generation_style == "Uncertainty Estimation":
            from rag_agent_advanced import answer_with_uncertainty
            final_output = await answer_with_uncertainty(None, query)

        duration = (time.time() - start_time) * 1000
        return {
            "status": "Success",
            "output": final_output,
            "duration": duration,
            "cost_label": estimate_cost(config),
            "name": config.name
        }
    except Exception as e:
        return {
            "status": "Error",
            "error": str(e),
            "duration": (time.time() - start_time) * 1000,
            "name": config.name
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
                        docs_dir = "documents"
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
            docs_dir = "documents"
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
                            documents_folder="documents",
                            clean_before_ingest=True
                        )
                        
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
                            st.markdown(f"""
                            <article class="strategy-container" aria-label="Results for {res['name']}">
                                <header style="margin-bottom: 15px;">
                                    <h3 style="margin: 0; font-size: 1.2em;">{res['name']}</h3>
                                    <div class="metric-container" style="margin-top: 8px;">
                                        <span class="metric-tag" title="Execution Time">⏱️ {res['duration']:.0f} ms</span>
                                        <span class="metric-tag" title="Estimated Cost">{res['cost_label']}</span>
                                    </div>
                                </header>
                                <section class="result-box" role="region" aria-label="Output content">
                                    {res['output']}
                                </section>
                            </article>
                            """, unsafe_allow_html=True)
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
    
    page = st.radio("Navigation", ["Ingestion Lab", "Strategy Lab"], index=0)
    st.divider()
    st.button("🎨 Toggle Theme", on_click=toggle_theme, use_container_width=True)
    st.caption(f"Current Theme: **{st.session_state.theme.title()}**")

if page == "Ingestion Lab":
    render_ingestion_page()
else:
    render_retrieval_page()
