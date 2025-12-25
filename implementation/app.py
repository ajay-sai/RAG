import streamlit as st
import asyncio
import time
import os
import sys
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# Add implementation directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import backend logic
from rag_agent_advanced import (
    initialize_db,
    close_db,
    db_pool,
    initialize_reranker,
    initialize_bm25,
    search_knowledge_base,
    search_with_multi_query,
    search_with_hybrid_retrieval,
    search_with_reranking,
    search_with_self_reflection,
)
from ingestion.ingest import DocumentIngestionPipeline
from utils.models import IngestionConfig
from utils.config_manager import save_active_config, load_active_config

# Page config
st.set_page_config(
    page_title="RAG Strategy Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Theme & Styling ---

if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

def toggle_theme():
    st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'

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

    .strategy-container {{
        border: 1px solid var(--card-border);
        border-radius: 10px;
        padding: 20px;
        background-color: var(--card-bg);
        margin-bottom: 20px;
        box-shadow: var(--shadow);
        color: var(--text-color);
        transition: all 0.3s ease;
    }}
    
    .result-box {{
        background-color: var(--result-bg);
        padding: 15px;
        border-radius: 8px;
        margin-top: 10px;
        border-left: 4px solid var(--result-border-left);
        color: var(--text-color);
    }}
    
    .metric-tag {{
        background-color: var(--metric-bg);
        color: var(--metric-text);
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.85em;
        margin-right: 8px;
        font-weight: 600;
        display: inline-block;
    }}

    /* Accessibility: Focus states */
    button:focus, input:focus, textarea:focus, select:focus {{
        outline: 2px solid #4CAF50;
        outline-offset: 2px;
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
    st.header("📥 Ingestion Lab")
    st.write("Configure how documents are processed, chunked, and embedded.")
    
    with st.container(border=True):
        st.subheader("Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.slider(
                "Chunk Size (tokens)", 
                min_value=100, max_value=2000, value=1000, step=100,
                help="Target size for each document chunk."
            )
            chunk_overlap = st.slider(
                "Chunk Overlap", 
                min_value=0, max_value=500, value=200, step=50,
                help="Number of overlapping tokens between adjacent chunks."
            )
            
        with col2:
            chunker_type = st.selectbox(
                "Chunker Type",
                ["semantic", "fixed", "adaptive"],
                index=0,
                help="Algorithm used to split text."
            )
            embedding_model = st.selectbox(
                "Embedding Model",
                ["text-embedding-3-small", "text-embedding-3-large"],
                index=0,
                help="Model used to generate vector embeddings."
            )
            
        contextual = st.checkbox(
            "Use Contextual Enrichment", 
            value=False,
            help="Uses an LLM to prepend document context to each chunk (Higher cost, better retrieval)."
        )

        if st.button("🔄 Run Ingestion Pipeline", type="primary"):
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
            st.info("Starting ingestion... This will clear existing data.")
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
                    
                    return await pipeline.ingest_documents(progress_callback=update_progress)

                import rag_agent_advanced
                rag_agent_advanced.db_pool = None # Reset pool
                
                results = asyncio.run(run_pipeline())
                
                st.success(f"Ingestion Complete! Processed {len(results)} documents.")
                st.json([{"file": r.title, "chunks": r.chunks_created} for r in results])
                
            except Exception as e:
                st.error(f"Ingestion Failed: {e}")

# --- Page: Retrieval Lab ---

def render_retrieval_page():
    st.header("🧪 Strategy Lab")
    st.markdown("Design and compare up to 3 RAG strategies side-by-side.")

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
                    # Informational only, based on active config
                    active_conf = load_active_config()
                    st.text_input(
                        "Chunking", 
                        value=active_conf.get("chunker_type", "semantic"), 
                        disabled=True, 
                        key=f"chunk_{i}"
                    )
                    st.text_input(
                        "Embedding",
                        value=active_conf.get("embedding_model", "small"),
                        disabled=True,
                        key=f"embed_{i}"
                    )
                with c2:
                    retrieval = st.selectbox(
                        "Retrieval",
                        ["Vector Search (Baseline)", "Multi-Query", "Hybrid (Vector + BM25)", "Self-Reflective RAG"],
                        key=f"retrieval_{i}",
                        help="Algorithm for finding relevant information."
                    )
                    rerank = st.checkbox("Reranking", key=f"rerank_{i}", help="Enable Cross-Encoder reranking.")

                llm = st.selectbox("LLM", ["gpt-4o-mini", "gpt-4o"], key=f"llm_{i}")
                gen_style = st.selectbox(
                    "Generation",
                    ["Standard", "Fact Verification", "Multi-Hop Reasoning", "Uncertainty Estimation"],
                    key=f"gen_{i}"
                )
                
                configs.append(StrategyConfig(
                    name=f"Strategy {i + 1}",
                    retrieval_method=retrieval,
                    reranking=rerank,
                    llm_model=llm,
                    generation_style=gen_style,
                    chunking_strategy=active_conf.get("chunker_type", "semantic")
                ))

    # Run Button
    if st.button("🚀 Run Comparison", type="primary", use_container_width=True):
        if not user_query:
            st.warning("Please enter a query first.")
        else:
            import rag_agent_advanced
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
    page = st.radio("Navigation", ["Retrieval Lab", "Ingestion Lab"], index=0)
    st.divider()
    st.button("Toggle Theme", on_click=toggle_theme)
    st.write(f"Theme: **{st.session_state.theme.title()}**")

if page == "Ingestion Lab":
    render_ingestion_page()
else:
    render_retrieval_page()
