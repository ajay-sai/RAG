"""
Advanced RAG CLI Agent with Multiple Strategies
===============================================
Implements multiple RAG strategies:
- Query Expansion
- Re-ranking
- Agentic RAG (semantic search + full file retrieval)
- Multi-Query RAG
- Self-Reflective RAG
- Context-aware chunking (via Docling HybridChunker - already in ingestion)
"""

import asyncio
try:
    import asyncpg  # type: ignore[import]
except Exception:
    asyncpg = None
import logging
import os
import sys
from typing import Any, List, Dict
from dataclasses import dataclass

from dotenv import load_dotenv
try:
    from pydantic_ai import Agent, RunContext  # type: ignore[import]
except Exception:
    # pydantic_ai is optional in test environments
    class Agent:
        def __init__(self, *a, **k):
            pass
    class RunContext:
        @classmethod
        def __class_getitem__(cls, item):
            return cls
        def __init__(self, *a, **k):
            pass
try:
    from sentence_transformers import CrossEncoder  # type: ignore[import]
except Exception:
    class CrossEncoder:
        def __init__(self, *a, **k):
            pass
        def predict(self, pairs):
            return [0.0] * len(pairs)

try:
    from rank_bm25 import BM25Okapi  # type: ignore[import]
except Exception:
    class BM25Okapi:
        def __init__(self, corpus):
            self._corpus = corpus
        def get_scores(self, tokenized_query):
            return [0.0] * len(self._corpus)
import numpy as np

# Load environment variables
load_dotenv(".env")

# Export placeholder names for tests that monkeypatch module-level symbols
# Tests often patch 'rag_agent_advanced.create_embedder' and 'rag_agent_advanced.AsyncOpenAI'
# Provide simple placeholders so monkeypatch.setattr works during test runs.
create_embedder = None
class AsyncOpenAI:
    def __init__(self, *a, **k):
        raise RuntimeError("AsyncOpenAI placeholder used without monkeypatch in tests")


logger = logging.getLogger(__name__)

# Global database pool
db_pool = None


def _extract_total_tokens(resp) -> int | None:
    """Extract total token count from common response shapes.

    Supports OpenAI-like response objects and dict-style responses.
    Returns None if token usage is not available.
    """
    try:
        # OpenAI style: resp.usage.total_tokens
        if hasattr(resp, 'usage') and getattr(resp.usage, 'total_tokens', None) is not None:
            return int(getattr(resp.usage, 'total_tokens'))
    except Exception:
        pass

    try:
        # dict-like: resp['usage']['total_tokens']
        if isinstance(resp, dict) and resp.get('usage') and isinstance(resp['usage'], dict):
            if 'total_tokens' in resp['usage']:
                return int(resp['usage']['total_tokens'])
    except Exception:
        pass

    try:
        # resp.get('usage', {}).get('total_tokens') fallback
        usage = getattr(resp, 'get', lambda k, d=None: None)('usage')
        if isinstance(usage, dict) and 'total_tokens' in usage:
            return int(usage['total_tokens'])
    except Exception:
        pass

    return None

# Initialize cross-encoder for re-ranking
reranker = None

# In-memory cache for BM25
bm25_index = None
bm25_chunks = None


async def initialize_bm25():
    """Initialize in-memory BM25 index."""
    global bm25_index, bm25_chunks
    if bm25_index is None:
        logger.info("Initializing BM25 index...")
        if not db_pool:
            await initialize_db()

        async with db_pool.acquire() as conn:
            chunks_records = await conn.fetch("SELECT id::text as chunk_id, content FROM chunks")
        
        if not chunks_records:
            logger.warning("No chunks found in DB to build BM25 index.")
            bm25_chunks = []
            return

        bm25_chunks = [
            {"chunk_id": str(r["chunk_id"]), "content": r["content"]}
            for r in chunks_records
        ]
        
        tokenized_corpus = [doc["content"].split(" ") for doc in bm25_chunks]
        bm25_index = BM25Okapi(tokenized_corpus)
        logger.info(f"BM25 index initialized with {len(bm25_chunks)} documents.")


async def initialize_db():
    """Initialize database connection pool."""
    global db_pool
    if not db_pool:
        db_pool = await asyncpg.create_pool(
            os.getenv("DATABASE_URL"),
            min_size=2,
            max_size=10,
            command_timeout=60
        )
        logger.info("Database connection pool initialized")


async def close_db():
    """Close database connection pool."""
    global db_pool
    if db_pool:
        await db_pool.close()
        logger.info("Database connection pool closed")


def initialize_reranker():
    """Initialize cross-encoder model for re-ranking."""
    global reranker
    if reranker is None:
        logger.info("Loading cross-encoder model for re-ranking...")
        reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        logger.info("Cross-encoder loaded")


# ======================
# STRATEGY 1: QUERY EXPANSION
# ======================

async def expand_query_variations(
    ctx: RunContext[None], query: str
) -> List[str]:
    """
    Generate multiple variations of a query for better retrieval.

    Args:
        query: Original search query

    Returns:
        List of query variations including the original
    """
    from openai import AsyncOpenAI  # type: ignore[import]
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    expansion_prompt = f"""Generate 3 different variations of this search query.
Each variation should capture a different perspective or phrasing while maintaining
the same intent.

Original query: {query}

Return only the 3 variations, one per line, without numbers or bullets."""

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": expansion_prompt}],
            temperature=0.7
        )

        variations_text = response.choices[0].message.content.strip()
        variations = [v.strip() for v in variations_text.split('\n') if v.strip()]

        # Return original + variations
        return [query] + variations[:3]

    except Exception as e:
        logger.error(f"Query expansion failed: {e}")
        return [query]  # Fallback to original query


# ======================
# STRATEGY 2 & 3: MULTI-QUERY RAG (parallel search with variations)
# ======================

async def search_single_query(query: str, limit: int):
    """Helper for parallel execution to avoid shared connection issues."""
    from ingestion.embedder import create_embedder
    embedder = create_embedder()
    
    query_embedding = await embedder.embed_query(query)
    embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
    
    async with db_pool.acquire() as conn:
        return await conn.fetch(
            """
            SELECT * FROM match_chunks($1::vector, $2)
            """,
            embedding_str,
            limit
        )

async def search_with_multi_query(
    ctx: RunContext[None], query: str, limit: int = 5
) -> str:
    """
    Search using multiple query variations in parallel (Multi-Query RAG).

    This combines query expansion with parallel execution for better recall.

    Args:
        query: The search query
        limit: Results per query variation

    Returns:
        Formatted deduplicated search results
    """
    try:
        if not db_pool:
            await initialize_db()

        # Generate query variations
        queries = await expand_query_variations(ctx, query)
        logger.info(f"Multi-query search with {len(queries)} variations")

        # Execute searches in parallel using helper
        search_tasks = [search_single_query(q, limit) for q in queries]
        results_lists = await asyncio.gather(*search_tasks)

        # Collect all results
        all_results = []
        for results in results_lists:
            all_results.extend(results)

        if not all_results:
            return "No relevant information found."

        # Deduplicate by chunk ID and keep highest similarity
        seen = {}
        for row in all_results:
            chunk_id = row['chunk_id']
            if chunk_id not in seen or row['similarity'] > seen[chunk_id]['similarity']:
                seen[chunk_id] = row

        unique_results = sorted(
            seen.values(), key=lambda x: x['similarity'], reverse=True
        )[:limit]

        # Format results
        response_parts = []
        for i, row in enumerate(unique_results, 1):
            response_parts.append(
                f"[Source: {row['document_title']}]\n{row['content']}\n"
            )

        return (
            f"Found {len(response_parts)} relevant results:\n\n"
            + "\n---\n".join(response_parts)
        )

    except Exception as e:
        logger.error(f"Multi-query search failed: {e}", exc_info=True)
        return f"Search error: {str(e)}"


async def search_with_multi_query_meta(ctx: RunContext[None], query: str, limit: int = 5):
    """Return formatted multi-query results plus metadata."""
    try:
        if not db_pool:
            await initialize_db()
        queries = await expand_query_variations(ctx, query)
        logger.info(f"Multi-query (meta) with {len(queries)} variations")
        search_tasks = [search_single_query(q, limit) for q in queries]
        results_lists = await asyncio.gather(*search_tasks)
        all_results = []
        for results in results_lists:
            all_results.extend(results)
        if not all_results:
            return {"formatted": "No relevant information found.", "meta": {"queries": queries, "total_results": 0}}
        seen = {}
        for row in all_results:
            chunk_id = row['chunk_id']
            if chunk_id not in seen or row['similarity'] > seen[chunk_id]['similarity']:
                seen[chunk_id] = row
        unique_results = sorted(seen.values(), key=lambda x: x['similarity'], reverse=True)[:limit]
        response_parts = []
        top_sources = []
        for row in unique_results:
            response_parts.append(f"[Source: {row['document_title']}\n]{row['content']}\n")
            top_sources.append(row.get('document_title'))
        formatted = (f"Found {len(response_parts)} relevant results:\n\n" + "\n---\n".join(response_parts))
        meta = {"queries": queries, "requested_limit": limit, "returned": len(response_parts), "top_sources": top_sources}
        return {"formatted": formatted, "meta": meta}
    except Exception as e:
        logger.error(f"Multi-query (meta) failed: {e}", exc_info=True)
        return {"formatted": f"Search error: {str(e)}", "meta": {"error": str(e)}}


async def search_with_reranking_meta(ctx: RunContext[None], query: str, limit: int = 5):
    """Return reranking formatted results plus metadata."""
    try:
        if not db_pool:
            await initialize_db()
        initialize_reranker()
        from ingestion.embedder import create_embedder
        embedder = create_embedder()
        query_embedding = await embedder.embed_query(query)
        embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
        candidate_limit = min(limit * 4, 20)
        async with db_pool.acquire() as conn:
            results = await conn.fetch(
                """
                SELECT * FROM match_chunks($1::vector, $2)
                """,
                embedding_str,
                candidate_limit
            )
        if not results:
            return {"formatted": "No relevant information found.", "meta": {"candidates": 0}}
        pairs = [[query, row['content']] for row in results]
        scores = reranker.predict(pairs)
        reranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)[:limit]
        response_parts = []
        top_sources = []
        rerank_scores = []
        for row, score in reranked:
            response_parts.append(f"[Source: {row['document_title']} | Relevance: {score:.2f}]\n{row['content']}\n")
            top_sources.append(row['document_title'])
            rerank_scores.append(float(score))
        formatted = (f"Found {len(response_parts)} highly relevant results:\n\n" + "\n---\n".join(response_parts))
        meta = {"candidates_considered": len(results), "returned": len(response_parts), "top_sources": top_sources, "rerank_scores": rerank_scores}
        return {"formatted": formatted, "meta": meta}
    except Exception as e:
        logger.error(f"Re-ranking (meta) failed: {e}", exc_info=True)
        return {"formatted": f"Search error: {str(e)}", "meta": {"error": str(e)}}


async def search_knowledge_base_meta(ctx: RunContext[None], query: str, limit: int = 5):
    """Return semantic search formatted results plus metadata."""
    try:
        if not db_pool:
            await initialize_db()
        from ingestion.embedder import create_embedder
        embedder = create_embedder()
        query_embedding = await embedder.embed_query(query)
        embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
        async with db_pool.acquire() as conn:
            results = await conn.fetch(
                """
                SELECT * FROM match_chunks($1::vector, $2)
                """,
                embedding_str,
                limit
            )
        if not results:
            return {"formatted": "No relevant information found in the knowledge base for your query.", "meta": {"returned": 0}}
        response_parts = []
        top_sources = []
        for i, row in enumerate(results, 1):
            response_parts.append(f"[Source: {row['document_title']}\n]{row['content']}\n")
            top_sources.append(row['document_title'])
        formatted = (f"Found {len(response_parts)} relevant results:\n\n" + "\n---\n".join(response_parts))
        meta = {"returned": len(response_parts), "top_sources": top_sources}
        return {"formatted": formatted, "meta": meta}
    except Exception as e:
        logger.error(f"Knowledge base search (meta) failed: {e}", exc_info=True)
        return {"formatted": f"Search error: {str(e)}", "meta": {"error": str(e)}}

# ======================
# STRATEGY 3: RE-RANKING
# ======================

async def search_with_reranking(
    ctx: RunContext[None], query: str, limit: int = 5
) -> str:
    """
    Two-stage retrieval: Fast vector search + precise cross-encoder re-ranking.

    Args:
        query: The search query
        limit: Final number of results to return after re-ranking

    Returns:
        Formatted re-ranked search results
    """
    try:
        if not db_pool:
            await initialize_db()

        initialize_reranker()

        # Stage 1: Fast vector retrieval (retrieve more candidates)
        from ingestion.embedder import create_embedder
        embedder = create_embedder()
        query_embedding = await embedder.embed_query(query)
        embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'

        # Retrieve 20 candidates for re-ranking
        candidate_limit = min(limit * 4, 20)

        async with db_pool.acquire() as conn:
            results = await conn.fetch(
                """
                SELECT * FROM match_chunks($1::vector, $2)
                """,
                embedding_str,
                candidate_limit
            )

        if not results:
            return "No relevant information found."

        # Stage 2: Re-rank with cross-encoder
        logger.info(f"Re-ranking {len(results)} candidates")

        pairs = [[query, row['content']] for row in results]
        scores = reranker.predict(pairs)

        # Combine results with new scores
        reranked = sorted(
            zip(results, scores),
            key=lambda x: x[1],
            reverse=True
        )[:limit]

        # Format results
        response_parts = []
        for i, (row, score) in enumerate(reranked, 1):
            response_parts.append(
                f"[Source: {row['document_title']} | "
                f"Relevance: {score:.2f}]\n{row['content']}\n"
            )

        return (
            f"Found {len(response_parts)} highly relevant results:\n\n"
            + "\n---\n".join(response_parts)
        )

    except Exception as e:
        logger.error(f"Re-ranking search failed: {e}", exc_info=True)
        return f"Search error: {str(e)}"


# ======================
# STRATEGY 4: AGENTIC RAG (Semantic Search + Full File Retrieval)
# ======================

async def search_knowledge_base(
    ctx: RunContext[None], query: str, limit: int = 5
) -> str:
    """
    Standard semantic search over chunks.

    Args:
        query: The search query
        limit: Maximum number of results

    Returns:
        Formatted search results
    """
    try:
        if not db_pool:
            await initialize_db()

        from ingestion.embedder import create_embedder
        embedder = create_embedder()
        query_embedding = await embedder.embed_query(query)
        embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'

        async with db_pool.acquire() as conn:
            results = await conn.fetch(
                """
                SELECT * FROM match_chunks($1::vector, $2)
                """,
                embedding_str,
                limit
            )

        if not results:
            return "No relevant information found in the knowledge base for your query."

        response_parts = []
        for i, row in enumerate(results, 1):
            response_parts.append(
                f"[Source: {row['document_title']}]\n{row['content']}\n"
            )

        return (
            f"Found {len(response_parts)} relevant results:\n\n"
            + "\n---\n".join(response_parts)
        )

    except Exception as e:
        logger.error(f"Knowledge base search failed: {e}", exc_info=True)
        return f"Search error: {str(e)}"


async def retrieve_full_document(
    ctx: RunContext[None], document_title: str
) -> str:
    """
    Retrieve the full content of a specific document by title.

    Use this when chunks don't provide enough context or when you need
    to see the complete document.

    Args:
        document_title: The title of the document to retrieve

    Returns:
        Full document content
    """
    try:
        if not db_pool:
            await initialize_db()

        async with db_pool.acquire() as conn:
            result = await conn.fetchrow(
                """
                SELECT title, content, source
                FROM documents
                WHERE title ILIKE $1
                LIMIT 1
                """,
                f"%{document_title}%"
            )

        if not result:
            # Try to list available documents
            async with db_pool.acquire() as conn:
                docs = await conn.fetch(
                    """
                    SELECT title FROM documents
                    ORDER BY created_at DESC
                    LIMIT 10
                    """
                )

            doc_list = "\n- ".join([doc['title'] for doc in docs])
            return (
                f"Document '{document_title}' not found. "
                f"Available documents:\n- {doc_list}"
            )

        return (
            f"**Document: {result['title']}**\n\n"
            f"Source: {result['source']}\n\n{result['content']}"
        )

    except Exception as e:
        logger.error(f"Full document retrieval failed: {e}", exc_info=True)
        return f"Error retrieving document: {str(e)}"


# ======================
# STRATEGY 5: SELF-REFLECTIVE RAG
# ======================

async def search_with_self_reflection(
    ctx: RunContext[None], query: str, limit: int = 5
) -> str:
    """
    Self-reflective search: evaluate results and refine if needed.

    This implements a simple self-reflection loop:
    1. Perform initial search
    2. Grade relevance of results
    3. If results are poor, refine query and search again

    Args:
        query: The search query
        limit: Number of results to return

    Returns:
        Formatted search results with reflection metadata
    """
    try:
        if not db_pool:
            await initialize_db()

        from openai import AsyncOpenAI  # type: ignore[import]
        from ingestion.embedder import create_embedder

        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        embedder = create_embedder()

        # Initial search
        query_embedding = await embedder.embed_query(query)
        embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'

        async with db_pool.acquire() as conn:
            results = await conn.fetch(
                """
                SELECT * FROM match_chunks($1::vector, $2)
                """,
                embedding_str,
                limit
            )

        if not results:
            return "No relevant information found."

        # Self-reflection: Grade relevance
        grade_prompt = f"""Query: {query}

Retrieved Documents:
{chr(10).join([f"{i+1}. {r['content'][:200]}..." for i, r in enumerate(results)])}

Grade the overall relevance of these documents to the query on a scale of 1-5:
1 = Not relevant at all
2 = Slightly relevant
3 = Moderately relevant
4 = Relevant
5 = Highly relevant

Respond with only a single number (1-5) and a brief reason."""

        try:
            grade_response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": grade_prompt}],
                temperature=0
            )

            grade_text = grade_response.choices[0].message.content.strip()
            grade_score = int(grade_text.split()[0])

        except Exception as e:
            logger.warning(f"Grading failed, proceeding with results: {e}")
            grade_score = 3  # Assume moderate relevance

        # If relevance is low, refine query
        if grade_score < 3:
            logger.info(f"Low relevance score ({grade_score}), refining query")

            refine_prompt = f"""The query "{query}" returned low-relevance results.
Suggest an improved, more specific query that might find better results.
Respond with only the improved query, nothing else."""

            try:
                refine_response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": refine_prompt}],
                    temperature=0.7
                )

                refined_query = refine_response.choices[0].message.content.strip()
                logger.info(f"Refined query: {refined_query}")

                # Search again with refined query
                refined_embedding = await embedder.embed_query(refined_query)
                refined_embedding_str = '[' + ','.join(map(str, refined_embedding)) + ']'

                async with db_pool.acquire() as conn:
                    results = await conn.fetch(
                        """
                        SELECT * FROM match_chunks($1::vector, $2)
                        """,
                        refined_embedding_str,
                        limit
                    )

                reflection_note = (
                    f"\n[Reflection: Refined query from '{query}' "
                    f"to '{refined_query}']\n"
                )

            except Exception as e:
                logger.warning(f"Query refinement failed: {e}")
                reflection_note = "\n[Reflection: Initial results had low relevance]\n"
        else:
            reflection_note = (
                f"\n[Reflection: Results deemed relevant (score: {grade_score}/5)]\n"
            )


        # Format final results
        response_parts = []
        for i, row in enumerate(results, 1):
            response_parts.append(
                f"[Source: {row['document_title']}]\n{row['content']}\n"
            )

        return (
            reflection_note + f"Found {len(response_parts)} results:\n\n"
            + "\n---\n".join(response_parts)
        )

    except Exception as e:
        logger.error(f"Self-reflective search failed: {e}", exc_info=True)
        return f"Search error: {str(e)}"


async def search_with_self_reflection_meta(ctx: RunContext[None], query: str, limit: int = 5):
    """Wrapper returning formatted results and metadata for self-reflective search."""
    try:
        # We'll re-run the existing function but instrument token usage where possible
        from openai import AsyncOpenAI  # type: ignore[import]
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Run the existing flow but capture grade and refine usages
        if not db_pool:
            await initialize_db()

        # Run a shallow copy of core logic to capture tokens
        from ingestion.embedder import create_embedder
        embedder = create_embedder()

        # Initial search
        query_embedding = await embedder.embed_query(query)

        # Use existing search to get results
        async with db_pool.acquire() as conn:
            results = await conn.fetch(
                """
                SELECT * FROM match_chunks($1::vector, $2)
                """,
                '[' + ','.join(map(str, query_embedding)) + ']',
                limit
            )

        meta = {
            "returned": len(results),
            "embedding_tokens": getattr(embedder, 'last_usage', None)
        }

        if not results:
            return {"formatted": "No relevant information found.", "meta": meta}

        # Grade relevance using LLM (capture usage)
        docs_list = [f"{i+1}. {r['content'][:200]}..." for i, r in enumerate(results)]
        docs_joined = "\n".join(docs_list)
        grade_prompt = (
            f"Query: {query}\n\nRetrieved Documents:\n{docs_joined}\n\n"
            "Grade the overall relevance of these documents to the query on a scale of 1-5:"
        )
        grade_res = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": grade_prompt}],
            temperature=0
        )
        grade_tokens = _extract_total_tokens(grade_res)
        try:
            grade_score = int(grade_res.choices[0].message.content.strip().split()[0])
        except Exception:
            grade_score = None

        meta.update({"grade_score": grade_score, "grade_tokens": grade_tokens})

        # If low score, attempt refine and capture tokens
        refined_query = None
        refine_tokens = None
        if grade_score is not None and grade_score < 3:
            refine_prompt = f"The query \"{query}\" returned low-relevance results. Suggest an improved, more specific query."
            refine_res = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": refine_prompt}],
                temperature=0.7
            )
            refine_tokens = _extract_total_tokens(refine_res)
            refined_query = refine_res.choices[0].message.content.strip()
            meta.update({"refined_query": refined_query, "refine_tokens": refine_tokens})

            # Compute aggregated token totals where possible
            token_components = []
            if isinstance(meta.get('embedding_tokens'), int):
                token_components.append(int(meta['embedding_tokens']))
            if isinstance(meta.get('grade_tokens'), int):
                token_components.append(int(meta['grade_tokens']))
            if isinstance(meta.get('refine_tokens'), int):
                token_components.append(int(meta['refine_tokens']))
            if token_components:
                meta['total_tokens'] = sum(token_components)
                meta['tokens_breakdown'] = {
                    'embedding_tokens': meta.get('embedding_tokens'),
                    'grade_tokens': meta.get('grade_tokens'),
                    'refine_tokens': meta.get('refine_tokens')
                }

        # Format results
        response_parts = [f"[Source: {r['document_title']}]\n{r['content']}\n" for r in results]
        formatted = (f"Found {len(response_parts)} results (self-reflection).\n\n" + "\n---\n".join(response_parts))

        return {"formatted": formatted, "meta": meta}

    except Exception as e:
        logger.error(f"Self-reflective search (meta) failed: {e}", exc_info=True)
        return {"formatted": f"Search error: {str(e)}", "meta": {"error": str(e)}}

# ======================
# STRATEGY 6: HYBRID RETRIEVAL (Dense + Sparse)
# ======================

async def search_with_hybrid_retrieval(
    ctx: RunContext[None], query: str, limit: int = 10
) -> str:
    """
    Combines dense (vector) and sparse (BM25) retrieval.
    """
    try:
        if not db_pool:
            await initialize_db()
        if not bm25_index:
            await initialize_bm25()
        
        if not bm25_index:
            return "Hybrid search unavailable: Index could not be initialized (database might be empty)."

        # 1. Dense retrieval (from existing vector search)
        from ingestion.embedder import create_embedder
        embedder = create_embedder()
        query_embedding = await embedder.embed_query(query)
        embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'

        async with db_pool.acquire() as conn:
            dense_results = await conn.fetch(
                "SELECT *, 1 - (embedding <=> $1) as similarity FROM chunks ORDER BY similarity DESC LIMIT $2",
                embedding_str,
                limit * 2
            )

        # 2. Sparse retrieval (BM25)
        tokenized_query = query.split(" ")
        bm25_scores = bm25_index.get_scores(tokenized_query)
        top_n_indices = np.argsort(bm25_scores)[::-1][:limit * 2]
        
        sparse_results = [
            {**bm25_chunks[i], "score": bm25_scores[i]} for i in top_n_indices
        ]

        # 3. Merge and re-rank (Reciprocal Rank Fusion)
        fused_scores = {}
        k = 60  # RRF constant

        for i, doc in enumerate(dense_results):
            # Fix: Use 'id' from chunks table
            chunk_id = str(doc["id"])
            if chunk_id not in fused_scores:
                fused_scores[chunk_id] = 0
            fused_scores[chunk_id] += 1 / (k + i + 1)

        for i, doc in enumerate(sparse_results):
            chunk_id = doc["chunk_id"]
            if chunk_id not in fused_scores:
                fused_scores[chunk_id] = 0
            fused_scores[chunk_id] += 1 / (k + i + 1)

        sorted_fused = sorted(
            fused_scores.items(), key=lambda item: item[1], reverse=True
        )[:limit]
        
        # Fetch full chunk data for top results
        top_chunk_ids = [item[0] for item in sorted_fused]
        if not top_chunk_ids:
            return "No relevant information found."

        async with db_pool.acquire() as conn:
            unique_results = await conn.fetch(
                """
                SELECT c.*, d.title as document_title 
                FROM chunks c 
                JOIN documents d ON c.document_id = d.id 
                WHERE c.id = ANY($1::uuid[])
                """, 
                top_chunk_ids
            )

        response_parts = [
            f"[Source: {r['document_title']}]\n{r['content']}\n"
            for r in unique_results
        ]
        return (
            f"Found {len(response_parts)} results via hybrid search:\n\n"
            + "\n---\n".join(response_parts)
        )

    except Exception as e:
        logger.error(f"Hybrid search failed: {e}", exc_info=True)
        return f"Search error: {str(e)}"


async def search_with_hybrid_retrieval_meta(ctx: RunContext[None], query: str, limit: int = 10):
    """Wrapper that returns formatted results and metadata for hybrid retrieval."""
    try:
        if not db_pool:
            await initialize_db()
        if not bm25_index:
            await initialize_bm25()

        if not bm25_index:
            return {"formatted": "Hybrid search unavailable: Index could not be initialized.", "meta": {"available": False}}

        # Dense retrieval (capture embedding token usage)
        from ingestion.embedder import create_embedder
        embedder = create_embedder()
        query_embedding = await embedder.embed_query(query)
        embedding_tokens = getattr(embedder, 'last_usage', None)
        embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'

        async with db_pool.acquire() as conn:
            dense_results = await conn.fetch(
                "SELECT *, 1 - (embedding <=> $1) as similarity FROM chunks ORDER BY similarity DESC LIMIT $2",
                embedding_str,
                limit * 2
            )

        dense_count = len(dense_results) if dense_results else 0

        # Sparse retrieval
        tokenized_query = query.split(" ")
        bm25_scores = bm25_index.get_scores(tokenized_query)
        # Guard against mismatched sizes or empty BM25 chunks (stubbed in tests)
        sparse_results = []
        sparse_count = 0
        if bm25_chunks and bm25_scores:
            # Ensure consistent lengths
            if len(bm25_scores) != len(bm25_chunks):
                min_len = min(len(bm25_scores), len(bm25_chunks))
                bm25_scores = bm25_scores[:min_len]
                top_n_indices = np.argsort(bm25_scores)[::-1][:limit * 2]
                sparse_results = [{**bm25_chunks[i], "score": float(bm25_scores[i])} for i in top_n_indices if i < len(bm25_chunks)]
            else:
                top_n_indices = np.argsort(bm25_scores)[::-1][:limit * 2]
                sparse_results = [{**bm25_chunks[i], "score": float(bm25_scores[i])} for i in top_n_indices]
            sparse_count = len(sparse_results)

        # Merge and RRF
        fused_scores = {}
        k = 60
        for i, doc in enumerate(dense_results):
            chunk_id = str(doc["id"])
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0) + 1 / (k + i + 1)

        for i, doc in enumerate(sparse_results):
            chunk_id = doc["chunk_id"]
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0) + 1 / (k + i + 1)

        sorted_fused = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)[:limit]
        candidates_considered = len(fused_scores)
        top_chunk_ids = [item[0] for item in sorted_fused]

        if not top_chunk_ids:
            return {"formatted": "No relevant information found.", "meta": {"dense_count": dense_count, "sparse_count": sparse_count, "embedding_tokens": embedding_tokens}}

        async with db_pool.acquire() as conn:
            unique_results = await conn.fetch(
                """
                SELECT c.*, d.title as document_title 
                FROM chunks c 
                JOIN documents d ON c.document_id = d.id 
                WHERE c.id = ANY($1::uuid[])
                """,
                top_chunk_ids
            )

        response_parts = [f"[Source: {r['document_title']}]\n{r['content']}\n" for r in unique_results]

        meta = {
            "dense_count": dense_count,
            "sparse_count": sparse_count,
            "candidates_considered": candidates_considered,
            "embedding_tokens": embedding_tokens,
            "top_sources": [r['document_title'] for r in unique_results][:5]
        }

        # Compute total_tokens explicitly (hybrid uses embeddings only)
        if isinstance(embedding_tokens, int):
            meta['total_tokens'] = int(embedding_tokens)

        return {"formatted": (f"Found {len(response_parts)} results via hybrid search:\n\n" + "\n---\n".join(response_parts)), "meta": meta}

    except Exception as e:
        logger.error(f"Hybrid search (meta) failed: {e}", exc_info=True)
        return {"formatted": f"Search error: {str(e)}", "meta": {"error": str(e)}}

# ======================
# STRATEGY 7: FACT VERIFICATION
# ======================

async def answer_with_fact_verification(ctx: RunContext[None], query: str) -> str:
    """
    Generates an answer and then verifies claims against retrieved evidence.
    """
    try:
        from openai import AsyncOpenAI  # type: ignore[import]
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # 1. Retrieve initial context
        context_str = await search_knowledge_base(ctx, query, limit=10)

        # 2. Generate an answer
        answer_prompt = f"Context:\n{context_str}\n\nQuestion: {query}\nAnswer:"
        answer_res = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": answer_prompt}]
        )
        answer = answer_res.choices[0].message.content.strip()

        # 3. Extract claims
        claim_prompt = (
            f"Answer: {answer}\n\nExtract factual claims from the "
            "answer above as a numbered list."
        )
        claim_res = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": claim_prompt}]
        )
        claims_str = claim_res.choices[0].message.content.strip()

        # 4. Verify claims
        verify_prompt = f"Context:\n{context_str}\n\nClaims:\n{claims_str}\n\nVerify each claim based *only* on the context. Respond with SUPPORTED, CONTRADICTED, or NEUTRAL for each."
        verify_res = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": verify_prompt}]
        )
        verification = verify_res.choices[0].message.content.strip()

        return f"{answer}\n\n---\nFact Verification:\n{verification}"

    except Exception as e:
        logger.error(f"Fact verification failed: {e}", exc_info=True)
        return f"Error during fact verification: {str(e)}"


# ======================
# STRATEGY 8: MULTI-HOP REASONING
# ======================

async def answer_with_multi_hop(
    ctx: RunContext[None], query: str, hops: int = 2
) -> str:
    """
    Answers complex questions by performing iterative retrieval.

    Args:
        query: The initial complex query
        hops: Number of retrieval-generation hops

    Returns:
        A comprehensive answer derived from multiple sources.
    """
    try:
        from openai import AsyncOpenAI  # type: ignore[import]
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        accumulated_context = []
        current_query = query
        reason_log = f"Multi-hop Reasoning Log for '{query}':\n"

        for i in range(hops):
            reason_log += f"\nHop {i+1}: Searching for '{current_query}'\n"
            
            # Retrieve documents
            results_str = await search_knowledge_base(ctx, current_query, limit=3)
            accumulated_context.append(results_str)
            
            # Generate next query
            if i < hops - 1:
                refine_prompt = f"""Based on the original question and the retrieved context, what is the next logical question to ask?

Original Question: {query}
Retrieved Context:
{" ".join(accumulated_context)}

Next Question:"""
                
                response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": refine_prompt}]
                )
                current_query = response.choices[0].message.content.strip()
                reason_log += f"Next query: {current_query}\n"

        # Final answer generation
        final_prompt = f"""Synthesize a comprehensive answer using all retrieved context.

Original Question: {query}
All Retrieved Context:
{" ".join(accumulated_context)}

Final Answer:"""
        
        final_response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": final_prompt}]
        )
        final_answer = final_response.choices[0].message.content.strip()

        return f"{reason_log}\n---\n{final_answer}"

    except Exception as e:
        logger.error(f"Multi-hop reasoning failed: {e}", exc_info=True)
        return f"Error during multi-hop reasoning: {str(e)}"


# ======================
# STRATEGY 9: UNCERTAINTY ESTIMATION
# ======================

async def answer_with_uncertainty(
    ctx: RunContext[None], query: str, num_responses: int = 3
) -> str:
    """
    Estimates uncertainty by generating multiple answers and checking for consistency.
    """
    try:
        from openai import AsyncOpenAI  # type: ignore[import]
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        from ingestion.embedder import create_embedder
        embedder = create_embedder()

        # 1. Retrieve context
        context = await search_knowledge_base(ctx, query, limit=5)

        # 2. Generate multiple responses
        responses = []
        for _ in range(num_responses):
            prompt = f"Based on this context, answer the query: '{query}'. Context: {context}"
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            responses.append(response.choices[0].message.content.strip())
        
        # 3. Estimate uncertainty using embedding similarity
        if len(responses) < 2:
            return f"Answer:\n{responses[0]}\n\n---\nUncertainty Score: N/A"

        # Fix: Use correct method name
        response_embeddings = await embedder.generate_embeddings_batch(responses)
        
        # Calculate cosine similarity between the first embedding and the rest
        similarities = [
            np.dot(response_embeddings[0], other_embedding) /
            (np.linalg.norm(response_embeddings[0]) * np.linalg.norm(other_embedding))
            for other_embedding in response_embeddings[1:]
        ]
        
        avg_similarity = sum(similarities) / len(similarities)
        uncertainty_score = 1.0 - avg_similarity

        return (
            f"{responses[0]}\n\n---\n"
            f"Uncertainty Score: {uncertainty_score:.2f} "
            "(0=confident, 1=uncertain)"
        )

    except Exception as e:
        logger.error(f"Uncertainty estimation failed: {e}", exc_info=True)
        return f"Error during uncertainty estimation: {str(e)}"


# ======================
# CREATE AGENT WITH ALL STRATEGIES
# ======================

agent = Agent(
    'openai:gpt-4o-mini',
    system_prompt="""You are an advanced knowledge assistant with multiple retrieval
strategies at your disposal.

AVAILABLE TOOLS:
1. search_knowledge_base - Standard semantic search over document chunks
2. retrieve_full_document - Get complete document when chunks aren't enough
3. search_with_multi_query - Use multiple query variations for better recall
4. search_with_reranking - Use two-stage retrieval for precision
5. search_with_self_reflection - Evaluate and refine search results automatically
6. search_with_hybrid_retrieval - Combine vector and keyword search (placeholder)
7. answer_with_fact_verification - Generate answer then verify claims (placeholder)
8. answer_with_multi_hop - Iterative retrieval for complex questions
9. answer_with_uncertainty - Estimate confidence by generating multiple answers

STRATEGY SELECTION GUIDE:
- Use search_knowledge_base for most queries (fast, reliable)
- Use retrieve_full_document when you need full context or found relevant chunks
- Use search_with_multi_query when query is ambiguous or could be interpreted
- Use search_with_reranking for precision-critical queries
- Use search_with_self_reflection for complex research questions
- Use search_with_hybrid_retrieval for queries with specific keywords
- Use answer_with_fact_verification for high-stakes domains requiring traceability
- Use answer_with_multi_hop for compositional questions
- Use answer_with_uncertainty when it's important to know model confidence

You can use multiple tools in sequence if needed. Be concise but thorough.""",
    tools=[
        search_knowledge_base,
        retrieve_full_document,
        search_with_multi_query,
        search_with_reranking,
        search_with_self_reflection,
        search_with_hybrid_retrieval,
        answer_with_fact_verification,
        answer_with_multi_hop,
        answer_with_uncertainty
    ]
)


async def run_cli():
    """Run the agent in an interactive CLI with streaming."""

    await initialize_db()

    print("=" * 60)
    print("Advanced RAG Knowledge Assistant")
    print("=" * 60)
    print("Multiple retrieval strategies available!")
    print("Type 'quit', 'exit', or press Ctrl+C to exit.")
    print("=" * 60)
    print()

    message_history = []

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nAssistant: Thank you for using the knowledge assistant. Goodbye!")
                break

            print("Assistant: ", end="", flush=True)

            try:
                async with agent.run_stream(
                    user_input,
                    message_history=message_history
                ) as result:
                    async for text in result.stream_text(delta=True):
                        print(text, end="", flush=True)

                    print()
                    message_history = result.all_messages()

            except KeyboardInterrupt:
                print("\n\n[Interrupted]")
                break
            except Exception as e:
                print(f"\n\nError: {e}")
                logger.error(f"Agent error: {e}", exc_info=True)

            print()

    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    finally:
        await close_db()


async def main():
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if not os.getenv("DATABASE_URL"):
        logger.error("DATABASE_URL environment variable is required")
        sys.exit(1)

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable is required")
        sys.exit(1)

    await run_cli()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nShutting down...")
