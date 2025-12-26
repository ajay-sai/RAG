import asyncio
import pytest  # type: ignore[import]
from app import StrategyConfig, execute_pipeline

async def dummy_hybrid(ctx, query, limit=5):
    return {"formatted": "Found 1 hybrid result.", "meta": {"total_tokens": 123, "dense_count": 4}}

async def dummy_self_reflection(ctx, query, limit=5):
    return {"formatted": "Found 1 self-reflection result.", "meta": {"total_tokens": 45, "grade_score": 4}}

import pytest  # type: ignore[import]

@pytest.mark.asyncio
async def test_hybrid_total_tokens(monkeypatch):
    # Patch the app-level function with an async callable so it can be awaited
    monkeypatch.setattr('app.search_with_hybrid_retrieval_meta', dummy_hybrid)

    config = StrategyConfig(
        name='Hybrid Test',
        retrieval_method='Hybrid (Vector + BM25)',
        reranking=False,
        llm_model='gpt-4o-mini',
        generation_style='Standard',
        chunking_strategy='semantic'
    )

    res = await execute_pipeline(config, 'test hybrid')
    assert res['status'] == 'Success'
    assert 'meta' in res
    assert res['meta'].get('total_tokens') == 123

@pytest.mark.asyncio
async def test_self_reflection_total_tokens(monkeypatch):
    # Patch the app-level function with an async callable so it can be awaited
    monkeypatch.setattr('app.search_with_self_reflection_meta', dummy_self_reflection)

    config = StrategyConfig(
        name='SelfTest',
        retrieval_method='Self-Reflective RAG',
        reranking=False,
        llm_model='gpt-4o-mini',
        generation_style='Standard',
        chunking_strategy='semantic'
    )

    res = await execute_pipeline(config, 'test self')
    assert res['status'] == 'Success'
    assert 'meta' in res
    assert res['meta'].get('total_tokens') == 45
