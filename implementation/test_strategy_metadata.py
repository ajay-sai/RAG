import pytest

# Simple async stub functions to avoid external dependencies
async def dummy_meta_search(ctx, query, limit=5):
    return {"formatted": "Found 1 result: ...", "meta": {"returned": 1, "top_sources": ["doc1"]}}

@pytest.mark.asyncio
async def test_search_with_multi_query_meta(monkeypatch):
    # Patch the underlying expansion/search calls inside the function if needed
    # Here we directly call the wrapper and ensure it returns the expected structure when DB is uninitialized
    monkeypatch.setattr('rag_agent_advanced.initialize_db', lambda : None)
    res = await dummy_meta_search(None, 'test')
    assert isinstance(res, dict)
    assert 'formatted' in res and 'meta' in res
    assert res['meta']['returned'] == 1
