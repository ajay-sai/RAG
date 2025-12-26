import asyncio
import pytest  # type: ignore[import]
from rag_agent_advanced import search_with_hybrid_retrieval_meta, search_with_self_reflection_meta

import pytest  # type: ignore[import]

@pytest.mark.asyncio
async def test_hybrid_meta_returns_structure(monkeypatch):
    # Patch initialize_bm25 & embedding to avoid DB dependencies
    class DummyEmbedder:
        async def embed_query(self, q):
            return [0.0]
    monkeypatch.setattr('rag_agent_advanced.create_embedder', lambda *a, **k: DummyEmbedder())

    # Patch DB responses: return empty results to trigger no-match path
    class DummyConn:
        async def fetch(self, *args, **kwargs):
            return []
    class DummyPool:
        def acquire(self):
            class Ctx:
                async def __aenter__(self_):
                    return DummyConn()
                async def __aexit__(self_, exc_type, exc, tb):
                    pass
            return Ctx()

    monkeypatch.setattr('rag_agent_advanced.db_pool', DummyPool())
    monkeypatch.setattr('rag_agent_advanced.bm25_index', type('X', (), {'get_scores': lambda self, q: [0]} )())
    monkeypatch.setattr('rag_agent_advanced.bm25_chunks', [])

    res = await search_with_hybrid_retrieval_meta(None, 'test')
    assert isinstance(res, dict)
    assert 'formatted' in res and 'meta' in res
    assert 'dense_count' in res['meta'] and 'sparse_count' in res['meta']


@pytest.mark.asyncio
async def test_self_reflection_meta_structure(monkeypatch):
    # Monkeypatch DB to return a fake result
    class DummyConn:
        async def fetch(self, *args, **kwargs):
            return [{'chunk_id': '1', 'content': 'abc', 'document_title': 'doc1'}]
    class DummyPool:
        def acquire(self):
            class Ctx:
                async def __aenter__(self_):
                    return DummyConn()
                async def __aexit__(self_, exc_type, exc, tb):
                    pass
            return Ctx()

    monkeypatch.setattr('rag_agent_advanced.db_pool', DummyPool())

    # Monkeypatch AsyncOpenAI client methods to return minimal objects with usage
    class DummyResp:
        def __init__(self, content, tokens=5):
            self.choices = [type('C', (), {'message': type('M', (), {'content': content})()})]
            self.usage = type('U', (), {'total_tokens': tokens})
    class DummyClient:
        async def chat_completion(self, *a, **k):
            return DummyResp('5')
        class chat:
            class completions:
                @staticmethod
                async def create(*a, **k):
                    return DummyResp('5')
    monkeypatch.setattr('rag_agent_advanced.AsyncOpenAI', lambda *a, **k: DummyClient())

    res = await search_with_self_reflection_meta(None, 'test')
    assert isinstance(res, dict)
    assert 'formatted' in res and 'meta' in res
    assert 'returned' in res['meta']
