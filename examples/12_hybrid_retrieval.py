"""Hybrid retrieval example: combine vector + BM25 and rerank"""
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o', system_prompt='Hybrid retrieval demo')

@agent.tool
def hybrid_search(query: str) -> str:
    # Dense vector candidates
    dense = vector_search(query, top_k=50)  # placeholder

    # Sparse BM25 candidates
    sparse = bm25_search(query, top_k=50)  # placeholder

    # Merge, dedupe, and rerank
    merged = merge_and_dedup(dense, sparse)
    final = rerank_with_cross_encoder(query, merged)[:10]
    return "\n\n".join([r['content'] for r in final])

if __name__ == '__main__':
    print(hybrid_search('How do I reset my password?'))
