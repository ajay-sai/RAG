````markdown
# Hybrid Retrieval (Dense + Sparse)

## Resource
**Hybrid retrieval: Combining dense vectors with sparse retrieval (BM25)**
https://www.elastic.co/guide/en/elasticsearch/guide/current/hybrid-retrieval.html

## What It Is
Hybrid retrieval combines semantic (dense) vector search with traditional sparse keyword retrieval such as BM25 or TF-IDF. The goal is to capture both paraphrased semantic matches (vectors) and precise keyword matches (sparse). Systems typically merge and re-rank results from both sources.

## Simple Example
```python
# Dense candidates (vector search)
dense_results = vector_search(query, top_k=50)

# Sparse candidates (BM25)
sparse_results = bm25_search(query, top_k=50)

# Merge + deduplicate + re-rank
merged = deduplicate_preserve_score(dense_results + sparse_results)
final = rerank_with_cross_encoder(query, merged)[:10]
```

## Pros
- Covers paraphrases and exact keyword matches.
- Better recall on queries containing specific terms (e.g., named entities, codes).

## Cons
- More complex infra (need both vector DB and search index).
- Additional latency when merging and re-ranking.

## When to Use It
Use for domain data where both semantic similarity and exact term matches matter (legal, software docs, product catalogs).

````
