````markdown
# Adaptive Chunking

## Resource
**Adaptive, content-aware chunk sizes for efficient retrieval**
https://arxiv.org/abs/2010.08296

## What It Is
Adaptive chunking varies chunk size based on local content density and semantic boundaries. Short chunks for dense factual sections, longer chunks for narrative or background. This improves retrieval efficiency and reduces noise.

## Simple Example
```python
for section in document_sections:
    density = estimate_information_density(section)
    chunk_size = 200 if density > 0.7 else 800
    chunks += chunk_text(section, chunk_size)
```

## Pros
- Better precision with fewer irrelevant tokens.

## Cons
- Slightly more complex ingestion logic.

## When to Use It
Large heterogeneous documents with variable information density (technical manuals, legal contracts).

````
