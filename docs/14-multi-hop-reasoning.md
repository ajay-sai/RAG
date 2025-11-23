````markdown
# Multi-hop Reasoning

## Resource
**Multi-hop retrieval for compositional QA**
https://arxiv.org/abs/2004.14071

## What It Is
Multi-hop RAG performs iterative retrieval where the output of one retrieval or generation step forms the next query. This allows answering questions that require combining information from multiple documents or knowledge sources.

## Simple Example
```python
context = []
query = initial_question
for step in range(3):
    results = vector_search(query, top_k=5)
    context.append(results)
    query = rewrite_query_with_context(initial_question, context)

final_answer = llm_generate(initial_question, context)
```

## Pros
- Can solve compositional questions that no single document answers.

## Cons
- More expensive and complex; needs careful termination criteria.

## When to Use It
Use for multi-step QA, research assistants, or tasks requiring causal chains across documents.

````
