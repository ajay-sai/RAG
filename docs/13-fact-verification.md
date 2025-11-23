````markdown
# Fact Verification and Attribution

## Resource
**Fact-checking LLM outputs: retrieval + claim verification**
https://arxiv.org/abs/2302.01359

## What It Is
Fact verification augments RAG with an explicit verification step: after the model generates a claim or an answer, a verifier (either a smaller LLM or a rule-based module) checks each claim against retrieved passages and returns confidence and citations.

## Simple Example
```python
answer = llm_generate(prompt_with_docs)
claims = extract_claims(answer)
verifications = [verify_claim(claim, evidence_db) for claim in claims]
return attach_citations(answer, verifications)
```

## Pros
- Improves faithfulness and traceability of answers.
- Provides users with citations and confidence scores.

## Cons
- Extra retrieval and compute overhead.
- Requires good claim-extraction heuristics to be most effective.

## When to Use It
Use in high-stakes domains (medical, legal, finance) or when regulatory traceability is required.

````
