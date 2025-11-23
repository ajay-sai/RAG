````markdown
# Uncertainty Estimation & Calibration

## Resource
**Calibrating model confidence and uncertainty**
https://arxiv.org/abs/2002.03402

## What It Is
Estimate and calibrate the model's confidence on generated answers using techniques like: ensemble predictions, Monte Carlo dropout, or verification with retrieved evidence. Provide a scalar confidence and highlight low-confidence statements.

## Simple Example
```python
answers = [llm_generate(prompt) for _ in range(5)]
confidence = measure_agreement(answers)
if confidence < 0.6:
    evidence = retrieve_additional_evidence(query)
    answer = llm_generate(prompt + evidence)

return answer, confidence
```

## Pros
- Helps users know when to trust answers and triggers safe fallbacks.

## Cons
- Extra compute for ensembles or multiple passes.

## When to Use It
High-risk applications and interfaces that must present uncertainty transparently.

````
