"""Fact verification example: generate answer then verify claims"""
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o', system_prompt='Fact verification demo')

@agent.tool
def verify_and_answer(query: str) -> str:
    # Generate an answer from context
    docs = vector_search(query, top_k=10)
    answer = llm_generate_with_docs(query, docs)

    # Extract claims and verify each against retrieved evidence
    claims = extract_claims(answer)
    verifications = []
    for c in claims:
        evidence = retrieve_supporting_evidence(c, top_k=5)
        ok, confidence = verify_claim_with_evidence(c, evidence)
        verifications.append({'claim': c, 'ok': ok, 'confidence': confidence})

    # Attach simple citation block
    citations = [f"{v['claim']} -> {v['ok']} (conf={v['confidence']})" for v in verifications]
    return answer + "\n\nVerifications:\n" + "\n".join(citations)

if __name__ == '__main__':
    print(verify_and_answer('When was the company founded?'))
