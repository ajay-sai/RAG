"""Multi-hop retrieval example: iterative retrieval loop"""
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o', system_prompt='Multi-hop demo')

@agent.tool
def multi_hop_search(question: str) -> str:
    context = []
    current_query = question
    for _ in range(3):
        results = vector_search(current_query, top_k=5)
        context.extend(results)
        # rewrite next query to focus on missing links
        current_query = rewrite_query_with_evidence(question, context)

    # Final answer using accumulated context
    return llm_generate_with_docs(question, context)

if __name__ == '__main__':
    print(multi_hop_search('How does our refund policy apply to subscription upgrades?'))
