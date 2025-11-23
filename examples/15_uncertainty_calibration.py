"""Uncertainty calibration example: ensembles and agreement scoring"""
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o', system_prompt='Uncertainty demo')

@agent.tool
def answer_with_uncertainty(query: str) -> str:
    # Run several independent generations to measure agreement
    responses = [llm_generate(query) for _ in range(5)]
    confidence = measure_agreement_score(responses)
    best = choose_most_consistent(responses)
    return f"Confidence: {confidence:.2f}\n\nAnswer:\n{best}"

if __name__ == '__main__':
    print(answer_with_uncertainty('What are the side effects of drug X?'))
