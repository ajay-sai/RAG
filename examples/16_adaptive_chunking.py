"""Adaptive chunking example: vary chunk sizes by information density"""
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o', system_prompt='Adaptive chunking demo')

@agent.tool
def adaptive_chunk_document(document: str) -> list:
    sections = split_into_sections(document)
    chunks = []
    for s in sections:
        density = estimate_information_density(s)
        size = 200 if density > 0.7 else 800
        chunks.extend(chunk_text(s, size))
    # Return first few chunks as a demo
    return chunks[:10]

if __name__ == '__main__':
    doc = open('../docs/11-fine-tuned-embeddings.md').read()
    print(adaptive_chunk_document(doc))
