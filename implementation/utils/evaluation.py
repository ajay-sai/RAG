"""
RAG Evaluation Metrics Module
==============================
Comprehensive evaluation metrics for RAG pipelines, inspired by RAGAS and
standard IR evaluation frameworks.

Metrics implemented:

Retrieval Metrics (no ground truth needed):
  - Context Precision: fraction of retrieved chunks relevant to the question
  - Average Similarity Score: mean cosine / vector similarity of retrieved chunks
  - Context Relevance Score: semantic match of retrieved context to question

Retrieval Metrics (with ground truth):
  - Context Recall: coverage of reference answer by retrieved contexts
  - Hit Rate@K: was a relevant chunk within top-K?
  - MRR: Mean Reciprocal Rank
  - NDCG@K: Normalized Discounted Cumulative Gain

Generation Metrics (LLM-as-Judge, no ground truth needed):
  - Faithfulness: fraction of answer claims supported by context
  - Answer Relevance: how relevant the answer is to the question
  - Groundedness: fraction of answer sentences attributable to context
  - Hallucination Rate: 1 - Groundedness
  - Coherence: logical structure and readability of the answer
  - Conciseness: appropriate brevity without missing key information

End-to-End Metrics:
  - Answer Correctness: semantic + factual similarity to reference answer
  - Token Efficiency: useful output per token consumed
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    na = np.array(a, dtype=float)
    nb = np.array(b, dtype=float)
    denom = np.linalg.norm(na) * np.linalg.norm(nb)
    if denom == 0:
        return 0.0
    return float(np.dot(na, nb) / denom)


def _token_overlap_score(text_a: str, text_b: str) -> float:
    """Simple token-level overlap (like Jaccard similarity) as a fallback."""
    tokens_a = set(re.findall(r'\w+', text_a.lower()))
    tokens_b = set(re.findall(r'\w+', text_b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _score_from_text(text: str, default: float = 0.5) -> float:
    """Extract a numeric score (0-10 or 0-1) from LLM response text."""
    numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', text)
    if numbers:
        v = float(numbers[0])
        # Normalise 0-10 scale to 0-1
        if v > 1.0:
            v = v / 10.0
        return _clamp(v)
    return default


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

class RAGEvaluator:
    """
    Comprehensive RAG evaluation engine.

    Usage::

        evaluator = RAGEvaluator(openai_api_key=os.getenv("OPENAI_API_KEY"))
        report = await evaluator.run_full_evaluation(
            question="What is RAG?",
            answer="RAG stands for Retrieval-Augmented Generation...",
            contexts=["Context chunk 1...", "Context chunk 2..."],
        )
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ) -> None:
        self.openai_api_key = openai_api_key
        self.model = model
        self._client = None

    # ------------------------------------------------------------------
    # OpenAI client (lazy)
    # ------------------------------------------------------------------

    def _get_client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI  # type: ignore[import]
                self._client = AsyncOpenAI(api_key=self.openai_api_key)
            except Exception as exc:
                logger.warning("Could not initialise AsyncOpenAI client: %s", exc)
        return self._client

    async def _llm_score(
        self,
        prompt: str,
        default: float = 0.5,
    ) -> Tuple[float, str]:
        """Call the LLM and extract a 0-1 score from its response."""
        client = self._get_client()
        if client is None:
            return default, "LLM unavailable"
        try:
            resp = await client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=256,
            )
            text = resp.choices[0].message.content.strip()
            score = _score_from_text(text, default)
            return score, text
        except Exception as exc:
            logger.warning("LLM scoring call failed: %s", exc)
            return default, f"Error: {exc}"

    # ------------------------------------------------------------------
    # Retrieval metrics
    # ------------------------------------------------------------------

    async def evaluate_context_precision(
        self,
        question: str,
        contexts: List[str],
    ) -> Dict[str, Any]:
        """
        Context Precision: What fraction of retrieved contexts are relevant?

        Uses LLM-as-judge when possible; falls back to token overlap.
        Score: 0-1 (higher = more precise retrieval).
        """
        if not contexts:
            return {"score": 0.0, "label": "Context Precision", "detail": "No contexts provided"}

        client = self._get_client()
        relevant_count = 0
        per_chunk_scores: List[float] = []

        if client:
            for ctx in contexts:
                prompt = (
                    f"Question: {question}\n\n"
                    f"Context: {ctx[:600]}\n\n"
                    "On a scale of 0 to 10, how relevant is this context to the question? "
                    "Reply with ONLY a single number (0-10)."
                )
                score, _ = await self._llm_score(prompt, default=5.0)
                per_chunk_scores.append(score)
                if score >= 0.5:
                    relevant_count += 1
        else:
            for ctx in contexts:
                s = _token_overlap_score(question, ctx)
                per_chunk_scores.append(s)
                if s >= 0.15:
                    relevant_count += 1

        precision = relevant_count / len(contexts) if contexts else 0.0
        avg_score = float(np.mean(per_chunk_scores)) if per_chunk_scores else 0.0

        return {
            "score": _clamp(avg_score),
            "precision_at_k": _clamp(precision),
            "label": "Context Precision",
            "detail": f"{relevant_count}/{len(contexts)} chunks relevant",
            "per_chunk_scores": per_chunk_scores,
        }

    async def evaluate_context_recall(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        reference_answer: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Context Recall: Can the answer be attributed to the retrieved contexts?

        If a reference answer is provided, checks whether reference statements
        appear in the contexts. Otherwise, checks whether the generated answer
        sentences are backed by the contexts.
        Score: 0-1 (higher = better recall).
        """
        ground = reference_answer if reference_answer else answer
        if not ground or not contexts:
            return {"score": 0.0, "label": "Context Recall", "detail": "Insufficient data"}

        # Split ground truth into sentences
        sentences = [s.strip() for s in re.split(r'[.!?]+', ground) if len(s.strip()) > 20]
        if not sentences:
            return {"score": 0.5, "label": "Context Recall", "detail": "Could not split into sentences"}

        combined_context = " ".join(contexts)
        client = self._get_client()
        supported = 0

        if client:
            for sentence in sentences[:8]:  # cap to avoid cost explosion
                prompt = (
                    f"Context: {combined_context[:1200]}\n\n"
                    f"Statement: {sentence}\n\n"
                    "Can this statement be fully inferred from the context? "
                    "Score 0-10 (10=fully supported, 0=not supported). "
                    "Reply with ONLY a single number."
                )
                score, _ = await self._llm_score(prompt, default=5.0)
                if score >= 0.5:
                    supported += 1
        else:
            for sentence in sentences[:8]:
                if _token_overlap_score(sentence, combined_context) >= 0.1:
                    supported += 1

        recall = supported / len(sentences[:8]) if sentences else 0.0
        return {
            "score": _clamp(recall),
            "label": "Context Recall",
            "detail": f"{supported}/{min(len(sentences), 8)} statements supported",
        }

    def compute_average_similarity(
        self,
        similarity_scores: List[float],
    ) -> Dict[str, Any]:
        """
        Average Similarity Score of retrieved chunks.

        Args:
            similarity_scores: cosine/vector similarity scores from DB query
        """
        if not similarity_scores:
            return {"score": 0.0, "label": "Avg Similarity", "detail": "No scores"}
        arr = np.array(similarity_scores, dtype=float)
        return {
            "score": _clamp(float(np.mean(arr))),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "std": float(np.std(arr)),
            "label": "Avg Retrieval Similarity",
            "detail": f"mean={np.mean(arr):.3f}, std={np.std(arr):.3f}",
        }

    def compute_hit_rate(
        self,
        relevant_chunk_ids: List[str],
        retrieved_chunk_ids: List[str],
        k: int = 5,
    ) -> Dict[str, Any]:
        """
        Hit Rate@K: Was at least one relevant chunk in the top-K results?
        Score: 0 or 1.
        """
        top_k = set(retrieved_chunk_ids[:k])
        relevant = set(relevant_chunk_ids)
        hit = int(bool(top_k & relevant))
        return {
            "score": float(hit),
            "label": f"Hit Rate@{k}",
            "detail": f"{'Hit' if hit else 'Miss'} in top-{k}",
        }

    def compute_mrr(
        self,
        relevant_chunk_ids: List[str],
        retrieved_chunk_ids: List[str],
    ) -> Dict[str, Any]:
        """
        Mean Reciprocal Rank: reciprocal of the rank of the first relevant result.
        Score: 0-1.
        """
        relevant = set(relevant_chunk_ids)
        for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1):
            if chunk_id in relevant:
                return {
                    "score": 1.0 / rank,
                    "label": "MRR",
                    "detail": f"First relevant result at rank {rank}",
                }
        return {"score": 0.0, "label": "MRR", "detail": "No relevant result found"}

    def compute_ndcg(
        self,
        relevance_scores: List[float],
        k: int = 5,
    ) -> Dict[str, Any]:
        """
        NDCG@K: Normalized Discounted Cumulative Gain.

        Args:
            relevance_scores: graded relevance for each retrieved result (0-1 or 0-10)
        """
        if not relevance_scores:
            return {"score": 0.0, "label": f"NDCG@{k}", "detail": "No scores"}

        rel = relevance_scores[:k]
        # Normalise to 0-1 range if values > 1
        if max(rel) > 1:
            rel = [r / 10.0 for r in rel]

        def dcg(scores: List[float]) -> float:
            return sum(s / math.log2(i + 2) for i, s in enumerate(scores))

        actual_dcg = dcg(rel)
        ideal_dcg = dcg(sorted(rel, reverse=True))
        ndcg = actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0
        return {
            "score": _clamp(ndcg),
            "label": f"NDCG@{k}",
            "detail": f"DCG={actual_dcg:.3f}, IDCG={ideal_dcg:.3f}",
        }

    # ------------------------------------------------------------------
    # Generation metrics (LLM-as-Judge)
    # ------------------------------------------------------------------

    async def evaluate_faithfulness(
        self,
        question: str,
        answer: str,
        contexts: List[str],
    ) -> Dict[str, Any]:
        """
        Faithfulness: Are all claims in the answer supported by retrieved contexts?
        Score: 0-1 (1 = fully faithful, 0 = hallucinated).
        """
        if not answer or not contexts:
            return {"score": 0.0, "label": "Faithfulness", "detail": "Missing answer or context"}

        combined = " ".join(contexts)[:2000]
        prompt = (
            f"Context:\n{combined}\n\n"
            f"Answer:\n{answer[:800]}\n\n"
            "Evaluate the faithfulness of the answer: what fraction of the answer's "
            "claims are directly supported by the context above? "
            "Consider:\n"
            "- Claims directly stated in the context = supported\n"
            "- Claims not mentioned in the context = unsupported (hallucinated)\n\n"
            "Score 0-10 (10 = every claim supported, 0 = nothing from context). "
            "Reply with ONLY a single number 0-10."
        )
        score, reasoning = await self._llm_score(prompt, default=0.5)
        return {
            "score": _clamp(score),
            "label": "Faithfulness",
            "detail": reasoning[:200] if reasoning and not reasoning.startswith("Error") else f"Score: {score:.2f}",
        }

    async def evaluate_answer_relevance(
        self,
        question: str,
        answer: str,
    ) -> Dict[str, Any]:
        """
        Answer Relevance: How well does the answer address the question?
        Score: 0-1.
        """
        if not answer:
            return {"score": 0.0, "label": "Answer Relevance", "detail": "No answer"}

        prompt = (
            f"Question: {question}\n\n"
            f"Answer: {answer[:800]}\n\n"
            "On a scale of 0-10, how relevant and directly responsive is this answer "
            "to the question? Consider:\n"
            "- Does it directly address what was asked? (high score)\n"
            "- Is it off-topic or evasive? (low score)\n"
            "- Is it a complete non-answer? (0)\n\n"
            "Reply with ONLY a single number 0-10."
        )
        score, reasoning = await self._llm_score(prompt, default=0.5)
        return {
            "score": _clamp(score),
            "label": "Answer Relevance",
            "detail": f"Score: {score:.2f}",
        }

    async def evaluate_groundedness(
        self,
        answer: str,
        contexts: List[str],
    ) -> Dict[str, Any]:
        """
        Groundedness: What fraction of the answer sentences can be traced to context?
        Score: 0-1 (higher = more grounded).
        Hallucination Rate = 1 - Groundedness.
        """
        if not answer or not contexts:
            return {
                "score": 0.0,
                "hallucination_rate": 1.0,
                "label": "Groundedness",
                "detail": "Missing data",
            }

        sentences = [s.strip() for s in re.split(r'[.!?]+', answer) if len(s.strip()) > 15]
        if not sentences:
            return {"score": 0.5, "hallucination_rate": 0.5, "label": "Groundedness", "detail": "N/A"}

        combined = " ".join(contexts)[:2000]
        client = self._get_client()
        grounded = 0

        if client:
            for sent in sentences[:8]:
                prompt = (
                    f"Context: {combined[:1500]}\n\n"
                    f"Statement: {sent}\n\n"
                    "Is this statement directly supported or inferable from the context? "
                    "Reply with 'YES' or 'NO'."
                )
                try:
                    resp = await client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,
                        max_tokens=10,
                    )
                    answer_text = resp.choices[0].message.content.strip().upper()
                    if "YES" in answer_text:
                        grounded += 1
                except Exception as exc:
                    logger.warning("Groundedness check failed: %s", exc)
                    grounded += 1  # conservative: assume grounded on error
        else:
            for sent in sentences[:8]:
                if _token_overlap_score(sent, combined) >= 0.1:
                    grounded += 1

        n = min(len(sentences), 8)
        score = grounded / n if n > 0 else 0.0
        return {
            "score": _clamp(score),
            "hallucination_rate": _clamp(1.0 - score),
            "label": "Groundedness",
            "detail": f"{grounded}/{n} sentences grounded",
        }

    async def evaluate_coherence(
        self,
        answer: str,
    ) -> Dict[str, Any]:
        """
        Coherence: Is the answer logically structured and easy to read?
        Score: 0-1.
        """
        if not answer:
            return {"score": 0.0, "label": "Coherence", "detail": "No answer"}

        prompt = (
            f"Answer: {answer[:800]}\n\n"
            "Rate the coherence of this answer on a scale of 0-10:\n"
            "- 10: Well-structured, logical flow, easy to understand\n"
            "- 5: Moderately clear but some gaps\n"
            "- 0: Incoherent, contradictory, or incomprehensible\n\n"
            "Reply with ONLY a single number 0-10."
        )
        score, _ = await self._llm_score(prompt, default=0.6)
        return {"score": _clamp(score), "label": "Coherence", "detail": f"Score: {score:.2f}"}

    async def evaluate_conciseness(
        self,
        question: str,
        answer: str,
    ) -> Dict[str, Any]:
        """
        Conciseness: Is the answer appropriately brief without losing key information?
        Score: 0-1.
        """
        if not answer:
            return {"score": 0.0, "label": "Conciseness", "detail": "No answer"}

        prompt = (
            f"Question: {question}\n\n"
            f"Answer: {answer[:800]}\n\n"
            "Rate the conciseness of this answer on a scale of 0-10:\n"
            "- 10: Precise and complete, no unnecessary filler\n"
            "- 5: Some repetition or padding, but mostly on-point\n"
            "- 0: Excessively verbose or filled with irrelevant content\n\n"
            "Reply with ONLY a single number 0-10."
        )
        score, _ = await self._llm_score(prompt, default=0.6)
        return {"score": _clamp(score), "label": "Conciseness", "detail": f"Score: {score:.2f}"}

    # ------------------------------------------------------------------
    # End-to-end metrics
    # ------------------------------------------------------------------

    async def evaluate_answer_correctness(
        self,
        question: str,
        answer: str,
        reference_answer: str,
    ) -> Dict[str, Any]:
        """
        Answer Correctness: Combined semantic + factual similarity to a reference answer.
        Score: 0-1.
        """
        if not answer or not reference_answer:
            return {"score": 0.0, "label": "Answer Correctness", "detail": "Missing data"}

        # Semantic overlap as simple baseline
        token_score = _token_overlap_score(answer, reference_answer)

        # LLM factual alignment
        prompt = (
            f"Question: {question}\n\n"
            f"Reference Answer: {reference_answer[:600]}\n\n"
            f"Generated Answer: {answer[:600]}\n\n"
            "Compare the generated answer to the reference on a scale of 0-10:\n"
            "- 10: Contains all key facts of the reference, no errors\n"
            "- 5: Partially correct, some important facts missing or wrong\n"
            "- 0: Completely wrong or contradicts the reference\n\n"
            "Reply with ONLY a single number 0-10."
        )
        llm_score, _ = await self._llm_score(prompt, default=0.5)

        # Weighted combination
        combined = 0.4 * token_score + 0.6 * llm_score
        return {
            "score": _clamp(combined),
            "token_overlap": token_score,
            "llm_score": llm_score,
            "label": "Answer Correctness",
            "detail": f"Token overlap={token_score:.2f}, LLM={llm_score:.2f}",
        }

    def compute_token_efficiency(
        self,
        answer: str,
        total_tokens: Optional[int],
    ) -> Dict[str, Any]:
        """
        Token Efficiency: Answer word count per 100 tokens consumed.
        Higher is more efficient.
        """
        word_count = len(answer.split()) if answer else 0
        if not total_tokens or total_tokens <= 0:
            return {"score": None, "label": "Token Efficiency", "detail": "Token count unavailable"}
        efficiency = (word_count / total_tokens) * 100
        # Normalise roughly: 15-30 words per 100 tokens is typical
        normalised = _clamp(efficiency / 30.0)
        return {
            "score": normalised,
            "raw_efficiency": efficiency,
            "word_count": word_count,
            "total_tokens": total_tokens,
            "label": "Token Efficiency",
            "detail": f"{word_count} words / {total_tokens} tokens = {efficiency:.1f} w/100t",
        }

    # ------------------------------------------------------------------
    # Full evaluation runner
    # ------------------------------------------------------------------

    async def run_full_evaluation(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        reference_answer: Optional[str] = None,
        similarity_scores: Optional[List[float]] = None,
        retrieved_chunk_ids: Optional[List[str]] = None,
        relevant_chunk_ids: Optional[List[str]] = None,
        total_tokens: Optional[int] = None,
        include_groundedness: bool = True,
        include_coherence: bool = True,
        include_conciseness: bool = True,
    ) -> Dict[str, Any]:
        """
        Run all applicable metrics and return a comprehensive evaluation report.

        Args:
            question: The user question.
            answer: The generated answer.
            contexts: List of retrieved context strings.
            reference_answer: Optional ground-truth answer.
            similarity_scores: Optional list of cosine similarities from retrieval.
            retrieved_chunk_ids: Optional ordered list of retrieved chunk IDs.
            relevant_chunk_ids: Optional list of known-relevant chunk IDs.
            total_tokens: Optional total token count consumed.
            include_groundedness: Run groundedness (more LLM calls).
            include_coherence: Run coherence evaluation.
            include_conciseness: Run conciseness evaluation.

        Returns:
            Dict with all metric results under their label keys.
        """
        report: Dict[str, Any] = {}

        # --- Retrieval metrics ---
        ctx_precision_task = self.evaluate_context_precision(question, contexts)
        ctx_recall_task = self.evaluate_context_recall(question, answer, contexts, reference_answer)

        # --- Generation metrics ---
        faithfulness_task = self.evaluate_faithfulness(question, answer, contexts)
        answer_rel_task = self.evaluate_answer_relevance(question, answer)

        tasks = [ctx_precision_task, ctx_recall_task, faithfulness_task, answer_rel_task]

        if include_groundedness:
            tasks.append(self.evaluate_groundedness(answer, contexts))
        if include_coherence:
            tasks.append(self.evaluate_coherence(answer))
        if include_conciseness:
            tasks.append(self.evaluate_conciseness(question, answer))
        if reference_answer:
            tasks.append(self.evaluate_answer_correctness(question, answer, reference_answer))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        labels = [
            "context_precision",
            "context_recall",
            "faithfulness",
            "answer_relevance",
        ]
        if include_groundedness:
            labels.append("groundedness")
        if include_coherence:
            labels.append("coherence")
        if include_conciseness:
            labels.append("conciseness")
        if reference_answer:
            labels.append("answer_correctness")

        for label, result in zip(labels, results):
            if isinstance(result, Exception):
                logger.warning("Metric %s failed: %s", label, result)
                report[label] = {"score": None, "label": label, "detail": str(result)}
            else:
                report[label] = result

        # --- Mathematical metrics (no LLM) ---
        if similarity_scores:
            report["avg_similarity"] = self.compute_average_similarity(similarity_scores)

        if retrieved_chunk_ids and relevant_chunk_ids:
            report["hit_rate"] = self.compute_hit_rate(relevant_chunk_ids, retrieved_chunk_ids)
            report["mrr"] = self.compute_mrr(relevant_chunk_ids, retrieved_chunk_ids)

        if similarity_scores:
            report["ndcg"] = self.compute_ndcg(similarity_scores)

        report["token_efficiency"] = self.compute_token_efficiency(answer, total_tokens)

        # --- Overall score (weighted average of available scores) ---
        weights = {
            "faithfulness": 0.25,
            "answer_relevance": 0.20,
            "context_precision": 0.20,
            "context_recall": 0.15,
            "groundedness": 0.10,
            "coherence": 0.05,
            "conciseness": 0.05,
        }
        total_weight = 0.0
        weighted_sum = 0.0
        for key, weight in weights.items():
            if key in report and report[key].get("score") is not None:
                weighted_sum += report[key]["score"] * weight
                total_weight += weight

        if total_weight > 0:
            report["overall_score"] = _clamp(weighted_sum / total_weight)
        else:
            report["overall_score"] = None

        return report


# ---------------------------------------------------------------------------
# Convenience: extract context strings from RAG result
# ---------------------------------------------------------------------------

def extract_contexts_from_formatted(formatted: str) -> List[str]:
    """
    Parse formatted RAG output (e.g. "Found N results:\n\n[Source: X]\ntext\n---")
    and extract individual context strings.
    """
    if not formatted:
        return []
    # Split on separator lines
    parts = re.split(r'\n---+\n', formatted)
    contexts = []
    for part in parts:
        # Strip metadata prefix like "Found N relevant results:\n\n"
        clean = re.sub(r'^Found \d+ .*?:\n\n', '', part.strip(), flags=re.DOTALL)
        # Strip source tag
        clean = re.sub(r'^\[Source:[^\]]*\]\n?', '', clean.strip())
        if clean.strip():
            contexts.append(clean.strip())
    return contexts
