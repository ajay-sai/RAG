"""
Tests for the RAG Evaluation Metrics Module (utils/evaluation.py)
"""

import asyncio
import math
import os
import sys
import types
import pytest

# Ensure implementation directory is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.evaluation import (
    RAGEvaluator,
    extract_contexts_from_formatted,
    _cosine_similarity,
    _token_overlap_score,
    _clamp,
    _score_from_text,
)


# ---------------------------------------------------------------------------
# Unit tests: helper utilities
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_clamp_below_zero(self):
        assert _clamp(-0.5) == 0.0

    def test_clamp_above_one(self):
        assert _clamp(1.5) == 1.0

    def test_clamp_in_range(self):
        assert _clamp(0.5) == 0.5

    def test_cosine_similarity_identical(self):
        v = [1.0, 0.0, 0.0]
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self):
        assert abs(_cosine_similarity([1, 0], [0, 1])) < 1e-6

    def test_cosine_similarity_zero_vector(self):
        assert _cosine_similarity([0, 0], [1, 0]) == 0.0

    def test_token_overlap_identical(self):
        text = "the quick brown fox"
        assert abs(_token_overlap_score(text, text) - 1.0) < 1e-6

    def test_token_overlap_no_overlap(self):
        assert _token_overlap_score("apple orange", "car bus train") == 0.0

    def test_token_overlap_partial(self):
        score = _token_overlap_score("apple orange", "apple banana")
        assert 0 < score < 1

    def test_score_from_text_integer(self):
        assert abs(_score_from_text("8") - 0.8) < 1e-6

    def test_score_from_text_decimal(self):
        assert abs(_score_from_text("0.75") - 0.75) < 1e-6

    def test_score_from_text_no_number(self):
        assert _score_from_text("no number here", default=0.3) == 0.3

    def test_score_from_text_scale_ten(self):
        # "7/10" → extracts 7 → normalises to 0.7
        assert abs(_score_from_text("7") - 0.7) < 1e-6


# ---------------------------------------------------------------------------
# Unit tests: extract_contexts_from_formatted
# ---------------------------------------------------------------------------

class TestExtractContexts:
    def test_empty_string(self):
        assert extract_contexts_from_formatted("") == []

    def test_none_like_empty(self):
        assert extract_contexts_from_formatted(None) == []  # type: ignore[arg-type]

    def test_single_chunk(self):
        formatted = "Found 1 relevant results:\n\n[Source: Doc A]\nThis is the content.\n"
        contexts = extract_contexts_from_formatted(formatted)
        assert len(contexts) == 1
        assert "This is the content." in contexts[0]

    def test_multiple_chunks(self):
        formatted = (
            "Found 2 relevant results:\n\n"
            "[Source: Doc A]\nFirst chunk.\n"
            "\n---\n"
            "[Source: Doc B]\nSecond chunk.\n"
        )
        contexts = extract_contexts_from_formatted(formatted)
        assert len(contexts) == 2
        assert any("First" in c for c in contexts)
        assert any("Second" in c for c in contexts)


# ---------------------------------------------------------------------------
# RAGEvaluator tests — using monkeypatched LLM client
# ---------------------------------------------------------------------------

def _make_dummy_client(score_text: str = "8", yes_no: str = "YES"):
    """Create a dummy AsyncOpenAI client."""

    class DummyChoice:
        def __init__(self, content):
            self.message = types.SimpleNamespace(content=content)

    class DummyCompletions:
        async def create(self, *args, **kwargs):
            # Return YES/NO for yes-no prompts, otherwise score
            prompt = kwargs.get("messages", [{}])[-1].get("content", "")
            if "YES" in prompt or "NO" in prompt or "supported" in prompt.lower():
                return types.SimpleNamespace(choices=[DummyChoice(yes_no)])
            return types.SimpleNamespace(choices=[DummyChoice(score_text)])

    class DummyChat:
        completions = DummyCompletions()

    class DummyClient:
        chat = DummyChat()

    return DummyClient()


@pytest.mark.asyncio
async def test_evaluate_context_precision_with_llm(monkeypatch):
    evaluator = RAGEvaluator(openai_api_key="test")
    monkeypatch.setattr(evaluator, "_get_client", lambda: _make_dummy_client("7"))
    result = await evaluator.evaluate_context_precision(
        question="What is RAG?",
        contexts=["RAG is Retrieval-Augmented Generation.", "Unrelated content about weather."],
    )
    assert "score" in result
    assert 0 <= result["score"] <= 1
    assert result["label"] == "Context Precision"
    assert "per_chunk_scores" in result
    assert len(result["per_chunk_scores"]) == 2


@pytest.mark.asyncio
async def test_evaluate_context_precision_no_llm():
    """Without LLM, falls back to token overlap."""
    evaluator = RAGEvaluator(openai_api_key=None)
    evaluator._client = None  # ensure no client

    # Patch _get_client to return None
    evaluator._get_client = lambda: None

    result = await evaluator.evaluate_context_precision(
        question="machine learning models",
        contexts=["deep learning models and machine learning", "the weather is nice today"],
    )
    assert "score" in result
    assert 0 <= result["score"] <= 1


@pytest.mark.asyncio
async def test_evaluate_context_precision_empty_contexts():
    evaluator = RAGEvaluator(openai_api_key=None)
    result = await evaluator.evaluate_context_precision("question", [])
    assert result["score"] == 0.0


@pytest.mark.asyncio
async def test_evaluate_context_recall_with_llm(monkeypatch):
    evaluator = RAGEvaluator(openai_api_key="test")
    monkeypatch.setattr(evaluator, "_get_client", lambda: _make_dummy_client("8"))
    result = await evaluator.evaluate_context_recall(
        question="What is RAG?",
        answer="RAG is Retrieval-Augmented Generation. It combines retrieval with generation.",
        contexts=["RAG is Retrieval-Augmented Generation.", "It uses a retriever and a generator."],
    )
    assert "score" in result
    assert 0 <= result["score"] <= 1
    assert result["label"] == "Context Recall"


@pytest.mark.asyncio
async def test_evaluate_faithfulness(monkeypatch):
    evaluator = RAGEvaluator(openai_api_key="test")
    monkeypatch.setattr(evaluator, "_get_client", lambda: _make_dummy_client("9"))
    result = await evaluator.evaluate_faithfulness(
        question="What is RAG?",
        answer="RAG combines retrieval and generation.",
        contexts=["RAG is Retrieval-Augmented Generation."],
    )
    assert "score" in result
    assert 0 <= result["score"] <= 1
    assert result["label"] == "Faithfulness"


@pytest.mark.asyncio
async def test_evaluate_faithfulness_no_context():
    evaluator = RAGEvaluator(openai_api_key=None)
    result = await evaluator.evaluate_faithfulness("Q", "A", [])
    assert result["score"] == 0.0


@pytest.mark.asyncio
async def test_evaluate_answer_relevance(monkeypatch):
    evaluator = RAGEvaluator(openai_api_key="test")
    monkeypatch.setattr(evaluator, "_get_client", lambda: _make_dummy_client("8"))
    result = await evaluator.evaluate_answer_relevance(
        question="What is RAG?",
        answer="RAG stands for Retrieval-Augmented Generation.",
    )
    assert "score" in result
    assert 0 <= result["score"] <= 1
    assert result["label"] == "Answer Relevance"


@pytest.mark.asyncio
async def test_evaluate_groundedness(monkeypatch):
    evaluator = RAGEvaluator(openai_api_key="test")
    monkeypatch.setattr(evaluator, "_get_client", lambda: _make_dummy_client(yes_no="YES"))
    result = await evaluator.evaluate_groundedness(
        answer="RAG is useful. It retrieves relevant documents.",
        contexts=["RAG retrieves documents from a knowledge base."],
    )
    assert "score" in result
    assert "hallucination_rate" in result
    assert abs(result["score"] + result["hallucination_rate"] - 1.0) < 0.01
    assert result["label"] == "Groundedness"


@pytest.mark.asyncio
async def test_evaluate_coherence(monkeypatch):
    evaluator = RAGEvaluator(openai_api_key="test")
    monkeypatch.setattr(evaluator, "_get_client", lambda: _make_dummy_client("8"))
    result = await evaluator.evaluate_coherence("This is a clear and well-structured answer.")
    assert "score" in result
    assert result["label"] == "Coherence"


@pytest.mark.asyncio
async def test_evaluate_conciseness(monkeypatch):
    evaluator = RAGEvaluator(openai_api_key="test")
    monkeypatch.setattr(evaluator, "_get_client", lambda: _make_dummy_client("7"))
    result = await evaluator.evaluate_conciseness(
        question="What is RAG?",
        answer="RAG combines retrieval and generation.",
    )
    assert "score" in result
    assert result["label"] == "Conciseness"


@pytest.mark.asyncio
async def test_evaluate_answer_correctness(monkeypatch):
    evaluator = RAGEvaluator(openai_api_key="test")
    monkeypatch.setattr(evaluator, "_get_client", lambda: _make_dummy_client("9"))
    result = await evaluator.evaluate_answer_correctness(
        question="What is RAG?",
        answer="RAG is Retrieval-Augmented Generation.",
        reference_answer="RAG stands for Retrieval-Augmented Generation, combining retrieval with LLMs.",
    )
    assert "score" in result
    assert 0 <= result["score"] <= 1
    assert result["label"] == "Answer Correctness"
    assert "token_overlap" in result
    assert "llm_score" in result


# ---------------------------------------------------------------------------
# Mathematical / statistical metric tests (no LLM)
# ---------------------------------------------------------------------------

class TestMathMetrics:
    def setup_method(self):
        self.evaluator = RAGEvaluator(openai_api_key=None)

    def test_compute_average_similarity_normal(self):
        result = self.evaluator.compute_average_similarity([0.9, 0.8, 0.7])
        assert abs(result["score"] - 0.8) < 0.01
        assert "min" in result and "max" in result and "std" in result

    def test_compute_average_similarity_empty(self):
        result = self.evaluator.compute_average_similarity([])
        assert result["score"] == 0.0

    def test_compute_hit_rate_hit(self):
        result = self.evaluator.compute_hit_rate(
            relevant_chunk_ids=["a", "b"],
            retrieved_chunk_ids=["c", "a", "d"],
            k=5,
        )
        assert result["score"] == 1.0

    def test_compute_hit_rate_miss(self):
        result = self.evaluator.compute_hit_rate(
            relevant_chunk_ids=["x"],
            retrieved_chunk_ids=["a", "b", "c"],
            k=3,
        )
        assert result["score"] == 0.0

    def test_compute_mrr_first_rank(self):
        result = self.evaluator.compute_mrr(
            relevant_chunk_ids=["a"],
            retrieved_chunk_ids=["a", "b", "c"],
        )
        assert abs(result["score"] - 1.0) < 1e-6

    def test_compute_mrr_second_rank(self):
        result = self.evaluator.compute_mrr(
            relevant_chunk_ids=["b"],
            retrieved_chunk_ids=["a", "b", "c"],
        )
        assert abs(result["score"] - 0.5) < 1e-6

    def test_compute_mrr_not_found(self):
        result = self.evaluator.compute_mrr(
            relevant_chunk_ids=["z"],
            retrieved_chunk_ids=["a", "b", "c"],
        )
        assert result["score"] == 0.0

    def test_compute_ndcg_perfect(self):
        # Perfect order: highest relevance first
        result = self.evaluator.compute_ndcg([1.0, 0.8, 0.5], k=3)
        assert abs(result["score"] - 1.0) < 1e-6

    def test_compute_ndcg_worst(self):
        # Reverse order: lowest relevance first
        result = self.evaluator.compute_ndcg([0.5, 0.8, 1.0], k=3)
        assert result["score"] < 1.0

    def test_compute_ndcg_empty(self):
        result = self.evaluator.compute_ndcg([])
        assert result["score"] == 0.0

    def test_compute_token_efficiency_basic(self):
        answer = "This is a test answer with ten words here now done"
        result = self.evaluator.compute_token_efficiency(answer, total_tokens=100)
        assert result["score"] is not None
        assert result["word_count"] == len(answer.split())
        assert "raw_efficiency" in result

    def test_compute_token_efficiency_no_tokens(self):
        result = self.evaluator.compute_token_efficiency("some answer", total_tokens=None)
        assert result["score"] is None


# ---------------------------------------------------------------------------
# Full evaluation runner test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_full_evaluation(monkeypatch):
    evaluator = RAGEvaluator(openai_api_key="test")
    monkeypatch.setattr(evaluator, "_get_client", lambda: _make_dummy_client("7", yes_no="YES"))

    report = await evaluator.run_full_evaluation(
        question="What is RAG?",
        answer="RAG is Retrieval-Augmented Generation, combining retrieval with LLMs.",
        contexts=["RAG retrieves relevant documents and passes them to an LLM."],
        similarity_scores=[0.92, 0.85, 0.78],
        total_tokens=200,
    )

    # Check all expected keys present
    assert "faithfulness" in report
    assert "answer_relevance" in report
    assert "context_precision" in report
    assert "context_recall" in report
    assert "groundedness" in report
    assert "coherence" in report
    assert "conciseness" in report
    assert "avg_similarity" in report
    assert "ndcg" in report
    assert "token_efficiency" in report
    assert "overall_score" in report

    # Verify scores are in range
    for key in ["faithfulness", "answer_relevance", "context_precision", "context_recall",
                "groundedness", "coherence", "conciseness"]:
        s = report[key].get("score")
        assert s is not None, f"{key} score is None"
        assert 0 <= s <= 1, f"{key} score out of range: {s}"

    # Overall score
    assert report["overall_score"] is not None
    assert 0 <= report["overall_score"] <= 1


@pytest.mark.asyncio
async def test_run_full_evaluation_with_reference(monkeypatch):
    evaluator = RAGEvaluator(openai_api_key="test")
    monkeypatch.setattr(evaluator, "_get_client", lambda: _make_dummy_client("8", yes_no="YES"))

    report = await evaluator.run_full_evaluation(
        question="What is RAG?",
        answer="RAG combines retrieval with generation.",
        contexts=["RAG stands for Retrieval-Augmented Generation."],
        reference_answer="RAG is Retrieval-Augmented Generation.",
    )

    assert "answer_correctness" in report
    assert 0 <= report["answer_correctness"]["score"] <= 1


@pytest.mark.asyncio
async def test_run_full_evaluation_no_llm():
    """Evaluator should degrade gracefully without a working LLM."""
    evaluator = RAGEvaluator(openai_api_key=None)
    evaluator._get_client = lambda: None  # type: ignore

    report = await evaluator.run_full_evaluation(
        question="What is RAG?",
        answer="RAG is Retrieval-Augmented Generation.",
        contexts=["RAG combines retrieval with generation."],
    )

    # Should still return a report with default/fallback scores
    assert isinstance(report, dict)
    assert "faithfulness" in report
    assert "overall_score" in report


# ---------------------------------------------------------------------------
# App structure tests
# ---------------------------------------------------------------------------

class TestAppStructureExtended:
    """Additional structure tests for the updated app.py."""

    def _read_app(self):
        with open(os.path.join(os.path.dirname(__file__), 'app.py'), 'r') as f:
            return f.read()

    def test_evaluation_lab_in_navigation(self):
        content = self._read_app()
        assert '"Evaluation Lab"' in content, "Evaluation Lab not in navigation"

    def test_render_evaluation_page_exists(self):
        content = self._read_app()
        assert 'def render_evaluation_page()' in content

    def test_eval_available_flag(self):
        content = self._read_app()
        assert 'EVAL_AVAILABLE' in content

    def test_evaluation_imports(self):
        content = self._read_app()
        assert 'from utils.evaluation import RAGEvaluator' in content

    def test_scorecard_renderer_exists(self):
        content = self._read_app()
        assert '_render_eval_scorecard' in content

    def test_batch_evaluation_tab(self):
        content = self._read_app()
        assert 'Batch Evaluation' in content

    def test_strategy_comparison_tab(self):
        content = self._read_app()
        assert 'Strategy Comparison' in content

    def test_faithfulness_metric_mentioned(self):
        content = self._read_app()
        assert 'Faithfulness' in content

    def test_answer_relevance_metric_mentioned(self):
        content = self._read_app()
        assert 'Answer Relevance' in content

    def test_groundedness_metric_mentioned(self):
        content = self._read_app()
        assert 'Groundedness' in content

    def test_context_precision_metric_mentioned(self):
        content = self._read_app()
        assert 'Context Precision' in content

    def test_ndcg_metric_mentioned(self):
        content = self._read_app()
        assert 'NDCG' in content


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
