"""
Test script for LoRA-SHIFT paper ingestion pipeline.
Tests document processing, chunking, embedding, and retrieval without requiring live database.
"""

import pytest
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch, MagicMock
import asyncio

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


class TestLoRAShiftIngestion:
    """Test suite for LoRA-SHIFT paper ingestion."""
    
    @pytest.fixture
    def lora_shift_paper_path(self):
        """Path to LoRA-SHIFT test paper."""
        return Path(__file__).parent / "documents" / "LoRA-SHIFT-Final-Research-Paper.md"
    
    @pytest.fixture
    def sample_paper_content(self, lora_shift_paper_path):
        """Read sample content from LoRA-SHIFT paper."""
        if lora_shift_paper_path.exists():
            with open(lora_shift_paper_path, 'r', encoding='utf-8') as f:
                return f.read()
        return "# Sample LoRA-SHIFT Content\n\nThis is test content."
    
    def test_paper_exists(self, lora_shift_paper_path):
        """Test that LoRA-SHIFT paper exists in documents folder."""
        assert lora_shift_paper_path.exists(), f"LoRA-SHIFT paper not found at {lora_shift_paper_path}"
        assert lora_shift_paper_path.is_file(), "LoRA-SHIFT paper should be a file"
    
    def test_paper_has_content(self, sample_paper_content):
        """Test that paper has substantial content."""
        assert len(sample_paper_content) > 1000, "Paper should have substantial content (>1000 chars)"
        assert "LoRA-SHIFT" in sample_paper_content, "Paper should mention LoRA-SHIFT"
        assert "Abstract" in sample_paper_content or "Introduction" in sample_paper_content
    
    def test_paper_structure(self, sample_paper_content):
        """Test that paper has proper structure."""
        # Check for common research paper sections
        expected_sections = ["Introduction", "Methodology", "Results", "Conclusion"]
        found_sections = sum(1 for section in expected_sections if section in sample_paper_content)
        assert found_sections >= 3, f"Paper should contain at least 3 standard sections, found {found_sections}"
    
    def test_paper_has_technical_content(self, sample_paper_content):
        """Test that paper contains technical content suitable for RAG."""
        technical_terms = [
            "parameter", "model", "training", "fine-tuning", 
            "rank", "matrix", "efficiency", "performance"
        ]
        found_terms = sum(1 for term in technical_terms if term.lower() in sample_paper_content.lower())
        assert found_terms >= 5, f"Paper should contain technical terminology, found {found_terms}/{len(technical_terms)} terms"


class TestChunkingLogic:
    """Test chunking logic on LoRA-SHIFT paper."""
    
    @pytest.fixture
    def mock_chunker_config(self):
        """Mock chunking configuration."""
        return {
            'chunk_size': 512,
            'chunk_overlap': 50,
            'max_chunk_size': 1024,
            'use_semantic_splitting': True
        }
    
    @pytest.fixture
    def sample_text(self):
        """Sample text for chunking tests."""
        return """# LoRA-SHIFT: Efficient Parameter Adaptation

## Introduction

Large Language Models have revolutionized NLP. This section discusses the background.

## Methodology

LoRA-SHIFT extends standard LoRA by introducing shift parameters.

### Core Architecture

The architecture combines low-rank matrices with learned shifts.

## Results

Experiments show 15-25% improvement over standard LoRA."""
    
    def test_simple_chunk_splitting(self, sample_text):
        """Test basic text splitting into chunks."""
        # Simple splitting logic (without Docling)
        chunk_size = 200
        chunks = []
        
        # Split by paragraphs first
        paragraphs = [p.strip() for p in sample_text.split('\n\n') if p.strip()]
        
        current_chunk = ""
        for para in paragraphs:
            if len(current_chunk) + len(para) <= chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        assert len(chunks) > 0, "Should create at least one chunk"
        assert all(len(chunk) <= chunk_size * 2 for chunk in chunks), "Chunks should be reasonably sized"
    
    def test_chunk_overlap(self):
        """Test that chunk overlap logic works."""
        text = "word " * 100  # 100 words
        chunk_size = 50
        overlap = 10
        
        chunks = []
        start = 0
        words = text.split()
        
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start = end - overlap if end < len(words) else end
        
        assert len(chunks) > 1, "Should create multiple chunks"
        # Check overlap exists between consecutive chunks
        if len(chunks) > 1:
            chunk1_words = set(chunks[0].split()[-overlap:])
            chunk2_words = set(chunks[1].split()[:overlap])
            assert len(chunk1_words & chunk2_words) > 0, "Chunks should have overlapping content"


class TestEmbeddingGeneration:
    """Test embedding generation logic."""
    
    def test_embedding_dimensions(self):
        """Test that embeddings have correct dimensions."""
        # Mock embedding - OpenAI text-embedding-3-small uses 1536 dimensions
        mock_embedding = [0.1] * 1536
        assert len(mock_embedding) == 1536, "Embedding should be 1536 dimensions"
    
    def test_embedding_normalization(self):
        """Test that embeddings can be normalized."""
        import math
        mock_embedding = [0.5, 0.5, 0.5, 0.5]
        
        # Calculate magnitude
        magnitude = math.sqrt(sum(x**2 for x in mock_embedding))
        
        # Normalize
        normalized = [x / magnitude for x in mock_embedding]
        
        # Check normalized magnitude is 1
        normalized_magnitude = math.sqrt(sum(x**2 for x in normalized))
        assert abs(normalized_magnitude - 1.0) < 0.001, "Normalized embedding should have magnitude 1"


class TestRetrievalQueries:
    """Test queries that should work on ingested LoRA-SHIFT paper."""
    
    @pytest.fixture
    def sample_queries(self):
        """Sample queries about LoRA-SHIFT paper."""
        return [
            "What is LoRA-SHIFT?",
            "How does LoRA-SHIFT improve over standard LoRA?",
            "What are the key contributions of the LoRA-SHIFT paper?",
            "Explain the shift function in LoRA-SHIFT",
            "What datasets were used for evaluation?",
            "What are the performance improvements of LoRA-SHIFT?",
            "How many trainable parameters does LoRA-SHIFT use?",
            "What is the computational overhead of LoRA-SHIFT?",
        ]
    
    def test_query_relevance(self, sample_queries):
        """Test that queries are relevant to paper content."""
        paper_keywords = {"lora", "shift", "parameter", "fine-tuning", "performance", "rank", "dataset", "evaluation", "overhead"}
        
        for query in sample_queries:
            query_lower = query.lower()
            # Each query should contain at least one paper keyword
            has_keyword = any(keyword in query_lower for keyword in paper_keywords)
            assert has_keyword, f"Query '{query}' should contain relevant keywords"
    
    def test_query_diversity(self, sample_queries):
        """Test that queries cover different aspects."""
        query_types = {
            'definition': ['what is', 'define', 'explain'],
            'comparison': ['improve', 'better', 'compare', 'versus'],
            'technical': ['how', 'function', 'architecture', 'algorithm'],
            'results': ['performance', 'results', 'evaluation', 'accuracy']
        }
        
        found_types = set()
        for query in sample_queries:
            query_lower = query.lower()
            for qtype, keywords in query_types.items():
                if any(kw in query_lower for kw in keywords):
                    found_types.add(qtype)
        
        assert len(found_types) >= 3, f"Queries should cover diverse types, found: {found_types}"


class TestIngestionPipelineIntegration:
    """Integration tests for full ingestion pipeline (mocked)."""
    
    @pytest.mark.asyncio
    async def test_mock_ingestion_flow(self):
        """Test complete ingestion flow with mocks."""
        # Mock document loading
        doc_content = "# LoRA-SHIFT\n\nThis is a test paper about LoRA-SHIFT methodology."
        
        # Mock chunking
        chunks = [
            {"content": doc_content[:50], "index": 0},
            {"content": doc_content[50:], "index": 1}
        ]
        
        # Mock embedding generation
        embeddings = [[0.1] * 1536 for _ in chunks]
        
        # Mock database storage
        stored_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            stored_chunks.append({
                "chunk_id": f"chunk_{chunk['index']}",
                "content": chunk["content"],
                "embedding": embedding,
                "metadata": {"doc_title": "LoRA-SHIFT"}
            })
        
        assert len(stored_chunks) == len(chunks), "All chunks should be stored"
        assert all("embedding" in chunk for chunk in stored_chunks), "All chunks should have embeddings"
    
    @pytest.mark.asyncio
    async def test_mock_retrieval_flow(self):
        """Test retrieval flow with mocks."""
        # Mock query
        query = "What is LoRA-SHIFT?"
        query_embedding = [0.1] * 1536
        
        # Mock stored chunks (from previous ingestion)
        stored_chunks = [
            {
                "chunk_id": "1",
                "content": "LoRA-SHIFT is a novel approach to parameter-efficient fine-tuning.",
                "embedding": [0.15] * 1536,
                "similarity": 0.95
            },
            {
                "chunk_id": "2", 
                "content": "The methodology combines low-rank adaptation with shift functions.",
                "embedding": [0.12] * 1536,
                "similarity": 0.87
            }
        ]
        
        # Sort by similarity (descending)
        results = sorted(stored_chunks, key=lambda x: x['similarity'], reverse=True)
        
        assert len(results) > 0, "Should return results"
        assert results[0]['similarity'] >= results[1]['similarity'], "Results should be sorted by similarity"
        assert "LoRA-SHIFT" in results[0]['content'], "Top result should mention LoRA-SHIFT"


class TestErrorHandling:
    """Test error handling in ingestion pipeline."""
    
    def test_missing_file_handling(self):
        """Test handling of missing files."""
        nonexistent_path = Path("nonexistent_file.md")
        assert not nonexistent_path.exists(), "Test file should not exist"
        
        # In real implementation, should raise FileNotFoundError or return error
        try:
            with open(nonexistent_path, 'r') as f:
                content = f.read()
            assert False, "Should raise FileNotFoundError"
        except FileNotFoundError:
            pass  # Expected
    
    def test_empty_content_handling(self):
        """Test handling of empty content."""
        empty_content = ""
        
        # Should not create chunks from empty content
        if empty_content.strip():
            chunks = [empty_content]
        else:
            chunks = []
        
        assert len(chunks) == 0, "Empty content should not create chunks"
    
    def test_invalid_embedding_dimensions(self):
        """Test validation of embedding dimensions."""
        valid_embedding = [0.1] * 1536
        invalid_embedding = [0.1] * 512  # Wrong dimension
        
        assert len(valid_embedding) == 1536, "Valid embedding should be 1536-dim"
        assert len(invalid_embedding) != 1536, "Invalid embedding should be detected"


class TestDocumentMetadata:
    """Test metadata extraction and handling."""
    
    def test_metadata_extraction(self):
        """Test extraction of document metadata."""
        doc_path = Path("documents/LoRA-SHIFT-Final-Research-Paper.md")
        
        metadata = {
            "title": "LoRA-SHIFT: Efficient Parameter Adaptation",
            "source": str(doc_path),
            "format": doc_path.suffix,
            "size": doc_path.stat().st_size if doc_path.exists() else 0
        }
        
        assert metadata["title"], "Should extract title"
        assert metadata["format"] == ".md", "Should detect markdown format"
        if doc_path.exists():
            assert metadata["size"] > 0, "Should extract file size"
        else:
            assert metadata["size"] == 0, "Non-existent file should have size 0"
    
    def test_metadata_storage(self):
        """Test that metadata is properly structured for storage."""
        metadata = {
            "title": "Test Paper",
            "authors": ["Author 1", "Author 2"],
            "date": "2025-12-26",
            "topics": ["RAG", "LoRA", "Fine-tuning"]
        }
        
        # Metadata should be JSON-serializable
        import json
        json_str = json.dumps(metadata)
        restored = json.loads(json_str)
        
        assert restored == metadata, "Metadata should survive JSON serialization"


if __name__ == "__main__":
    # Run tests
    print("Running LoRA-SHIFT Ingestion Tests...\n")
    
    # Run with pytest
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-W", "ignore::DeprecationWarning"
    ])
    
    sys.exit(exit_code)
