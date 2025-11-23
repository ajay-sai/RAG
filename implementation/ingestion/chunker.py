"""
Docling HybridChunker implementation for intelligent document splitting.

This module uses Docling's built-in HybridChunker which combines:
- Token-aware chunking (uses actual tokenizer)
- Document structure preservation (headings, sections, tables)
- Semantic boundary respect (paragraphs, code blocks)
- Contextualized output (chunks include heading hierarchy)

Benefits over custom chunking:
- Fast (no LLM API calls)
- Token-precise (not character-based estimates)
- Better for RAG (chunks include document context)
- Battle-tested (maintained by Docling team)
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from dotenv import load_dotenv
from transformers import AutoTokenizer
from docling.chunking import HybridChunker
from docling_core.types.doc import DoclingDocument

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class ChunkingConfig:
    """Configuration for chunking."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_chunk_size: int = 2000
    min_chunk_size: int = 100
    use_semantic_splitting: bool = True
    preserve_structure: bool = True
    max_tokens: int = 512

    def __post_init__(self):
        """Validate configuration."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
        if self.min_chunk_size <= 0:
            raise ValueError("Minimum chunk size must be positive")


@dataclass
class DocumentChunk:
    """Represents a document chunk with optional embedding."""
    content: str
    index: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any]
    token_count: Optional[int] = None
    embedding: Optional[List[float]] = None

    def __post_init__(self):
        """Calculate token count if not provided."""
        if self.token_count is None:
            # Rough estimation: ~4 characters per token
            self.token_count = len(self.content) // 4


class DoclingHybridChunker:
    """
    Docling HybridChunker wrapper for intelligent document splitting.
    """

    def __init__(self, config: ChunkingConfig):
        self.config = config
        model_id = "sentence-transformers/all-MiniLM-L6-v2"
        logger.info(f"Initializing tokenizer: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.chunker = HybridChunker(
            tokenizer=self.tokenizer,
            max_tokens=config.max_tokens,
            merge_peers=True
        )

        logger.info(
            f"HybridChunker initialized (max_tokens={config.max_tokens})"
        )

    async def chunk_document(
        self,
        content: str,
        title: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
        docling_doc: Optional[DoclingDocument] = None
    ) -> List[DocumentChunk]:
        if not content.strip():
            return []

        base_metadata = {
            "title": title,
            "source": source,
            "chunk_method": "hybrid",
            **(metadata or {})
        }

        if docling_doc is None:
            logger.warning(
                "No DoclingDocument provided, using simple fallback"
            )
            return self._simple_fallback_chunk(content, base_metadata)

        try:
            chunk_iter = self.chunker.chunk(dl_doc=docling_doc)
            chunks = list(chunk_iter)
            document_chunks = []
            current_pos = 0

            for i, chunk in enumerate(chunks):
                contextualized_text = self.chunker.contextualize(chunk=chunk)
                token_count = len(self.tokenizer.encode(contextualized_text))
                chunk_metadata = {
                    **base_metadata,
                    "total_chunks": len(chunks),
                    "token_count": token_count,
                    "has_context": True
                }
                start_char = current_pos
                end_char = start_char + len(contextualized_text)
                document_chunks.append(DocumentChunk(
                    content=contextualized_text.strip(),
                    index=i,
                    start_char=start_char,
                    end_char=end_char,
                    metadata=chunk_metadata,
                    token_count=token_count
                ))
                current_pos = end_char

            logger.info(
                f"Created {len(document_chunks)} chunks with HybridChunker"
            )
            return document_chunks

        except Exception as e:
            logger.error(
                f"HybridChunker failed: {e}, falling back to simple chunking"
            )
            return self._simple_fallback_chunk(content, base_metadata)

    def _simple_fallback_chunk(
        self,
        content: str,
        base_metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        chunks = []
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        start = 0
        chunk_index = 0

        while start < len(content):
            end = start + chunk_size
            if end >= len(content):
                chunk_text = content[start:]
            else:
                chunk_end = end
                for i in range(
                    end, max(start + self.config.min_chunk_size, end - 200), -1
                ):
                    if i < len(content) and content[i] in '.!?\n':
                        chunk_end = i + 1
                        break
                chunk_text = content[start:chunk_end]
                end = chunk_end

            if chunk_text.strip():
                token_count = len(self.tokenizer.encode(chunk_text))
                chunks.append(DocumentChunk(
                    content=chunk_text.strip(),
                    index=chunk_index,
                    start_char=start,
                    end_char=end,
                    metadata={
                        **base_metadata,
                        "chunk_method": "simple_fallback",
                        "total_chunks": -1
                    },
                    token_count=token_count
                ))
                chunk_index += 1
            start = end - overlap

        for chunk in chunks:
            chunk.metadata["total_chunks"] = len(chunks)

        logger.info(f"Created {len(chunks)} chunks using simple fallback")
        return chunks


class SimpleChunker:
    """
    Simple non-semantic chunker for faster processing without Docling.
    """

    def __init__(self, config: ChunkingConfig):
        self.config = config

    async def chunk_document(
        self,
        content: str,
        title: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[DocumentChunk]:
        if not content.strip():
            return []

        base_metadata = {
            "title": title,
            "source": source,
            "chunk_method": "simple",
            **(metadata or {})
        }
        import re
        paragraphs = re.split(r'\n\s*\n', content)
        chunks = []
        current_chunk = ""
        current_pos = 0
        chunk_index = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            potential_chunk = (
                current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            )

            if len(potential_chunk) <= self.config.chunk_size:
                current_chunk = potential_chunk
            else:
                if current_chunk:
                    chunks.append(self._create_chunk(
                        current_chunk,
                        chunk_index,
                        current_pos,
                        current_pos + len(current_chunk),
                        base_metadata.copy()
                    ))
                    current_pos += len(current_chunk)
                    chunk_index += 1
                current_chunk = paragraph

        if current_chunk:
            chunks.append(self._create_chunk(
                current_chunk,
                chunk_index,
                current_pos,
                current_pos + len(current_chunk),
                base_metadata.copy()
            ))

        for chunk in chunks:
            chunk.metadata["total_chunks"] = len(chunks)
        return chunks

    def _create_chunk(
        self,
        content: str,
        index: int,
        start_pos: int,
        end_pos: int,
        metadata: Dict[str, Any]
    ) -> DocumentChunk:
        return DocumentChunk(
            content=content.strip(),
            index=index,
            start_char=start_pos,
            end_char=end_pos,
            metadata=metadata
        )


class AdaptiveChunker:
    """
    Adaptive chunker that varies chunk size based on content density.
    """
    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.dense_threshold = 2.0
        self.dense_size = 300
        self.sparse_size = 1200

    def _estimate_density(self, text: str) -> float:
        if not text:
            return 0.0
        num_sentences = text.count('.') + text.count('!') + text.count('?')
        return (num_sentences / len(text)) * 100 if len(text) > 0 else 0

    async def chunk_document(
        self,
        content: str,
        title: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[DocumentChunk]:
        import re
        paragraphs = re.split(r'\n\s*\n', content)
        chunks = []
        chunk_index = 0
        current_pos = 0

        for p in paragraphs:
            if not p.strip():
                continue

            density = self._estimate_density(p)
            target_size = (
                self.dense_size if density > self.dense_threshold
                else self.sparse_size
            )
            for i in range(0, len(p), target_size):
                chunk_text = p[i:i+target_size]
                chunk_meta = {
                    "title": title, "source": source,
                    "chunk_method": "adaptive",
                    "density": f"{density:.2f}",
                    "target_size": target_size,
                    **(metadata or {})
                }
                start = current_pos + i
                end = start + len(chunk_text)
                chunks.append(DocumentChunk(
                    content=chunk_text.strip(),
                    index=chunk_index,
                    start_char=start,
                    end_char=end,
                    metadata=chunk_meta,
                ))
                chunk_index += 1
            current_pos += len(p) + 2

        for chunk in chunks:
            chunk.metadata["total_chunks"] = len(chunks)
        return chunks


def create_chunker(config: ChunkingConfig, chunker_type: str = "semantic"):
    """
    Create appropriate chunker based on configuration.
    """
    if chunker_type == "semantic":
        return DoclingHybridChunker(config)
    elif chunker_type == "simple":
        return SimpleChunker(config)
    elif chunker_type == "adaptive":
        return AdaptiveChunker(config)
    else:
        logger.warning(
            f"Unknown chunker type '{chunker_type}', falling back to semantic."
        )
        return DoclingHybridChunker(config)
