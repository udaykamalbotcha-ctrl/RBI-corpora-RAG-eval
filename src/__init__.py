"""RBI Regulatory Corpora RAG Evaluation Package.

Modules:
- ingest: Hierarchical regulatory PDF parsing and clause metadata extraction.
- chunker: Fixed-size, recursive, and clause-aware chunking strategies.
- vector_store: ChromaDB container/persistent store integration and vector indexing.
"""

from src.ingest import RegulatoryClause, RegulatoryPDFParser
from src.chunker import (
    BaseChunker,
    Chunk,
    FixedSizeChunker,
    RecursiveChunker,
    ClauseAwareChunker,
    create_chunker,
)
from src.vector_store import RegulatoryVectorStore

__all__ = [
    "RegulatoryClause",
    "RegulatoryPDFParser",
    "BaseChunker",
    "Chunk",
    "FixedSizeChunker",
    "RecursiveChunker",
    "ClauseAwareChunker",
    "create_chunker",
    "RegulatoryVectorStore",
]
