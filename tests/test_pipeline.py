"""Unit and Integration Tests for RBI Regulatory Corpora RAG Evaluation Pipeline."""

import os
import sys
from pathlib import Path
import pytest

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest import RegulatoryPDFParser, RegulatoryClause
from src.chunker import FixedSizeChunker, RecursiveChunker, ClauseAwareChunker, create_chunker
from src.vector_store import RegulatoryVectorStore


@pytest.fixture(scope="session")
def sample_clause():
    return RegulatoryClause(
        clause_id="test_doc_p6_seq1",
        source_doc="Test_RBI_Directions_2026.pdf",
        doc_title="Reserve Bank of India Directions, 2026",
        chapter="Chapter-II: Scope of Application",
        section="A. Covered Scope",
        clause_number="Paragraph 6",
        text=(
            "6. A bank shall calculate the capital requirements for Credit Valuation Adjustment (CVA) "
            "risk on a standalone basis for all covered transactions in both banking book and trading book. "
            "It includes CVA for a bank's entire portfolio of covered transactions and eligible CVA hedges. "
            "Trades pertaining to SFTs shall be excluded from the calculation of the CVA capital charge."
        ),
        page_start=5,
        page_end=5,
    )


def test_fixed_size_chunker(sample_clause):
    chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk_clause(sample_clause)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 100
        assert c.strategy == "fixed"
        assert c.clause_number == "Paragraph 6"
        assert c.chapter == "Chapter-II: Scope of Application"


def test_recursive_chunker(sample_clause):
    chunker = RecursiveChunker(chunk_size=120, chunk_overlap=30)
    chunks = chunker.chunk_clause(sample_clause)
    assert len(chunks) >= 1
    for c in chunks:
        assert c.strategy == "recursive"
        assert c.source_doc == sample_clause.source_doc
        assert c.clause_number == sample_clause.clause_number


def test_clause_aware_chunker(sample_clause):
    chunker = ClauseAwareChunker(max_clause_size=1000)
    chunks = chunker.chunk_clause(sample_clause)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.strategy == "clause_aware"
    assert c.clause_number == "Paragraph 6"
    assert "Credit Valuation Adjustment" in c.text
    assert c.metadata.get("is_atomic_clause") is True


def test_chunker_factory():
    assert isinstance(create_chunker("fixed"), FixedSizeChunker)
    assert isinstance(create_chunker("recursive"), RecursiveChunker)
    assert isinstance(create_chunker("clause_aware"), ClauseAwareChunker)


def test_ingest_data_folder():
    data_dir = "data"
    if not Path(data_dir).exists():
        pytest.skip("data directory does not exist")
    parser = RegulatoryPDFParser()
    clauses = parser.parse_corpus(data_dir)
    assert len(clauses) > 0
    # Validate that clauses have structural metadata
    for c in clauses[:10]:
        assert c.source_doc.endswith(".pdf")
        assert c.chapter != ""
        assert c.clause_number.startswith("Paragraph ")
        assert len(c.text) > 0
