"""Unit and Integration Tests for RBI Regulatory Corpora RAG Evaluation Pipeline."""

import os
import sys
import json
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


def test_golden_dataset():
    golden_path = Path("golden_set/questions.jsonl")
    assert golden_path.exists(), "golden_set/questions.jsonl does not exist"

    with open(golden_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    assert len(records) == 60, f"Expected 60 questions, got {len(records)}"

    types = set(r["type"] for r in records)
    expected_types = {
        "factual_single_clause",
        "multi_clause_calculation",
        "paraphrased_adversarial",
        "unanswerable_refusal",
    }
    assert types == expected_types

    for r in records:
        assert "id" in r and r["id"].startswith("q_")
        assert len(r["question"]) > 10
        assert len(r["expected_answer"]) > 5
        assert r["difficulty"] in ["Easy", "Medium", "Hard"]
        if r["type"] == "unanswerable_refusal":
            assert r["source_document"] == "None"
            assert r["clause_number"] == "None"
        else:
            assert r["source_document"].endswith(".pdf")
            assert r["clause_number"].startswith("Paragraph ")
