# RBI Regulatory Corpora RAG Evaluation Pipeline (`RBI-corpora-RAG-eval`)

[![CI / Regression Evaluation Gate](https://github.com/udaykamalbotcha-ctrl/RBI-corpora-RAG-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/udaykamalbotcha-ctrl/RBI-corpora-RAG-eval/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![ChromaDB](https://img.shields.io/badge/Vector%20Store-ChromaDB-purple.svg)](https://www.trychroma.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade Retrieval-Augmented Generation (RAG) evaluation benchmark tailored for complex, hierarchical financial regulatory texts issued by the **Reserve Bank of India (RBI)** (such as the *Credit Valuation Adjustment (CVA) Framework Directions 2026* and *Commercial Banks Forthcoming Instructions 2025/2026*).

This project empirically demonstrates how **chunking strategies** (Fixed-Size vs. Recursive vs. Regulatory Clause-Aware) fundamentally alter semantic retrieval precision and downstream LLM answer quality on legal and prudential mandates.

---

## Repository Structure

```
/bank-reg-rag-eval
├── /data                    # Houses CVA 2026, 2025 Instructions, & 2026 Amendments PDFs
├── /golden_set              # Ground-truth questions & clause targets (questions.jsonl, Phase 2)
├── /src
│   ├── __init__.py          # Package interface
│   ├── ingest.py            # Hierarchical PDF parsing & clause metadata extraction
│   ├── chunker.py           # Fixed, recursive, & clause-aware chunking splitters
│   └── vector_store.py      # ChromaDB setup (persistent & container) + embedding pipeline
├── /eval                    # Evaluation runner & retrieval metrics (Phase 5)
├── /reports                 # JSON test logs & benchmark output reports
├── /tests                   # Automated unit & integration test suite
│   └── test_pipeline.py
├── .github/workflows
│   └── ci.yml               # CI/CD regression gate (Phase 10)
├── docker-compose.yml       # Local ChromaDB container instance
├── requirements.txt         # Production Python dependencies
└── README.md
```

---

## Core Capabilities Implemented (Phase 1)

### 1. Hierarchical Regulatory Ingestion (`src/ingest.py`)
Regulatory prose cannot be ingested as generic flat text without losing legal context. The parser isolates:
- **Document Metadata**: Title, notification number, issuance date.
- **Chapters & Schedules**: (e.g., `Chapter-I: Preliminary`, `Chapter-II: Scope of Application`, `Annex I`).
- **Sections / Headings**: (e.g., `A. Short Title`, `C. Definitions`, `B. Alternate Treatment`).
- **Numbered Paragraph Clauses**: (e.g., `Paragraph 4`, `Paragraph 6`, `Paragraph 14`).
- **Sub-clauses & Provisos**: (e.g., `(1) 'Banking Book'`, `(2) 'Covered Transactions'`).
- **Noise Filtering**: Cleans running header banners (`भारतीय रिज़र्व बैंक RESERVE BANK OF INDIA`), Table of Contents dots, and standalone page numbers.

### 2. Multi-Strategy Chunking (`src/chunker.py`)
To benchmark how chunk boundaries impact retrieval:
1. **Fixed-Size Chunking (`FixedSizeChunker`)**:
   - Strictly character-based sliding window (default 500 characters, 50 character overlap).
   - Ignores sentences and clause boundaries; demonstrates truncation and boundary-splitting failure modes.
2. **Recursive Chunking (`RecursiveChunker`)**:
   - Recursively breaks text on hierarchical separators (`["\n\n", "\n", ". ", "; ", ", ", " "]`).
   - Targets 600 characters with 100 character overlap, preserving syntactic readability.
3. **Clause-Aware Structural Chunking (`ClauseAwareChunker`)**:
   - Respects regulatory boundaries by keeping complete numbered clauses intact as atomic searchable units.
   - For extra-large clauses (like definitions in Paragraph 4), it decomposes cleanly along sub-clauses while injecting parent chapter and clause breadcrumbs (`[Chapter-I > Paragraph 4 > C. Definitions]`).

### 3. Embeddings & Vector Storage (`src/vector_store.py`)
- **Embedding Model**: Local open-source `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors, normalized for cosine similarity).
- **Dual Vector Store Modes**:
  - **Docker Container Mode**: Via `docker-compose.yml` (`http://localhost:8000`).
  - **Persistent Local Mode**: Zero-dependency local file persistence (`./chroma_db`). The client automatically detects if the container is running and falls back to persistent mode seamlessly.
- **Clause-Tagged Retrieval**: Every indexed vector carries structured metadata (`clause_number`, `chapter`, `section`, `source_doc`, `page_start`, `page_end`, `strategy`).
- **Retrieval Validation**: Integrated test query verifying that questions such as *"What are Covered Transactions under the CVA framework?"* retrieve the exact paragraph (`Paragraph 6`) with full lineage tags.

---

## Quickstart Guide

### 1. Prerequisites
- Python 3.10+ (Python 3.12 recommended)
- Docker & Docker Compose (Optional, for running ChromaDB in container mode)
- Git

### 2. Environment Setup

```bash
# Clone the repository
git clone https://github.com/udaykamalbotcha-ctrl/RBI-corpora-RAG-eval.git
cd RBI-corpora-RAG-eval

# Create and activate a virtual environment
python -m venv .venv
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On macOS / Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. (Optional) Run ChromaDB Container

```bash
docker compose up -d
```
ChromaDB will be available at `http://localhost:8000`. If you do not run Docker, the vector store automatically runs in local persistent client mode using `./chroma_db`.

---

## Running the Pipeline

### Step 1: Ingest & Parse PDFs
Inspect extracted chapters, sections, and paragraphs from the `data/` folder:

```bash
python src/ingest.py data
```

### Step 2: Run End-to-End Ingestion, Indexing & Validation
Run the full pipeline with your chosen chunking strategy (`clause_aware`, `recursive`, or `fixed`):

```bash
# Clause-aware chunking (recommended for regulatory evaluation)
python src/vector_store.py data clause_aware

# Recursive chunking
python src/vector_store.py data recursive

# Fixed-size chunking
python src/vector_store.py data fixed
```

### Example Validation Output
```text
=================================================================
RUNNING RETRIEVAL VALIDATION TEST
=================================================================
Query: "What are Covered Transactions under the CVA framework?"

Retrieved 3 matching chunks:

[1] Score: 0.6912
    Clause Tag : Paragraph 6
    Chapter    : Chapter-II: Scope of Application
    Source Doc : Reserve Bank of India (Commercial Banks – Credit Valuation Adjustment Framework) Directions, 2026.pdf
    Strategy   : clause_aware
    Text Preview: [Chapter-II: Scope of Application > Paragraph 6]
(CVA) risk on a standalone basis for all covered transactions in both banking book and trading book. It includes CVA for a bank’s e...
-------------------------------------------------------
[PASS] Validation succeeded: All retrieved chunks contain valid clause tags & hierarchy metadata.
```

---

## Running Tests

Execute the comprehensive test suite:

```bash
pytest tests/ -v
```

All 5 test suites validate parser integrity, boundary adherence for each chunking strategy, and vector store metadata consistency.

---

## Roadmap

- [x] **Phase 1: Ingestion, Chunking Strategies & Vector Store Infrastructure**
- [ ] **Phase 2: Golden Question-Answer Set Construction (`golden_set/questions.jsonl`)**
- [ ] **Phase 3: Hybrid Retrieval & Re-ranking (Dense + BM25 + Cross-Encoder)**
- [ ] **Phase 4: Agentic Reasoning & Regulatory Discrepancy Resolution**
- [ ] **Phase 5: Evaluation Framework (Faithfulness, Answer Relevance, Context Recall)**
- [ ] **Phase 10: CI/CD Quality Gate & Regression Prevention**
