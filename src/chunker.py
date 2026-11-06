"""Chunking Strategies Module for RBI Regulatory Corpora.

Implements three distinct chunking strategies to evaluate their impact on RAG performance:
1. Fixed-Size Chunking:
   Splits text strictly by character length (e.g., 500 characters with 50 character overlap)
   without respecting sentence or clause boundaries.
2. Recursive Chunking:
   Splits text recursively based on hierarchical separators (paragraphs, newlines, spaces)
   targeting a standard block size while preserving semantic continuity.
3. Clause-Aware (Semantic/Structural) Chunking:
   Custom regulatory splitter that keeps complete numbered clauses (e.g. Paragraph 4,
   Paragraph 6, Paragraph 14) intact as single searchable units, preserving legal context.
"""

from abc import ABC, abstractmethod
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel, Field
from src.ingest import RegulatoryClause


class Chunk(BaseModel):
    """Represents a text chunk produced by any chunking strategy."""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    text: str = Field(..., description="Chunk text content")
    strategy: str = Field(..., description="Chunking strategy: fixed, recursive, or clause_aware")
    source_doc: str = Field(..., description="Source PDF filename")
    doc_title: str = Field(default="", description="Formal document title")
    chapter: str = Field(default="Unknown", description="Chapter heading")
    section: Optional[str] = Field(default=None, description="Section heading")
    clause_number: str = Field(default="Unknown", description="Clause / Paragraph number")
    sub_clause: Optional[str] = Field(default=None, description="Sub-clause / definition tag")
    page_start: int = Field(default=1, description="Source page start")
    page_end: int = Field(default=1, description="Source page end")
    chunk_index: int = Field(default=0, description="Sequential index of this chunk within parent unit")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary additional metadata")

    def to_chroma_metadata(self) -> Dict[str, Any]:
        """Flatten chunk metadata into ChromaDB-compatible types (str, int, float, bool)."""
        chroma_meta = {
            "chunk_id": self.chunk_id,
            "strategy": self.strategy,
            "source_doc": self.source_doc,
            "doc_title": self.doc_title,
            "chapter": self.chapter or "Unknown",
            "section": self.section or "",
            "clause_number": self.clause_number or "Unknown",
            "sub_clause": self.sub_clause or "",
            "page_start": int(self.page_start),
            "page_end": int(self.page_end),
            "chunk_index": int(self.chunk_index),
            "char_length": len(self.text),
            "word_count": len(self.text.split()),
        }
        for k, v in self.metadata.items():
            if isinstance(v, (str, int, float, bool)):
                chroma_meta[k] = v
        return chroma_meta


class BaseChunker(ABC):
    """Abstract base class for chunking strategies."""

    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name

    @abstractmethod
    def chunk_clause(self, clause: RegulatoryClause) -> List[Chunk]:
        """Chunk a single RegulatoryClause into one or more Chunks."""
        pass

    def chunk_clauses(self, clauses: List[RegulatoryClause]) -> List[Chunk]:
        """Chunk a collection of RegulatoryClauses."""
        all_chunks = []
        for clause in clauses:
            all_chunks.extend(self.chunk_clause(clause))
        return all_chunks


class FixedSizeChunker(BaseChunker):
    """Fixed-Size Character Splitter.

    Splits text strictly by character length with sliding window overlap,
    without respecting sentence or clause boundaries.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        super().__init__(strategy_name="fixed")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.step = chunk_size - chunk_overlap

    def chunk_clause(self, clause: RegulatoryClause) -> List[Chunk]:
        text = clause.text
        if not text:
            return []

        chunks: List[Chunk] = []
        chunk_idx = 0
        start = 0
        total_len = len(text)

        while start < total_len:
            end = min(start + self.chunk_size, total_len)
            chunk_text = text[start:end]

            chunk_id = f"{clause.clause_id}_fix_{chunk_idx}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    strategy=self.strategy_name,
                    source_doc=clause.source_doc,
                    doc_title=clause.doc_title,
                    chapter=clause.chapter,
                    section=clause.section,
                    clause_number=clause.clause_number,
                    sub_clause=clause.sub_clause,
                    page_start=clause.page_start,
                    page_end=clause.page_end,
                    chunk_index=chunk_idx,
                    metadata={"start_char": start, "end_char": end},
                )
            )
            chunk_idx += 1
            if end == total_len:
                break
            start += self.step

        return chunks


class RecursiveChunker(BaseChunker):
    """Recursive Character / Structural Splitter.

    Splits text recursively based on a hierarchical separator list
    (double newline, single newline, sentence end, comma, space)
    targeting a specified block size with overlap.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "; ", ", ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None,
    ):
        super().__init__(strategy_name="recursive")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using the first matching separator."""
        final_chunks: List[str] = []
        separator = separators[-1]
        new_separators = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator != "" else list(text)

        good_splits: List[str] = []
        for s in splits:
            if not s:
                continue
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if new_separators:
                    sub_splits = self._split_text(s, new_separators)
                    good_splits.extend(sub_splits)
                else:
                    good_splits.append(s)

        # Merge small splits up to chunk_size with overlap
        merged: List[str] = []
        current: List[str] = []
        current_len = 0

        for piece in good_splits:
            piece_len = len(piece) + (len(separator) if current else 0)
            if current_len + piece_len > self.chunk_size and current:
                merged_text = separator.join(current).strip()
                if merged_text:
                    merged.append(merged_text)
                # Keep overlap if possible
                overlap_tokens: List[str] = []
                overlap_len = 0
                for rev_piece in reversed(current):
                    if overlap_len + len(rev_piece) <= self.chunk_overlap:
                        overlap_tokens.insert(0, rev_piece)
                        overlap_len += len(rev_piece)
                    else:
                        break
                current = overlap_tokens
                current_len = sum(len(x) for x in current)

            current.append(piece)
            current_len += len(piece)

        if current:
            merged_text = separator.join(current).strip()
            if merged_text:
                merged.append(merged_text)

        return merged

    def chunk_clause(self, clause: RegulatoryClause) -> List[Chunk]:
        text = clause.text
        if not text:
            return []

        raw_pieces = self._split_text(text, self.separators)
        chunks: List[Chunk] = []

        for idx, piece in enumerate(raw_pieces):
            chunk_id = f"{clause.clause_id}_rec_{idx}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=piece,
                    strategy=self.strategy_name,
                    source_doc=clause.source_doc,
                    doc_title=clause.doc_title,
                    chapter=clause.chapter,
                    section=clause.section,
                    clause_number=clause.clause_number,
                    sub_clause=clause.sub_clause,
                    page_start=clause.page_start,
                    page_end=clause.page_end,
                    chunk_index=idx,
                )
            )
        return chunks


class ClauseAwareChunker(BaseChunker):
    """Clause-Aware (Semantic/Structural) Splitter.

    Groups text explicitly by regulatory boundaries:
    - Retains entire numbered clauses (e.g. Paragraph 4, Paragraph 6, Paragraph 14)
      intact as atomic searchable units.
    - If a clause is exceptionally large (e.g., Definitions or complex Annex schedules
      exceeding `max_clause_size`), decomposes along numbered sub-clauses
      (e.g., '(1)', '(2)', '(3)') while retaining the parent clause context.
    """

    RE_SUBCLAUSE_SPLIT = re.compile(r'(?=\s*\([0-9a-zivx]+\)\s+[‘\'"]?[A-Za-z])', re.IGNORECASE)

    def __init__(self, max_clause_size: int = 1800):
        super().__init__(strategy_name="clause_aware")
        self.max_clause_size = max_clause_size

    def chunk_clause(self, clause: RegulatoryClause) -> List[Chunk]:
        text = clause.text
        if not text:
            return []

        # If clause fits comfortably within threshold, keep intact as atomic unit
        if len(text) <= self.max_clause_size:
            chunk_id = f"{clause.clause_id}_clause_0"
            return [
                Chunk(
                    chunk_id=chunk_id,
                    text=text,
                    strategy=self.strategy_name,
                    source_doc=clause.source_doc,
                    doc_title=clause.doc_title,
                    chapter=clause.chapter,
                    section=clause.section,
                    clause_number=clause.clause_number,
                    sub_clause=clause.sub_clause,
                    page_start=clause.page_start,
                    page_end=clause.page_end,
                    chunk_index=0,
                    metadata={"is_atomic_clause": True},
                )
            ]

        # For long clauses (like Definitions), split by sub-clauses
        sub_pieces = self.RE_SUBCLAUSE_SPLIT.split(text)
        if len(sub_pieces) <= 1:
            # Fallback if no sub-clauses: keep as whole or large paragraph units
            chunk_id = f"{clause.clause_id}_clause_0"
            return [
                Chunk(
                    chunk_id=chunk_id,
                    text=text,
                    strategy=self.strategy_name,
                    source_doc=clause.source_doc,
                    doc_title=clause.doc_title,
                    chapter=clause.chapter,
                    section=clause.section,
                    clause_number=clause.clause_number,
                    sub_clause=clause.sub_clause,
                    page_start=clause.page_start,
                    page_end=clause.page_end,
                    chunk_index=0,
                    metadata={"is_atomic_clause": True},
                )
            ]

        lead_in = sub_pieces[0].strip()
        chunks: List[Chunk] = []
        chunk_idx = 0

        for piece in sub_pieces[1:]:
            piece = piece.strip()
            if not piece:
                continue

            # Detect subclause tag e.g. '(2)'
            tag_match = re.match(r'^\(([0-9a-zivx]+)\)', piece, re.IGNORECASE)
            sub_tag = f"({tag_match.group(1)})" if tag_match else None

            # Prepend context header to retain semantic self-containment
            context_header = f"[{clause.chapter} > {clause.clause_number}"
            if clause.section:
                context_header += f" > {clause.section}"
            context_header += "]\n"

            chunk_text = context_header + piece
            chunk_id = f"{clause.clause_id}_clause_{chunk_idx}"

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    strategy=self.strategy_name,
                    source_doc=clause.source_doc,
                    doc_title=clause.doc_title,
                    chapter=clause.chapter,
                    section=clause.section,
                    clause_number=clause.clause_number,
                    sub_clause=sub_tag or clause.sub_clause,
                    page_start=clause.page_start,
                    page_end=clause.page_end,
                    chunk_index=chunk_idx,
                    metadata={"is_atomic_clause": False, "lead_in": lead_in[:100]},
                )
            )
            chunk_idx += 1

        return chunks if chunks else [
            Chunk(
                chunk_id=f"{clause.clause_id}_clause_0",
                text=text,
                strategy=self.strategy_name,
                source_doc=clause.source_doc,
                doc_title=clause.doc_title,
                chapter=clause.chapter,
                section=clause.section,
                clause_number=clause.clause_number,
                sub_clause=clause.sub_clause,
                page_start=clause.page_start,
                page_end=clause.page_end,
                chunk_index=0,
                metadata={"is_atomic_clause": True},
            )
        ]


def create_chunker(strategy: str, **kwargs) -> BaseChunker:
    """Factory function for instantiating a chunker by strategy name."""
    norm = strategy.lower().strip().replace("-", "_")
    if norm in ("fixed", "fixed_size"):
        return FixedSizeChunker(**kwargs)
    elif norm in ("recursive", "recursive_chunking"):
        return RecursiveChunker(**kwargs)
    elif norm in ("clause", "clause_aware", "structural", "semantic"):
        return ClauseAwareChunker(**kwargs)
    else:
        raise ValueError(
            f"Unknown chunking strategy '{strategy}'. Supported strategies: 'fixed', 'recursive', 'clause_aware'"
        )
