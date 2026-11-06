"""RBI Regulatory Corpora Ingestion and Clause Metadata Extraction Module.

This module provides hierarchical parsing of Indian regulatory directions
(e.g., Reserve Bank of India Commercial Banks Directions and Amendments).
It isolates:
  - Document metadata (Title, Reference Number, Date)
  - Chapters (e.g., Chapter-I: Preliminary, Chapter-II: Scope of Application)
  - Sections / Headings (e.g., A. Short Title and Commencement, C. Definitions)
  - Numbered Paragraph Clauses (e.g., Paragraph 4, Paragraph 6, Paragraph 14)
  - Sub-clauses, Explanations, and Provisos
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel, Field
import pypdf


class RegulatoryClause(BaseModel):
    """Represents a structured regulatory clause with hierarchical metadata."""
    clause_id: str = Field(..., description="Unique deterministic identifier for the clause")
    source_doc: str = Field(..., description="Source PDF filename")
    doc_title: str = Field(..., description="Formal document title")
    chapter: str = Field(default="Preamble / Introduction", description="Chapter heading")
    section: Optional[str] = Field(default=None, description="Section heading under the chapter")
    clause_number: str = Field(..., description="Paragraph / Clause identifier, e.g., 'Paragraph 4'")
    sub_clause: Optional[str] = Field(default=None, description="Sub-clause or definition tag, e.g., '(2)'")
    heading: Optional[str] = Field(default=None, description="Clause title/heading if present")
    text: str = Field(..., description="Extracted clause text")
    page_start: int = Field(..., description="Starting page in source document (1-indexed)")
    page_end: int = Field(..., description="Ending page in source document (1-indexed)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional arbitrary metadata")

    def to_metadata_dict(self) -> Dict[str, Any]:
        """Convert clause metadata to Chroma-compatible flat dictionary."""
        base_meta = {
            "clause_id": self.clause_id,
            "source_doc": self.source_doc,
            "doc_title": self.doc_title,
            "chapter": self.chapter or "Unknown",
            "section": self.section or "",
            "clause_number": self.clause_number,
            "sub_clause": self.sub_clause or "",
            "heading": self.heading or "",
            "page_start": self.page_start,
            "page_end": self.page_end,
            "char_count": len(self.text),
            "word_count": len(self.text.split()),
        }
        for k, v in self.metadata.items():
            if isinstance(v, (str, int, float, bool)):
                base_meta[k] = v
        return base_meta


class RegulatoryPDFParser:
    """Hierarchical parser for RBI regulatory documents retaining full clause lineage."""

    # Regex patterns for structural boundary detection
    RE_TOC_LINE = re.compile(r'\.{4,}\s*\d+\s*$', re.MULTILINE)
    RE_HEADER_BANNER = re.compile(
        r'^\s*(?:भारतीय\s+रि[जज़]र्व\s+बैंक|RESERVE\s+BANK\s+OF\s+INDIA|www\.rbi\.org\.in|_+).*$',
        re.IGNORECASE
    )
    RE_PAGE_NUMBER = re.compile(r'^\s*\d+\s*$')

    RE_CHAPTER = re.compile(
        r'^\s*(?:Chapter\s*[-–—:]\s*[IVXLCDM0-9]+|Chapter\s+[IVXLCDM0-9]+|Annex(?:ure)?\s+[IVXLCDM0-9]+|Schedule\s+[IVXLCDM0-9]+)'
        r'(?:\s*[-–—:]\s*(.*))?$',
        re.IGNORECASE
    )

    RE_SECTION = re.compile(
        r'^\s*([A-Z])\.\s+([A-Z0-9][A-Za-z0-9\s,\-\(\)/]+)$'
    )

    RE_PARAGRAPH_START = re.compile(
        r'^\s*(\d{1,3})\.\s+(.*)$'
    )

    RE_SUBCLAUSE_START = re.compile(
        r'^\s*\(([0-9]+|[a-z]|[ivx]+)\)\s+(.*)$',
        re.IGNORECASE
    )

    RE_EXPLANATION = re.compile(
        r'^\s*(?:Explanation|Provided\s+that|Note)(?:\s*\(?\d*\)?)?:\s*(.*)$',
        re.IGNORECASE
    )

    def __init__(self, clean_whitespace: bool = True):
        self.clean_whitespace = clean_whitespace

    def clean_page_lines(self, raw_text: str) -> List[str]:
        """Strip repetitive header banners, page numbers, and table of contents dots."""
        cleaned_lines = []
        for line in raw_text.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            # Remove RBI header banners
            if self.RE_HEADER_BANNER.match(stripped):
                continue
            # Remove standalone page numbers
            if self.RE_PAGE_NUMBER.match(stripped):
                continue
            # Remove Table of Contents dotted lines
            if self.RE_TOC_LINE.search(stripped):
                continue
            cleaned_lines.append(stripped)
        return cleaned_lines

    def extract_document_title(self, pages_text: List[str], fallback_name: str) -> str:
        """Attempt to extract formal title from the first two pages."""
        for page_idx in range(min(2, len(pages_text))):
            lines = pages_text[page_idx].split("\n")
            for line in lines:
                sline = line.strip()
                if "Reserve Bank of India" in sline and ("Directions" in sline or "Framework" in sline or "Instructions" in sline):
                    return sline
                if "Directions, 202" in sline or "Amendment Directions" in sline:
                    return sline
        return Path(fallback_name).stem

    def parse_pdf(self, file_path: str) -> List[RegulatoryClause]:
        """Parse a single regulatory PDF file into a list of structured clauses."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        reader = pypdf.PdfReader(str(file_path))
        num_pages = len(reader.pages)
        pages_raw = [reader.pages[i].extract_text() or "" for i in range(num_pages)]

        doc_title = self.extract_document_title(pages_raw, file_path.name)
        source_doc = file_path.name

        current_chapter = "Introduction / Preamble"
        current_section: Optional[str] = None
        current_para_num: Optional[str] = None
        current_para_heading: Optional[str] = None
        current_para_text_lines: List[str] = []
        current_para_page_start = 1
        current_para_page_end = 1

        clauses: List[RegulatoryClause] = []

        def flush_current_para():
            nonlocal current_para_num, current_para_heading, current_para_text_lines
            nonlocal current_para_page_start, current_para_page_end
            if current_para_num and current_para_text_lines:
                full_text = " ".join(current_para_text_lines).strip()
                # Remove excessive whitespace
                full_text = re.sub(r'\s+', ' ', full_text)
                
                # Derive distinctive document slug
                doc_clean = Path(source_doc).stem.lower()
                if "credit valuation" in doc_clean:
                    doc_prefix = "cva_2026"
                elif "amendment" in doc_clean:
                    doc_prefix = "forthcoming_amend_2026"
                elif "forthcoming" in doc_clean:
                    doc_prefix = "forthcoming_2025"
                else:
                    doc_prefix = re.sub(r'[^a-z0-9]+', '_', doc_clean)[:20].strip('_')

                clause_seq = len(clauses) + 1
                clause_id = f"{doc_prefix}_p{current_para_num}_seq{clause_seq}"
                clause = RegulatoryClause(
                    clause_id=clause_id,
                    source_doc=source_doc,
                    doc_title=doc_title,
                    chapter=current_chapter,
                    section=current_section,
                    clause_number=f"Paragraph {current_para_num}",
                    heading=current_para_heading,
                    text=full_text,
                    page_start=current_para_page_start,
                    page_end=current_para_page_end,
                )
                clauses.append(clause)
                current_para_text_lines = []

        in_toc = False

        for page_idx in range(num_pages):
            page_num = page_idx + 1
            raw_text = pages_raw[page_idx]
            lines = self.clean_page_lines(raw_text)

            for line in lines:
                # Detect Table of Contents block
                if "Table of Contents" in line or "Contents" == line:
                    in_toc = True
                    continue
                if in_toc:
                    if self.RE_TOC_LINE.search(line):
                        continue
                    # TOC typically finishes when "Introduction" or "Chapter" starts with actual body text
                    if "In exercise of the powers" in line or "Please refer to" in line or (
                        self.RE_CHAPTER.match(line) and page_num > 1
                    ):
                        in_toc = False

                # Check for Chapter heading
                chapter_match = self.RE_CHAPTER.match(line)
                if chapter_match and not in_toc:
                    flush_current_para()
                    current_chapter = line.strip()
                    current_section = None
                    current_para_num = None
                    continue

                # Check for Section heading (e.g. 'A. Short Title and Commencement')
                section_match = self.RE_SECTION.match(line)
                if section_match and not in_toc:
                    current_section = line.strip()
                    continue

                # Check for Paragraph start (e.g. '4. In these Directions...')
                para_match = self.RE_PARAGRAPH_START.match(line)
                if para_match and not in_toc:
                    num_candidate = para_match.group(1)
                    # Regulatory paragraph numbers are generally sequential integers (1..200)
                    if int(num_candidate) > 0:
                        flush_current_para()
                        current_para_num = num_candidate
                        current_para_page_start = page_num
                        current_para_page_end = page_num
                        rest_of_line = para_match.group(2).strip()
                        current_para_text_lines = [f"{current_para_num}. {rest_of_line}"] if rest_of_line else [f"{current_para_num}."]
                        continue

                # Continuing existing paragraph
                if current_para_num is not None:
                    current_para_page_end = page_num
                    current_para_text_lines.append(line)
                else:
                    # Content before first numbered paragraph (Introduction / Preamble)
                    if not in_toc and len(line) > 10:
                        if not current_para_num:
                            current_para_num = "0_preamble"
                            current_para_page_start = page_num
                            current_para_page_end = page_num
                            current_para_text_lines = [line]

        flush_current_para()
        return clauses

    def parse_corpus(self, data_dir: str) -> List[RegulatoryClause]:
        """Parse all regulatory PDF files in the specified directory."""
        data_path = Path(data_dir)
        if not data_path.is_dir():
            raise NotADirectoryError(f"Directory not found: {data_dir}")

        pdf_files = sorted(data_path.glob("*.pdf"))
        if not pdf_files:
            print(f"Warning: No PDF files found in {data_dir}")
            return []

        all_clauses = []
        for pdf_file in pdf_files:
            print(f"[Ingest] Parsing {pdf_file.name}...")
            clauses = self.parse_pdf(str(pdf_file))
            print(f"  -> Extracted {len(clauses)} clauses from {pdf_file.name}")
            all_clauses.extend(clauses)

        print(f"[Ingest] Total clauses parsed across {len(pdf_files)} documents: {len(all_clauses)}")
        return all_clauses


def run_cli_ingest(data_dir: str = "data"):
    """CLI utility function for running ingestion directly."""
    parser = RegulatoryPDFParser()
    clauses = parser.parse_corpus(data_dir)

    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    docs = {}
    for c in clauses:
        docs.setdefault(c.source_doc, []).append(c)

    for doc_name, doc_clauses in docs.items():
        print(f"\nDocument: {doc_name}")
        print(f"Total Clauses: {len(doc_clauses)}")
        chapters = list(dict.fromkeys(c.chapter for c in doc_clauses))
        print(f"Chapters ({len(chapters)}):")
        for ch in chapters:
            ch_clauses = [c for c in doc_clauses if c.chapter == ch]
            print(f"  * {ch} ({len(ch_clauses)} clauses)")

    if clauses:
        print("\n--- Sample Extracted Clause ---")
        sample = next((c for c in clauses if c.clause_number == "Paragraph 4"), clauses[0])
        print(f"ID: {sample.clause_id}")
        print(f"Source Doc: {sample.source_doc}")
        print(f"Chapter: {sample.chapter}")
        print(f"Section: {sample.section}")
        print(f"Clause: {sample.clause_number}")
        print(f"Pages: {sample.page_start} - {sample.page_end}")
        print(f"Text Snippet: {sample.text[:250]}...")
    return clauses


if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    run_cli_ingest(target_dir)
