"""Document Ingestion & Text Normalization for SEC Filing RAG.

Parses curated corporate annual 10-K filings, extracts regulatory metadata
headers, strips formatting noise, and outputs normalized FilingDocument objects.
"""

import os
import re
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field


class FilingDocument(BaseModel):
    """Schema representing an ingested regulatory SEC document."""
    doc_id: str = Field(description="Unique document identifier, e.g. NVDA-2024-10K")
    symbol: str = Field(description="Equity ticker symbol")
    company_name: str = Field(description="Full legal corporate name")
    filing_type: str = Field(default="10-K", description="SEC filing category")
    fiscal_year: int = Field(description="Reporting fiscal year")
    section: str = Field(default="Item 1A. Risk Factors", description="Document section title")
    source_url: str = Field(description="Official SEC EDGAR hyperlink")
    clean_text: str = Field(description="Normalized and cleaned narrative text")


def clean_text(raw_text: str) -> str:
    """Normalizes whitespace, removes header comments, and strips boilerplate noise."""
    # Remove comment lines starting with #
    lines = [line for line in raw_text.splitlines() if not line.strip().startswith("#")]
    text = "\n".join(lines)
    
    # Normalize multiple newlines and spaces
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_filing_file(file_path: Path) -> Optional[FilingDocument]:
    """Parses a curated filing text file, extracting metadata and clean body content."""
    if not file_path.exists():
        return None

    raw_content = file_path.read_text(encoding="utf-8")
    
    # Parse header metadata from top comment lines
    company_name = "Public Company"
    symbol = file_path.stem.split("_")[0].upper()
    fiscal_year = 2024
    source_url = "https://www.sec.gov/edgar"
    section = "Item 1A. Risk Factors"
    
    for line in raw_content.splitlines():
        line_clean = line.strip().lstrip("#").strip()
        if line.startswith("#") and "CORPORATION" in line or "INC." in line or "APPLE" in line or "MICROSOFT" in line or "TESLA" in line or "NVIDIA" in line:
            company_name = line_clean
        elif "Fiscal Year" in line_clean:
            match = re.search(r"20\d{2}", line_clean)
            if match:
                fiscal_year = int(match.group(0))
        elif "SEC EDGAR URL:" in line_clean:
            source_url = line_clean.replace("SEC EDGAR URL:", "").strip()
        elif "ITEM" in line_clean.upper() and "RISK FACTORS" in line_clean.upper():
            section = "Item 1A. Risk Factors"

    cleaned_body = clean_text(raw_content)
    doc_id = f"{symbol}-{fiscal_year}-10K"

    return FilingDocument(
        doc_id=doc_id,
        symbol=symbol,
        company_name=company_name,
        filing_type="10-K",
        fiscal_year=fiscal_year,
        section=section,
        source_url=source_url,
        clean_text=cleaned_body
    )


def load_all_filings(filings_dir: str = "data/filings") -> List[FilingDocument]:
    """Loads and parses all filing documents located in the filings directory."""
    dir_path = Path(filings_dir)
    if not dir_path.exists():
        return []

    documents: List[FilingDocument] = []
    for file_path in dir_path.glob("*.txt"):
        doc = parse_filing_file(file_path)
        if doc and doc.clean_text:
            documents.append(doc)
            
    return documents
