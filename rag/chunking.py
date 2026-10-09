"""Semantic Text Chunking for SEC Filing RAG.

Splits long corporate 10-K disclosures into coherent, semantically bounded
chunks using recursive character splitting with sliding window overlap.
"""

from typing import List
from pydantic import BaseModel, Field
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.ingestion import FilingDocument


class FilingChunk(BaseModel):
    """Schema representing an individual indexed chunk of a corporate filing."""
    chunk_id: str = Field(description="Unique chunk identifier, e.g. NVDA-2024-CHK-001")
    doc_id: str = Field(description="Parent document identifier")
    symbol: str = Field(description="Stock ticker symbol")
    company_name: str = Field(description="Legal company name")
    filing_type: str = Field(default="10-K", description="SEC filing category")
    fiscal_year: int = Field(description="Filing fiscal year")
    section: str = Field(description="Specific filing section, e.g. Item 1A. Risk Factors")
    source_url: str = Field(description="SEC EDGAR URL link")
    text: str = Field(description="Chunk body text content")
    char_start: int = Field(description="Character start index in parent document")
    char_end: int = Field(description="Character end index in parent document")


def chunk_document(
    doc: FilingDocument,
    chunk_size: int = 800,
    chunk_overlap: int = 150
) -> List[FilingChunk]:
    """Splits a single FilingDocument into overlapping FilingChunk objects.
    
    Args:
        doc: The parent FilingDocument.
        chunk_size: Maximum characters per chunk (default 800).
        chunk_overlap: Sliding window character overlap (default 150).
        
    Returns:
        List of FilingChunk objects with preserved metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        is_separator_regex=False
    )
    
    raw_chunks = splitter.split_text(doc.clean_text)
    filing_chunks: List[FilingChunk] = []
    
    current_char = 0
    for idx, text_segment in enumerate(raw_chunks, start=1):
        clean_segment = text_segment.strip()
        if not clean_segment:
            continue
            
        start_idx = doc.clean_text.find(clean_segment, max(0, current_char - chunk_overlap))
        if start_idx == -1:
            start_idx = current_char
        end_idx = start_idx + len(clean_segment)
        current_char = end_idx

        chunk_id = f"{doc.symbol}-{doc.fiscal_year}-CHK-{idx:03d}"
        
        filing_chunks.append(
            FilingChunk(
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                symbol=doc.symbol,
                company_name=doc.company_name,
                filing_type=doc.filing_type,
                fiscal_year=doc.fiscal_year,
                section=doc.section,
                source_url=doc.source_url,
                text=clean_segment,
                char_start=start_idx,
                char_end=end_idx
            )
        )
        
    return filing_chunks


def chunk_all_documents(
    documents: List[FilingDocument],
    chunk_size: int = 800,
    chunk_overlap: int = 150
) -> List[FilingChunk]:
    """Chunks a collection of FilingDocument instances."""
    all_chunks: List[FilingChunk] = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc, chunk_size, chunk_overlap))
    return all_chunks
