"""SEC Filing Search Tool for Financial Research Agent.

Exposes semantic vector search over audited SEC 10-K annual filings (specifically
Item 1A Risk Factors and legal disclosures) as a LangChain agent tool.
"""

from typing import Optional
from langchain_core.tools import tool
from rag.retrieval import retrieve_filing_chunks


@tool
def search_filing_rag(query: str, symbol: Optional[str] = None) -> str:
    """Search audited annual SEC 10-K filings and risk factors using semantic vector retrieval.

    Use this tool when the user asks about corporate business risks, regulatory threats,
    supply chain dependencies, legal proceedings, antitrust scrutiny, export controls,
    foundry concentration, or long-term disclosures in official annual 10-K filings.
    Currently indexed benchmark equities: AAPL (Apple), MSFT (Microsoft), NVDA (NVIDIA), TSLA (Tesla).

    Args:
        query: The semantic search query describing the risk or disclosure topic (e.g. 'export restrictions to China and Taiwan foundry dependence').
        symbol: Optional stock ticker symbol (e.g. 'NVDA', 'AAPL', 'MSFT', 'TSLA') to restrict search to a specific company.

    Returns:
        JSON string containing relevant retrieved filing chunks, similarity scores, and SEC citations.
    """
    response = retrieve_filing_chunks(query=query, symbol=symbol, top_k=3)
    return response.model_dump_json()
