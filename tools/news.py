"""Financial News Tool for Financial Research Agent.

Searches recent financial headlines, publisher attributions, publication dates,
and article URLs using Yahoo Finance Search API with fallback containment
and strict Pydantic validation.
"""

from datetime import datetime, timezone
from typing import Optional, List, Union, Dict, Any
import yfinance as yf
from pydantic import BaseModel, Field
from langchain_core.tools import tool


class NewsItem(BaseModel):
    """Schema representing an individual news dispatch."""
    title: str = Field(description="Headline of the news article")
    publisher: str = Field(description="Publishing outlet or news agency, e.g. Reuters, Bloomberg")
    url: str = Field(description="Direct URL link to the original article")
    published_date: Optional[str] = Field(default=None, description="Human-readable publication date/time in UTC")
    snippet: Optional[str] = Field(default=None, description="Short summary or lead excerpt of article")
    related_tickers: List[str] = Field(default_factory=list, description="Stock tickers associated with this story")


class NewsSearchResults(BaseModel):
    """Schema for successful news retrieval."""
    status: str = Field(default="success", description="Status indicator")
    query: str = Field(description="Search terms submitted")
    result_count: int = Field(description="Total number of articles returned")
    articles: List[NewsItem] = Field(default_factory=list, description="List of structured news items")
    timestamp: str = Field(description="ISO timestamp of query execution")
    source: str = Field(default="Yahoo Finance News Engine", description="News provider")


class NewsSearchError(BaseModel):
    """Schema for failed news retrieval."""
    status: str = Field(default="error", description="Status indicator")
    error_code: str = Field(description="Machine-readable error classification")
    query: str = Field(description="Search terms that caused the error")
    message: str = Field(description="Human-readable explanation of error")
    timestamp: str = Field(description="ISO timestamp of error")


def _format_timestamp(raw_time: Any) -> Optional[str]:
    """Converts unix timestamp or datetime into formatted UTC string."""
    if not raw_time:
        return None
    try:
        if isinstance(raw_time, (int, float)):
            dt = datetime.fromtimestamp(raw_time, tz=timezone.utc)
            return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        return str(raw_time)
    except Exception:
        return None


STOP_WORDS = {
    "why", "has", "have", "had", "what", "is", "are", "the", "a", "an",
    "for", "and", "of", "to", "in", "on", "at", "by", "with", "about",
    "recent", "recently", "moved", "down", "up", "shares", "price", "situation"
}


def _extract_fallback_keywords(query: str) -> Optional[str]:
    """Extracts high-signal keywords or company names from a verbose query."""
    tokens = [t.strip(",.?!:\"'") for t in query.split() if t.strip(",.?!:\"'")]
    meaningful = [t for t in tokens if t.lower() not in STOP_WORDS]
    if len(meaningful) >= 1:
        # Return the top 2 most prominent subject terms (e.g. 'NVIDIA AI')
        return " ".join(meaningful[:2])
    return None


def fetch_financial_news(query: str, max_results: int = 5) -> Union[NewsSearchResults, NewsSearchError]:
    """Searches for recent financial news articles matching a query or company ticker.
    
    Args:
        query: Company ticker, company name, or research topic (e.g. 'NVDA', 'Apple AI launch').
        max_results: Maximum number of articles to return (defaults to 5 to prevent context stuffing).
        
    Returns:
        NewsSearchResults with articles list, or NewsSearchError on failure.
    """
    now_str = datetime.now(timezone.utc).isoformat()
    clean_query = query.strip()
    
    if not clean_query:
        return NewsSearchError(
            error_code="EMPTY_QUERY",
            query=query,
            message="News search query cannot be empty.",
            timestamp=now_str
        )

    try:
        # Step 1: Query Yahoo Finance News Engine with primary query
        search = yf.Search(clean_query, news_count=max(1, min(max_results, 10)))
        raw_news = getattr(search, "news", []) or []

        # Step 2: Query Relaxation Fallback if long query returned 0 results
        if not raw_news and len(clean_query.split()) > 2:
            fallback_query = _extract_fallback_keywords(clean_query)
            if fallback_query and fallback_query.lower() != clean_query.lower():
                search = yf.Search(fallback_query, news_count=max(1, min(max_results, 10)))
                raw_news = getattr(search, "news", []) or []
        
        articles: List[NewsItem] = []
        for item in raw_news:
            if not isinstance(item, dict):
                continue
                
            title = item.get("title")
            link = item.get("link")
            if not title or not link:
                continue
                
            publisher = item.get("publisher", "Financial Media")
            raw_time = item.get("providerPublishTime")
            formatted_date = _format_timestamp(raw_time)
            tickers = item.get("relatedTickers", []) or []
            
            # Construct a clear excerpt snippet
            snippet = f"Story covering {', '.join(tickers) if tickers else clean_query} published by {publisher}."
            
            articles.append(
                NewsItem(
                    title=title.strip(),
                    publisher=publisher.strip(),
                    url=link.strip(),
                    published_date=formatted_date,
                    snippet=snippet,
                    related_tickers=tickers
                )
            )

        if not articles:
            return NewsSearchResults(
                status="success",
                query=clean_query,
                result_count=0,
                articles=[],
                timestamp=now_str,
                source="Yahoo Finance News Engine"
            )

        return NewsSearchResults(
            status="success",
            query=clean_query,
            result_count=len(articles),
            articles=articles[:max_results],
            timestamp=now_str,
            source="Yahoo Finance News Engine"
        )

    except Exception as e:
        return NewsSearchError(
            error_code="NEWS_FETCH_EXCEPTION",
            query=clean_query,
            message=f"Failed to fetch financial news for '{clean_query}': {str(e)}",
            timestamp=now_str
        )



@tool
def search_financial_news(query: str) -> str:
    """Search for recent financial news articles, market catalysts, and press reports.

    Use this tool when the user asks why a stock moved recently, what recent events occurred,
    market sentiment, regulatory inquiries, product launches, or earnings announcements.
    
    Args:
        query: Ticker symbol, company name, or catalyst topic (e.g. 'NVDA', 'Apple earnings').
        
    Returns:
        JSON string containing structured news dispatches with headlines, publishers, dates, and links.
    """
    result = fetch_financial_news(query, max_results=5)
    return result.model_dump_json()
