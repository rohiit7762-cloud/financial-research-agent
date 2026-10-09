"""Unit tests for Financial News Tool (tools/news.py)."""

import json
import pytest
from tools.news import fetch_financial_news, search_financial_news, NewsSearchResults, NewsSearchError


def test_fetch_news_valid_ticker():
    """Verify news extraction for a high-volume equity (AAPL)."""
    result = fetch_financial_news("AAPL", max_results=3)
    
    assert isinstance(result, NewsSearchResults)
    assert result.status == "success"
    assert result.result_count > 0
    assert len(result.articles) > 0
    
    first_article = result.articles[0]
    assert len(first_article.title) > 5
    assert len(first_article.publisher) > 0
    assert first_article.url.startswith("http")
    assert first_article.published_date is not None


def test_fetch_news_conceptual_topic():
    """Verify that conceptual research queries work beyond just ticker symbols."""
    result = fetch_financial_news("NVIDIA AI Blackwell GPU demand", max_results=3)
    
    assert isinstance(result, NewsSearchResults)
    assert result.status == "success"
    assert result.result_count > 0
    assert any("Nvidia" in a.title or "AI" in a.title or "Blackwell" in a.title or "GPU" in a.title for a in result.articles)


def test_fetch_news_empty_query():
    """Verify defensive rejection of empty queries."""
    result = fetch_financial_news("   ")
    
    assert isinstance(result, NewsSearchError)
    assert result.status == "error"
    assert result.error_code == "EMPTY_QUERY"


def test_fetch_news_gibberish_returns_zero():
    """Verify that queries with no results return empty list cleanly without error."""
    result = fetch_financial_news("ZXCVBNMASDFGHJKL99887766554433", max_results=3)
    
    assert isinstance(result, NewsSearchResults)
    assert result.status == "success"
    assert result.result_count == 0
    assert result.articles == []


def test_langchain_news_tool_invocation():
    """Verify invocation through the LangChain @tool interface."""
    raw_json = search_financial_news.invoke({"query": "Microsoft Azure cloud growth"})
    assert isinstance(raw_json, str)
    
    payload = json.loads(raw_json)
    assert payload["status"] == "success"
    assert "articles" in payload
    assert isinstance(payload["articles"], list)


if __name__ == "__main__":
    pytest.main(["-v", "tests/test_news.py"])
