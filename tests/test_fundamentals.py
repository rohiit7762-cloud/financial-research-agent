"""Unit tests for Fundamentals Tool (tools/fundamentals.py)."""

import json
import pytest
from tools.fundamentals import fetch_company_fundamentals, get_company_fundamentals, FundamentalMetrics, FundamentalsError


def test_fetch_fundamentals_valid_symbol():
    """Verify fundamentals extraction on a major equity (AAPL)."""
    result = fetch_company_fundamentals("AAPL")
    
    assert isinstance(result, FundamentalMetrics)
    assert result.status == "success"
    assert result.symbol == "AAPL"
    assert result.total_revenue is not None and result.total_revenue > 0
    assert result.total_revenue_formatted is not None
    assert ("B" in result.total_revenue_formatted or "T" in result.total_revenue_formatted)
    assert result.eps_trailing is not None
    assert result.operating_margin is not None
    assert result.total_debt is not None
    assert result.total_cash is not None
    assert result.free_cash_flow is not None


def test_fetch_fundamentals_symbol_sanitization():
    """Verify that leading $ and whitespace are handled cleanly."""
    result = fetch_company_fundamentals("  $msft  ")
    
    assert isinstance(result, FundamentalMetrics)
    assert result.status == "success"
    assert result.symbol == "MSFT"
    assert result.pe_ratio is not None and result.pe_ratio > 0


def test_fetch_fundamentals_invalid_symbol():
    """Verify graceful error containment for a bogus ticker."""
    result = fetch_company_fundamentals("XYZINVALIDBOGUS999")
    
    assert isinstance(result, FundamentalsError)
    assert result.status == "error"
    assert result.error_code in ["INVALID_TICKER", "INVALID_TICKER_FORMAT", "FUNDAMENTALS_FETCH_EXCEPTION"]
    assert "XYZINVALIDBOGUS999" in result.message or result.symbol == "XYZINVALIDBOGUS999"


def test_langchain_fundamentals_tool_invocation():
    """Verify invocation through the LangChain @tool interface."""
    raw_json = get_company_fundamentals.invoke({"symbol": "NVDA"})
    assert isinstance(raw_json, str)
    
    payload = json.loads(raw_json)
    assert payload["status"] == "success"
    assert payload["symbol"] == "NVDA"
    assert "total_revenue_formatted" in payload
    assert "free_cash_flow_formatted" in payload


if __name__ == "__main__":
    pytest.main(["-v", "tests/test_fundamentals.py"])
