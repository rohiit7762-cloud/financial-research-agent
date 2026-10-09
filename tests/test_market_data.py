"""Unit tests for Market Data Tool (tools/market_data.py)."""

import json
import pytest
from tools.market_data import fetch_market_data, get_market_data, MarketDataSuccess, MarketDataError


def test_fetch_market_data_valid_symbol():
    """Verify market data fetch for a known liquid ticker (AAPL)."""
    result = fetch_market_data("AAPL")
    
    assert isinstance(result, MarketDataSuccess)
    assert result.status == "success"
    assert result.symbol == "AAPL"
    assert result.current_price > 0
    assert result.currency in ["USD", "USD"]
    assert result.fifty_two_week_high is not None
    assert result.fifty_two_week_low is not None
    assert result.fifty_two_week_high >= result.fifty_two_week_low
    assert result.market_cap is not None
    assert result.market_cap_formatted is not None
    assert "$" in result.market_cap_formatted


def test_fetch_market_data_symbol_with_dollar_sign():
    """Verify that leading $ or lowercase characters are cleanly sanitized."""
    result = fetch_market_data("  $nvda  ")
    
    assert isinstance(result, MarketDataSuccess)
    assert result.status == "success"
    assert result.symbol == "NVDA"
    assert result.current_price > 0


def test_fetch_market_data_invalid_symbol():
    """Verify graceful error containment for a non-existent ticker."""
    result = fetch_market_data("XYZNOTAREALTICKER999")
    
    assert isinstance(result, MarketDataError)
    assert result.status == "error"
    assert result.error_code in ["INVALID_TICKER", "INVALID_TICKER_FORMAT", "DATA_FETCH_EXCEPTION"]
    assert "XYZNOTAREALTICKER999" in result.message or result.symbol == "XYZNOTAREALTICKER999"


def test_langchain_tool_invocation():
    """Verify invocation through the LangChain @tool interface."""
    raw_json = get_market_data.invoke({"symbol": "MSFT"})
    assert isinstance(raw_json, str)
    
    payload = json.loads(raw_json)
    assert payload["status"] == "success"
    assert payload["symbol"] == "MSFT"
    assert payload["current_price"] > 0
    assert "market_cap_formatted" in payload


if __name__ == "__main__":
    pytest.main(["-v", "tests/test_market_data.py"])
