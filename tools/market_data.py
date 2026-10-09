"""Market Data Tool for Financial Research Agent.

Fetches normalized, real-time market data quotes, price movements,
52-week trading ranges, and market capitalization using yfinance with
defensive Pydantic validation and error containment.
"""

from datetime import datetime, timezone
from typing import Optional, Union, Dict, Any
import yfinance as yf
from pydantic import BaseModel, Field
from langchain_core.tools import tool


class MarketDataSuccess(BaseModel):
    """Schema for successful market data retrieval."""
    status: str = Field(default="success", description="Status indicator")
    symbol: str = Field(description="Ticker symbol, e.g. AAPL")
    company_name: Optional[str] = Field(default=None, description="Company commercial name")
    currency: str = Field(default="USD", description="Currency of quotes")
    current_price: float = Field(description="Most recent traded market price")
    previous_close: Optional[float] = Field(default=None, description="Previous trading day closing price")
    change: Optional[float] = Field(default=None, description="Absolute price change from previous close")
    change_percent: Optional[float] = Field(default=None, description="Percentage change from previous close")
    volume: Optional[int] = Field(default=None, description="Latest daily trading volume")
    fifty_two_week_high: Optional[float] = Field(default=None, description="52-week highest traded price")
    fifty_two_week_low: Optional[float] = Field(default=None, description="52-week lowest traded price")
    market_cap: Optional[float] = Field(default=None, description="Raw market capitalization")
    market_cap_formatted: Optional[str] = Field(default=None, description="Formatted market cap, e.g. $3.25T or $850.40B")
    timestamp: str = Field(description="ISO timestamp of data retrieval")
    source: str = Field(default="Yahoo Finance", description="Underlying market data provider")


class MarketDataError(BaseModel):
    """Schema for failed market data retrieval."""
    status: str = Field(default="error", description="Status indicator")
    error_code: str = Field(description="Machine-readable error code")
    symbol: str = Field(description="Ticker symbol that caused error")
    message: str = Field(description="Human-readable explanation of failure")
    timestamp: str = Field(description="ISO timestamp of error")


def _format_market_cap(value: Optional[float], currency: str = "USD") -> Optional[str]:
    """Formats large numeric market cap into readable strings ($B, $T)."""
    if value is None or value <= 0:
        return None
    curr_symbol = "$" if currency == "USD" else f"{currency} "
    if value >= 1e12:
        return f"{curr_symbol}{value / 1e12:.2f}T"
    if value >= 1e9:
        return f"{curr_symbol}{value / 1e9:.2f}B"
    if value >= 1e6:
        return f"{curr_symbol}{value / 1e6:.2f}M"
    return f"{curr_symbol}{value:,.2f}"


def fetch_market_data(symbol: str) -> Union[MarketDataSuccess, MarketDataError]:
    """Fetches and normalizes market data for a given ticker symbol.
    
    Args:
        symbol: Equity ticker symbol (e.g. 'AAPL', 'MSFT', 'NVDA').
        
    Returns:
        MarketDataSuccess on success, or MarketDataError on failure.
    """
    now_str = datetime.now(timezone.utc).isoformat()
    clean_symbol = symbol.strip().upper().lstrip("$")
    
    if not clean_symbol or len(clean_symbol) > 10:
        return MarketDataError(
            error_code="INVALID_TICKER_FORMAT",
            symbol=symbol,
            message=f"'{symbol}' is not a valid stock ticker format.",
            timestamp=now_str
        )

    try:
        ticker = yf.Ticker(clean_symbol)
        fast_info = ticker.fast_info
        
        # Check if lastPrice exists and is valid
        last_price = None
        try:
            raw_price = fast_info.get("lastPrice") or fast_info.get("last_price")
            if raw_price is not None:
                last_price = round(float(raw_price), 2)
        except Exception:
            last_price = None

        if last_price is None or last_price <= 0:
            return MarketDataError(
                error_code="INVALID_TICKER",
                symbol=clean_symbol,
                message=f"Could not retrieve pricing data for symbol '{clean_symbol}'. The ticker may be invalid, delisted, or unavailable.",
                timestamp=now_str
            )

        # Extract remaining fields defensively from fast_info
        currency = str(fast_info.get("currency", "USD"))
        
        prev_close = None
        try:
            raw_prev = fast_info.get("previousClose") or fast_info.get("regularMarketPreviousClose")
            if raw_prev is not None:
                prev_close = round(float(raw_prev), 2)
        except Exception:
            pass

        # Compute change and change_percent
        change = None
        change_pct = None
        if prev_close and prev_close > 0:
            change = round(last_price - prev_close, 2)
            change_pct = round((change / prev_close) * 100, 2)

        volume = None
        try:
            raw_vol = fast_info.get("lastVolume")
            if raw_vol is not None:
                volume = int(raw_vol)
        except Exception:
            pass

        year_high = None
        try:
            raw_high = fast_info.get("yearHigh")
            if raw_high is not None:
                year_high = round(float(raw_high), 2)
        except Exception:
            pass

        year_low = None
        try:
            raw_low = fast_info.get("yearLow")
            if raw_low is not None:
                year_low = round(float(raw_low), 2)
        except Exception:
            pass

        market_cap = None
        try:
            raw_mc = fast_info.get("marketCap")
            if raw_mc is not None and float(raw_mc) > 0:
                market_cap = float(raw_mc)
        except Exception:
            pass

        # Retrieve company name from ticker.info if available, without failing if info throttles
        company_name = clean_symbol
        try:
            info = ticker.info
            company_name = info.get("shortName") or info.get("longName") or clean_symbol
        except Exception:
            pass

        formatted_mc = _format_market_cap(market_cap, currency)

        return MarketDataSuccess(
            symbol=clean_symbol,
            company_name=company_name,
            currency=currency,
            current_price=last_price,
            previous_close=prev_close,
            change=change,
            change_percent=change_pct,
            volume=volume,
            fifty_two_week_high=year_high,
            fifty_two_week_low=year_low,
            market_cap=market_cap,
            market_cap_formatted=formatted_mc,
            timestamp=now_str,
            source="Yahoo Finance"
        )

    except Exception as e:
        return MarketDataError(
            error_code="DATA_FETCH_EXCEPTION",
            symbol=clean_symbol,
            message=f"Failed to fetch market data for '{clean_symbol}': {str(e)}",
            timestamp=now_str
        )


@tool
def get_market_data(symbol: str) -> str:
    """Retrieve current stock price, daily change, volume, 52-week range, and market cap for an equity ticker.

    Use this tool when the user asks for stock quotes, current price, recent price movement,
    trading volume, valuation size, or 52-week trading bounds for a publicly traded company.
    
    Args:
        symbol: The stock ticker symbol (e.g. 'AAPL', 'MSFT', 'NVDA', 'TSLA').
        
    Returns:
        JSON string containing structured market metrics or an error payload.
    """
    result = fetch_market_data(symbol)
    return result.model_dump_json()
