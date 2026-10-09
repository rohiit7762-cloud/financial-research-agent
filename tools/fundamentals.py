"""Fundamentals Tool for Financial Research Agent.

Retrieves and normalizes audited financial health metrics, valuation ratios,
solvency metrics, and profitability figures from SEC-filed company statements
via yfinance with strict Pydantic validation.
"""

from datetime import datetime, timezone
from typing import Optional, Union, Dict, Any
import yfinance as yf
from pydantic import BaseModel, Field
from langchain_core.tools import tool


class FundamentalMetrics(BaseModel):
    """Schema for successful company fundamentals retrieval."""
    status: str = Field(default="success", description="Status indicator")
    symbol: str = Field(description="Ticker symbol, e.g. AAPL")
    company_name: Optional[str] = Field(default=None, description="Legal company name")
    currency: str = Field(default="USD", description="Reporting currency")
    sector: Optional[str] = Field(default=None, description="Economic sector, e.g. Technology")
    industry: Optional[str] = Field(default=None, description="Industry categorization")
    
    # Valuation Multiples
    pe_ratio: Optional[float] = Field(default=None, description="Trailing Price-to-Earnings ratio")
    forward_pe: Optional[float] = Field(default=None, description="Forward 1-year Price-to-Earnings ratio")
    peg_ratio: Optional[float] = Field(default=None, description="Price/Earnings-to-Growth ratio")
    eps_trailing: Optional[float] = Field(default=None, description="Diluted Earnings Per Share (TTM)")
    eps_forward: Optional[float] = Field(default=None, description="Consensus Estimated EPS (Forward)")
    
    # Income & Profitability
    total_revenue: Optional[float] = Field(default=None, description="Total revenue (Trailing 12 Months)")
    total_revenue_formatted: Optional[str] = Field(default=None, description="Formatted total revenue")
    net_income: Optional[float] = Field(default=None, description="Net income attributable to common shares")
    net_income_formatted: Optional[str] = Field(default=None, description="Formatted net income")
    operating_margin: Optional[float] = Field(default=None, description="Operating profit margin as a percentage")
    profit_margin: Optional[float] = Field(default=None, description="Net profit margin as a percentage")
    return_on_equity: Optional[float] = Field(default=None, description="Return on equity as a percentage")
    
    # Balance Sheet & Solvency
    total_debt: Optional[float] = Field(default=None, description="Total outstanding debt obligations")
    total_debt_formatted: Optional[str] = Field(default=None, description="Formatted total debt")
    total_cash: Optional[float] = Field(default=None, description="Cash and cash equivalents")
    total_cash_formatted: Optional[str] = Field(default=None, description="Formatted total cash")
    free_cash_flow: Optional[float] = Field(default=None, description="Operating cash flow minus CapEx (TTM)")
    free_cash_flow_formatted: Optional[str] = Field(default=None, description="Formatted free cash flow")
    debt_to_equity: Optional[float] = Field(default=None, description="Total debt divided by total shareholder equity")
    
    timestamp: str = Field(description="ISO timestamp of data retrieval")
    source: str = Field(default="Yahoo Finance (Audited Public Statements)", description="Source provenance")


class FundamentalsError(BaseModel):
    """Schema for failed fundamentals retrieval."""
    status: str = Field(default="error", description="Status indicator")
    error_code: str = Field(description="Error classification")
    symbol: str = Field(description="Ticker symbol requested")
    message: str = Field(description="Human-readable error details")
    timestamp: str = Field(description="ISO timestamp of error")


def _format_financial_amount(value: Optional[float], currency: str = "USD") -> Optional[str]:
    """Formats large balance sheet and income numbers into readable notations ($B, $M, $T)."""
    if value is None:
        return None
    curr_symbol = "$" if currency == "USD" else f"{currency} "
    abs_val = abs(value)
    sign = "-" if value < 0 else ""
    
    if abs_val >= 1e12:
        return f"{sign}{curr_symbol}{abs_val / 1e12:.2f}T"
    if abs_val >= 1e9:
        return f"{sign}{curr_symbol}{abs_val / 1e9:.2f}B"
    if abs_val >= 1e6:
        return f"{sign}{curr_symbol}{abs_val / 1e6:.2f}M"
    return f"{sign}{curr_symbol}{abs_val:,.2f}"


def fetch_company_fundamentals(symbol: str) -> Union[FundamentalMetrics, FundamentalsError]:
    """Retrieves selected audited financial statements and ratios for an equity symbol.
    
    Args:
        symbol: Stock ticker symbol (e.g. 'AAPL', 'MSFT', 'NVDA').
        
    Returns:
        FundamentalMetrics on success, or FundamentalsError on failure.
    """
    now_str = datetime.now(timezone.utc).isoformat()
    clean_symbol = symbol.strip().upper().lstrip("$")
    
    if not clean_symbol or len(clean_symbol) > 10:
        return FundamentalsError(
            error_code="INVALID_TICKER_FORMAT",
            symbol=symbol,
            message=f"'{symbol}' is not a valid equity ticker format.",
            timestamp=now_str
        )

    try:
        ticker = yf.Ticker(clean_symbol)
        info = ticker.info
        
        # Yahoo Finance returns empty or 1-key dict on invalid tickers
        if not info or len(info) <= 2 or (info.get("trailingPegRatio") is None and not info.get("totalRevenue") and not info.get("trailingPE")):
            # Double-check if fast_info exists, otherwise reject as invalid ticker
            if not getattr(ticker, "fast_info", None) or ticker.fast_info.get("lastPrice") is None:
                return FundamentalsError(
                    error_code="INVALID_TICKER",
                    symbol=clean_symbol,
                    message=f"Fundamental financial statements not found for '{clean_symbol}'. The symbol may be invalid or delisted.",
                    timestamp=now_str
                )

        currency = str(info.get("financialCurrency") or info.get("currency") or "USD")
        company_name = info.get("longName") or info.get("shortName") or clean_symbol
        sector = info.get("sector")
        industry = info.get("industry")

        # Valuation ratios
        pe_ratio = round(float(info["trailingPE"]), 2) if info.get("trailingPE") is not None else None
        forward_pe = round(float(info["forwardPE"]), 2) if info.get("forwardPE") is not None else None
        peg_ratio = round(float(info["pegRatio"]), 2) if info.get("pegRatio") is not None else None
        eps_trailing = round(float(info["trailingEps"]), 2) if info.get("trailingEps") is not None else None
        eps_forward = round(float(info["forwardEps"]), 2) if info.get("forwardEps") is not None else None

        # Income metrics
        raw_rev = info.get("totalRevenue")
        total_revenue = float(raw_rev) if raw_rev is not None else None
        rev_formatted = _format_financial_amount(total_revenue, currency)

        raw_ni = info.get("netIncomeToCommon")
        net_income = float(raw_ni) if raw_ni is not None else None
        ni_formatted = _format_financial_amount(net_income, currency)

        op_margin = round(float(info["operatingMargins"]) * 100, 2) if info.get("operatingMargins") is not None else None
        profit_margin = round(float(info["profitMargins"]) * 100, 2) if info.get("profitMargins") is not None else None
        roe = round(float(info["returnOnEquity"]) * 100, 2) if info.get("returnOnEquity") is not None else None

        # Balance sheet metrics
        raw_debt = info.get("totalDebt")
        total_debt = float(raw_debt) if raw_debt is not None else None
        debt_formatted = _format_financial_amount(total_debt, currency)

        raw_cash = info.get("totalCash")
        total_cash = float(raw_cash) if raw_cash is not None else None
        cash_formatted = _format_financial_amount(total_cash, currency)

        raw_fcf = info.get("freeCashflow")
        free_cash_flow = float(raw_fcf) if raw_fcf is not None else None
        fcf_formatted = _format_financial_amount(free_cash_flow, currency)

        debt_to_equity = round(float(info["debtToEquity"]), 2) if info.get("debtToEquity") is not None else None

        return FundamentalMetrics(
            symbol=clean_symbol,
            company_name=company_name,
            currency=currency,
            sector=sector,
            industry=industry,
            pe_ratio=pe_ratio,
            forward_pe=forward_pe,
            peg_ratio=peg_ratio,
            eps_trailing=eps_trailing,
            eps_forward=eps_forward,
            total_revenue=total_revenue,
            total_revenue_formatted=rev_formatted,
            net_income=net_income,
            net_income_formatted=ni_formatted,
            operating_margin=op_margin,
            profit_margin=profit_margin,
            return_on_equity=roe,
            total_debt=total_debt,
            total_debt_formatted=debt_formatted,
            total_cash=total_cash,
            total_cash_formatted=cash_formatted,
            free_cash_flow=free_cash_flow,
            free_cash_flow_formatted=fcf_formatted,
            debt_to_equity=debt_to_equity,
            timestamp=now_str,
            source="Yahoo Finance (Audited Public Statements)"
        )

    except Exception as e:
        return FundamentalsError(
            error_code="FUNDAMENTALS_FETCH_EXCEPTION",
            symbol=clean_symbol,
            message=f"Failed to fetch fundamentals for '{clean_symbol}': {str(e)}",
            timestamp=now_str
        )


@tool
def get_company_fundamentals(symbol: str) -> str:
    """Retrieve audited financial statements and ratios for an equity symbol.

    Use this tool when the user asks about company valuation, P/E ratio, EPS,
    total revenue, net income, profit margins, balance sheet health, debt load,
    cash reserves, free cash flow, or solvency.
    
    Args:
        symbol: The stock ticker symbol (e.g. 'AAPL', 'MSFT', 'NVDA', 'TSLA').
        
    Returns:
        JSON string containing structured financial health metrics or an error payload.
    """
    result = fetch_company_fundamentals(symbol)
    return result.model_dump_json()
