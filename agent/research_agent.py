"""Core LangGraph Research Agent Workflow.

Orchestrates the dynamic ReAct tool-calling loop, immutable evidence ledger,
tool execution bus, stopping conditions, and final report synthesis.
"""

import os
import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union, Tuple
from dotenv import load_dotenv

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage
)
from langgraph.graph import StateGraph, START, END

from agent.state import ResearchAgentState, EvidenceItem
from agent.prompts import SYSTEM_REASONING_PROMPT, SYNTHESIS_SYSTEM_PROMPT
from tools.market_data import get_market_data
from tools.fundamentals import get_company_fundamentals
from tools.news import search_financial_news
from tools.filing_search import search_filing_rag

load_dotenv()

# Complete list of exposed agent tools
AGENT_TOOLS = [
    get_market_data,
    get_company_fundamentals,
    search_financial_news,
    search_filing_rag
]

TOOL_MAP = {t.name: t for t in AGENT_TOOLS}


def _extract_text(content: Any) -> str:
    """Safely extracts raw text from string or message content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif hasattr(item, "text"):
                parts.append(getattr(item, "text"))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def get_llm(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.1,
    google_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None
):
    """Instantiates a tool-calling LLM instance from Google Gemini or OpenAI."""
    g_key = google_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    o_key = openai_api_key or os.getenv("OPENAI_API_KEY")

    # Priority 1: Google Gemini (default)
    if (provider == "google" or not provider) and g_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        target_model = model_name or os.getenv("DEFAULT_MODEL", "gemini-3.1-flash-lite")
        return ChatGoogleGenerativeAI(
            model=target_model,
            temperature=temperature,
            google_api_key=g_key
        )

    # Priority 2: OpenAI
    if (provider == "openai" or not provider) and o_key:
        from langchain_openai import ChatOpenAI
        target_model = model_name or "gpt-4o-mini"
        return ChatOpenAI(
            model=target_model,
            temperature=temperature,
            api_key=o_key
        )

    raise ValueError(
        "No active LLM API key detected. Please configure GOOGLE_API_KEY or OPENAI_API_KEY in your .env file."
    )


def _convert_tool_output_to_evidence(
    tool_name: str,
    tool_args: Dict[str, Any],
    raw_output: str,
    source_index: int
) -> Tuple[Optional[EvidenceItem], str]:
    """Parses raw tool output and constructs a verified EvidenceItem with UI activity string."""
    now_str = datetime.now(timezone.utc).isoformat()
    source_id = f"SRC-{source_index:02d}"

    try:
        data = json.loads(raw_output)
    except Exception:
        data = {"raw": raw_output}

    if tool_name == "get_market_data":
        symbol = data.get("symbol", tool_args.get("symbol", "EQUITY"))
        price = data.get("current_price", "N/A")
        currency = data.get("currency", "USD")
        change = data.get("change_percent", "N/A")
        mc = data.get("market_cap_formatted", "N/A")
        
        title = f"{symbol} Real-Time Market Data ({data.get('company_name', symbol)})"
        text = f"Current Price: {currency} {price} (Change: {change}%). 52W High: {data.get('fifty_two_week_high')}, 52W Low: {data.get('fifty_two_week_low')}, Market Cap: {mc}."
        activity = f"Retrieved market data for {symbol} ({currency} {price})"
        
        return EvidenceItem(
            source_id=source_id,
            source_type="market_data",
            tool_name=tool_name,
            title=title,
            url_or_ref=f"https://finance.yahoo.com/quote/{symbol}",
            retrieved_text=text,
            timestamp=now_str
        ), activity

    elif tool_name == "get_company_fundamentals":
        symbol = data.get("symbol", tool_args.get("symbol", "EQUITY"))
        pe = data.get("pe_ratio", "N/A")
        rev = data.get("total_revenue_formatted", "N/A")
        ni = data.get("net_income_formatted", "N/A")
        fcf = data.get("free_cash_flow_formatted", "N/A")
        debt = data.get("total_debt_formatted", "N/A")
        
        title = f"{symbol} Balance Sheet & Fundamentals ({data.get('company_name', symbol)})"
        text = f"Revenue (TTM): {rev}, Net Income: {ni}, Free Cash Flow: {fcf}, P/E: {pe}, Total Debt: {debt}, Operating Margin: {data.get('operating_margin')}%."
        activity = f"Retrieved company fundamentals for {symbol} (P/E: {pe}, Revenue: {rev})"
        
        return EvidenceItem(
            source_id=source_id,
            source_type="fundamentals",
            tool_name=tool_name,
            title=title,
            url_or_ref=f"https://finance.yahoo.com/quote/{symbol}/financials",
            retrieved_text=text,
            timestamp=now_str
        ), activity

    elif tool_name == "search_financial_news":
        articles = data.get("articles", [])
        query = data.get("query", tool_args.get("query", ""))
        article_titles = [a.get("title", "") for a in articles[:3]]
        
        title = f"Financial News Catalysts for '{query}'"
        text = " | ".join([f"{a.get('title')} ({a.get('publisher')}, {a.get('published_date')} - {a.get('url')})" for a in articles[:3]])
        activity = f"Searched recent financial news for '{query}' ({len(articles)} articles found)"
        
        url_ref = articles[0].get("url") if articles else "https://finance.yahoo.com/news"
        return EvidenceItem(
            source_id=source_id,
            source_type="news",
            tool_name=tool_name,
            title=title,
            url_or_ref=url_ref,
            retrieved_text=text if text else "No recent articles found.",
            timestamp=now_str
        ), activity

    elif tool_name == "search_filing_rag":
        results = data.get("results", [])
        query = data.get("query", tool_args.get("query", ""))
        symbol = data.get("symbol_filter") or tool_args.get("symbol", "10-K")
        
        title = f"SEC 10-K Filing Risk Factors ({symbol})"
        url_ref = results[0].get("source_url") if results else "https://www.sec.gov/edgar"
        
        excerpts = [f"[{r.get('chunk_id')}] {r.get('excerpt')}" for r in results]
        text = "\n\n".join(excerpts) if excerpts else "No matching filing disclosures retrieved."
        activity = f"Retrieved {len(results)} SEC 10-K risk disclosures for {symbol}"
        
        return EvidenceItem(
            source_id=source_id,
            source_type="filing_rag",
            tool_name=tool_name,
            title=title,
            url_or_ref=url_ref,
            retrieved_text=text,
            timestamp=now_str
        ), activity

    # Generic fallback
    return EvidenceItem(
        source_id=source_id,
        source_type="general",
        tool_name=tool_name,
        title=f"Tool Observation ({tool_name})",
        url_or_ref="Internal Tool Output",
        retrieved_text=raw_output[:500],
        timestamp=now_str
    ), f"Executed tool {tool_name}"


def build_research_agent(llm=None, max_iterations: int = 5):
    """Constructs and compiles the cyclic LangGraph Research Agent workflow."""
    if llm is None:
        llm = get_llm()

    llm_with_tools = llm.bind_tools(AGENT_TOOLS)

    def reasoning_node(state: ResearchAgentState) -> Dict[str, Any]:
        """Evaluates message history and decides next tool action(s) or termination."""
        messages = state["messages"]
        
        # Inject system prompt at beginning of conversational thread
        if not messages or not isinstance(messages[0], SystemMessage):
            system_msg = SystemMessage(content=SYSTEM_REASONING_PROMPT)
            call_messages = [system_msg] + messages
        else:
            call_messages = messages

        response: AIMessage = llm_with_tools.invoke(call_messages)
        new_iteration = state.get("iteration_count", 0) + 1
        
        return {
            "messages": [response],
            "iteration_count": new_iteration
        }

    def tool_execution_node(state: ResearchAgentState) -> Dict[str, Any]:
        """Dispatches requested tool calls, updates evidence ledger, and logs activity."""
        last_message: AIMessage = state["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", []) or []
        
        ledger = list(state.get("evidence_ledger", []))
        activity_log = list(state.get("tool_activity", []))
        tool_responses: List[ToolMessage] = []

        for call in tool_calls:
            tool_name = call["name"]
            tool_args = call.get("args", {})
            call_id = call.get("id", f"call_{len(tool_responses)}")

            tool_fn = TOOL_MAP.get(tool_name)
            if not tool_fn:
                err_output = json.dumps({"status": "error", "message": f"Tool '{tool_name}' does not exist."})
                tool_responses.append(ToolMessage(content=err_output, tool_call_id=call_id, name=tool_name))
                continue

            try:
                raw_result = tool_fn.invoke(tool_args)
                raw_result_str = raw_result if isinstance(raw_result, str) else json.dumps(raw_result)
            except Exception as e:
                raw_result_str = json.dumps({"status": "error", "message": f"Tool execution failed: {str(e)}"})

            # Transform raw output into immutable EvidenceItem
            evidence_item, activity_desc = _convert_tool_output_to_evidence(
                tool_name=tool_name,
                tool_args=tool_args,
                raw_output=raw_result_str,
                source_index=len(ledger) + 1
            )
            
            if evidence_item:
                ledger.append(evidence_item.model_dump())
            activity_log.append(f"✓ {activity_desc}")

            tool_responses.append(
                ToolMessage(
                    content=raw_result_str,
                    tool_call_id=call_id,
                    name=tool_name
                )
            )

        return {
            "messages": tool_responses,
            "evidence_ledger": ledger,
            "tool_activity": activity_log
        }

    def should_continue(state: ResearchAgentState) -> str:
        """Determines whether to execute another tool turn or transition to synthesis."""
        # 1. Enforce hard iteration ceiling
        max_iter = state.get("max_iterations", max_iterations)
        if state.get("iteration_count", 0) >= max_iter:
            return "synthesis"

        # 2. Check if agent emitted tool calls
        last_message = state["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", None)
        if not tool_calls:
            return "synthesis"

        return "tools"

    def synthesis_node(state: ResearchAgentState) -> Dict[str, Any]:
        """Compiles the final structured research report backed by the evidence ledger."""
        query = state["query"]
        ledger = state.get("evidence_ledger", [])

        # Format accumulated evidence ledger as structured context for synthesizer
        evidence_blocks = []
        for item in ledger:
            evidence_blocks.append(
                f"[{item['source_id']}] ({item['source_type'].upper()}) {item['title']}\n"
                f"Reference/URL: {item['url_or_ref']}\n"
                f"Factual Disclosures: {item['retrieved_text']}\n"
            )
        evidence_context = "\n---\n".join(evidence_blocks) if evidence_blocks else "No external evidence retrieved."

        user_prompt = (
            f"User Research Query: {query}\n\n"
            f"ACCUMULATED EVIDENCE LEDGER:\n{evidence_context}\n\n"
            "Please synthesize the final comprehensive equity research report following your structured format. "
            "Anchor every factual claim to its specific [SRC-xx] tag."
        )

        synthesis_messages = [
            SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]

        final_response: AIMessage = llm.invoke(synthesis_messages)
        
        # Calculate qualitative confidence score based on source variety
        source_types = {item.get("source_type") for item in ledger}
        if len(source_types) >= 3:
            confidence = "HIGH"
        elif len(source_types) >= 2:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        activity_log = list(state.get("tool_activity", []))
        activity_log.append("✓ Synthesized final research report with source citations")

        return {
            "final_report": _extract_text(final_response.content),
            "confidence_rating": confidence,
            "tool_activity": activity_log
        }

    # Assemble State Machine Graph
    builder = StateGraph(ResearchAgentState)
    builder.add_node("agent", reasoning_node)
    builder.add_node("tools", tool_execution_node)
    builder.add_node("synthesis", synthesis_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "synthesis": "synthesis"
        }
    )
    builder.add_edge("tools", "agent")
    builder.add_edge("synthesis", END)

    return builder.compile()


def run_research(
    query: str,
    ticker: Optional[str] = None,
    llm=None,
    max_iterations: int = 5
) -> ResearchAgentState:
    """Convenience helper to run an end-to-end research query through the agent."""
    agent_app = build_research_agent(llm=llm, max_iterations=max_iterations)
    
    initial_state: ResearchAgentState = {
        "query": query,
        "ticker": ticker,
        "messages": [HumanMessage(content=query)],
        "evidence_ledger": [],
        "tool_activity": ["Initiated research workflow"],
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "final_report": None,
        "confidence_rating": None,
        "errors": []
    }

    final_state = agent_app.invoke(initial_state)
    return final_state
