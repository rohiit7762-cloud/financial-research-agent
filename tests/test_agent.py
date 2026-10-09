"""Unit & Integration Tests for Agent Loop and State Transitions.

Tests:
1. Tool execution node and Evidence Ledger accumulation.
2. Iteration ceiling enforcement (stopping guard).
3. Routing conditions (tools vs. synthesis).
4. Multi-step mock LLM execution verifying dynamic multi-turn planning.
5. Live LLM execution if GOOGLE_API_KEY or OPENAI_API_KEY is configured.
"""

import os
import json
import pytest
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

from agent.state import ResearchAgentState, EvidenceItem
from agent.research_agent import (
    build_research_agent,
    run_research,
    _convert_tool_output_to_evidence,
    TOOL_MAP
)


def test_convert_tool_output_to_evidence_market_data():
    """Verify market data payload converts to normalized EvidenceItem with [SRC-01]."""
    mock_payload = json.dumps({
        "status": "success",
        "symbol": "AAPL",
        "company_name": "Apple Inc.",
        "currency": "USD",
        "current_price": 220.50,
        "change_percent": 1.25,
        "fifty_two_week_high": 237.23,
        "fifty_two_week_low": 164.08,
        "market_cap_formatted": "$3.35T"
    })
    
    item, activity = _convert_tool_output_to_evidence(
        tool_name="get_market_data",
        tool_args={"symbol": "AAPL"},
        raw_output=mock_payload,
        source_index=1
    )
    
    assert isinstance(item, EvidenceItem)
    assert item.source_id == "SRC-01"
    assert item.source_type == "market_data"
    assert "AAPL" in item.title
    assert "220.5" in item.retrieved_text
    assert "Retrieved market data for AAPL" in activity


def test_convert_tool_output_to_evidence_filing_rag():
    """Verify filing RAG payload converts to EvidenceItem."""
    mock_payload = json.dumps({
        "status": "success",
        "symbol_filter": "NVDA",
        "result_count": 1,
        "results": [{
            "chunk_id": "NVDA-2024-CHK-001",
            "symbol": "NVDA",
            "company_name": "NVIDIA Corporation",
            "section": "Item 1A. Risk Factors",
            "source_url": "https://www.sec.gov/edgar",
            "excerpt": "Export control regulations targeting advanced computing semiconductor chips..."
        }]
    })
    
    item, activity = _convert_tool_output_to_evidence(
        tool_name="search_filing_rag",
        tool_args={"query": "export controls", "symbol": "NVDA"},
        raw_output=mock_payload,
        source_index=2
    )
    
    assert item.source_id == "SRC-02"
    assert item.source_type == "filing_rag"
    assert "NVDA" in item.title
    assert "NVDA-2024-CHK-001" in item.retrieved_text
    assert "Retrieved 1 SEC 10-K risk disclosures for NVDA" in activity


class MockChatModel:
    """Mock LLM to test deterministic multi-step state graph transitions."""
    
    def __init__(self, step_responses, default_response=None):
        self.step_responses = step_responses
        self.default_response = default_response or AIMessage(content="# Default Synthesis Report")
        self.call_count = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        if self.call_count < len(self.step_responses):
            resp = self.step_responses[self.call_count]
        else:
            resp = self.default_response
        self.call_count += 1
        return resp


def test_multi_step_agent_execution_with_mock_llm():
    """Tests a full multi-turn agent execution:
    Turn 1: LLM requests get_market_data and search_financial_news.
    Turn 2: LLM observes results and calls search_filing_rag.
    Turn 3: LLM observes filing results and determines research is complete.
    Turn 4: Synthesis node compiles report.
    """
    turn_1_ai = AIMessage(
        content="I will check market data and news first.",
        tool_calls=[
            {"id": "call_1", "name": "get_market_data", "args": {"symbol": "NVDA"}},
            {"id": "call_2", "name": "search_financial_news", "args": {"query": "NVDA"}}
        ]
    )
    
    turn_2_ai = AIMessage(
        content="Market data and news retrieved. Now checking 10-K risk factors.",
        tool_calls=[
            {"id": "call_3", "name": "search_filing_rag", "args": {"query": "export restrictions China", "symbol": "NVDA"}}
        ]
    )

    turn_3_done = AIMessage(
        content="All evidence retrieved. I will now synthesize the final research report."
        # No tool_calls -> routes to synthesis
    )
    
    turn_4_synthesis = AIMessage(
        content="# Executive Summary\nNVIDIA is performing strongly [SRC-01] but faces export control risks [SRC-03]."
    )

    mock_llm = MockChatModel(
        step_responses=[turn_1_ai, turn_2_ai, turn_3_done, turn_4_synthesis]
    )
    
    final_state = run_research(
        query="Why has NVIDIA moved recently and what risks exist?",
        ticker="NVDA",
        llm=mock_llm,
        max_iterations=5
    )

    assert final_state["iteration_count"] == 3
    assert len(final_state["evidence_ledger"]) == 3
    assert final_state["evidence_ledger"][0]["source_id"] == "SRC-01"
    assert final_state["evidence_ledger"][1]["source_id"] == "SRC-02"
    assert final_state["evidence_ledger"][2]["source_id"] == "SRC-03"
    assert final_state["confidence_rating"] == "HIGH"
    assert "Executive Summary" in final_state["final_report"]
    assert len(final_state["tool_activity"]) >= 4


def test_iteration_guard_stops_infinite_loop():
    """Verify that hitting max_iterations forces synthesis even if LLM keeps requesting tools."""
    # Creates tool calls with distinct IDs for each turn
    tool_responses = [
        AIMessage(
            content=f"Turn {i} requesting tools.",
            tool_calls=[{"id": f"call_inf_{i}", "name": "get_market_data", "args": {"symbol": "AAPL"}}]
        )
        for i in range(1, 10)
    ]
    synthesis_ai = AIMessage(content="# Forced Synthesis Report")

    mock_llm = MockChatModel(
        step_responses=tool_responses,
        default_response=synthesis_ai
    )
    
    final_state = run_research(
        query="Endless query",
        ticker="AAPL",
        llm=mock_llm,
        max_iterations=3  # Stop strictly at 3 iterations
    )

    assert final_state["iteration_count"] == 3
    assert final_state["final_report"] is not None


@pytest.mark.skipif(
    not (os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY")),
    reason="Requires active GOOGLE_API_KEY or OPENAI_API_KEY in .env"
)
def test_live_agent_execution():
    """Live integration test against external LLM API (runs only when key is configured)."""
    final_state = run_research(
        query="What is Apple's current stock price?",
        ticker="AAPL",
        max_iterations=3
    )
    
    assert final_state["final_report"] is not None
    assert len(final_state["evidence_ledger"]) > 0


if __name__ == "__main__":
    pytest.main(["-v", "tests/test_agent.py"])
