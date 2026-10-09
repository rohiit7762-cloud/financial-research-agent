"""Agent State Schema for Financial Research Agent.

Defines the centralized, immutable state dictionary and evidence models
that flow through every node in the LangGraph cyclic state machine.
"""

from typing import TypedDict, List, Dict, Any, Optional, Annotated
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class EvidenceItem(BaseModel):
    """Schema representing an individual piece of supporting factual evidence."""
    source_id: str = Field(description="Unique citation identifier, e.g. SRC-01")
    source_type: str = Field(description="Origin category: market_data | fundamentals | news | filing_rag")
    tool_name: str = Field(description="Tool that retrieved this evidence")
    title: str = Field(description="Human-readable citation title")
    url_or_ref: str = Field(description="Verifiable URL or regulatory filing section")
    retrieved_text: str = Field(description="Normalized factual snippet or data summary")
    timestamp: str = Field(description="ISO timestamp of evidence acquisition")


class ResearchAgentState(TypedDict):
    """Central state passed across all LangGraph nodes."""
    query: str                                             # Original user research question
    ticker: Optional[str]                                  # Ticker symbol if identified
    messages: Annotated[List[BaseMessage], add_messages]   # Conversational message history
    evidence_ledger: List[Dict[str, Any]]                  # Append-only list of serialized EvidenceItem dicts
    tool_activity: List[str]                               # High-level activity log for UI (no private scratchpad)
    iteration_count: int                                   # Number of reasoning/tool cycles executed
    max_iterations: int                                    # Safety guardrail ceiling (default 5)
    final_report: Optional[str]                            # Synthesized structured markdown report
    confidence_rating: Optional[str]                       # Qualitative confidence: High | Medium | Low
    errors: List[str]                                      # Non-fatal error warnings log
