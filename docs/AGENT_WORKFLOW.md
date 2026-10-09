# Financial Research Agent — Agent Workflow & Execution Lifecycle

> **Document Purpose:** Detailed explanation of the agentic paradigm, state graph transitions, tool-calling mechanisms, and execution guarantees.  
> **Status:** Phase 0 Blueprint (`[PLANNED]` architecture)  
> **Interview Focus:** Defending why this system is genuinely agentic rather than a rule-based chatbot or linear chain.

---

## 1. What is an Agent? (And Why Our System Qualifies)

An **AI Agent** is an autonomous software system that uses a reasoning engine (such as a Large Language Model) to iteratively observe its environment, formulate plans, take actions via tools, inspect intermediate results, and adapt its behavior until a terminal goal is achieved.

### Why Our System Qualifies as a True Agent:
1. **Dynamic Decision-Making (No Hardcoded Routing):** There are no hardcoded `if/else` keyword trees (e.g., `if "price" in query: fetch_price()`). The LLM inspects the user prompt against semantic tool definitions and autonomously chooses which tools to execute.
2. **Cyclic Multi-Step Reasoning (ReAct Pattern):** The agent does not execute a static one-pass sequence. It can execute Tool A, observe that the result is incomplete or reveals new questions, and dynamically dispatch Tool B or Tool C.
3. **Observation-Driven Adaptation:** If a news search reveals a sudden regulatory lawsuit, the agent can autonomously query the 10-K filing RAG tool for legal risk factors without being prompted by the user to do so.
4. **Stateful Memory Across Steps:** The agent accumulates intermediate findings in a structured state ledger (`evidence_ledger`), using past observations to contextualize subsequent steps.
5. **Autonomous Stopping Criteria:** The agent inspects its accumulated evidence and decides for itself when it has gathered enough information to synthesize the final report.

---

## 2. Definitive Conceptual Distinctions for Interviews

| System Type | Core Mechanism | Can it Take Dynamic External Actions? | Can it Self-Correct or Multi-Step? | Example |
| :--- | :--- | :--- | :--- | :--- |
| **Traditional Chatbot** | Direct prompt $\to$ completion using internal static weights | No | No | Standard ChatGPT without plugins or browsing |
| **Simple RAG Pipeline** | Query $\to$ Vector search $\to$ Context injection $\to$ Answer | No (fixed single retrieval step) | No | PDF Q&A document summarizer |
| **Standalone Tool / Script** | Deterministic Python function with inputs and outputs | Yes (does one fixed job) | No | A script that downloads historical stock CSVs |
| **True AI Agent (Our System)** | Dynamic reasoning loop over state, tools, and observations | Yes (dynamic multi-tool selection) | Yes (iterative evaluation and multi-turn planning) | Our Financial Research Agent |

---

## 3. The Agent Lifecycle: Step-by-Step State Transitions

Our implementation utilizes **LangGraph** to model the agent lifecycle as an explicit state machine:

```
                    ┌────────────────────────┐
                    │      START NODE        │
                    │ User inputs query      │
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │      AGENT NODE        │◄─────────────────────────┐
                    │  LLM evaluates state   │                          │
                    │  & decides next action │                          │
                    └───────────┬────────────┘                          │
                                │                                       │
                      [Conditional Routing]                             │
                                │                                       │
            ┌───────────────────┴───────────────────┐                   │
            │ Does LLM want to call tools?          │                   │
            │ AND iteration_count < MAX_ITERATIONS? │                   │
            └─────────┬───────────────────┬─────────┘                   │
                     YES                  NO                            │
                      │                   │                             │
                      ▼                   ▼                             │
          ┌───────────────────────┐   ┌───────────────────────────┐     │
          │      TOOL NODE        │   │      SYNTHESIS NODE       │     │
          │  Executes tool(s)     │   │ Formulates structured     │     │
          │  Updates state ledger │   │ report with [SRC-xx] tags │     │
          └───────────┬───────────┘   └─────────────┬─────────────┘     │
                      │                             │                   │
                      └─────────────────────────────┘                   │
                                    │                                   │
                           Loop back to Agent Node                      ▼
                           for next reasoning turn                 ┌─────────┐
                                                                   │   END   │
                                                                   └─────────┘
```

### Trace of a Real Multi-Step Execution

**User Prompt:**  
*"Why has NVIDIA moved recently and what risks did the company highlight in its annual filing?"*

#### Iteration 1: Initial Assessment & Market Sensing
- **State Input:** Query text, empty evidence ledger, `iteration = 0`.
- **LLM Reasoning (Internal Thought):**  
  *"The user wants to know why NVIDIA moved recently. First, I need recent quantitative stock performance to see the magnitude and direction of the move. Second, I need recent news to find the catalysts. I will call `get_market_data` and `search_financial_news` for NVDA."*
- **Action Dispatched:**
  1. `get_market_data(symbol="NVDA")`
  2. `search_financial_news(query="NVIDIA stock price movement catalysts")`
- **Tool Observation:**
  - Market Data: NVDA up +4.8% over the past 5 trading days; 52-week range $45–$140; P/E ratio 62.
  - News: Articles report export restrictions on advanced AI accelerators to specific international markets and high demand for Blackwell architecture.
- **State Mutation:**
  - `evidence_ledger` appends `SRC-01` (Market Data) and `SRC-02` (News Articles).
  - `iteration_count` becomes `1`.

#### Iteration 2: Targeted Follow-up (Filing RAG)
- **State Input:** Query + Iteration 1 observations in message history + 2 evidence items.
- **LLM Reasoning (Internal Thought):**  
  *"I have market data and recent news catalysts. However, the user specifically asked what risks NVIDIA highlighted in its annual filing regarding these pressures. I should query the filing vector store focusing on regulatory export controls and supply chain concentration."*
- **Action Dispatched:**
  1. `search_filing_rag(symbol="NVDA", query="risks related to export controls, trade regulations, and manufacturing concentration")`
- **Tool Observation:**
  - Filing RAG: Retrieved 3 chunks from NVDA FY2024 10-K Item 1A: (a) US government licensing requirements for high-performance computing chips, (b) single-foundry dependency on TSMC in Taiwan, (c) component shortages for advanced packaging.
- **State Mutation:**
  - `evidence_ledger` appends `SRC-03` (10-K Item 1A Chunks).
  - `iteration_count` becomes `2`.

#### Iteration 3: Stopping Condition Check & Synthesis
- **State Input:** Query + complete tool observations covering market data, news, and 10-K risks.
- **LLM Reasoning (Internal Thought):**  
  *"All aspects of the user query have been addressed with high-signal evidence: price movement (+4.8%), news catalysts (Blackwell demand vs. export curbs), and filing risk disclosures (licensing bans and TSMC dependency). No further tools are needed. I will now synthesize the report."*
- **Action Dispatched:** None (No tool calls emitted).
- **Conditional Routing:** Evaluates `has_tool_calls == False` $\to$ Routes directly to `SynthesisNode`.
- **Synthesis Node Output:** Generates structured executive report citing `[SRC-01]`, `[SRC-02]`, and `[SRC-03]`, assigning `Confidence: High`.

---

## 4. The Agent State Schema (`agent/state.py`)

The state schema is the backbone of the agent. In LangGraph, state is a typed Python dictionary that flows through every node:

```python
from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

class EvidenceItem(BaseModel):
    source_id: str          # e.g., "SRC-01"
    tool_name: str          # e.g., "market_data", "sec_filing_rag"
    title: str              # Human-readable title
    url_or_ref: str         # Source link or 10-K section citation
    snippet: str            # Verbatim text or structured data excerpt
    timestamp: str          # Retrieval time

class ResearchAgentState(TypedDict):
    query: str                          # Original user query
    ticker: Optional[str]               # Extracted equity symbol (e.g., "NVDA")
    messages: List[BaseMessage]         # Full conversational & tool message history
    evidence_ledger: List[Dict[str, Any]] # Deduplicated evidence items for citation
    iteration_count: int                # Guard against infinite loops
    final_report: Optional[str]         # Synthesized markdown report
    confidence_rating: Optional[str]    # "High" | "Medium" | "Low"
    errors: List[str]                   # Non-fatal error logs (e.g., degraded news)
```

### Why Each Field Exists (Interview Defense):
- `query`: Preserves the ground-truth user goal across multi-turn state mutations.
- `ticker`: Allows tools to share a verified ticker symbol without re-parsing it each turn.
- `messages`: Standard LangChain message list containing `HumanMessage`, `AIMessage` (with tool calls), and `ToolMessage` (with tool results) required by native LLM APIs.
- `evidence_ledger`: Decouples raw conversation messages from presentation-ready citations.
- `iteration_count`: Enforces safety boundaries (`iteration_count <= 5`).
- `confidence_rating`: Provides qualitative safety signaling to the financial analyst.

---

## 5. Loop Termination & Infinite Loop Prevention

A critical flaw in naive agent implementations is the **infinite tool-calling loop** (e.g., tool returns empty result $\to$ LLM queries tool again $\to$ returns empty result forever).

### Our 3-Layer Termination Mechanism:
1. **Explicit LLM Self-Termination:** When the model judges its context sufficient, it emits conversational markdown without tool call tokens. The conditional router detects zero tool calls and transitions to `SynthesisNode`.
2. **Hard Iteration Ceiling (`MAX_ITERATIONS = 5`):** If an agent oscillates or queries excessively, the conditional edge forces a transition to `SynthesisNode` regardless of what the LLM requested:
   ```python
   def should_continue(state: ResearchAgentState) -> str:
       if state["iteration_count"] >= MAX_ITERATIONS:
           return "synthesis"  # Hard boundary reached
       last_message = state["messages"][-1]
       if not getattr(last_message, "tool_calls", None):
           return "synthesis"  # Agent decided it is finished
       return "tools"          # More tools requested
   ```
3. **Repeated Tool Call Detection:** If the agent attempts to call the exact same tool with the exact same arguments twice in a row, the tool execution node returns a cached warning: *"Tool already executed with these parameters. Please proceed with synthesis or select an alternative tool."*
