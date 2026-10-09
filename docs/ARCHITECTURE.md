# Financial Research Agent — System Architecture

> **Document Purpose:** Complete architectural specification, component topology, data flows, and failure containment mechanisms.  
> **Status Tagging:** Every component is tagged as `[IMPLEMENTED]`, `[PLANNED]`, or `[REFERENCE ARCHITECTURE — NOT IN OUR IMPLEMENTATION]`.  
> **Target Audience:** College Placement Technical Interviews (AI Engineer, Python Backend, Applied ML).

---

## 1. High-Level Architectural Diagram

```
                                  USER (Streamlit UI)
                                          │
                                          │ Natural Language Query
                                          ▼
                         ┌─────────────────────────────────┐
                         │   Streamlit Presentation Layer  │ [PLANNED]
                         │  - User Prompt & Model Selector │
                         │  - Activity Log (No raw scratch)│
                         │  - Executive Report & Citations │
                         └────────────────┬────────────────┘
                                          │
                                          │ Initial State (query, empty ledgers)
                                          ▼
                         ┌─────────────────────────────────┐
                         │     LangGraph State Machine     │ [PLANNED]
                         │          (Agent Loop)           │
                         └──────────────┬───▲──────────────┘
                                        │   │
                  ┌─────────────────────┘   └─────────────────────┐
                  ▼                                               │ Tool Result
       ┌────────────────────┐                       ┌─────────────┴───────────┐
       │   Planner / LLM    │                       │     Execution Node      │ [PLANNED]
       │ Tool Calling Node  │                       │  (Dispatches & Records) │
       └──────────┬─────────┘                       └─────────────▲───────────┘
                  │                                               │
                  │ Tool Call Request(s)                          │ Raw Output
                  ▼                                               │
       ┌──────────────────────────────────────────────────────────┴───────────┐
       │                          TOOL EXECUTION BUS                          │
       │                                                                      │
       │  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────────┐  │
       │  │   Market Data   │  │  Fundamentals   │  │  Financial News      │  │
       │  │    (yfinance)   │  │   (yfinance)    │  │ (yfinance/DuckDuckGo)│  │
       │  │   [PLANNED]     │  │   [PLANNED]     │  │      [PLANNED]       │  │
       │  └────────┬────────┘  └────────┬────────┘  └──────────┬───────────┘  │
       │           │                    │                      │              │
       │           └────────────────────┼──────────────────────┘              │
       │                                │                                     │
       │                                ▼                                     │
       │                   ┌──────────────────────────┐                       │
       │                   │    SEC Filing Search     │ [PLANNED]             │
       │                   │      (Local RAG)         │                       │
       │                   └────────────┬─────────────┘                       │
       │                                │                                     │
       └────────────────────────────────┼─────────────────────────────────────┘
                                        │
                                        │ Semantic Similarity Top-K
                                        ▼
                         ┌─────────────────────────────┐
                         │   Vector Store (FAISS)      │ [PLANNED]
                         │   Embeddings: MiniLM-L6-v2  │
                         │   Corpus: Curated 10-Ks     │
                         └─────────────────────────────┘
                                        │
                                        │ Chunks + Metadata
                                        ▼
                         ┌─────────────────────────────┐
                         │   Evidence Ledger Builder   │ [PLANNED]
                         │  - Assigns [Source ID]      │
                         │  - Normalizes schemas       │
                         │  - Prepares audit trail     │
                         └──────────────┬──────────────┘
                                        │
                                        │ Complete Evidence Ledger
                                        ▼
                         ┌─────────────────────────────┐
                         │    Report Synthesis Node    │ [PLANNED]
                         │  - Grounded Structured Gen  │
                         │  - In-text Citations        │
                         │  - Confidence Scoring       │
                         └──────────────┬──────────────┘
                                        │
                                        │ Final Markdown Report + Source Cards
                                        ▼
                                  USER (Streamlit UI)
```

---

## 2. Core Architectural Philosophy: Why One Agent vs. Multi-Agent Swarm?

In technical interviews, interviewers frequently ask:
> *"Why didn't you build a Multi-Agent Swarm with a Technical Analyst Agent, a Fundamental Analyst Agent, a Sentiment Agent, and a Master Manager Agent?"*

### Our Engineering Answer:
1. **Coordination Overhead & Token Waste:** In a multi-agent system, agents must communicate via natural language messages, consuming hundreds of thousands of prompt/completion tokens just coordinating handoffs. A single agent equipped with high-signal tools executes in $\frac{1}{4}$ the token footprint and latency.
2. **Cascading Failure & Stochastic Drift:** If Agent A (Market Analyst) hallucinates an incorrect ticker or date, Agent B (Sentiment Analyst) bases its search on garbage data. In a single agent loop with typed Pydantic state, state mutations are deterministic and observable.
3. **Auditability in Placement Evaluation:** A single-agent loop with LangGraph explicitly models the decision graph: `Agent -> Tools -> Agent -> Synthesis`. Every decision branch can be tested, mocked, and evaluated unit-by-unit.

| Dimension | Multi-Agent Swarm (Reference Architecture) | Single Tool-Calling Agent (Our Project) |
| :--- | :--- | :--- |
| **Topology** | 5–11 Autonomous Agents talking to each other | 1 Agent reasoning over 4 deterministic tools |
| **State Management** | Distributed message passing (Redis / Celery) | Centralized, immutable TypedDict state graph |
| **Token Cost** | Very High (repetitive system prompts per agent) | Minimal (single system prompt + tool schemas) |
| **Debugging** | Extremely difficult (non-deterministic agent chat) | Transparent (exact tool input/output logs) |
| **Interview Defensibility** | Low if built by a student (often copied boilerplate) | High (every transition and fallback is mastered) |

---

## 3. Deep Component Breakdown

### Component 1: Streamlit Presentation Layer `[PLANNED]`
- **WHAT:** An interactive web frontend allowing users to input tickers/questions, observe execution progress indicators, and view the final grounded report with expandable evidence cards.
- **WHY:** Provides an intuitive interface for end-users while keeping internal chain-of-thought private.
- **HOW:** Uses Streamlit reactive session state (`st.session_state`), displays progress status containers (`st.status`), and renders formatted markdown with interactive metrics and download options.
- **WHAT HAPPENS IF REMOVED:** The system would function as a pure Python CLI or REST API. The core research intelligence remains identical, proving strong separation of concerns between presentation and logic.

### Component 2: LangGraph State Machine & Agent Loop `[PLANNED]`
- **WHAT:** A cyclic state graph that coordinates LLM reasoning, tool dispatching, observation recording, and stopping condition checks.
- **WHY:** Unlike raw sequential chains (`Chain.run()`), real-world research requires conditional branching, dynamic loops (retrying a search or fetching news after observing market anomalies), and hard execution bounds (max iterations).
- **HOW:** 
  1. Node `agent_reasoning`: LLM receives message history + tool schemas and outputs tool calls or final answer.
  2. Conditional Edge: If tool calls exist and `iteration_count < MAX_ITERATIONS`, route to `tool_execution`. If no tool calls or limit reached, route to `report_synthesis`.
  3. Node `tool_execution`: Executes selected tools, formats results into `ToolMessage`, updates `EvidenceLedger`.
  4. Loop back to `agent_reasoning`.
- **WHAT HAPPENS IF REMOVED:** Without an agent loop, you are forced into hardcoded `if/else` procedural pipelines (`if "price" in query: fetch_price()`), losing dynamic multi-step problem solving.

### Component 3: Four Core Specialized Tools `[PLANNED]`
1. **Market Data Tool (`get_market_data`):**
   - **WHAT:** Fetches latest price, previous close, percentage change, 52-week high/low, and market cap via `yfinance`.
   - **WHY:** Provides real-time quantitative baseline without LLM knowledge cutoff errors.
   - **HOW:** Calls `yfinance.Ticker(symbol).fast_info` / `.info`, sanitizes missing fields, returns normalized Pydantic JSON.
   - **WHAT HAPPENS IF REMOVED:** Agent cannot evaluate current market valuation or recent price volatility.
2. **Company Fundamentals Tool (`get_company_fundamentals`):**
   - **WHAT:** Retrieves audited financial health metrics: Revenue, Net Income, Operating Margins, EPS, P/E Ratio, Total Debt, Free Cash Flow.
   - **WHY:** Evaluates structural solvency and profitability rather than speculative price movement.
   - **HOW:** Queries `yfinance.Ticker(symbol).financials`, `.balance_sheet`, `.info`, parses point-in-time metrics with strict null handling.
   - **WHAT HAPPENS IF REMOVED:** Agent cannot verify whether a stock price drop is backed by deteriorating fundamentals or pure market noise.
3. **Financial News Tool (`search_financial_news`):**
   - **WHAT:** Gathers recent news headlines, publisher sources, publication timestamps, and snippets.
   - **WHY:** Supplies qualitative catalysts (earnings surprises, regulatory actions, management changes) that explain quantitative moves.
   - **HOW:** Blends `yfinance.Ticker.news` with DuckDuckGo Search API fallback, filtered for financial domain credibility.
   - **WHAT HAPPENS IF REMOVED:** Agent cannot answer *"Why did the stock move today/this week?"*
4. **SEC Filing Search Tool (`search_filing_rag`):**
   - **WHAT:** Performs semantic vector search over audited 10-K/10-Q filings, targeting Item 1A (Risk Factors) and Item 7 (MD&A).
   - **WHY:** News covers short-term chatter; 10-K filings disclose legally binding, long-term operational and systemic risks.
   - **HOW:** Queries local FAISS vector index using dense embeddings (`all-MiniLM-L6-v2`) and returns top-$k$ relevant chunks with document metadata.
   - **WHAT HAPPENS IF REMOVED:** Agent hallucinates company risk profiles or relies solely on unverified public blog posts.

### Component 4: Vector Storage & Embedding Pipeline (FAISS + MiniLM) `[PLANNED]`
- **WHAT:** A lightweight vector retrieval subsystem storing chunked 10-K filings for target companies.
- **WHY:** LLMs cannot ingest 200-page 10-K filings in their entirety per user query without massive latency and context degradation ("lost in the middle").
- **HOW:** Documents are parsed into semantic chunks (800 characters with 150-character overlap), embedded locally into 384-dimensional dense vectors using `sentence-transformers/all-MiniLM-L6-v2`, and indexed using FAISS IndexFlatL2/IndexFlatIP.
- **WHAT HAPPENS IF REMOVED:** Agent loses access to deep audited corporate disclosures, reducing to a generic stock-quote chatbot.

### Component 5: Evidence Ledger & Audit Trail `[PLANNED]`
- **WHAT:** A centralized, deduplicated ledger where every tool output is converted into a structured `EvidenceItem`:
  ```python
  class EvidenceItem(BaseModel):
      source_id: str          # e.g., "SRC-01"
      tool_name: str          # e.g., "sec_filing_rag"
      title: str              # e.g., "Apple Inc. FY2023 10-K - Item 1A Risk Factors"
      url_or_ref: str         # e.g., "SEC EDGAR 0000320193-23-000106 / Chunk #42"
      snippet: str            # Verbatim excerpt or key metric table
      timestamp: str          # Retrieval time
  ```
- **WHY:** Prevents "untraceable hallucination." Every factual sentence in the final output must cite a `[SRC-xx]` identifier.
- **HOW:** Before returning tool observations to the LLM, the execution node appends validated items to the state's `evidence_ledger`.
- **WHAT HAPPENS IF REMOVED:** The LLM's synthesis becomes ungrounded; neither the user nor an interviewer can verify whether a stated statistic is real or invented.

### Component 6: Synthesis & Qualitative Confidence Scorer `[PLANNED]`
- **WHAT:** Final prompt stage that compiles accumulated evidence into an executive research memo and assigns a qualitative confidence rating (`High`, `Medium`, `Low`).
- **WHY:** Real analysts need to know if a report is based on complete data (e.g., all 4 tools succeeded) or degraded data (e.g., news API failed or filing was unavailable).
- **HOW:** The synthesizer inspects evidence diversity:
  - `HIGH`: Real-time market data + Fundamentals + News + Filings all retrieved and coherent.
  - `MEDIUM`: Core data retrieved, but one qualitative source (e.g., filing RAG) missing or ambiguous.
  - `LOW`: Single source only, or conflicting contradictory data detected.
- **WHAT HAPPENS IF REMOVED:** The user receives a flat text output with no indicator of analytical certainty.

---

## 4. End-to-End Data Flow

```
Step 1: User enters query: "Why has NVDA moved recently and what risks are in its 10-K?"
Step 2: Streamlit initializes ResearchAgentState:
        - query = "Why has NVDA moved recently and what risks are in its 10-K?"
        - ticker = None (to be extracted)
        - evidence_ledger = []
        - tool_history = []
        - iteration = 0
Step 3: Agent Reasoning Node:
        - LLM inspects query and tool definitions.
        - Identifies symbol "NVDA".
        - Decides to invoke: get_market_data(symbol="NVDA") and search_financial_news(query="NVIDIA stock catalyst")
Step 4: Tool Execution Node:
        - Executes yfinance market data fetch. Result: Stock up +4.2% today, P/E = 65.
        - Executes news search. Result: New AI chip announcement, supply constraints.
        - Appends SRC-01 (Market Data) and SRC-02 (News) to Evidence Ledger.
Step 5: Agent Reasoning Node (Second Turn):
        - LLM reviews observations: "I have market data and news. User also requested filing risks."
        - Decides to invoke: search_filing_rag(symbol="NVDA", query="supply chain concentration and export control risks")
Step 6: Tool Execution Node:
        - FAISS vector search retrieves 3 relevant chunks from NVDA 10-K Item 1A.
        - Appends SRC-03 (Filing Chunks) to Evidence Ledger.
Step 7: Agent Reasoning Node (Third Turn):
        - LLM evaluates: "All facets of the query have been addressed with high-signal evidence."
        - Decides: No further tools needed. Route to synthesis.
Step 8: Synthesis Node:
        - LLM constructs structured report with sections: Executive Summary, Market Dynamics, News Catalysts, 10-K Risk Analysis, Sources.
        - Attaches explicit citations [SRC-01], [SRC-02], [SRC-03].
        - Computes qualitative confidence: HIGH.
Step 9: Streamlit renders formatted report and interactive evidence drawer.
```

---

## 5. Failure Containment & Graceful Degradation

| Failure Mode | Failure Impact | Containment & Fallback Strategy |
| :--- | :--- | :--- |
| **Invalid Ticker (e.g. `$XYZABC`)** | yfinance returns empty dict / 404 | Tool catches `KeyError`/`HTTPError`, returns structured message: `{"status": "error", "error_code": "INVALID_TICKER", "message": "Ticker XYZABC not found. Did you mean..."}`. Agent informs user without crashing. |
| **News API Down or Rate-Limited** | News tool fails or returns timeout | Tool catches timeout, automatically falls back to secondary search provider (DuckDuckGo News) or returns `{"status": "degraded", "warning": "Live news temporarily unavailable"}`. Agent continues report using filings & market data with `Medium` confidence. |
| **Filing Not in Vector DB** | Symbol filing not yet ingested | RAG tool returns `{"status": "not_indexed", "message": "10-K filing for SYMBOL is not in the curated index."}`. Agent reports market data and news, explicitly noting filing unavailability. |
| **LLM Tool-Calling Loop Overflow** | Infinite loop or repetitive tool calls | State machine strictly enforces `iteration_count <= MAX_ITERATIONS` (default 5). If exceeded, forced transition to `SynthesisNode` with whatever evidence has been gathered so far. |
| **Network Timeout / Model Outage** | LLM API fails | Streamlit UI displays user-friendly error notification with diagnostic advice (check API key / connectivity) rather than an unhandled Python stack trace. |

---

## 6. Interview Checklist: What You Must Be Ready to Draw on a Whiteboard

1. **Draw the State Graph:** Box for Agent Reasoning $\to$ Conditional diamond (Tools needed? Yes/No) $\to$ Tool Execution Box $\to$ Loop back $\to$ Synthesis Box.
2. **Explain the State Dictionary:** Explain what fields are tracked (`messages`, `evidence_ledger`, `iteration_count`).
3. **Justify FAISS over Cloud Vector DB:** *"For a curated corpus of benchmark corporate filings, an in-memory/local FAISS FlatL2 index gives sub-5ms retrieval with zero network latency, zero hosting cost, and complete offline reproducibility."*
