# Financial Research Agent — Project Development Log

> **Document Purpose:** Chronological engineering diary tracking implementation progress, technical challenges encountered, architectural solutions applied, and placement interview questions generated per stage.  
> **Rule:** Updated iteratively at the conclusion of every development phase.

---

## Log Entry 01: Phase 0 — Requirements & Architectural Blueprint

- **Date:** October 5, 2026
- **Stage:** Phase 0: Requirements Analysis, Workspace Inspection & Architectural Blueprint
- **Files Created / Modified:**
  - `docs/PROJECT_OVERVIEW.md` (System requirements, scope, comparison with reference architecture)
  - `docs/ARCHITECTURE.md` (ASCII topology, data flows, component breakdown, failure containment)
  - `docs/TECHNICAL_NOTES.md` (7-point encyclopedic breakdowns for 19 core concepts)
  - `docs/AGENT_WORKFLOW.md` (Agent definition, ReAct loop, state schemas, infinite loop safeguards)
  - `docs/RAG.md` (12-stage RAG pipeline, embeddings theory, chunking mathematics, FAISS design)
  - `docs/DECISIONS.md` (ADRs covering single vs multi-agent, LangGraph, FAISS, MiniLM, yfinance, Streamlit)
  - `docs/PROJECT_LOG.md` (This development tracking log)
  - `docs/INTERVIEW_PREP.md` (30s/1m/2m pitches, technical Q&A across 15 domains, tricky questions)
  - `docs/LEARNING_CHECKLIST.md` (Student mastery tracker across engineering concepts)

### What Was Implemented:
1. Conducted automated workspace inspection: Verified Python 3.12 64-bit runtime environment on Windows.
2. Verified API environment: Determined absence of global API keys, establishing the necessity for a localized `.env` configuration pattern.
3. Formulated complete architectural specification for a focused, single-agent tool-calling system contrasting against the bloated 11-agent reference architecture (`gsaini/financial-research-analyst-agent`).
4. Architected the 4 core tool abstractions: `get_market_data`, `get_company_fundamentals`, `search_financial_news`, and `search_filing_rag`.
5. Designed the immutable `EvidenceLedger` schema to enforce factual provenance and prevent untraceable hallucinations.
6. Specified the LangGraph cyclic state machine with hard iteration boundaries (`MAX_ITERATIONS = 5`).

### Concepts Learned & Solidified:
- Why multi-agent swarms frequently degrade into token-wasting conversational chatter rather than reliable analytical execution.
- How LangGraph replaces deprecated, black-box `AgentExecutor` with inspectable, typed state machines.
- The mathematical distinction between cosine similarity and Euclidean distance in L2-normalized vector spaces.
- Why recursive character chunking with sliding window overlap is required to preserve legal sentence semantics across 10-K sections.
- How Pydantic v2 schemas provide runtime type safety against unpredictable third-party API responses.

### Problems & Challenges Encountered:
- *Challenge 1:* Deciding how to balance deep SEC 10-K filing retrieval without forcing the student to build a complex, multi-gigabyte SEC EDGAR scraping pipeline.
  - *Solution:* Curated a focused set of benchmark technology filings (AAPL, MSFT, NVDA, TSLA) focusing specifically on Item 1A (Risk Factors) and Item 7 (MD&A), which gives 100% of the real-world textual complexity while keeping the FAISS index under 15MB.
- *Challenge 2:* Determining the proper framework for the agent loop.
  - *Solution:* Selected LangGraph for its native state machine modeling, allowing cyclic execution with transparent stopping boundaries and clean integration with Streamlit status widgets.

### Important Architectural Decisions Made:
- Standardized on a single tool-calling agent rather than a multi-agent system (ADR-001).
- Chose local CPU-bound FAISS and `sentence-transformers/all-MiniLM-L6-v2` for zero-cost, offline-reproducible RAG (ADR-003, ADR-004).
- Standardized on Streamlit to maintain a 100% Python codebase (ADR-006).

### Placement Interview Questions Generated:
1. *Why did you build a single agent with 4 tools instead of a multi-agent team (e.g. Fundamental Agent + Sentiment Agent)?*
2. *What is the difference between an agent loop and a hardcoded DAG pipeline?*
3. *Why is FAISS IndexFlatL2 optimal for a few thousand chunks compared to an approximate index like IVF?*
4. *How do you prevent the agent from getting stuck in an infinite tool-calling loop?*
5. *Why is RAG superior to fine-tuning for financial filings?*

---

## Log Entry 02: Phase 1 — Project Setup & Environment Verification

- **Date:** October 5, 2026
- **Stage:** Phase 1: Environment Setup, Dependency Pinning & Verification
- **Files Created / Modified:**
  - `requirements.txt` (Pinned core dependencies: LangGraph, yfinance, FAISS, Sentence Transformers, Pydantic, Streamlit)
  - `.env.example` (Config template for Gemini/OpenAI API keys, models, and paths)
  - `.gitignore` (Excludes `.env`, `.venv`, and binary caches)
  - `agent/__init__.py`, `tools/__init__.py`, `rag/__init__.py`, `evaluation/__init__.py`, `tests/__init__.py`
  - `tests/test_setup.py` (Automated environment sanity test suite)
  - `docs/PROJECT_LOG.md` (Updated with Phase 1 log)
  - `docs/LEARNING_CHECKLIST.md` (Updated setup progress)

### What Was Implemented:
1. Created complete modular directory hierarchy (`agent/`, `tools/`, `rag/`, `app/`, `data/filings/`, `data/vector_store/`, `evaluation/`, `tests/`).
2. Initialized isolated Python 3.12 64-bit virtual environment (`.venv`).
3. Installed and pinned all required dependencies with zero version conflicts.
4. Created and executed `tests/test_setup.py` which dynamically verified that all 11 critical modules import cleanly and report valid version strings.

### Concepts Learned & Solidified:
- Clean modular packaging in Python (`__init__.py` and absolute imports).
- Separation of configuration from code via `.env` and `python-dotenv`.
- How Windows C++ wheels for `faiss-cpu` and `torch` interact with Python 3.12.
- The importance of automated setup verification scripts before developing domain logic.

### Problems & Challenges Encountered:
- *Challenge:* Large binary wheels (`torch` 124MB, `scipy` 36MB) temporarily dropped during network streaming.
  - *Solution:* pip's built-in download resumption automatically resumed chunk downloads cleanly, resulting in 100% successful wheel installations without manual intervention.

### Important Architectural Decisions Made:
- Pinned LangChain Core v1.4+, LangGraph v1.2+, and Pydantic v2.13+ to ensure native tool-calling compatibility and sub-millisecond data validation.
- Configured `.gitignore` to prevent leaking API keys or committing binary FAISS indices to Git.

### Placement Interview Questions Generated:
1. *Why is it critical to isolate project dependencies inside a virtual environment?*
2. *Why do we separate `.env` from `.env.example` in production repositories?*
3. *How does Pydantic v2 achieve significantly higher validation performance than Pydantic v1?*

---

## Log Entry 03: Phase 2 — Market Data Tool Implementation & Unit Testing

- **Date:** October 5, 2026
- **Stage:** Phase 2: Market Data Tool (`tools/market_data.py`)
- **Files Created / Modified:**
  - `tools/market_data.py` (Pydantic models, yfinance fast_info extractor, LangChain @tool)
  - `tests/test_market_data.py` (Unit tests covering valid tickers, sanitization, invalid tickers, @tool invocation)
  - `docs/PROJECT_LOG.md` (Updated with Phase 2 log)
  - `docs/LEARNING_CHECKLIST.md` (Marked tools/market_data.py completed)

### What Was Implemented:
1. Engineered `fetch_market_data(symbol)` using `yfinance.Ticker.fast_info` for low-latency market metric extraction (current price, previous close, daily delta, volume, 52-week range, market cap).
2. Built defensive Pydantic v2 schemas (`MarketDataSuccess` and `MarketDataError`), guaranteeing typed outputs with zero runtime crashes on missing or null API fields.
3. Implemented market cap formatting helper converting raw numbers into human-readable strings (e.g. `$3.25T`, `$850.40B`).
4. Exposed `get_market_data` as a LangChain `@tool` with comprehensive docstrings optimized for LLM function calling.
5. Developed automated test suite `tests/test_market_data.py` with 4 test cases; verified 100% pass rate in 4.98 seconds.

### Concepts Learned & Solidified:
- Discovered that `yfinance.fast_info` keys use camelCase (`lastPrice`, `yearHigh`, `yearLow`, `previousClose`), differing from legacy `ticker.info` snake_case keys.
- Handled yfinance's unhandled `KeyError: 'currentTradingPeriod'` when querying invalid/delisted tickers, safely catching it and returning a structured `MarketDataError`.
- Cleaned input tickers by trimming whitespace, uppercasing, and removing leading `$`.

### Problems & Challenges Encountered:
- *Challenge:* `fast_info['fiftyTwoWeekHigh']` raised `KeyError` during initial testing.
  - *Solution:* Inspected `fast_info.keys()` and identified that yfinance names this field `'yearHigh'` and `'yearLow'`. Adjusted extractor accordingly.
- *Challenge:* Invalid tickers raise 404 HTTP errors in yfinance scrapers and emit noisy stdout warnings.
  - *Solution:* Wrapped extraction in a defensive try/except block that checks for null prices and returns a standardized `INVALID_TICKER` payload without crashing.

### Important Architectural Decisions Made:
- Chose `fast_info` over `ticker.info` for the primary quote extraction because `fast_info` executes in a fraction of the time without downloading full corporate metadata packets.
- Serialized output as JSON strings for LLM tool consumption while preserving Pydantic model objects for direct internal Python calls.

### Placement Interview Questions Generated:
1. *Why did you prefer `fast_info` over `info` in yfinance?*
2. *What happens to your agent if Yahoo Finance returns an HTTP 404 for a ticker?*
3. *Why does the tool return a JSON string rather than a raw Python dictionary to the LLM?*

---

## Log Entry 04: Phase 3 — Company Fundamentals Tool Implementation & Unit Testing

- **Date:** October 5, 2026
- **Stage:** Phase 3: Fundamentals Tool (`tools/fundamentals.py`)
- **Files Created / Modified:**
  - `tools/fundamentals.py` (Pydantic models, valuation ratios, income & balance sheet extractor, LangChain @tool)
  - `tests/test_fundamentals.py` (Unit tests covering valid tickers, sanitization, invalid tickers, @tool invocation)
  - `docs/PROJECT_LOG.md` (Updated with Phase 3 log)
  - `docs/LEARNING_CHECKLIST.md` (Marked tools/fundamentals.py completed)

### What Was Implemented:
1. Engineered `fetch_company_fundamentals(symbol)` to extract key financial metrics: Trailing P/E, Forward P/E, PEG Ratio, Trailing EPS, Forward EPS, Total Revenue, Net Income, Operating Margins, Net Profit Margins, Return on Equity, Total Debt, Total Cash, Free Cash Flow, and Debt-to-Equity.
2. Formulated strict Pydantic v2 schemas (`FundamentalMetrics` and `FundamentalsError`).
3. Implemented `_format_financial_amount` to convert large financial quantities into clear human-readable strings (e.g. `$466.82B`, `$128.93B`, `-$2.40B`).
4. Exposed `get_company_fundamentals` as a LangChain `@tool` with an explicit docstring guiding the LLM when valuation, income, or solvency questions are asked.
5. Implemented automated test suite `tests/test_fundamentals.py` with 4 test cases; verified 100% pass rate in 4.11 seconds.

### Concepts Learned & Solidified:
- Distinguishing valuation multiples (P/E, PEG) from accounting statement figures (balance sheet, income statement).
- Yahoo Finance's return format for invalid tickers when calling `.info` (returns an almost empty dictionary with `{'trailingPegRatio': None}`).
- Defensive field access with `.get()` and type coercion to prevent `NoneType` arithmetic crashes.

### Problems & Challenges Encountered:
- *Challenge:* Certain companies (e.g., fast-growing tech companies or biotech) have negative earnings, causing trailing P/E to be `None` or negative.
  - *Solution:* Declared all financial ratio fields as `Optional[float] = None`, ensuring that missing ratios do not cause Pydantic validation errors while preventing the LLM from fabricating missing metrics.

### Important Architectural Decisions Made:
- Excluded raw multi-year balance sheet tabular dumps in favor of normalized high-signal scalar metrics to minimize LLM context window consumption.
- Sourced free cash flow directly from operating cash flow minus CapEx, which is the primary solvency indicator used by institutional analysts.

### Placement Interview Questions Generated:
1. *Why did you separate Market Data from Fundamentals into two distinct tools?*
2. *How do you handle companies with negative earnings or missing P/E ratios?*
3. *Why is Free Cash Flow often preferred over Net Income when evaluating corporate health?*

---

## Log Entry 05: Phase 4 — Financial News Tool Implementation & Query Relaxation Fallback

- **Date:** October 5, 2026
- **Stage:** Phase 4: Financial News Tool (`tools/news.py`)
- **Files Created / Modified:**
  - `tools/news.py` (Pydantic models, Yahoo Finance News Search, query relaxation fallback, LangChain @tool)
  - `tests/test_news.py` (Unit tests covering ticker queries, multi-term conceptual search, empty queries, zero results, @tool invocation)
  - `docs/PROJECT_LOG.md` (Updated with Phase 4 log)
  - `docs/LEARNING_CHECKLIST.md` (Marked tools/news.py completed)

### What Was Implemented:
1. Built `fetch_financial_news(query, max_results)` powered by `yfinance.Search`, retrieving verified financial news headlines, publisher sources, publication timestamps in UTC, URLs, and related tickers.
2. Built defensive Pydantic v2 schemas: `NewsItem`, `NewsSearchResults`, and `NewsSearchError`.
3. Engineered an intelligent **Query Relaxation Fallback** mechanism: if an agent passes a verbose, multi-term natural language query (e.g. *"NVIDIA AI Blackwell GPU demand"*), the tool automatically extracts the core company and subject terms and re-queries to ensure continuous high recall.
4. Exposed `search_financial_news` as a LangChain `@tool` with an explicit docstring directing the LLM to invoke it when investigating recent market catalysts, news, or sentiment.
5. Implemented automated test suite `tests/test_news.py` with 5 test cases; verified 100% pass rate in 3.92 seconds.

### Concepts Learned & Solidified:
- DuckDuckGo News endpoint rate-limits anonymous scraping with HTTP 403, whereas `yfinance.Search` provides robust, zero-cost, real-time financial news coverage with zero rate-limit blocks.
- Real-world query impedance mismatch: LLMs frequently pass conversational sentences to search tools; intelligent query relaxation bridges the gap between natural language prompts and strict search engine indexing.

### Problems & Challenges Encountered:
- *Challenge:* `duckduckgo_search.DDGS().news()` raised `RatelimitException: 403 Ratelimit`.
  - *Solution:* Switched primary news retrieval to `yfinance.Search(query, news_count=...)`, which proved 100% reliable and required zero API keys.
- *Challenge:* Multi-word queries with >4 words returned 0 results in Yahoo Finance Search.
  - *Solution:* Implemented `_extract_fallback_keywords()` that strips stop words and re-queries top subject entities if initial results are empty.

### Important Architectural Decisions Made:
- Capped results to 5 articles (`max_results=5`) to prevent flooding the LLM context window with redundant news paragraphs.
- Formatted Unix timestamps to standard human-readable `YYYY-MM-DD HH:MM:SS UTC` for seamless in-text citation and evidence tracking.

### Placement Interview Questions Generated:
1. *What is "Query Relaxation" and why is it essential when building agentic search tools?*
2. *Why did you limit news output to 5 articles instead of returning all available search results?*
3. *How do you handle news sources that are rate-limited or return 403 errors?*

---

## Log Entry 06: Phase 5 — SEC Filing RAG Subsystem Implementation & Integration Testing

- **Date:** October 5, 2026
- **Stage:** Phase 5: SEC Filing RAG Pipeline & Retrieval Verification
- **Files Created / Modified:**
  - `data/filings/nvda_10k_risk_factors.txt` (Audited NVIDIA FY2024 Item 1A risk factors)
  - `data/filings/aapl_10k_risk_factors.txt` (Audited Apple FY2023 Item 1A risk factors)
  - `data/filings/msft_10k_risk_factors.txt` (Audited Microsoft FY2024 Item 1A risk factors)
  - `data/filings/tsla_10k_risk_factors.txt` (Audited Tesla FY2023 Item 1A risk factors)
  - `rag/ingestion.py` (Document parsing, normalization, metadata extraction)
  - `rag/chunking.py` (Recursive character splitting: 800 chars, 150 overlap)
  - `rag/embeddings.py` (SentenceTransformers all-MiniLM-L6-v2 singleton, 384-dim normalized embeddings)
  - `rag/vector_store.py` (FAISS IndexFlatIP cosine similarity index, persistence to disk)
  - `rag/retrieval.py` (Retrieval engine, symbol-based filtering, lazy index loading)
  - `tools/filing_search.py` (LangChain @tool exposing `search_filing_rag`)
  - `tests/test_rag.py` (Comprehensive 9-test unit and integration suite)
  - `docs/PROJECT_LOG.md` (Updated with Phase 5 log)
  - `docs/LEARNING_CHECKLIST.md` (Marked RAG modules and tool completed)

### What Was Implemented:
1. Ingested audited SEC 10-K Item 1A (Risk Factors) for 4 benchmark public equities: AAPL, MSFT, NVDA, and TSLA.
2. Engineered text cleaning and recursive character chunking with sliding window overlap (800 character window, 150 character overlap), preserving section titles, SEC accession URLs, and chunk IDs across every passage.
3. Wrapped `sentence-transformers/all-MiniLM-L6-v2` generating 384-dimensional dense vectors on CPU with unit L2 normalization.
4. Built `FAISSVectorStore` using `IndexFlatIP` (Inner Product on normalized vectors = Cosine Similarity), complete with binary disk persistence (`index.faiss`) and companion metadata ledger (`metadata.json`).
5. Implemented `retrieve_filing_chunks()` featuring optional ticker symbol filtering and graceful warning responses for unindexed corporate tickers.
6. Exposed `search_filing_rag` as an agent tool and verified 9 out of 9 tests passing in 17.16 seconds.

### Concepts Learned & Solidified:
- **Inner Product vs. L2 Distance:** When embedding vectors are L2-normalized ($\|\mathbf{v}\| = 1$), FAISS `IndexFlatIP` computes the exact Cosine Similarity ($1.0$ = identical, $0.0$ = orthogonal), eliminating the need for square root computations.
- **Precision Retrieval:** Verified that querying *"US export controls on high performance GPU chips to China"* cleanly surfaces NVIDIA's BIS licensing disclosures as the #1 hit with $>0.50$ cosine similarity, without retrieving irrelevant Apple or Tesla chunks.
- **Decoupled Architecture:** Built the RAG subsystem as a standalone, testable Python package before connecting it to the agent, ensuring zero coupling with the LLM reasoning graph.

### Problems & Challenges Encountered:
- *Challenge:* Initial download of `all-MiniLM-L6-v2` took ~2 minutes over slow network connection.
  - *Solution:* Implemented singleton caching in `rag/embeddings.py` so the model is only downloaded and loaded into memory once.
- *Challenge:* Section header parsing case sensitivity caused `assert doc.section == "Item 1A. Risk Factors"` to fail because raw document had uppercase `ITEM 1A. RISK FACTORS`.
  - *Solution:* Standardized section extraction using regex case-insensitive matching in `rag/ingestion.py`.

### Important Architectural Decisions Made:
- Persisted vector store to disk (`data/vector_store/index.faiss`) so subsequent runs load in <5ms without re-chunking or re-embedding documents.
- Preserved official SEC EDGAR URLs in chunk metadata to support verifiable in-text report citations.

### Placement Interview Questions Generated:
1. *Why does Inner Product equal Cosine Similarity on normalized vectors?*
2. *Why did you choose 800 characters for chunk size and 150 for overlap?*
3. *What is the computational complexity of FAISS IndexFlat search?*

---

## Log Entry 07: Phase 6 — LangGraph Agent State Machine & Multi-Step Reasoning Loop

- **Date:** October 5, 2026
- **Stage:** Phase 6: Agent Core, State Graph, and Cyclic Tool-Calling Loop
- **Files Created / Modified:**
  - `agent/state.py` (TypedDict state schema, `EvidenceItem` model, `add_messages` reducer)
  - `agent/prompts.py` (System persona, dynamic tool calling policy, grounding rules, synthesis format)
  - `agent/research_agent.py` (LangGraph state graph: `agent` $\leftrightarrow$ `tools` $\to$ `synthesis`, model factory, tool dispatch bus)
  - `tests/test_agent.py` (Deterministic multi-step test suite with mock LLM and iteration guardrails)
  - `docs/PROJECT_LOG.md` (Updated with Phase 6 log)
  - `docs/LEARNING_CHECKLIST.md` (Marked agent/state.py, prompts.py, research_agent.py completed)

### What Was Implemented:
1. Designed `ResearchAgentState` with typed channels: `query`, `ticker`, `messages` (with LangGraph's `add_messages` reducer), `evidence_ledger`, `tool_activity`, `iteration_count`, `max_iterations`, `final_report`, `confidence_rating`, and `errors`.
2. Built `agent/prompts.py` establishing the Senior Equity Research Analyst persona, explicit tool guidelines, anti-hallucination negative constraints, and mandatory `[SRC-xx]` in-text citation rules.
3. Implemented `build_research_agent()` using LangGraph:
   - Node `agent`: Invokes LLM with bound tool schemas (`get_market_data`, `get_company_fundamentals`, `search_financial_news`, `search_filing_rag`) and increments `iteration_count`.
   - Node `tools`: Intercepts `tool_calls`, dispatches Python functions, translates raw JSON into verified `EvidenceItem` objects with sequential IDs (`SRC-01`, `SRC-02`, etc.), records user-friendly tool activity, and returns `ToolMessage` payloads.
   - Conditional Edge `should_continue`: Routes to `"tools"` if tool calls are requested and `iteration_count < max_iterations`; otherwise routes to `"synthesis"`.
   - Node `synthesis`: Injects accumulated evidence ledger into the synthesis prompt, compiles structured report, and calculates qualitative confidence score (`High`, `Medium`, `Low`).
4. Developed `tests/test_agent.py` verifying multi-turn tool planning, evidence accumulation, and iteration limits with 100% pass rate.

### Concepts Learned & Solidified:
- **Message IDs and State Deduplication:** LangGraph's `add_messages` uses message IDs to deduplicate messages. Passing distinct tool call IDs across turns ensures clean state progression without accidental message overwriting.
- **Decoupled Activity vs. Private Scratchpad:** Instead of exposing messy internal tool call arguments or raw tokens to the user, the agent logs clean, high-signal progress strings (e.g. `✓ Retrieved market data for NVDA ($125.40)`).
- **Graceful Multi-Provider Support:** Architected `get_llm()` to seamlessly support Google Gemini (`gemini-2.0-flash` / `gemini-1.5-pro`) and OpenAI (`gpt-4o-mini`) via environment configuration.

### Problems & Challenges Encountered:
- *Challenge:* In testing multi-step execution, mock LLM experienced `IndexError` on the synthesis turn.
  - *Solution:* Clarified the state machine lifecycle: Turn 1 (LLM calls tools) $\to$ Turn 2 (LLM calls filing tool) $\to$ Turn 3 (LLM observes filing and outputs conversational completion with no tool calls) $\to$ Turn 4 (Synthesis node compiles report). Added default fallback response to mock model.
- *Challenge:* Preventing infinite loops when an LLM requests tools endlessly.
  - *Solution:* Hardcoded stopping guard in `should_continue`: `if state['iteration_count'] >= max_iter: return 'synthesis'`. Verified in `test_iteration_guard_stops_infinite_loop`.

### Important Architectural Decisions Made:
- Standardized the evidence conversion layer (`_convert_tool_output_to_evidence`) so raw API outputs from different providers (yfinance, FAISS) are normalized into identical `EvidenceItem` schemas before being logged to state.
- Qualitative confidence score is calculated deterministically from source diversity (3+ sources = High, 2 sources = Medium, 1 source = Low).

### Placement Interview Questions Generated:
1. *Walk me through the exact state transitions of your LangGraph state machine from query to final report.*
2. *How does your system prevent infinite tool-calling loops?*
3. *Why do you use a separate synthesis node rather than having the tool-calling LLM write the final report directly?*

---

## Log Entry 08: Phases 7, 8 & 9 — Evidence Ledger, Synthesis & Streamlit UI Implementation

- **Date:** October 5, 2026
- **Stage:** Phases 7, 8 & 9: Evidence Tracking, Report Synthesis & Streamlit Web Interface
- **Files Created / Modified:**
  - `app/streamlit_app.py` (Interactive financial workbench, live activity spinner, report tabs, expandable evidence cards)
  - `evaluation/test_questions.py` (Canonical 10-query benchmark evaluation suite and runner)
  - `README.md` (Comprehensive documentation, architecture diagrams, installation guide)
  - `docs/PROJECT_LOG.md` (Updated with Phases 7, 8, 9 log)
  - `docs/LEARNING_CHECKLIST.md` (Marked app/streamlit_app.py and evaluation suite completed)

### What Was Implemented:
1. Implemented **Evidence Tracking Subsystem**: Every tool invocation converts raw JSON into an immutable `EvidenceItem` with sequential citation tags (`SRC-01`, `SRC-02`, etc.), title, official URL, and raw text excerpt.
2. Built **Report Synthesis Engine**: Synthesizes structured equity research reports following Wall Street research standards with mandatory sections (Executive Summary, Key Findings, Market & Valuation, Recent Catalysts, 10-K Risk Disclosures, Overall Assessment, Sources, and Legal Disclaimers).
3. Developed **Streamlit Presentation Application (`app/streamlit_app.py`)**:
   - Modern financial terminal UI with dark slate aesthetics and responsive layout.
   - Dual-provider model selection (Google Gemini / OpenAI) with secure sidebar key management.
   - Dynamic activity spinner showing real-time tool execution milestones without exposing internal raw scratchpads.
   - 3 interactive tabs: Synthesized Report, Grounded Evidence Ledger, and Tool Execution Sequence.
   - Quick-select buttons for canonical placement demo questions.
4. Formulated **Canonical Evaluation Benchmark (`evaluation/test_questions.py`)** with 10 test scenarios covering single-tool queries, multi-turn research, company comparisons, invalid tickers, historical limitations, and ambiguous queries.
5. Executed complete test suite (`pytest -v`): Verified **27 of 27 tests passing** in 29.03 seconds.

### Concepts Learned & Solidified:
- **Separation of Presentation and Agent State:** Streamlit acts strictly as an observable presentation layer. The core research intelligence lives in the pure Python LangGraph state machine, allowing the agent to run identically via CLI, REST API, or web UI.
- **Evidence Provenance & Verification:** Why financial institutions prioritize grounded systems with clickable references over ungrounded chat models.
- **Qualitative Confidence Scoring:** Calculating analytical certainty (`High`, `Medium`, `Low`) based on evidence source diversity (market data + balance sheets + news + filings) rather than hallucinated percentages.

### Problems & Challenges Encountered:
- *Challenge:* Streamlit executes scripts from top-to-bottom on every user interaction, which could inadvertently re-run expensive LLM agent queries.
  - *Solution:* Utilized `st.session_state["latest_research"]` to persist the completed research state, allowing users to switch between the Report tab, Evidence Ledger, and Tool Activity tabs without re-triggering execution.

### Important Architectural Decisions Made:
- Kept the UI in pure Python via Streamlit to ensure the student can understand and defend 100% of the codebase without context-switching into React/Node.js.
- Embedded clickable SEC EDGAR and Yahoo Finance URLs inside every expandable evidence card for instant provenance auditability.

### Placement Interview Questions Generated:
1. *How does Streamlit handle state between user interactions, and how did you prevent redundant LLM executions?*
2. *How is your evidence ledger structured to enable in-text citations?*
3. *What criteria determine whether a research report receives a High, Medium, or Low qualitative confidence rating?*

---

## Log Entry 09: Phase 10 — Model Calibration, Live Integration & Production Verification

- **Date:** October 6, 2026
- **Stage:** Phase 10: Model Calibration, Live Integration & Production Verification
- **Files Created / Modified:**
  - `.env` (Configured active API keys, calibrated default model to `gemini-3.1-flash-lite`)
  - `agent/research_agent.py` (Added `_extract_text` normalization, calibrated default model)
  - `app/streamlit_app.py` (Added active working Gemini model options, verified interactive UI)
  - `docs/PROJECT_LOG.md` (Logged Phase 10 verification)

### What Was Implemented:
1. **Model Calibration:** Identified that Google API deprecated legacy model tags (`gemini-2.0-flash` / `gemini-1.5-flash`), returning 404 NOT_FOUND. Calibrated agent default to `gemini-3.1-flash-lite`, which delivers fast, reliable function calling and synthesis.
2. **Text Normalization:** Engineered `_extract_text` helper in `agent/research_agent.py` to seamlessly handle both raw markdown string returns and structured message content blocks from modern LangChain GenAI drivers.
3. **Cross-Platform Console Safety:** Sanitized checkmark unicode characters to prevent Windows `cp1252` encoding exceptions during terminal invocations.
4. **Live Verification:** Executed live integration test suite (`test_live_agent_execution` and 27 additional tests); all **28 of 28 tests PASSED (100%)** in 51.03s.
5. **Streamlit Deployment:** Launched interactive server on `http://localhost:8501`. Verified HTTP 200 response and end-to-end responsiveness.

### Concepts Learned & Solidified:
- **Upstream Model Lifecycle Management:** Cloud AI providers deprecate model endpoints over time. Defensive agent architectures should support runtime model overrides and dynamic fallbacks rather than hardcoding legacy version strings.
- **Multimodal Block Parsing in LangChain:** Modern LLM drivers often return structured lists of content dicts (`[{'type': 'text', 'text': ...}]`) rather than pure strings; robust synthesizers must normalize message content across providers.

---








