# Autonomous Financial Research Agent

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![FAISS](https://img.shields.io/badge/Vector_DB-FAISS-green.svg)](https://github.com/facebookresearch/faiss)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)

An autonomous, multi-tool AI equity research agent designed for high-signal equity analysis, factual grounding, and auditability.

Given open-ended natural language investment research queries, the agent autonomously coordinates **real-time market data**, **audited balance sheet fundamentals**, **financial news search**, and a **local SEC 10-K filing RAG pipeline** inside an explicit cyclic state machine, compiling verifiable research memos with **in-text citations `[SRC-xx]`** and **qualitative confidence scoring**.

---

## 1. System Architecture

```
                                  USER (Streamlit UI)
                                          │
                                          │ Natural Language Query
                                          ▼
                         ┌─────────────────────────────────┐
                         │   Streamlit Presentation Layer  │
                         │  - Model Selector (Gemini/OpenAI)│
                         │  - High-Level Activity Spinner  │
                         │  - Executive Report & Citations │
                         └────────────────┬────────────────┘
                                          │
                                          │ Initial ResearchAgentState
                                          ▼
                         ┌─────────────────────────────────┐
                         │     LangGraph State Machine     │
                         │         (Reasoning Loop)        │
                         └──────────────┬───▲──────────────┘
                                        │   │
                  ┌─────────────────────┘   └─────────────────────┐
                  ▼                                               │ Tool Result
       ┌────────────────────┐                       ┌─────────────┴───────────┐
       │   Agent Node       │                       │     Tool Execution      │
       │ Evaluates history  │                       │ Normalizes & logs to    │
       │ & selects tool(s)  │                       │ immutable evidence list │
       └──────────┬─────────┘                       └─────────────▲───────────┘
                  │                                               │
                  │ Tool Call Request(s)                          │ Raw Output
                  ▼                                               │
       ┌──────────────────────────────────────────────────────────┴───────────┐
       │                          TOOL EXECUTION BUS                          │
       │                                                                      │
       │  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────────┐  │
       │  │   Market Data   │  │  Fundamentals   │  │  Financial News      │  │
       │  │    (yfinance)   │  │   (yfinance)    │  │ (yfinance Search)    │  │
       │  └────────┬────────┘  └────────┬────────┘  └──────────┬───────────┘  │
       │           │                    │                      │              │
       │           └────────────────────┼──────────────────────┘              │
       │                                │                                     │
       │                                ▼                                     │
       │                   ┌──────────────────────────┐                       │
       │                   │    SEC Filing Search     │                       │
       │                   │    (Local FAISS RAG)     │                       │
       │                   └────────────┬─────────────┘                       │
       │                                │                                     │
       └────────────────────────────────┼─────────────────────────────────────┘
                                        │
                                        │ Semantic Similarity Top-K
                                        ▼
                         ┌─────────────────────────────┐
                         │   Vector Store (FAISS)      │
                         │   Embeddings: MiniLM-L6-v2  │
                         │   Corpus: Curated 10-Ks     │
                         └─────────────────────────────┘
                                        │
                                        │ Chunks + Metadata
                                        ▼
                         ┌─────────────────────────────┐
                         │   Evidence Ledger Builder   │
                         │  - Assigns [SRC-01], etc.   │
                         │  - Normalizes schemas       │
                         └──────────────┬──────────────┘
                                        │
                                        │ Complete Evidence Ledger
                                        ▼
                         ┌─────────────────────────────┐
                         │    Report Synthesis Node    │
                         │  - Grounded Structured Gen  │
                         │  - In-text Citations        │
                         │  - Confidence: HIGH/MED/LOW │
                         └──────────────┬──────────────┘
                                        │
                                        │ Final Markdown Report + Evidence Drawer
                                        ▼
                                  USER (Streamlit UI)
```

---

## 2. Our Architecture vs. Large Multi-Agent Reference Designs

To defend this project in placement interviews, you must understand how our architecture compares to large-scale enterprise reference designs (such as the `gsaini/financial-research-analyst-agent` reference repository):

| Dimension | Enterprise Reference Architecture | Our Implementation |
| :--- | :--- | :--- |
| **Topology** | 11+ Distributed Agents chatting with each other | Single High-Capability Agent with dynamic 4-tool calling loop |
| **State** | Redis message queues + Celery workers | Centralized immutable TypedDict state machine (LangGraph) |
| **Tokens & Latency**| Very High (redundant inter-agent chatter) | Low ($\frac{1}{4}$ token footprint; executes in seconds) |
| **Vector DB** | Cloud Pinecone / Milvus cluster | Local in-process FAISS (`IndexFlatIP`) running on CPU |
| **Embeddings** | Cloud OpenAI `text-embedding-ada-002` | Local Hugging Face `all-MiniLM-L6-v2` (Zero-cost, offline) |
| **Defensibility** | Hard to defend; frequent non-deterministic drops | 100% whiteboard-defensible; deterministic state transitions |

---

## 3. Four Core Specialized Tools

1. **Market Data Tool (`tools/market_data.py`):**
   - Retrieves live stock quotes, previous close, daily delta, volume, 52-week trading bounds, and formatted market cap (`$3.25T`) via `yfinance.fast_info`.
   - Defensive error containment returning `MarketDataError` on invalid or delisted tickers.
2. **Company Fundamentals Tool (`tools/fundamentals.py`):**
   - Extracts audited balance sheet and valuation metrics: Trailing/Forward P/E, PEG Ratio, Diluted EPS, Total Revenue, Net Income, Operating Margins, Total Debt, Cash Reserves, Free Cash Flow, and Debt-to-Equity.
3. **Financial News Tool (`tools/news.py`):**
   - Queries verified financial news media via `yfinance.Search`.
   - Features **Intelligent Query Relaxation Fallback**: automatically strips conversational stop words from verbose LLM queries to maintain high recall.
4. **SEC Filing RAG Tool (`tools/filing_search.py`):**
   - Ingests audited annual SEC 10-K filings (Item 1A Risk Factors) for benchmark equities (AAPL, MSFT, NVDA, TSLA).
   - Recursive character chunking (800 chars, 150 overlap) embedded into 384-dimensional dense vectors using `all-MiniLM-L6-v2` and indexed with FAISS.

---

## 4. Evidence Tracking & Anti-Hallucination Guardrails

- **Immutable Evidence Ledger:** Every piece of retrieved data is serialized into an `EvidenceItem` with a unique ID (`[SRC-01]`, `[SRC-02]`).
- **Strict Grounding Prompt:** The synthesis node is strictly instructed that the evidence ledger is its sole source of truth and mandates in-text citation tags.
- **Qualitative Confidence Scoring:** Evaluates report reliability as `HIGH`, `MEDIUM`, or `LOW` based on cross-source evidence coverage.
- **Infinite Loop Safeguards:** Hard iteration ceiling (`MAX_ITERATIONS = 5`) ensures the agent can never enter runaway execution cycles.

---

## 5. Quickstart & Installation

### Step 1: Clone and Set Up Virtual Environment
```bash
# Clone the repository
cd financial-research-agent

# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Install pinned dependencies
pip install -r requirements.txt
```

### Step 2: Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Edit `.env` and add your **Google Gemini API Key** (free at [Google AI Studio](https://aistudio.google.com/)) or **OpenAI API Key**:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### Step 3: Run the Streamlit Application
```bash
streamlit run app/streamlit_app.py
```
The application will launch locally at `http://localhost:8501`.

---

## 6. Automated Test Suite

We maintain 100% test coverage across all tools, RAG subsystems, and agent state transitions:

```bash
# Run all unit and integration tests
pytest -v
```

Test Breakdown:
- `tests/test_setup.py`: Validates all 11 dependencies and environment health.
- `tests/test_market_data.py`: Validates quote extraction, sanitization, and invalid ticker containment.
- `tests/test_fundamentals.py`: Validates valuation ratios, balance sheet extraction, and error handling.
- `tests/test_news.py`: Validates news queries, conceptual search, and query relaxation fallbacks.
- `tests/test_rag.py`: Validates document ingestion, chunking, FAISS persistence, and semantic precision.
- `tests/test_agent.py`: Validates multi-step LangGraph execution, evidence logging, and iteration limits.

---

## 7. Complete Documentation Suite

All core architectural and design specifications are available in the [`docs/`](docs/) directory:

- [**`docs/PROJECT_OVERVIEW.md`**](docs/PROJECT_OVERVIEW.md) — Scope, problem statement, features, and boundary conditions.
- [**`docs/ARCHITECTURE.md`**](docs/ARCHITECTURE.md) — Component breakdown, ASCII diagrams, data flows, and failure containment.

*(Note: In-depth private interview preparation notes, 19-concept technical breakdowns, and personal evaluation logs are maintained locally by the author for placement defense).*

---

## 8. Compliance & Legal Disclaimer

*This application is built strictly for research and educational purposes as part of an undergraduate engineering portfolio. It does not constitute financial, investment, legal, or tax advice. Market quotes are subject to third-party delays.*
