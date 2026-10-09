# Financial Research Agent — Architectural Decision Log (ADR)

> **Document Purpose:** Explicit architectural trade-off records for all major technical decisions.  
> **Format for Every Entry:**  
> - **Decision:** What was chosen  
> - **Context & Rationale:** Why it was chosen  
> - **Alternatives Considered:** What else was evaluated  
> - **Why Alternatives Were Rejected:** Concrete reasons for rejection  
> - **Trade-Offs & Downsides:** Real-world compromises accepted  
> - **Interview Defense:** How to articulate this decision in 60 seconds

---

## ADR-001: Orchestration Topology — Single Agent with Tool Calling vs. Multi-Agent Swarm

- **Decision:** Build a single high-capability Research Agent running an explicit cyclic LangGraph state machine, rather than a multi-agent swarm (e.g., Sentiment Agent + Technical Agent + Fundamental Agent).
- **Context & Rationale:** The reference architecture (`gsaini/financial-research-analyst-agent`) employs 10+ distributed agents coordinated via message queues. For an undergraduate placement portfolio, multi-agent swarms introduce severe non-deterministic failure modes, token bloat (each agent requiring repetitive system prompts), and impossible debugging during a live 30-minute interview.
- **Alternatives Considered:**
  1. *Multi-Agent Swarm (CrewAI / AutoGen / LangGraph Multi-Agent):* Multiple conversational personas chatting with each other.
  2. *Hardcoded Procedural Pipeline (`if-else` scripts):* Fixed rules determining tool execution.
- **Why Alternatives Were Rejected:**
  - *Swarm rejected because:* Agents spend 70% of their tokens discussing handoffs with each other rather than doing research. Failures are stochastic and impossible to reproduce reliably on a laptop during an interview.
  - *Hardcoded pipeline rejected because:* It is not agentic. It cannot dynamically re-plan or follow up on unexpected tool observations.
- **Trade-Offs & Downsides:** A single agent has a larger combined tool schema in its context window (4 tools), requiring clear, orthogonal tool descriptions to prevent tool confusion.
- **Interview Defense:**  
  *"I intentionally chose a single agent with a cyclic state machine over a multi-agent swarm. In equity research, what matters is deterministic data collection and provenance, not simulated conversational debates between personas. A single agent executes in 1/4 the token footprint, runs with zero distributed infrastructure, and has a transparent state graph that can be fully defended on a whiteboard."*

---

## ADR-002: Orchestration Framework — LangGraph over Legacy LangChain `AgentExecutor`

- **Decision:** Use **LangGraph** with an explicit TypedDict state graph to govern the agent execution loop, rather than LangChain's legacy `AgentExecutor` or raw Python loops.
- **Context & Rationale:** `AgentExecutor` is a legacy black box that hides state transitions, makes intercepting intermediate tool observations difficult, and was officially deprecated by the LangChain team in favor of graph-based state machines.
- **Alternatives Considered:**
  1. *Legacy `AgentExecutor` (`langchain.agents`):* Monolithic pre-packaged loop.
  2. *Custom Python `while` loop:* Raw code executing tools until the LLM returns no function calls.
- **Why Alternatives Were Rejected:**
  - *`AgentExecutor` rejected because:* Deprecated, rigid, black-box logging, difficult to attach custom evidence ledgers or enforce complex routing conditions.
  - *Raw `while` loop rejected because:* Reinvents state checkpointing, conditional edge routing, and tool message schemas that LangGraph standardizes across the enterprise ecosystem.
- **Trade-Offs & Downsides:** LangGraph introduces a slight learning curve regarding state channels, reducers, and graph compilation.
- **Interview Defense:**  
  *"LangGraph models the agent as a discrete finite state machine with inspectable nodes and typed state transitions. This gave us total control over iteration caps, error routing, and real-time UI status updates without relying on deprecated black-box executors."*

---

## ADR-003: Vector Indexing Engine — Local FAISS over Cloud Vector Databases (Pinecone / Milvus)

- **Decision:** Use **FAISS (`faiss-cpu`)** running locally in-process, serialized to disk, rather than a managed cloud vector database.
- **Context & Rationale:** Our filing corpus consists of curated annual 10-Ks (~2,000 chunks). An in-process FAISS `IndexFlatL2` computes exact brute-force Euclidean distance in under 2 milliseconds on a single CPU core.
- **Alternatives Considered:**
  1. *Pinecone / Weaviate Cloud:* Managed cloud vector SaaS.
  2. *PostgreSQL with `pgvector`:* Relational DB with vector extensions.
  3. *ChromaDB:* Embedded Python vector database.
- **Why Alternatives Were Rejected:**
  - *Pinecone/Weaviate rejected because:* Introduces 40–100ms cloud network latency per search, requires paid API keys, and creates external points of failure during an offline interview demo.
  - *pgvector rejected because:* Running a local PostgreSQL daemon adds unnecessary DevOps overhead to what should be an effortlessly cloneable repository.
  - *ChromaDB rejected because:* Chroma's SQLite/telemetry dependencies frequently cause compilation and lock conflicts on Windows environments. FAISS-CPU installs with a single clean wheel.
- **Trade-Offs & Downsides:** FAISS does not natively provide persistent document metadata storage out-of-the-box (we solved this by maintaining a companion JSON/pickle metadata ledger mapped by vector index IDs).
- **Interview Defense:**  
  *"For a focused corpus of benchmark corporate filings, FAISS provides sub-2ms retrieval with zero network latency, zero infrastructure cost, and 100% offline portability. Adding a cloud vector database would add cost and failure points without improving retrieval quality."*

---

## ADR-004: Embedding Model — `sentence-transformers/all-MiniLM-L6-v2` over OpenAI Embeddings

- **Decision:** Embed SEC filing chunks locally using the open-source **`all-MiniLM-L6-v2`** model (384 dimensions) via HuggingFace `sentence-transformers`.
- **Context & Rationale:** Financial document retrieval requires fast, consistent, zero-cost vector transformations that do not depend on external API quotas or billing.
- **Alternatives Considered:**
  1. *OpenAI `text-embedding-3-small` (1536 dims):* Cloud API embedding.
  2. *BGE-Large / E5-Mistral:* Heavy open-source 1024-dim embedding models.
- **Why Alternatives Were Rejected:**
  - *OpenAI Embeddings rejected because:* Every indexing and search operation burns paid API credits and fails if the developer is offline or out of quota.
  - *BGE-Large rejected because:* Requires significant VRAM or slow CPU inference (>150ms per query), degrading local performance.
- **Trade-Offs & Downsides:** 384 dimensions capture slightly less nuanced semantic context than 1536-dimensional models, but benchmark recall on English legal/financial text remains above 90% for top-5 retrieval.
- **Interview Defense:**  
  *"MiniLM-L6-v2 runs locally on CPU in <10ms, produces compact 384-dimensional vectors that keep our FAISS index footprint under 15MB, and ensures the entire RAG pipeline runs completely free without external API dependencies."*

---

## ADR-005: Market & Fundamental Data Provider — `yfinance` over Proprietary Financial APIs

- **Decision:** Use **`yfinance`** as the primary market data and balance-sheet metrics provider.
- **Context & Rationale:** Provides instantaneous quotes, 52-week statistics, historical volatility, and audited financial statements for all US equities without requiring subscription contracts or credit card registration.
- **Alternatives Considered:**
  1. *Alpha Vantage / Financial Modeling Prep (FMP):* REST APIs with free tiers.
  2. *Bloomberg / FactSet API:* Institutional financial feeds.
- **Why Alternatives Were Rejected:**
  - *Alpha Vantage rejected because:* Free tier is severely rate-limited (5 requests per minute, 500 per day), causing agent multi-step loops to fail instantly under evaluation testing.
  - *Bloomberg rejected because:* Costs $25,000/year and is impossible to run on a personal student portfolio.
- **Trade-Offs & Downsides:** `yfinance` relies on scraping Yahoo Finance endpoints, meaning schema changes on Yahoo's side can occasionally break field names. We mitigated this by wrapping all extraction in defensive Pydantic models with null-safe fallbacks.
- **Interview Defense:**  
  *"yfinance gave us comprehensive coverage of price, volume, ratios, and balance sheets with zero API key barriers. To protect against schema volatility, we engineered a defensive Pydantic sanitization layer that validates types and prevents null-pointer crashes."*

---

## ADR-006: Presentation Layer — Streamlit over React/FastAPI Full-Stack

- **Decision:** Build the user interface using **Streamlit**, keeping the entire codebase in unified Python.
- **Context & Rationale:** The purpose of the placement project is to demonstrate mastery of AI agent architectures, RAG retrieval mechanics, and Python backend engineering. A React/Next.js frontend adds substantial JavaScript/CSS boilerplate without adding any AI engineering value.
- **Alternatives Considered:**
  1. *FastAPI Backend + React / TypeScript Frontend:* Decoupled web architecture.
  2. *Gradio:* Alternative Python ML UI library.
  3. *Pure CLI:* Terminal command-line interface.
- **Why Alternatives Were Rejected:**
  - *FastAPI + React rejected because:* Splits development time into frontend state management, CORS configuration, and NPM packaging rather than perfecting the agent reasoning loop and RAG evaluation.
  - *Gradio rejected because:* Gradio's layout model is rigid and optimized for single-function inputs/outputs (like Stable Diffusion image generators) rather than rich financial dashboards with collapsible source drawers and dynamic status containers.
  - *CLI rejected because:* Harder for non-technical interviewers or recruiters to quickly interact with and appreciate the live tool execution activity.
- **Trade-Offs & Downsides:** Streamlit re-runs scripts on user interaction, requiring careful usage of `st.session_state` to avoid re-triggering costly LLM agent executions on UI state changes.
- **Interview Defense:**  
  *"Streamlit allowed me to build a high-polish, reactive financial workbench with real-time tool activity logs and interactive citation drawers while keeping 100% of the codebase in Python, allowing me to master and defend every single line of code in the interview."*

---

## ADR-007: Curated 10-K Filing Scope over Full SEC EDGAR Firehose

- **Decision:** Index a curated corpus of benchmark technology equities (AAPL, MSFT, NVDA, TSLA) focusing on Item 1A (Risk Factors) and Item 7 (MD&A), rather than crawling the entire SEC EDGAR database.
- **Context & Rationale:** Ingesting all 10,000+ public company filings requires hundreds of gigabytes of storage, multi-day background workers, and heavy parsing infrastructure. Curating benchmark companies provides full real-world 10-K complexity while maintaining a 100% testable, fast-loading portfolio project.
- **Alternatives Considered:**
  1. *Full SEC EDGAR Crawler:* Ingesting all US public company filings.
  2. *Toy Synthetic Text Files:* Writing 2-paragraph fake company descriptions.
- **Why Alternatives Were Rejected:**
  - *Full EDGAR rejected because:* Massive data engineering bloat unrelated to testing the core agent and RAG reasoning loops.
  - *Toy text rejected because:* Fails to demonstrate real-world parsing challenges (messy legal disclaimers, boilerplate risk language, complex terminology).
- **Trade-Offs & Downsides:** The system only supports deep filing RAG for the indexed benchmark companies (for other tickers, the agent gracefully informs the user and falls back to market data + live news).
- **Interview Defense:**  
  *"I used real, audited SEC 10-K filings from major benchmark equities to capture true legalistic language complexity, but scoped the corpus to keep the vector store lightweight and instantly runnable on any interviewer's laptop."*
