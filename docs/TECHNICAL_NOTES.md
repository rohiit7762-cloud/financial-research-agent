# Financial Research Agent — Technical Notes & Concepts Encyclopedia

> **Document Purpose:** Complete technical reference and interview revision notes for all core AI, software engineering, and data concepts used in this project.  
> **Format for Every Concept:**  
> 1. Definition  
> 2. Why We Use It  
> 3. How It Works Mechanically  
> 4. How OUR Project Uses It  
> 5. Alternatives Considered  
> 6. Why We Chose Our Approach  
> 7. Target Interview Questions & Model Answers  

---

## Table of Contents
1. [Large Language Model (LLM)](#1-large-language-model-llm)
2. [AI Agent](#2-ai-agent)
3. [Tool (Agent Tool)](#3-tool-agent-tool)
4. [Tool Calling (Function Calling)](#4-tool-calling-function-calling)
5. [Agent Loop (ReAct / Cyclic Execution)](#5-agent-loop-react--cyclic-execution)
6. [LangGraph / LangChain](#6-langgraph--langchain)
7. [Retrieval-Augmented Generation (RAG)](#7-retrieval-augmented-generation-rag)
8. [Vector Embeddings](#8-vector-embeddings)
9. [Vector Database & Similarity Index (FAISS)](#9-vector-database--similarity-index-faiss)
10. [Semantic Search vs. Keyword Search](#10-semantic-search-vs-keyword-search)
11. [Text Chunking & Sliding Window Overlap](#11-text-chunking--sliding-window-overlap)
12. [Prompt Engineering & System Prompt Design](#12-prompt-engineering--system-prompt-design)
13. [Structured Outputs & Schema Validation](#13-structured-outputs--schema-validation)
14. [Pydantic v2](#14-pydantic-v2)
15. [APIs & Financial Data Ingestion (`yfinance`)](#15-apis--financial-data-ingestion-yfinance)
16. [Evidence Tracking & Provenance Ledger](#16-evidence-tracking--provenance-ledger)
17. [Groundedness & Context Adherence](#17-groundedness--context-adherence)
18. [Hallucination Mitigation in Financial AI](#18-hallucination-mitigation-in-financial-ai)
19. [Streamlit Presentation Layer](#19-streamlit-presentation-layer)

---

### 1. Large Language Model (LLM)
1. **Definition:** An autoregressive transformer-based neural network trained on vast text corpora to predict next-token probability distributions given an input sequence.
2. **Why We Use It:** To comprehend open-ended natural language research queries, reason about which domain tools to call, and synthesize multi-source evidence into coherent financial prose.
3. **How It Works:** Given a context window of tokens, tokens are mapped to vector embeddings, processed through multi-head self-attention and feed-forward layers, and decoded into probability distributions over vocabulary tokens.
4. **How OUR Project Uses It:** As the reasoning brain of the agent (e.g., `gemini-2.0-flash` or `gemini-1.5-pro` via LangChain's `ChatGoogleGenerativeAI` or OpenAI compatible). It selects tools and synthesizes the final report.
5. **Alternatives:** Fine-tuned BERT/RoBERTa models, rule-based keyword matchers, local open-source LLMs (Llama 3 8B, Mistral 7B via Ollama).
6. **Why We Chose Our Approach:** Modern hosted frontier models provide native tool-calling JSON schemas with near-zero syntax errors, avoiding local GPU hardware requirements for placement demonstrations.
7. **Interview Questions:**
   - *Q: Why not use a fine-tuned smaller model instead of a frontier LLM?*  
     *A:* Fine-tuning adjusts model weights for a static task but does not solve the fundamental need to retrieve real-time external data (stock prices move by the second). Native tool-calling capability in frontier models is far more flexible.

---

### 2. AI Agent
1. **Definition:** An autonomous entity that perceives its environment through inputs, makes reasoned decisions using an internal model, and takes actions via external tools over multiple steps to achieve an objective.
2. **Why We Use It:** Financial research is not a one-turn question-answering task; it requires dynamic planning (e.g., if price drop is unexplained by fundamentals, query recent news).
3. **How It Works:** The agent maintains a memory state, evaluates goals against current observations, selects an action (tool call), receives feedback (observation), and repeats until stopping criteria are satisfied.
4. **How OUR Project Uses It:** Single Research Agent maintaining a LangGraph state graph.
5. **Alternatives:** Static procedural scripts (`if-else`), linear chains (`PromptTemplate | LLM`), Multi-agent swarms (AutoGen, CrewAI).
6. **Why We Chose Our Approach:** A single agent with typed state achieves genuine autonomy without the token bloat and debugging chaos of multi-agent swarms.
7. **Interview Questions:**
   - *Q: What distinguishes your agent from a standard chatbot?*  
     *A:* A chatbot produces a direct one-shot response from its static weights. Our agent executes an autonomous loop with external tool dispatch, observation ingestion, and multi-turn planning before generating an answer.

---

### 3. Tool (Agent Tool)
1. **Definition:** A modular, executable software function exposed to an LLM via a structured schema (name, description, typed input arguments) that extends the model's capabilities beyond text generation.
2. **Why We Use It:** To give the LLM real-time sensing of the world: live stock quotes, balance sheets, news articles, and vector database queries.
3. **How It Works:** The function signature and docstring are converted into a JSON Schema provided in the LLM's system payload. The LLM returns a structured JSON payload requesting execution rather than raw conversational text.
4. **How OUR Project Uses It:** Exactly 4 tools: `get_market_data`, `get_company_fundamentals`, `search_financial_news`, `search_filing_rag`.
5. **Alternatives:** Feeding all data into the prompt up front (context stuffing), hardcoded function calls.
6. **Why We Chose Our Approach:** Four distinct, orthogonal tools keep the agent's action space clean and prevent decision paralysis or tool confusion.
7. **Interview Questions:**
   - *Q: How does the model know when and how to call a tool?*  
     *A:* Via the tool's JSON schema description. High-quality descriptions explain *what* the tool does, *when* it should be called, and *what format* arguments (e.g., uppercase ticker string) must take.

---

### 4. Tool Calling (Function Calling)
1. **Definition:** A fine-tuned capability in modern LLMs where the model outputs valid, machine-readable JSON matching a specified schema instead of free-form conversational text.
2. **Why We Use It:** To replace brittle regex parsing of model outputs with deterministic, schema-validated execution.
3. **How It Works:** During training, the model is taught special tokens (e.g., `<tool_call>`) that wrap serialized JSON containing `name` and `arguments`. The host runtime intercepts these tokens and invokes the Python function.
4. **How OUR Project Uses It:** Leverages LangChain's `.bind_tools([market_data, fundamentals, news, filing_search])` interface.
5. **Alternatives:** Text prompt parsing ("If you want to use a tool, write CALL_TOOL: name(arg)"), ReAct regex extraction.
6. **Why We Chose Our Approach:** Native tool calling achieves >98% schema validity compared to brittle text-based regex parsing.
7. **Interview Questions:**
   - *Q: Does the LLM execute the Python code directly?*  
     *A:* No! The LLM only generates a structured text request (JSON). The host application runtime safely validates the JSON against Pydantic and executes the Python function in the local environment.

---

### 5. Agent Loop (ReAct / Cyclic Execution)
1. **Definition:** A feedback loop where an LLM cycles through **Thought** (reasoning), **Action** (tool execution), and **Observation** (tool output) until the task is complete.
2. **Why We Use It:** Real-world research requires compounding observations. An observation from Tool A (e.g., "P/E ratio has spiked to 120") informs what to query in Tool B (e.g., search news for "earnings contraction").
3. **How It Works:** State is initialized $\to$ LLM decides tool $\to$ Tool executes $\to$ State receives observation $\to$ Loop evaluates if termination condition is met $\to$ LLM synthesizes report.
4. **How OUR Project Uses It:** Implemented using LangGraph nodes (`agent` $\leftrightarrow$ `tools`) with a conditional routing edge and a hard ceiling of `MAX_ITERATIONS = 5`.
5. **Alternatives:** Fixed DAG (Directed Acyclic Graph) pipelines, single-pass chains.
6. **Why We Chose Our Approach:** A cyclic loop handles dynamic problem complexity while the hard iteration cap guarantees protection against infinite loops and runaway billing.
7. **Interview Questions:**
   - *Q: How do you prevent infinite loops in the agent cycle?*  
     *A:* Through explicit termination guards in the state machine: (1) if the LLM emits a final response without tool calls, or (2) if `iteration_count >= MAX_ITERATIONS`, the system forcibly transitions to the report synthesis node.

---

### 6. LangGraph / LangChain
1. **Definition:** LangChain provides standard abstractions for LLMs, prompts, and tools. LangGraph is an orchestration library built on top of LangChain that models cyclic, stateful multi-step agent applications as a graph of nodes and edges.
2. **Why We Use It:** To move away from black-box agent abstractions (like `AgentExecutor`) toward an explicit, inspectable, deterministic state machine.
3. **How It Works:** A graph is defined with a `State` schema (TypedDict). Nodes represent Python functions that receive state and return state updates. Edges dictate conditional transitions based on state values.
4. **How OUR Project Uses It:** Defines an agent graph with an `agent_node`, a `tool_node`, and a `synthesis_node`.
5. **Alternatives:** Raw Python while-loops, CrewAI, AutoGen, LlamaIndex Workflows.
6. **Why We Chose Our Approach:** LangGraph is the current industry gold standard for production agent architectures, offering fine-grained observability, clean state management, and clear interview visualizability.
7. **Interview Questions:**
   - *Q: Why did you pick LangGraph instead of LangChain's legacy AgentExecutor?*  
     *A:* `AgentExecutor` was a monolithic, rigid black box deprecated by LangChain. LangGraph allows full developer control over state transitions, cyclic execution, human-in-the-loop inspection, and graceful error handling.

---

### 7. Retrieval-Augmented Generation (RAG)
1. **Definition:** An AI design pattern that supplements an LLM's prompt with dynamically retrieved, domain-specific text passages from an external knowledge base before text generation.
2. **Why We Use It:** Corporate 10-K filings are hundreds of pages long and contain legally binding risk factors that LLMs either hallucinate or omit due to context window limits and training cutoffs.
3. **How It Works:** Offline: Documents $\to$ Chunks $\to$ Embeddings $\to$ Vector DB. Online: Query $\to$ Query Embedding $\to$ Similarity Search $\to$ Top-K Chunks injected into LLM context window.
4. **How OUR Project Uses It:** A dedicated tool (`search_filing_rag`) retrieves relevant excerpts from audited SEC 10-K filings (Item 1A Risk Factors) for target companies.
5. **Alternatives:** Fine-tuning an LLM on 10-K documents, context stuffing (dumping full 200-page PDF into 1M-token context).
6. **Why We Chose Our Approach:** RAG provides immediate provenance (we know the exact page/chunk), zero retraining cost when new filings drop, and avoids the catastrophic "lost-in-the-middle" attention degradation of massive context windows.
7. **Interview Questions:**
   - *Q: Why RAG over fine-tuning for financial filings?*  
     *A:* Fine-tuning teaches a model style and vocabulary, but is terrible for factual recall and cannot provide source citations. RAG injects the exact factual text at inference time with verifiable chunk citations.

---

### 8. Vector Embeddings
1. **Definition:** Dense, low-dimensional mathematical representations (arrays of floating-point numbers) of text that capture semantic and conceptual meaning in a shared geometric vector space.
2. **Why We Use It:** To enable semantic search: finding text chunks that share conceptual meaning with a query even if they don't share exact keywords (e.g., matching "supply chain disruptions" with "component vendor shortages in East Asia").
3. **How It Works:** A bi-encoder transformer model projects text tokens through learned attention layers into a fixed-size vector (e.g., 384 dimensions for `all-MiniLM-L6-v2`). Texts with similar meanings end up with high cosine similarity.
4. **How OUR Project Uses It:** `sentence-transformers/all-MiniLM-L6-v2` embeds 10-K filing chunks and incoming agent search queries.
5. **Alternatives:** OpenAI `text-embedding-3-small`, Cohere Embed, BM25 (sparse keyword frequency).
6. **Why We Chose Our Approach:** `all-MiniLM-L6-v2` is completely open-source, runs blazing fast on CPU (<10ms per query), requires zero API keys, and has a proven track record for enterprise RAG benchmarks.
7. **Interview Questions:**
   - *Q: What does embedding dimension mean, and why does 384 matter?*  
     *A:* The dimension is the length of the vector representation. Higher dimensions (e.g., 1536) capture finer semantic nuances but require 4x more RAM and slower distance calculations. 384 dimensions strikes the perfect balance of speed and semantic capture for focused filings.

---

### 9. Vector Database & Similarity Index (FAISS)
1. **Definition:** A specialized data structure and retrieval engine optimized for fast similarity search across millions of high-dimensional dense vectors using distance metrics (Cosine Similarity, Euclidean Distance).
2. **Why We Use It:** Standard relational databases (SQL) cannot efficiently compute nearest-neighbor geometric distances across high-dimensional vectors.
3. **How It Works:** Stores vectors in memory or disk. When a query vector arrives, FAISS computes vector inner products (dot product / L2 distance) using highly optimized C++ routines (AVX instructions).
4. **How OUR Project Uses It:** `faiss-cpu` indexes our pre-chunked 10-K filing library locally.
5. **Alternatives:** ChromaDB, Pinecone, Qdrant, Weaviate, PostgreSQL pgvector.
6. **Why We Chose Our Approach:** FAISS is a lightweight, zero-dependency, local C++ library that runs with zero cloud setup or background servers, making the project 100% self-contained for offline evaluation.
7. **Interview Questions:**
   - *Q: What is the difference between FAISS IndexFlatL2 and an approximate index like IndexIVFFlat?*  
     *A:* `IndexFlatL2` computes exact brute-force Euclidean distance, guaranteeing 100% recall with $O(N)$ complexity. For our dataset of thousands of filing chunks, brute-force search takes <2 milliseconds, eliminating the need for lossy approximate clustering (IVF).

---

### 10. Semantic Search vs. Keyword Search
1. **Definition:** Keyword search (e.g., BM25) matches exact literal token occurrences; semantic search matches conceptual meaning regardless of specific phrasing.
2. **Why We Use It:** In SEC filings, companies rarely use conversational words like "problems" or "bad stuff"; they use formal terminology like "material adverse effect," "counterparty credit risk," or "regulatory enforcement headwinds."
3. **How It Works:** Queries and documents are mapped into the same continuous vector space. Distance between points reflects conceptual proximity.
4. **How OUR Project Uses It:** Enables user queries like *"What supply chain issues does Apple have?"* to retrieve filing sections titled *"Dependence on third-party manufacturing in foreign jurisdictions."*
5. **Alternatives:** Pure BM25 search, Hybrid Search (Dense + Sparse with Reciprocal Rank Fusion).
6. **Why We Chose Our Approach:** Dense semantic search captures the legalistic paraphrasing inherent in SEC disclosures. (Hybrid search is documented as a future enterprise scalability improvement).
7. **Interview Questions:**
   - *Q: When would semantic search fail where keyword search succeeds?*  
     *A:* Semantic search can struggle with exact alphanumeric identifiers (e.g., "Section 404(b)", product codes like "M3 Max", or financial figures like "$12.4B"). In production, a Hybrid Search combining BM25 and Dense Retrieval solves this.

---

### 11. Text Chunking & Sliding Window Overlap
1. **Definition:** The process of splitting long documents into smaller, coherent text segments (chunks) with a fixed character/token length and an overlapping boundary region.
2. **Why We Use It:** (1) Embedding models have token limits (e.g., 512 tokens), and (2) Embedding an entire 200-page document into one vector dilutes semantic specificity into generic noise.
3. **How It Works:** Recursive character splitting breaks text along logical boundaries (paragraphs `\n\n`, sentences `\n`, periods `.`), maintaining a chunk size of ~800 characters and a chunk overlap of 150 characters.
4. **How OUR Project Uses It:** Pre-processes SEC 10-K filings into searchable sections, preserving chapter/section metadata with every chunk.
5. **Alternatives:** Fixed-token splitting, Semantic chunking (splitting on embedding cosine distance shifts), Agentic chunking.
6. **Why We Chose Our Approach:** Recursive splitting with overlap is fast, deterministic, and preserves cross-boundary context without losing the beginning or ending of complex sentences.
7. **Interview Questions:**
   - *Q: Why do you need chunk overlap?*  
     *A:* Without overlap, a critical sentence or thought spanning the boundary between Chunk A and Chunk B gets sliced in half, corrupting the semantic meaning of both chunks and destroying retrieval accuracy.

---

### 12. Prompt Engineering & System Prompt Design
1. **Definition:** The discipline of designing structured, constraint-enforcing textual instructions provided to an LLM to guide its reasoning, tone, format, and grounding behavior.
2. **Why We Use It:** To enforce financial rigor, prevent speculative trading advice, mandate source citations, and direct tool calling strategy.
3. **How It Works:** The system prompt specifies: (1) Persona/Role (Senior Financial Research Analyst), (2) Available capabilities, (3) Mandatory citation schema (`[SRC-xx]`), (4) Negative constraints (never fabricate numbers, never provide financial advice).
4. **How OUR Project Uses It:** Modular prompt templates in `agent/prompts.py` separating the agent's reasoning policy from the synthesis formatting guidelines.
5. **Alternatives:** Few-shot prompting, chain-of-thought prompting without tools, zero-shot unconstrained prompts.
6. **Why We Chose Our Approach:** Structured role-based system prompts with clear negative constraints and explicit citation syntax produce professional, repeatable financial research outputs.
7. **Interview Questions:**
   - *Q: How do you prevent your agent from giving financial advice?*  
     *A:* By embedding strict negative constraints in the system prompt ("You are an educational research assistant, not a financial advisor; never recommend buying/selling securities") and programmatically appending a standardized compliance disclaimer to every generated report.

---

### 13. Structured Outputs & Schema Validation
1. **Definition:** Constraining an LLM's response to conform strictly to a predefined schema (such as JSON or a Pydantic class) rather than unconstrained markdown text.
2. **Why We Use It:** Downstream software systems and UI widgets require structured data fields (e.g., `ticker`, `price`, `sources`, `confidence_rating`) to render interactive tables and cards.
3. **How It Works:** Under the hood, LLM providers use grammar-based constrained sampling or JSON mode to enforce that generated tokens must conform to the specified JSON Schema grammar.
4. **How OUR Project Uses It:** Normalizes all tool outputs and evidence items into validated Pydantic models before logging to the state ledger.
5. **Alternatives:** Asking the LLM to "format as JSON" in the prompt and using `json.loads()`.
6. **Why We Chose Our Approach:** Prompt-only JSON requests frequently break due to markdown backticks or trailing commas; schema-enforced Pydantic validation guarantees runtime type safety.
7. **Interview Questions:**
   - *Q: What do you do if an LLM returns malformed JSON during tool calling?*  
     *A:* We catch validation errors using Pydantic, capture the raw string, and trigger an automated fallback or retry message instructing the LLM to correct its schema syntax.

---

### 14. Pydantic v2
1. **Definition:** The standard data validation and settings management library for Python, utilizing Python type annotations and a high-performance Rust core (`pydantic-core`).
2. **Why We Use It:** Guarantees strict type safety, parsing, serialization, and schema generation across tools, agent state, and UI representations.
3. **How It Works:** Declares data models inheriting from `BaseModel`. At instantiation, inputs are validated against type hints; invalid fields raise typed validation exceptions.
4. **How OUR Project Uses It:** Defines schemas for `MarketDataResult`, `FundamentalMetrics`, `NewsItem`, `FilingChunk`, `EvidenceItem`, and `ResearchReport`.
5. **Alternatives:** Standard Python `dataclasses`, `marshmallow`, `attrs`, untyped Python dictionaries.
6. **Why We Chose Our Approach:** Pydantic is natively integrated with LangChain, LangGraph, and FastAPI, providing 10x-50x faster validation speed in v2 due to its Rust backend.
7. **Interview Questions:**
   - *Q: What is the difference between a standard Python dataclass and a Pydantic model?*  
     *A:* Dataclasses only store type annotations without enforcing runtime validation (assigning a string to an integer field succeeds). Pydantic actively validates, coerces, and sanitizes data at runtime, raising exceptions if rules are violated.

---

### 15. APIs & Financial Data Ingestion (`yfinance`)
1. **Definition:** Application Programming Interfaces providing programmatic access to financial market data feeds, balance sheet statements, and corporate announcements.
2. **Why We Use It:** To provide reliable quantitative market data without expensive $25,000/year Bloomberg Terminal subscriptions.
3. **How It Works:** `yfinance` queries Yahoo Finance's publicly accessible backend API endpoints, parsing raw JSON responses into clean pandas DataFrames and Python dictionaries.
4. **How OUR Project Uses It:** In `tools/market_data.py` and `tools/fundamentals.py`, we sanitize and extract high-signal fields (P/E, Market Cap, 52W Range, Free Cash Flow).
5. **Alternatives:** Alpha Vantage, Polygon.io, Financial Modeling Prep (FMP), Bloomberg/FactSet.
6. **Why We Chose Our Approach:** Zero API key setup required, zero rate-limit billing risks, and comprehensive coverage of US equities, making the project effortlessly runnable for any interviewer.
7. **Interview Questions:**
   - *Q: How would you handle rate-limiting or API deprecation in production?*  
     *A:* In production, we would wrap API calls in an exponential backoff decorator (e.g., `tenacity`), implement an API gateway with caching (Redis), and configure fallback providers (e.g., Polygon.io if yfinance throttles).

---

### 16. Evidence Tracking & Provenance Ledger
1. **Definition:** An append-only data ledger that records the exact source, tool origin, timestamp, URL, and raw text snippet of every piece of data retrieved during an agentic session.
2. **Why We Use It:** In finance, an answer without provenance is unusable. The evidence ledger bridges raw tool observations and the final synthesized report.
3. **How It Works:** Every tool execution serializes its payload into standardized `EvidenceItem` objects. The synthesizer references these items by ID (`[SRC-01]`, `[SRC-02]`), and the UI renders them as clickable citations.
4. **How OUR Project Uses It:** Maintained in `ResearchAgentState["evidence_ledger"]` throughout the agent cycle.
5. **Alternatives:** Relying on the LLM to remember where it found data in its scratchpad, discarding intermediate outputs.
6. **Why We Chose Our Approach:** Guarantees 100% auditability. If an interviewer asks "Where did this $45B revenue figure come from?", you can click `[SRC-02]` and show the exact API payload timestamp.
7. **Interview Questions:**
   - *Q: How does evidence tracking help reduce hallucinations?*  
     *A:* By feeding the accumulated evidence ledger directly into the synthesis prompt and instructing the model to *only* make assertions that can be anchored to a specific `[SRC-xx]` tag. Any statement without a tag is flagged during post-processing.

---

### 17. Groundedness & Context Adherence
1. **Definition:** A metric of factual consistency measuring the degree to which an LLM's generated assertions are strictly supported by retrieved reference materials rather than parametric memory.
2. **Why We Use It:** Prevents the agent from regurgitating outdated pre-training knowledge when real-time numbers are right in front of it.
3. **How It Works:** The synthesizer is given strict instructions: "Treat the evidence ledger as your sole source of truth. If a metric is not present in the evidence ledger, state that it is unavailable."
4. **How OUR Project Uses It:** Enforced via system prompt rules and validated during the synthesis phase.
5. **Alternatives:** Allowing the LLM to freely blend its pre-trained memory with tool results.
6. **Why We Chose Our Approach:** Grounding is the single most critical attribute required by hedge funds and equity research desks deploying generative AI systems.
7. **Interview Questions:**
   - *Q: Can RAG completely eliminate hallucinations?*  
     *A:* No. RAG dramatically reduces hallucinations by providing relevant context, but LLMs can still misinterpret numbers, extrapolate unsupported conclusions, or hallucinate citations. True safety requires strict post-generation verification.

---

### 18. Hallucination Mitigation in Financial AI
1. **Definition:** Engineering techniques designed to detect, minimize, or eliminate false, fabricated, or ungrounded factual assertions made by a generative model.
2. **Why We Use It:** In financial engineering, a hallucinated debt figure or invented CEO statement can lead to disastrous investment decisions or legal liabilities.
3. **How It Works:** We employ a 4-layer defense:
   - Layer 1: Schema normalization (tools return typed data, not loose text).
   - Layer 2: Grounding prompts with negative constraints ("Do not infer unstated numbers").
   - Layer 3: Mandatory in-text source tagging (`[SRC-xx]`).
   - Layer 4: Qualitative confidence rating reflecting evidence completeness.
4. **How OUR Project Uses It:** Embedded throughout the tool contracts, agent loop, and final synthesis templates.
5. **Alternatives:** Fine-tuning RLHF with severe length penalties (which makes models overly cautious and unhelpful).
6. **Why We Chose Our Approach:** Multi-layered defense-in-depth provides verifiable safety without sacrificing analytical depth.
7. **Interview Questions:**
   - *Q: What would you do if an interviewer claims their AI has "0% hallucinations"?*  
     *A:* I would respectfully clarify that autoregressive models are probabilistic token predictors, meaning non-zero hallucination risk is mathematically inherent. High-reliability systems don't claim 0% hallucinations; they implement verification layers, deterministic tools, and provenance tracking.

---

### 19. Streamlit Presentation Layer
1. **Definition:** An open-source Python framework that converts Python scripts into interactive, reactive web user interfaces.
2. **Why We Use It:** Allows AI engineers to build high-polish, data-driven web applications without writing complex React/TypeScript frontend code.
3. **How It Works:** Streamlit runs the Python script from top to bottom whenever user state updates, caching heavy objects (`st.cache_resource`) and maintaining session state (`st.session_state`).
4. **How OUR Project Uses It:** In `app/streamlit_app.py`, provides the research input form, real-time activity spinners, executive report display, and interactive evidence drawers.
5. **Alternatives:** FastAPI + React, Gradio, Flask + Jinja2, Next.js.
6. **Why We Chose Our Approach:** Streamlit allows 100% Python development, enabling our entire system to be inspected, understood, and defended by a single engineer in a placement interview.
7. **Interview Questions:**
   - *Q: How does Streamlit handle state between re-runs?*  
     *A:* Via `st.session_state`, a dictionary-like object that persists variables (e.g., query history, cached agent instances, retrieved evidence) across user interactions without re-triggering expensive tool executions.
