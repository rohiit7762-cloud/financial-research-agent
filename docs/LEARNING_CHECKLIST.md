# Financial Research Agent — Student Learning & Mastery Checklist

> **Document Purpose:** Self-assessment checklist to track technical mastery across every software architecture, AI/ML, and system engineering concept in this project before placement interviews.  
> **Grading Scale:**  
> - `[ ]` Not Yet Studied  
> - `[-]` In Progress / Partially Clear  
> - `[x]` Fully Mastered (Can explain clearly on a whiteboard under interview pressure)

---

## 1. High-Level AI & Agent Fundamentals
- [x] **LLM Mechanics:** Tokenization, autoregressive sampling, context windows, and KV cache.
- [x] **Agent Definition:** What makes a system an agent vs. a linear chain or simple chatbot.
- [x] **Agent Loop (ReAct):** Thought $\to$ Action $\to$ Observation cyclic execution flow.
- [x] **Tool Abstraction:** How Python functions are converted into JSON Schemas for LLM function calling.
- [x] **Native Tool Calling:** How LLMs generate structured tool payloads instead of free-text regex strings.
- [x] **Multi-Step Problem Solving:** How intermediate observations guide subsequent tool choices.
- [x] **Infinite Loop Safeguards:** Hard iteration caps (`MAX_ITERATIONS`), repeated parameter detection, and terminal transitions.

---

## 2. RAG & Information Retrieval (IR)
- [x] **RAG Pipeline Stages:** Document acquisition, parsing, cleaning, chunking, embedding, indexing, retrieval, synthesis.
- [x] **Embeddings:** Vector representation of semantic meaning in high-dimensional space ($\mathbb{R}^{384}$).
- [x] **Vector Distance Metrics:** Mathematical differences between Cosine Similarity, Inner Product, and Euclidean (L2) distance.
- [x] **Recursive Chunking:** Why chunking is required (token limits, attention density) and why sliding window overlap (150 chars) prevents sentence fragmentation.
- [x] **Metadata Preservation:** Why chunk IDs, source URLs, and section titles are vital for legal citations and hybrid filtering.
- [x] **Vector Database (FAISS):** How in-memory `IndexFlatL2` works and why exact search is optimal for small-to-medium corpora (<5,000 chunks).
- [x] **RAG vs. Fine-Tuning:** When to use RAG (facts, recency, citations) vs. fine-tuning (style, tone, task format).
- [x] **RAG vs. Long-Context LLMs:** Why targeted retrieval outperforms 1M-token context stuffing (Lost-in-the-Middle, cost, latency).

---

## 3. Software Architecture & Python Engineering
- [x] **Single-Agent vs. Multi-Agent Swarms:** Why a single agent with tools outperforms a multi-agent swarm in token efficiency, latency, and debuggability.
- [x] **LangGraph State Graph:** Modeling agents as state graphs with TypedDict states, nodes, and conditional edges.
- [x] **Pydantic v2:** Runtime data validation, strict type coercion, and defensive schema design.
- [x] **Defensive Error Handling:** Catching third-party API failures, mapping to structured error JSON, and graceful degradation.
- [x] **Evidence Ledger:** Centralized provenance tracking with `[SRC-xx]` source attribution.
- [x] **Streamlit Engineering:** Reactive session state management (`st.session_state`), progress spinners, and interactive UI component layout.
- [x] **Security & Environment Management:** Handling API keys via `.env` / `python-dotenv` without committing secrets to Git.

---

## 4. Source Code Mastery (Target Files)
- [x] `agent/state.py`: TypedDict state definition, `EvidenceItem` schema, and message reducers.
- [x] `agent/prompts.py`: System prompt, role constraints, citation syntax, and synthesis templates.
- [x] `agent/research_agent.py`: LangGraph graph construction, node definitions, conditional routing.
- [x] `tools/market_data.py`: `yfinance` market data fetcher with Pydantic validation.
- [x] `tools/fundamentals.py`: Balance sheet and valuation metrics extractor.
- [x] `tools/news.py`: Financial news query and snippet parser.
- [x] `tools/filing_search.py`: Tool interface querying the FAISS vector index.
- [x] `rag/ingestion.py`: Parsing curated 10-Ks and cleaning legal boilerplate.
- [x] `rag/chunking.py`: Recursive character text splitting with overlap.
- [x] `rag/embeddings.py`: SentenceTransformers embedding wrapper.
- [x] `rag/vector_store.py`: FAISS index serialization and metadata persistence.
- [x] `rag/retrieval.py`: Top-K similarity search and formatting.
- [x] `app/streamlit_app.py`: Streamlit UI, status activity indicators, and citation cards.
- [x] `evaluation/test_questions.py`: 10-query benchmark suite verifying tool selection and citation accuracy.

---

## 5. Interview Defense Milestones
- [x] Master the 30-Second, 1-Minute, and 2-Minute project pitches.
- [x] Ability to sketch the complete LangGraph state machine on a whiteboard.
- [x] Ability to write out the mathematical formula for Cosine Similarity.
- [x] Ability to defend why FAISS and `all-MiniLM-L6-v2` were chosen over Pinecone and OpenAI embeddings.
- [x] Ability to explain how the evidence ledger prevents untraceable hallucinations.
- [x] Prepared answers for tricky interview traps (trading bot claims, text-to-SQL security risks).
