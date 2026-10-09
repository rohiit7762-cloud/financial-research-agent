# Financial Research Agent — Retrieval-Augmented Generation (RAG) Architecture

> **Document Purpose:** Complete technical blueprint of the SEC Filing RAG pipeline, covering document parsing, semantic chunking, vector indexing, retrieval mathematics, source attribution, and failure boundaries.  
> **Status:** Phase 0 Blueprint (`[PLANNED]` architecture)  
> **Target Audience:** Technical Placement Interviewers evaluating Applied AI, NLP, and Information Retrieval fundamentals.

---

## 1. End-to-End RAG Pipeline Architecture

```
OFFLINE INGESTION PIPELINE (Build Time)
┌─────────────────────────────────┐
│ 1. Document Acquisition         │ Curated SEC 10-K filings (AAPL, MSFT, NVDA, TSLA)
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│ 2. Document Parsing             │ Extract Item 1A (Risk Factors) & Item 7 (MD&A)
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│ 3. Text Cleaning & Normalization│ Strip HTML tags, XBRL markers, tabular noise
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│ 4. Semantic Chunking            │ Recursive character splitting (800 chars, 150 overlap)
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│ 5. Dense Embedding Generation   │ sentence-transformers/all-MiniLM-L6-v2 (384-dim)
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│ 6. Local Vector Storage         │ FAISS IndexFlatL2 + JSON Metadata Ledger
└─────────────────────────────────┘

ONLINE RETRIEVAL & SYNTHESIS PIPELINE (Runtime)
┌─────────────────────────────────┐
│ 7. Query Formulation            │ Agent generates targeted search phrase
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│ 8. Query Embedding              │ Transform query into 384-dim vector using same model
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│ 9. Similarity Search            │ FAISS computes Euclidean / Cosine distances
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│ 10. Top-K Retrieval & Filtering │ Extract Top 3–5 chunks with similarity scores
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│ 11. Context Construction        │ Format chunks with Source IDs into LLM prompt
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│ 12. Grounded Generation         │ LLM synthesizes report using [SRC-xx] citations
└─────────────────────────────────┘
```

---

## 2. Theoretical Foundations: What You Must Defend in an Interview

### 2.1 What are Embeddings & Why are They Required?
An **embedding** is a mapping of discrete linguistic tokens or passages into a continuous, high-dimensional vector space ($\mathbb{R}^d$, where $d = 384$ for `all-MiniLM-L6-v2`) such that vectors that are close in space share conceptual and semantic meaning.

Computers and mathematical indexes cannot compute mathematical distance on raw ASCII characters. You cannot perform `distance("lawsuit", "litigation")` using basic algebra. Once converted into dense floating-point vectors:
$$\mathbf{v}_{\text{lawsuit}} \approx \mathbf{v}_{\text{litigation}}$$
Their cosine similarity is close to $1.0$, allowing mathematical algorithms to retrieve conceptually synonymous documents without relying on exact character matches.

### 2.2 What is Semantic Similarity?
Semantic similarity is a metric measuring the conceptual proximity of two texts regardless of their specific lexical phrasing. In our vector space, similarity is measured using either **Cosine Similarity** or **Euclidean Distance (L2)**:

$$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$$

$$\text{Euclidean Distance (L2)}(\mathbf{u}, \mathbf{v}) = \sqrt{\sum_{i=1}^{d} (u_i - v_i)^2}$$

When embedding vectors are L2-normalized ($\|\mathbf{u}\| = 1$), maximizing cosine similarity is mathematically identical to minimizing Euclidean distance.

### 2.3 Why is Chunking Required?
1. **Transformer Attention & Token Bounds:** Embedding models (like `all-MiniLM-L6-v2`) have a maximum input token length (typically 256–512 tokens). If you pass an entire 150-page 10-K filing into the model, all text beyond the 512th token is silently truncated and lost.
2. **Semantic Dilution (Information Smearing):** If you attempt to condense a 200-page document into a single 384-dimensional vector, the vector becomes an undifferentiated average of every topic in the company, losing all specific factual details.
3. **Retrieval Precision:** When answering a question about *semiconductor export restrictions*, you want to feed the LLM the exact 2 paragraphs discussing export rules, not 40 unrelated pages about employee health insurance.

### 2.4 What Does Chunk Overlap Do?
Chunk overlap ensures continuity of information across chunk boundaries.
- **Example without overlap (Chunk Size: 500, Overlap: 0):**
  - Chunk 1 ends with: *"The company's primary single-source supplier for advanced GPU packaging is"*
  - Chunk 2 begins with: *"TSMC located in Taiwan, which presents geopolitical risks."*
  - **Result:** Neither chunk contains the complete proposition. A search for *"Who manufactures NVIDIA's GPUs?"* fails on both chunks.
- **Example with overlap (Chunk Size: 500, Overlap: 150):**
  - The sentence spans across both chunks, ensuring that the semantic relationship between supplier and geopolitical risk is preserved in at least one intact vector representation.

### 2.5 Why Metadata Matters
Retrieving raw text is insufficient for financial compliance. Every chunk in our FAISS store is bundled with an immutable metadata dictionary:
```json
{
  "chunk_id": "NVDA-2024-10K-ITEM1A-042",
  "symbol": "NVDA",
  "company_name": "NVIDIA Corporation",
  "filing_type": "10-K",
  "fiscal_year": 2024,
  "section": "Item 1A. Risk Factors",
  "source_url": "https://www.sec.gov/ix?doc=/Archives/edgar/data/1045810/000104581024000029/nvda-20240128.htm",
  "char_start": 48200,
  "char_end": 49000
}
```
Metadata enables **hybrid filtering** (e.g., *only search chunks where `symbol == "AAPL"` and `fiscal_year == 2024`*) and provides the exact URL and filing section for in-text citations.

---

## 3. RAG vs. Fine-Tuning: The Classic Interview Question

| Evaluation Criteria | RAG (Our Choice) | Fine-Tuning (Alternative) |
| :--- | :--- | :--- |
| **Knowledge Recency** | Real-time (instant update by indexing new filing) | Static (requires re-training/fine-tuning cycle) |
| **Auditability & Citations** | Perfect (returns exact text chunk and SEC URL) | Impossible (model generates from parametric weights) |
| **Hallucination Rate** | Low (strictly grounded in retrieved prompt context) | High (fine-tuned models still confabulate numbers) |
| **Computational Cost** | Low (CPU vector search in milliseconds) | High (requires high-VRAM GPUs like A100/H100) |
| **Purpose** | Information Retrieval & Factual Grounding | Style, Tone, and Specialized Grammar Adaptation |

**Interview Pitch:**  
*"You do not fine-tune a model to memorize dynamic financial numbers; you fine-tune to teach a model a specialized linguistic behavior or task format. For factual recall, audited corporate disclosures, and compliance-grade citations, RAG is the industry standard because it provides verifiable provenance and zero retraining cost when new quarterly filings are released."*

---

## 4. Why RAG is Better than Long-Context LLMs

Modern LLMs advertise 1M to 2M token context windows (e.g., Gemini 1.5 Pro). An interviewer might ask:
> *"Why bother building a RAG pipeline when you can just dump the entire 200-page 10-K directly into Gemini's 1-million token context window?"*

### Our Engineering Defense:
1. **The "Lost in the Middle" Phenomenon:** Extensive NLP benchmarks (Liu et al., 2023) prove that LLM attention mechanisms degrade when extracting nuanced factual details placed in the middle of massive 100k+ token contexts. Targeted chunk retrieval maximizes attention density.
2. **Inference Latency:** Processing a 300,000-token prompt takes 15–30 seconds to prefill the KV cache before the first output token is generated. RAG retrieves 3 chunks (1,500 tokens) and generates in under 2 seconds.
3. **Financial Cost:** At scale, querying 300k tokens per user question costs $0.30 to $1.50 per query in API fees. RAG costs <$0.001 per query.
4. **Multi-Source Synthesis:** A user question often compares 2 or 3 companies across market data, news, and filings. Stuffing multiple 200-page filings plus news articles quickly overwhelms even large context limits.

---

## 5. Chunking Strategy & Implementation Details

Our chunking strategy uses LangChain's `RecursiveCharacterTextSplitter`:
- **Target Chunk Size:** 800 characters (~150–200 words). Large enough to contain a complete legal risk argument, small enough to maintain high semantic density.
- **Overlap:** 150 characters (~25–35 words). Prevents splitting compound legal sentences across boundaries.
- **Separators Hierarchy:** `["\n\n", "\n", ". ", " ", ""]`. Prioritizes paragraph breaks first, sentence boundaries second, word boundaries third.

---

## 6. Vector Database Architecture: Why FAISS?

We use **FAISS (`faiss-cpu`)** (Facebook AI Similarity Search) with an `IndexFlatL2` index:
- **Zero Cloud Infrastructure:** Runs entirely in-process within Python memory, backed by serialized binary files on disk (`data/vector_store/index.faiss`).
- **Exact Nearest-Neighbor Search:** For our curated library of target corporate filings (~2,000 chunks across 4 major equities), an exact brute-force search executes in **< 2 milliseconds** on an ordinary CPU.
- **No Vector Cloud Latency or Cost:** Cloud vector databases (Pinecone, Qdrant Cloud) introduce 40–100ms of network overhead and recurring subscription costs, which are completely unnecessary for a localized placement portfolio project.

---

## 7. Known Limitations & Production Enhancements

In an interview, acknowledging engineering trade-offs proves seniority:

| Current Implementation Limitation | How Production Systems Solve It |
| :--- | :--- |
| **Curated 10-K Ingestion:** Only indexes benchmark equities (AAPL, MSFT, NVDA, TSLA) | Real-time automated SEC EDGAR crawler using SEC EDGAR API with daily ticker ingestion. |
| **Dense Semantic Search Only:** Struggles with exact financial figures or tax code numbers | **Hybrid Search:** Combines BM25 (sparse keyword frequency) with FAISS (dense vectors) via Reciprocal Rank Fusion (RRF). |
| **No Re-Ranking Step:** Top-K retrieved directly from vector distance | **Cross-Encoder Re-Ranker:** Uses Cohere Re-rank or `bge-reranker-large` to re-score Top-10 chunks down to Top-3 for higher precision. |
| **Plain Text Extraction:** Discards complex financial multi-column tables in Item 8 | **Multi-Modal Document Parsing:** Uses Unstructured.io or Azure Form Recognizer to convert financial tables into Markdown/HTML tables before chunking. |
