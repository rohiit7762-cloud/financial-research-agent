"""Streamlit Application for Financial Research Agent.

Interactive equity research workbench featuring:
- Autonomous multi-tool reasoning activity tracker
- Grounded executive research memo with in-text [SRC-xx] citations
- Expandable evidence ledger with clickable SEC and financial sources
- Model selection (Gemini / OpenAI) with API key configuration
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path to enable absolute imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from dotenv import load_dotenv

from agent.research_agent import run_research, get_llm
from rag.retrieval import get_vector_store

load_dotenv()

# ============================================================
# PAGE CONFIGURATION & STYLING
# ============================================================
st.set_page_config(
    page_title="Financial Research Agent | AI Equity Analyst",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Financial Terminal CSS Theme
st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .badge-high {
        background-color: #065f46;
        color: #34d399;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.8rem;
    }
    .badge-medium {
        background-color: #854d0e;
        color: #fde047;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.8rem;
    }
    .badge-low {
        background-color: #991b1b;
        color: #f87171;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.8rem;
    }
    .evidence-box {
        background-color: #0f172a;
        border-left: 4px solid #38bdf8;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 10px;
    }
    .source-tag {
        font-weight: 700;
        color: #38bdf8;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR CONFIGURATION
# ============================================================
with st.sidebar:
    st.title("⚙️ Agent Controls")
    st.markdown("Autonomous multi-tool equity research engine.")
    st.divider()

    # Provider & Model Selection
    st.subheader("1. LLM Provider")
    provider = st.selectbox(
        "Select Provider",
        options=["Google Gemini (Recommended)", "OpenAI"],
        index=0
    )

    if "Gemini" in provider:
        target_provider = "google"
        default_model = "gemini-3.1-flash-lite"
        model_name = st.selectbox(
            "Gemini Model",
            options=["gemini-3.1-flash-lite", "gemini-3.8-flash", "gemini-flash-latest"],
            index=0
        )
        api_key_input = st.text_input(
            "Google API Key",
            type="password",
            value=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "",
            help="Free key available at https://aistudio.google.com/"
        )
    else:
        target_provider = "openai"
        default_model = "gpt-4o-mini"
        model_name = st.selectbox(
            "OpenAI Model",
            options=["gpt-4o-mini", "gpt-4o"],
            index=0
        )
        api_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY") or "",
            help="Key available at https://platform.openai.com/api-keys"
        )

    st.divider()
    st.subheader("2. Agent Hyperparameters")
    max_iter = st.slider("Max Tool Iterations", min_value=1, max_value=8, value=5, help="Hard ceiling to prevent infinite tool loops")
    temp = st.slider("Temperature", min_value=0.0, max_value=0.7, value=0.1, step=0.05, help="Low temperature enforces grounded factual retrieval")

    st.divider()
    st.subheader("3. Knowledge Base Status")
    try:
        store = get_vector_store()
        indexed_symbols = sorted(list({c.symbol for c in store.chunks}))
        st.success(f"✓ FAISS Store Active: {store.total_vectors} chunks")
        st.caption(f"Indexed 10-K Filings: {', '.join(indexed_symbols)}")
    except Exception as e:
        st.warning(f"Vector Store Initializing: {e}")

    st.divider()
    st.caption("Developed for Placement Portfolio by Rohit Kumar • Python 3.12 • LangGraph • FAISS")

# ============================================================
# MAIN APPLICATION INTERFACE
# ============================================================
st.title("📈 Autonomous Financial Research Agent")
st.markdown(
    "Ask open-ended equity research questions. The agent autonomously coordinates **Market Data**, "
    "**Balance Sheet Fundamentals**, **Financial News Search**, and **SEC 10-K Filing RAG** "
    "to synthesize an auditable, source-cited research memo."
)

# Quick Demo Prompts
st.markdown("**Quick Demo Questions:**")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🔍 Why has NVIDIA moved recently?", use_container_width=True):
        st.session_state["user_query"] = "Why has NVIDIA moved recently and what catalysts are driving it?"
        st.session_state["ticker_input"] = "NVDA"
with col2:
    if st.button("⚖️ Compare AAPL vs MSFT fundamentals", use_container_width=True):
        st.session_state["user_query"] = "Compare Apple and Microsoft fundamentals including P/E, revenue, free cash flow and debt."
        st.session_state["ticker_input"] = "AAPL"
with col3:
    if st.button("⚠️ What risks did Apple cite in its 10-K?", use_container_width=True):
        st.session_state["user_query"] = "What risks did Apple highlight in its latest annual 10-K filing regarding supply chains and antitrust?"
        st.session_state["ticker_input"] = "AAPL"

# Query Input Form
with st.form("research_query_form"):
    query_text = st.text_area(
        "Ask a financial research question:",
        value=st.session_state.get("user_query", ""),
        placeholder="e.g. Analyze NVIDIA considering recent stock performance, fundamentals, recent news, and 10-K filing risks.",
        height=90
    )
    
    col_t, col_btn = st.columns([1, 4])
    with col_t:
        ticker_val = st.text_input(
            "Primary Ticker (Optional)",
            value=st.session_state.get("ticker_input", ""),
            placeholder="e.g. NVDA"
        )
    with col_btn:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        submit_button = st.form_submit_button("🚀 Run Research Agent", use_container_width=True, type="primary")

# ============================================================
# EXECUTION & RESULTS PRESENTATION
# ============================================================
if submit_button:
    if not query_text.strip():
        st.warning("Please enter a research question.")
    elif not api_key_input.strip():
        st.error(
            f"Please enter your {provider.split()[0]} API Key in the left sidebar to run the agent. "
            "You can obtain a free Gemini API key at https://aistudio.google.com/."
        )
    else:
        # Configure chosen API key in environment
        if target_provider == "google":
            os.environ["GOOGLE_API_KEY"] = api_key_input.strip()
            os.environ["GEMINI_API_KEY"] = api_key_input.strip()
        else:
            os.environ["OPENAI_API_KEY"] = api_key_input.strip()

        # Step 1: Display High-Level Research Activity Spinner
        with st.status("🤖 Financial Research Agent is executing...", expanded=True) as status_box:
            st.write("Initializing agent state machine and tool schemas...")
            
            try:
                llm = get_llm(
                    provider=target_provider,
                    model_name=model_name,
                    temperature=temp,
                    google_api_key=api_key_input if target_provider == "google" else None,
                    openai_api_key=api_key_input if target_provider == "openai" else None
                )

                final_state = run_research(
                    query=query_text.strip(),
                    ticker=ticker_val.strip() if ticker_val else None,
                    llm=llm,
                    max_iterations=max_iter
                )

                # Render discrete activity milestones without exposing messy raw scratchpad
                for activity in final_state.get("tool_activity", []):
                    st.write(activity)

                status_box.update(label="✓ Research Completed Successfully!", state="complete", expanded=False)
                st.session_state["latest_research"] = final_state

            except Exception as e:
                status_box.update(label=f"⚠️ Execution Encountered Error: {str(e)}", state="error")
                st.error(f"Error details: {str(e)}")

# Display Persisted Results
if "latest_research" in st.session_state:
    res = st.session_state["latest_research"]
    final_report = res.get("final_report", "No report generated.")
    confidence = res.get("confidence_rating", "MEDIUM")
    ledger = res.get("evidence_ledger", [])
    activity = res.get("tool_activity", [])

    st.divider()

    # Top Header Summary
    col_hdr, col_conf, col_count = st.columns([3, 1, 1])
    with col_hdr:
        st.subheader("📑 Research Intelligence Report")
        st.caption(f"Query: *\"{res.get('query')}\"*")
    with col_conf:
        st.markdown("<div class='metric-title'>Qualitative Confidence</div>", unsafe_allow_html=True)
        badge_class = f"badge-{confidence.lower()}"
        st.markdown(f"<span class='{badge_class}'>{confidence} CONFIDENCE</span>", unsafe_allow_html=True)
    with col_count:
        st.markdown("<div class='metric-title'>Evidence Sources</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{len(ledger)} Sources</div>", unsafe_allow_html=True)

    # Main Tabs: Report vs. Evidence Ledger vs. Activity Log
    tab_report, tab_evidence, tab_activity = st.tabs([
        "📄 Synthesized Report",
        f"🔍 Evidence Ledger ({len(ledger)})",
        "⚙️ Tool Activity Log"
    ])

    with tab_report:
        st.markdown(final_report)
        st.divider()
        st.info("💡 **Compliance Note:** This analysis is generated by an autonomous AI agent for research and educational purposes and does not constitute financial advice.")

    with tab_evidence:
        st.markdown("### 🗂️ Grounded Evidence Ledger")
        st.caption("Every factual assertion in the synthesized report maps directly to one of these immutable evidence records.")
        
        if not ledger:
            st.write("No external evidence was logged for this session.")
        else:
            for item in ledger:
                src_id = item.get("source_id", "SRC")
                src_type = item.get("source_type", "GENERAL").upper()
                title = item.get("title", "Untitled Evidence")
                url = item.get("url_or_ref", "#")
                snippet = item.get("retrieved_text", "")
                ts = item.get("timestamp", "")

                with st.expander(f"📌 [{src_id}] {src_type}: {title}"):
                    st.markdown(f"**Source Origin:** `{item.get('tool_name')}`")
                    st.markdown(f"**Reference / Link:** [{url}]({url})")
                    st.markdown(f"**Retrieved Timestamp:** `{ts}`")
                    st.markdown("**Verbatim Excerpt / Structured Metrics:**")
                    st.code(snippet, language="text")

    with tab_activity:
        st.markdown("### 📋 Tool Execution Sequence")
        st.caption("High-level observable tool actions without private chain-of-thought.")
        for step in activity:
            st.markdown(f"- {step}")

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    "Financial Research Agent • Built with Python, LangGraph, yfinance, FAISS, and Streamlit • "
    "Placement Portfolio Project"
)
