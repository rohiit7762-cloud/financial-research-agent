"""Phase 1 Setup Verification Script.

Tests that all core libraries can be imported and their versions logged.
"""

import sys
import importlib

REQUIRED_MODULES = [
    ("langchain", "LangChain Core Abstractions"),
    ("langgraph", "LangGraph Cyclic State Machine"),
    ("langchain_google_genai", "Google Gemini Tool-Calling Adapter"),
    ("langchain_openai", "OpenAI Tool-Calling Adapter"),
    ("yfinance", "Yahoo Finance Market Data API"),
    ("duckduckgo_search", "DuckDuckGo Financial News Search"),
    ("sentence_transformers", "Dense Sentence Transformer Embeddings"),
    ("faiss", "FAISS C++ Vector Similarity Index"),
    ("pydantic", "Pydantic v2 Data Validation"),
    ("pandas", "Pandas DataFrames"),
    ("streamlit", "Streamlit Presentation Framework"),
]

def test_imports():
    print(f"Testing environment on Python {sys.version.split()[0]}...\n")
    all_passed = True
    for module_name, desc in REQUIRED_MODULES:
        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, "__version__", "unknown")
            print(f"  [OK] {module_name:<25} v{version:<10} ({desc})")
        except Exception as e:
            print(f"  [FAIL] {module_name:<25} ERROR: {e}")
            all_passed = False
            
    assert all_passed, "One or more critical dependencies failed to import."
    print("\nPhase 1 Environment Setup: 100% HEALTHY & VERIFIED.")

if __name__ == "__main__":
    test_imports()
