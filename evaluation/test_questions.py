"""Evaluation Benchmark Suite for Financial Research Agent.

Defines the 10 canonical placement evaluation test cases:
1. "What is Apple's current stock price?" (Expected: Market Data only)
2. "Why has Apple moved recently?" (Expected: Market Data + News)
3. "What risks did Apple mention in its latest 10-K?" (Expected: Filing RAG)
4. "Compare Apple and Microsoft fundamentals." (Expected: Fundamentals for both)
5. "Analyze NVIDIA's recent outlook." (Expected: Agent determines tools)
6. "Analyze NVIDIA considering stock performance, fundamentals, recent news and filing risks." (Expected: 4 tools)
7. "What is the stock price of $XYZINVALID999?" (Expected: Graceful error containment)
8. "What was NVIDIA's revenue in the year 1985?" (Expected: Insufficient data notice)
9. Simulated API failure (Expected: Graceful degradation)
10. "Tell me about tech stocks." (Ambiguous query -> Stated assumptions)
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from agent.research_agent import run_research, get_llm
from agent.state import ResearchAgentState


class EvaluationTestCase(BaseModel):
    """Specification of a benchmark research test case."""
    test_id: str
    query: str
    ticker: Optional[str] = None
    expected_tools: List[str]
    expected_evidence_types: List[str]
    evaluation_criteria: str


BENCHMARK_TEST_SUITE: List[EvaluationTestCase] = [
    EvaluationTestCase(
        test_id="TEST-01",
        query="What is Apple's current stock price?",
        ticker="AAPL",
        expected_tools=["get_market_data"],
        expected_evidence_types=["market_data"],
        evaluation_criteria="Agent should invoke only market data without fetching balance sheets or news."
    ),
    EvaluationTestCase(
        test_id="TEST-02",
        query="Why has Apple moved recently?",
        ticker="AAPL",
        expected_tools=["get_market_data", "search_financial_news"],
        expected_evidence_types=["market_data", "news"],
        evaluation_criteria="Agent recognizes that price movement requires both quantitative quote and recent news catalysts."
    ),
    EvaluationTestCase(
        test_id="TEST-03",
        query="What risks did Apple mention in its latest 10-K filing?",
        ticker="AAPL",
        expected_tools=["search_filing_rag"],
        expected_evidence_types=["filing_rag"],
        evaluation_criteria="Agent routes directly to filing RAG to extract Item 1A risk disclosures."
    ),
    EvaluationTestCase(
        test_id="TEST-04",
        query="Compare Apple and Microsoft fundamentals including P/E and revenue.",
        ticker=None,
        expected_tools=["get_company_fundamentals"],
        expected_evidence_types=["fundamentals"],
        evaluation_criteria="Agent must invoke fundamentals tool for BOTH AAPL and MSFT."
    ),
    EvaluationTestCase(
        test_id="TEST-05",
        query="Analyze NVIDIA's recent outlook and catalysts.",
        ticker="NVDA",
        expected_tools=["get_market_data", "search_financial_news"],
        expected_evidence_types=["market_data", "news"],
        evaluation_criteria="Agent autonomously selects relevant market and sentiment tools."
    ),
    EvaluationTestCase(
        test_id="TEST-06",
        query="Analyze NVIDIA considering stock performance, fundamentals, recent news and filing risks.",
        ticker="NVDA",
        expected_tools=["get_market_data", "get_company_fundamentals", "search_financial_news", "search_filing_rag"],
        expected_evidence_types=["market_data", "fundamentals", "news", "filing_rag"],
        evaluation_criteria="Full multi-source synthesis exercising all four specialized tools."
    ),
    EvaluationTestCase(
        test_id="TEST-07",
        query="What is the current stock price and P/E ratio for $XYZINVALID999?",
        ticker="XYZINVALID999",
        expected_tools=["get_market_data"],
        expected_evidence_types=[],
        evaluation_criteria="Agent catches invalid ticker gracefully without unhandled exceptions."
    ),
    EvaluationTestCase(
        test_id="TEST-08",
        query="What was NVIDIA's revenue and operating margin in the year 1985?",
        ticker="NVDA",
        expected_tools=["get_company_fundamentals"],
        expected_evidence_types=[],
        evaluation_criteria="Agent clearly states that historical 1985 data is unavailable rather than hallucinating figures."
    ),
    EvaluationTestCase(
        test_id="TEST-09",
        query="Check market data for TSLA assuming network degradation.",
        ticker="TSLA",
        expected_tools=["get_market_data"],
        expected_evidence_types=["market_data"],
        evaluation_criteria="Agent handles API degradation gracefully."
    ),
    EvaluationTestCase(
        test_id="TEST-10",
        query="Tell me about tech stocks right now.",
        ticker=None,
        expected_tools=["search_financial_news"],
        expected_evidence_types=["news"],
        evaluation_criteria="Handles open-ended ambiguous query by making a clearly stated assumption (e.g. focusing on major tech leaders)."
    )
]


def run_evaluation_benchmark(llm=None, max_tests: int = 10) -> List[Dict[str, Any]]:
    """Executes benchmark test cases and logs tool selection, evidence, and report quality."""
    results: List[Dict[str, Any]] = []

    print(f"\n{'='*70}\nSTARTING BENCHMARK EVALUATION ({min(len(BENCHMARK_TEST_SUITE), max_tests)} TESTS)\n{'='*70}")

    for idx, test in enumerate(BENCHMARK_TEST_SUITE[:max_tests], start=1):
        print(f"\nRunning [{test.test_id}] \"{test.query}\"...")
        try:
            state: ResearchAgentState = run_research(
                query=test.query,
                ticker=test.ticker,
                llm=llm,
                max_iterations=5
            )

            actual_tools = list({item.get("tool_name") for item in state.get("evidence_ledger", [])})
            actual_sources = [item.get("source_id") for item in state.get("evidence_ledger", [])]
            report = state.get("final_report", "")

            # Check if expected tools were called
            tool_match = all(tool in actual_tools for tool in test.expected_tools) if test.expected_tools else True

            results.append({
                "test_id": test.test_id,
                "query": test.query,
                "expected_tools": test.expected_tools,
                "actual_tools": actual_tools,
                "tool_match": tool_match,
                "evidence_count": len(actual_sources),
                "confidence_rating": state.get("confidence_rating"),
                "status": "PASS" if state.get("final_report") else "FAIL"
            })
            print(f"  Result: {results[-1]['status']} | Tools: {actual_tools} | Confidence: {state.get('confidence_rating')}")

        except Exception as e:
            results.append({
                "test_id": test.test_id,
                "query": test.query,
                "status": "ERROR",
                "error": str(e)
            })
            print(f"  Result: ERROR -> {e}")

    return results


if __name__ == "__main__":
    print(f"Loaded {len(BENCHMARK_TEST_SUITE)} canonical benchmark test cases.")
    for t in BENCHMARK_TEST_SUITE:
        print(f"[{t.test_id}] {t.query} -> Expected: {t.expected_tools}")
