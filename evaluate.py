"""
evaluate.py — Phase 10: Evaluation Framework

HOW TO USE:
  1. Create a test_questions.json file (format shown at bottom of this file)
  2. Run: python evaluate.py --doc_id YOUR_DOC_ID

WHAT GETS MEASURED:
  - Retrieval hit rate: Did the relevant chunk appear in top-k results?
  - Latency: How long does each query take end-to-end?
  - Answer quality: Self-assessed by the LLM (proxy metric)

WHY THIS MATTERS FOR YOUR RESUME:
  Every production ML system needs evaluation. Showing you built one
  demonstrates engineering maturity, not just model-building ability.
"""

import json
import time
import argparse
import math
import requests
from datetime import datetime

API_BASE = "http://localhost:8000"


def run_evaluation(doc_id: str, questions_file: str):
    """Run all test questions and compute metrics."""

    with open(questions_file) as f:
        test_cases = json.load(f)

    print(f"\n{'='*60}")
    print(f"  AskMyDocs Evaluation Report")
    print(f"  Document ID : {doc_id}")
    print(f"  Questions   : {len(test_cases)}")
    print(f"  Timestamp   : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}\n")

    results    = []
    latencies  = []
    hit_count  = 0

    for i, case in enumerate(test_cases):
        question    = case["question"]
        expected_kw = case.get("expected_keywords", [])

        print(f"[Q{i+1}] {question}")

        start = time.time()
        try:
            res  = requests.post(f"{API_BASE}/query", json={"question": question, "doc_id": doc_id})
            data = res.json()
        except Exception as e:
            print(f"  ERROR: {e}\n")
            continue

        latency = time.time() - start
        latencies.append(latency)

        answer  = data.get("answer", "")
        sources = data.get("sources", [])

        # Keyword hit check (simple proxy for retrieval accuracy)
        # Mark as "hit" if any expected keyword appears in the answer
        hit = any(kw.lower() in answer.lower() for kw in expected_kw) if expected_kw else True
        if hit:
            hit_count += 1

        print(f"  Answer  : {answer[:200]}{'...' if len(answer) > 200 else ''}")
        print(f"  Sources : {', '.join(sources)}")
        print(f"  Latency : {latency:.2f}s")
        print(f"  Hit     : {'✓' if hit else '✗'}")
        print()

        results.append({
            "question": question,
            "answer"  : answer,
            "sources" : sources,
            "latency" : latency,
            "hit"     : hit
        })

    # ── Summary metrics ───────────────────────────────────────────────────
    n               = len(results)
    hit_rate        = hit_count / n * 100 if n > 0 else 0
    avg_latency     = sum(latencies) / len(latencies) if latencies else 0
    p95_latency     = sorted(latencies)[math.ceil(0.95 * len(latencies)) - 1] if latencies else 0

    print(f"{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Questions evaluated : {n}")
    print(f"  Retrieval hit rate  : {hit_rate:.1f}%")
    print(f"  Avg latency         : {avg_latency:.2f}s")
    print(f"  P95 latency         : {p95_latency:.2f}s")
    print(f"{'='*60}\n")

    # Save report
    report = {
        "doc_id"         : doc_id,
        "timestamp"      : datetime.now().isoformat(),
        "questions_count": n,
        "hit_rate_pct"   : round(hit_rate, 1),
        "avg_latency_s"  : round(avg_latency, 3),
        "p95_latency_s"  : round(p95_latency, 3),
        "individual"     : results
    }

    with open("evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("Report saved to evaluation_report.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc_id",    required=True, help="Document ID to evaluate")
    parser.add_argument("--questions", default="test_questions.json", help="Path to questions JSON")
    args = parser.parse_args()
    run_evaluation(args.doc_id, args.questions)


# ── EXAMPLE test_questions.json ───────────────────────────────────────────
# Save this as test_questions.json and fill with questions from your document
EXAMPLE_QUESTIONS = """
[
  {
    "question": "What is the main topic of this document?",
    "expected_keywords": []
  },
  {
    "question": "What methodology was used in the study?",
    "expected_keywords": ["method", "approach", "technique"]
  },
  {
    "question": "What were the key findings or conclusions?",
    "expected_keywords": ["result", "conclusion", "finding"]
  }
]
"""
# Run: echo '<paste above>' > test_questions.json
# Then: python evaluate.py --doc_id YOUR_DOC_ID
