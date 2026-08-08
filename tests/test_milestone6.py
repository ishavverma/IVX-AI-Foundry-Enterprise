"""
AFE — Milestone 6 Verification Script

Tests Evaluation benchmark runner (MMLU, HumanEval, GSM8K),
Registry asset management, and OpenAI-compatible Inference Server.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add root directory to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from core.database.engine import init_db, get_session
from core.database.models.evaluation import EvaluationRun, RegistryAsset, InferenceEndpoint
from engines.evaluation.evaluator import BenchmarkEvaluator
from engines.inference.server import InferenceServer


def run_tests():
    print("=======================================================")
    print("   AFE MILESTONE 6: EVALUATION, REGISTRY & SERVING    ")
    print("=======================================================")

    init_db()

    # ── 1. Test Benchmark Evaluator ──
    print("\n1. Testing Benchmark Evaluator:")
    eval_id = "test-eval-mmlu-01"
    with get_session() as session:
        old = session.get(EvaluationRun, eval_id)
        if old:
            session.delete(old)
            session.flush()
        ev = EvaluationRun(id=eval_id, benchmark_name="MMLU", status="running")
        session.add(ev)

    res_mmlu = BenchmarkEvaluator.run_benchmark(eval_id, "mmlu")
    print(f"   MMLU Score: {res_mmlu.overall_score}% across categories: {list(res_mmlu.category_scores.keys())}")
    assert res_mmlu.overall_score > 50.0

    eval_he_id = "test-eval-he-01"
    with get_session() as session:
        old_he = session.get(EvaluationRun, eval_he_id)
        if old_he:
            session.delete(old_he)
            session.flush()
        ev_he = EvaluationRun(id=eval_he_id, benchmark_name="HumanEval", status="running")
        session.add(ev_he)

    res_he = BenchmarkEvaluator.run_benchmark(eval_he_id, "humaneval")
    print(f"   HumanEval Pass@1: {res_he.pass_at_1 * 100:.1f}%")
    assert res_he.pass_at_1 > 0.40

    # ── 2. Test Inference Server (Streaming & Complete) ──
    print("\n2. Testing Inference Server:")
    comp_res = InferenceServer.generate_complete(
        messages=[{"role": "user", "content": "Explain hybrid linear scaling."}],
        max_tokens=64,
    )
    print(f"   OpenAI Format Response: {comp_res['choices'][0]['message']['content'][:80]}...")
    assert "choices" in comp_res
    assert comp_res["object"] == "chat.completion"

    chunks = list(InferenceServer.generate_stream(prompt="Hello", max_tokens=10))
    print(f"   Streamed {len(chunks)} chunks successfully.")
    assert len(chunks) > 5

    # ── 3. Test Registry Asset ──
    print("\n3. Testing Registry Asset Catalog:")
    with get_session() as session:
        reg_count = session.query(RegistryAsset).count()
        print(f"   Total Registered Assets: {reg_count}")

    print("\n=======================================================")
    print("         ALL MILESTONE 6 UNIT TESTS PASSED!          ")
    print("=======================================================")


if __name__ == "__main__":
    run_tests()
