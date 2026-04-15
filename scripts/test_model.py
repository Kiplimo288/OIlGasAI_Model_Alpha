
"""
test_model.py
---------------
Quick smoke test to verify OilGasAI-Model-Alpha is loading and responding correctly.

Usage:
    python scripts/test_model.py
    python scripts/test_model.py --api   # test HF API backend
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from dotenv import load_dotenv

load_dotenv()

TEST_QUERIES = [
    "What are EPA Subpart W reporting requirements for centrifugal compressors?",
    "How do I apply drift correction to a low-cost methane sensor deployed for 30 days?",
    "What is the LDAR leak threshold under NSPS OOOOa for valves?",
    "Explain OGMP 2.0 Level 5 measurement-based quantification requirements.",
    "Who are you and what can you help me with?",
]


def run_tests(backend: dict):
    from inference.chat import chat

    logger.info(f"Running smoke tests (backend: {backend['mode']})")
    results = []

    for i, query in enumerate(TEST_QUERIES, 1):
        logger.info(f"\n[{i}/{len(TEST_QUERIES)}] Query: {query[:80]}...")
        try:
            response = chat(query, backend)
            logger.info(f"Response ({len(response)} chars): {response[:200]}...")
            results.append({"query": query, "status": "PASS", "response_len": len(response)})
        except Exception as e:
            logger.error(f"FAILED: {e}")
            results.append({"query": query, "status": "FAIL", "error": str(e)})

    passed = sum(1 for r in results if r["status"] == "PASS")
    logger.info(f"\n{'='*50}")
    logger.info(f"Results: {passed}/{len(TEST_QUERIES)} passed")
    for r in results:
        status = "✅" if r["status"] == "PASS" else "❌"
        logger.info(f"  {status} {r['query'][:60]}...")

    return passed == len(TEST_QUERIES)


def main():
    parser = argparse.ArgumentParser(description="OilGasAI Model Alpha smoke test")
    parser.add_argument("--api", action="store_true", help="Use HF Inference API backend")
    args = parser.parse_args()

    import os
    if args.api:
        os.environ["USE_HF_API"] = "true"

    from inference.load_model import get_inference_backend
    backend = get_inference_backend()

    success = run_tests(backend)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
