"""
batch_inference.py
------------------
Run bulk compliance queries against OilGasAI-Model-Alpha.
Useful for processing multiple wells, compressors, or sensor sites at once.
"""

import json
import csv
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from .chat import chat


def run_batch(
    queries: list[str],
    backend: dict,
    output_path: str = "output/batch_results.jsonl",
    system_prompt: str | None = None,
) -> list[dict]:
    """
    Run a list of queries through OilGasAI Model Alpha and save results.

    Args:
        queries: List of query strings
        backend: Dict from get_inference_backend()
        output_path: Where to write JSONL results
        system_prompt: Optional custom system prompt

    Returns:
        List of {"query": ..., "response": ...} dicts
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    results = []

    logger.info(f"Running batch inference on {len(queries)} queries...")

    with open(output_path, "w") as f:
        for i, query in enumerate(tqdm(queries, desc="OilGasAI Model Alpha Batch")):
            try:
                kwargs = {"message": query, "backend": backend}
                if system_prompt:
                    kwargs["system_prompt"] = system_prompt

                response = chat(**kwargs)
                result = {"index": i, "query": query, "response": response, "status": "ok"}
            except Exception as e:
                logger.error(f"Query {i} failed: {e}")
                result = {"index": i, "query": query, "response": None, "status": "error", "error": str(e)}

            results.append(result)
            f.write(json.dumps(result) + "\n")

    logger.info(f"Batch complete. Results saved to {output_path}")
    return results


def batch_from_csv(
    csv_path: str,
    query_column: str,
    backend: dict,
    output_path: str = "output/batch_results.jsonl",
) -> list[dict]:
    """
    Load queries from a CSV file and run batch inference.

    Args:
        csv_path: Path to CSV file
        query_column: Column name containing queries
        backend: Dict from get_inference_backend()
        output_path: Where to write JSONL results

    Returns:
        List of result dicts
    """
    queries = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            queries.append(row[query_column])

    logger.info(f"Loaded {len(queries)} queries from {csv_path}")
    return run_batch(queries, backend, output_path)
