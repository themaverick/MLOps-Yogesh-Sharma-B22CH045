import numpy as np
import logging
from typing import List, Dict, Any
from src.data.schema import SearchResult

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SearchEvaluator:
    """
    Utility to calculate search metrics like Precision@k and Recall@k.
    """
    @staticmethod
    def calculate_precision_at_k(actual_ids: List[str], predicted_results: List[SearchResult], k: int) -> float:
        """
        Calculate Precision@K.
        """
        if not actual_ids or not predicted_results:
            return 0.0
            
        top_k_preds = [res.id for res in predicted_results[:k]]
        relevant_preds = [pid for pid in top_k_preds if pid in actual_ids]
        
        return len(relevant_preds) / k

    @staticmethod
    def calculate_ndcg(actual_ids: List[str], predicted_results: List[SearchResult], k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at K.
        Assumes binary relevance (1 if in actual_ids, 0 otherwise).
        """
        if not actual_ids or not predicted_results:
            return 0.0
            
        dcg = 0.0
        for i, res in enumerate(predicted_results[:k]):
            rel = 1 if res.id in actual_ids else 0
            dcg += rel / np.log2(i + 2)
            
        # Ideal DCG: all actual items are at the top
        idcg = 0.0
        for i in range(min(len(actual_ids), k)):
            idcg += 1 / np.log2(i + 2)
            
        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def run_benchmark(engine: Any, benchmark_data: List[Dict[str, Any]], k: int = 5):
        """
        Run search benchmark across a set of labeled queries.
        benchmark_data format: [{"query": "...", "expected_ids": ["id1", "id2"]}]
        """
        logger.info(f"Running benchmark on {len(benchmark_data)} queries...")
        
        metrics = {
            "p_at_k": [],
            "ndcg": [],
            "latency": []
        }
        
        import time
        for item in benchmark_data:
            start_time = time.time()
            results = engine.search(item["query"], top_k=k)
            latency = time.time() - start_time
            
            p_k = SearchEvaluator.calculate_precision_at_k(item["expected_ids"], results, k)
            ndcg_k = SearchEvaluator.calculate_ndcg(item["expected_ids"], results, k)
            
            metrics["p_at_k"].append(p_k)
            metrics["ndcg"].append(ndcg_k)
            metrics["latency"].append(latency)
            
        avg_metrics = {
            "avg_precision_at_k": np.mean(metrics["p_at_k"]),
            "avg_ndcg": np.mean(metrics["ndcg"]),
            "avg_latency_ms": np.mean(metrics["latency"]) * 1000
        }
        
        logger.info(f"Benchmark Results: {avg_metrics}")
        return avg_metrics

if __name__ == "__main__":
    # Mock test
    class MockResult:
        def __init__(self, id): self.id = id
        
    actual = ["paper1", "paper2"]
    predicted = [MockResult("paper1"), MockResult("paper3"), MockResult("paper2")]
    
    evaluator = SearchEvaluator()
    print(f"P@3: {evaluator.calculate_precision_at_k(actual, predicted, 3)}") # Should be 2/3 = 0.66
    print(f"NDCG@3: {evaluator.calculate_ndcg(actual, predicted, 3)}")
