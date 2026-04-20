import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from src.data.schema import ResearchPaper, SearchResult
from src.search.vector_search import VectorSearchEngine

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridSearchEngine:
    def __init__(self, 
                 vector_engine: VectorSearchEngine,
                 reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.vector_engine = vector_engine
        self.reranker = CrossEncoder(reranker_model)
        self.bm25 = None
        self.papers = []
        self.tokenized_corpus = []

    def load_corpus(self):
        """
        Load papers and prepare BM25 index.
        """
        if not self.vector_engine.papers:
            self.vector_engine.load_papers()
        
        self.papers = self.vector_engine.papers
        # Tokenize abstracts for BM25
        self.tokenized_corpus = [p.abstract.lower().split() for p in self.papers]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info(f"BM25 index built on {len(self.papers)} papers.")

    def keyword_search(self, query: str, top_k: int = 50) -> List[SearchResult]:
        """
        Perform BM25 keyword search.
        """
        if self.bm25 is None:
            self.load_corpus()

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        max_score = max(scores) if any(scores) else 1
        for idx in top_indices:
            if scores[idx] > 0:
                paper = self.papers[idx]
                results.append(SearchResult(
                    **paper.model_dump(),
                    score=float(scores[idx] / max_score), # Normalized score
                    keyword_score=float(scores[idx] / max_score),
                    retrieval_method="keyword"
                ))
        return results

    def hybrid_search(self, query: str, top_k: int = 10, rrf_k: int = 60) -> List[SearchResult]:
        """
        Combine Semantic and Keyword search using Reciprocal Rank Fusion (RRF).
        """
        vector_results = self.vector_engine.search(query, top_k=50)
        keyword_results = self.keyword_search(query, top_k=50)

        # RRF Scoring
        rrf_scores = {} # paper_id -> score
        paper_map = {}

        for rank, res in enumerate(vector_results):
            rrf_scores[res.id] = rrf_scores.get(res.id, 0) + 1.0 / (rrf_k + rank + 1)
            paper_map[res.id] = res

        for rank, res in enumerate(keyword_results):
            rrf_scores[res.id] = rrf_scores.get(res.id, 0) + 1.0 / (rrf_k + rank + 1)
            paper_map[res.id] = res

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        hybrid_results = []
        for pid, score in sorted_ids:
            res = paper_map[pid]
            # Create a model dump to ensure we don't accidentally share state
            dump = res.model_dump()
            # Remove keys we want to override to avoid multiple values error
            dump.pop('score', None)
            dump.pop('retrieval_method', None)
            dump.pop('rrf_score', None)
            
            # Ensure semantic and keyword scores are preserved if they were already set
            hybrid_res = SearchResult(
                **dump,
                score=score,
                rrf_score=score,
                retrieval_method="hybrid"
            )
            hybrid_results.append(hybrid_res)
            
        return hybrid_results

    def rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """
        Rerank results using a Cross-Encoder.
        """
        if not results:
            return []

        logger.info(f"Reranking {len(results)} results...")
        pairs = [[query, f"{r.title}. {r.abstract}"] for r in results]
        rerank_scores = self.reranker.predict(pairs)

        for res, score in zip(results, rerank_scores):
            res.score = float(score)
            res.rerank_score = float(score)
            res.retrieval_method += "+rerank"

        # Sort by rerank score
        results.sort(key=lambda x: x.score, reverse=True)
        return results

if __name__ == "__main__":
    v_engine = VectorSearchEngine()
    v_engine.load_index()
    
    h_engine = HybridSearchEngine(v_engine)
    h_engine.load_corpus()
    
    test_query = "large language models for scientific search"
    logger.info(f"Testing Hybrid search + Reranking for: '{test_query}'")
    
    hybrid_results = h_engine.hybrid_search(test_query, top_k=10)
    reranked_results = h_engine.rerank(test_query, hybrid_results)
    
    for r in reranked_results[:5]:
        logger.info(f"[{r.score:.4f}] {r.title} ({r.retrieval_method})")
