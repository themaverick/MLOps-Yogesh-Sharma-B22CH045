import json
import logging
import numpy as np
import faiss
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from src.data.schema import ResearchPaper, SearchResult

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorSearchEngine:
    def __init__(self, 
                 model_name: str = "finetuned_model_fixed", 
                 processed_data_path: str = "data/processed/papers_cleaned.jsonl",
                 index_path: str = "data/processed/faiss_index.bin"):
        self.processed_data_path = Path(processed_data_path)
        self.index_path = Path(index_path)
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.papers: List[ResearchPaper] = []
        self.id_map: Dict[int, str] = {} # Map index ID to paper ID

    def load_papers(self):
        """
        Load processed papers from JSONL.
        """
        if not self.processed_data_path.exists():
            logger.error(f"Processed data {self.processed_data_path} not found.")
            return

        self.papers = []
        with open(self.processed_data_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    self.papers.append(ResearchPaper(**data))
                except Exception as e:
                    logger.error(f"Error loading paper: {e}")
        logger.info(f"Loaded {len(self.papers)} papers.")

    def build_index(self):
        """
        Generate embeddings and build FAISS index.
        """
        if not self.papers:
            self.load_papers()

        if not self.papers:
            logger.error("No papers to index.")
            return

        logger.info("Generating embeddings for abstracts...")
        abstracts = [p.abstract for p in self.papers]
        # Use titles + abstracts for better semantic context
        texts = [f"{p.title}. {p.abstract}" for p in self.papers]
        
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        # Normalize for cosine similarity (FlatL2 on normalized is equivalent to Cosine)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        # Save index
        faiss.write_index(self.index, str(self.index_path))
        logger.info(f"FAISS index built and saved to {self.index_path}")

    def load_index(self):
        """
        Load FAISS index from disk.
        """
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            if not self.papers:
                self.load_papers()
            logger.info("FAISS index loaded.")
        else:
            logger.info("Index not found. Building index...")
            self.build_index()

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Perform semantic search.
        """
        if self.index is None:
            self.load_index()

        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.papers):
                paper = self.papers[idx]
                results.append(SearchResult(
                    **paper.model_dump(),
                    score=float(1 - score/2), # Heuristic to map L2 distance to [0,1] similarity
                    semantic_score=float(1 - score/2),
                    retrieval_method="semantic"
                ))
        return results

if __name__ == "__main__":
    engine = VectorSearchEngine()
    engine.build_index()
    
    # Test search
    test_query = "problems with embedding drift in semantic search"
    logger.info(f"Testing search for: '{test_query}'")
    results = engine.search(test_query, top_k=3)
    for r in results:
        logger.info(f"[{r.score:.4f}] {r.title}")
