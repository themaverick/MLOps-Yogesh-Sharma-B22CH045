import numpy as np
import logging
from typing import List, Optional
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DriftDetector:
    """
    Monitors the similarity between incoming search queries and the document corpus.
    Detects if the user search intent is 'drifting' away from the indexed data distribution.
    """
    def __init__(self, 
                 model: SentenceTransformer, 
                 corpus_embeddings: np.ndarray, 
                 window_size: int = 50,
                 threshold: float = 0.3):
        self.model = model
        # Calculate the centroid (average vector) of the indexed corpus
        self.baseline_centroid = np.mean(corpus_embeddings, axis=0)
        self.window_size = window_size
        self.threshold = threshold
        self.query_buffer: List[np.ndarray] = []
        
        logger.info("DriftDetector initialized with corpus baseline centroid.")

    def add_query(self, query_text: str):
        """
        Encode a query and add it to the monitoring buffer.
        """
        query_embedding = self.model.encode([query_text])[0]
        self.query_buffer.append(query_embedding)
        
        # Maintain window size
        if len(self.query_buffer) > self.window_size:
            self.query_buffer.pop(0)

    def calculate_drift_score(self) -> float:
        """
        Calculate the cosine distance between the corpus centroid and the current query window centroid.
        A higher score means more drift.
        """
        if len(self.query_buffer) < 5: # Need a minimum number of queries for a stable estimate
            return 0.0
            
        current_centroid = np.mean(self.query_buffer, axis=0)
        # Cosine distance = 1 - Cosine Similarity
        # Range: [0, 2], where 0 is identical direction
        drift_score = cosine(self.baseline_centroid, current_centroid)
        return float(drift_score)

    def check_drift(self) -> dict:
        """
        Check if the current drift score exceeds the threshold.
        """
        score = self.calculate_drift_score()
        is_drifting = score > self.threshold
        
        status = {
            "drift_score": score,
            "threshold": self.threshold,
            "is_drifting": is_drifting,
            "buffer_size": len(self.query_buffer)
        }
        
        if is_drifting:
            logger.warning(f"⚠️ EMBEDDING DRIFT DETECTED: Score {score:.4f} exceeds threshold {self.threshold}")
        
        return status

if __name__ == "__main__":
    # Simple test simulation
    # In a real scenario, we'd load embeddings from FAISS
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Simulate a corpus about 'Python Programming'
    corpus_texts = ["python code", "learn python", "dictionary in python", "python list", "flask web development"]
    corpus_embeddings = model.encode(corpus_texts)
    
    detector = DriftDetector(model, corpus_embeddings, window_size=10, threshold=0.15)
    
    # 1. Simulate queries consistent with the corpus
    logger.info("Simulating consistent queries...")
    for q in ["write python", "python variable", "python loops"]:
        detector.add_query(q)
    print(f"Initial Status: {detector.check_drift()}")
    
    # 2. Simulate drift (queries about unrelated topics)
    logger.info("Simulating drifted queries...")
    for q in ["cooking recipe", "gardening tips", "yoga poses", "best football player", "how to bake cake"]:
        detector.add_query(q)
    print(f"Drifted Status: {detector.check_drift()}")
