import arxiv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from src.data.schema import ResearchPaper

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ArxivIngestor:
    def __init__(self, raw_data_path: str = "data/raw/papers.jsonl"):
        self.raw_data_path = Path(raw_data_path)
        self.raw_data_path.parent.mkdir(parents=True, exist_ok=True)

    def fetch_papers(self, query: str, max_results: int = 100) -> List[ResearchPaper]:
        """
        Fetch papers from ArXiv and validate them against the ResearchPaper schema.
        """
        logger.info(f"Fetching {max_results} papers for query: '{query}'")
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )

        validated_papers = []
        for result in tqdm(search.results(), total=max_results, desc="Ingesting ArXiv"):
            try:
                paper = ResearchPaper(
                    id=result.entry_id.split('/')[-1],
                    title=result.title,
                    abstract=result.summary.replace('\n', ' '),
                    authors=[author.name for author in result.authors],
                    published_date=result.published.date(),
                    categories=result.categories,
                    doi=result.doi,
                    url=result.pdf_url
                )
                validated_papers.append(paper)
            except Exception as e:
                logger.error(f"Validation failed for paper {result.entry_id}: {e}")
        
        return validated_papers

    def save_to_jsonl(self, papers: List[ResearchPaper]):
        """
        Save validated papers to a JSONL file.
        """
        logger.info(f"Saving {len(papers)} papers to {self.raw_data_path}")
        with open(self.raw_data_path, "a", encoding="utf-8") as f:
            for paper in papers:
                # Use json.dumps with pydantic's json() method
                f.write(paper.model_dump_json() + "\n")

if __name__ == "__main__":
    # Expanded queries for better coverage
    queries = [
        "semantic search", "embedding drift", "vector database", 
        "large language models", "information retrieval", 
        "natural language processing", "graph neural networks",
        "reinforcement learning", "computer vision transformer",
        "generative ai", "multimodal learning"
    ]
    ingestor = ArxivIngestor()
    
    for q in queries:
        papers = ingestor.fetch_papers(query=q, max_results=200) # 200 per query = ~2,200 papers total
        ingestor.save_to_jsonl(papers)
    
    logger.info("Ingestion complete.")
