import json
import logging
import re
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
from src.data.schema import ResearchPaper

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, input_path: str = "data/raw/papers.jsonl", output_path: str = "data/processed/papers_cleaned.jsonl"):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def clean_text(self, text: str) -> str:
        """
        Perform basic text cleaning: lowercasing, removing extra whitespace, and special characters.
        """
        # Remove LaTeX-style commands if any
        text = re.sub(r'\$.*?\$', '', text)
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def process(self):
        """
        Read raw JSONL, clean fields, and save to processed JSONL.
        """
        if not self.input_path.exists():
            logger.error(f"Input file {self.input_path} does not exist.")
            return

        processed_count = 0
        with open(self.input_path, "r", encoding="utf-8") as f_in, \
             open(self.output_path, "w", encoding="utf-8") as f_out:
            
            for line in tqdm(f_in, desc="Preprocessing papers"):
                try:
                    data = json.loads(line)
                    # Use Pydantic to ensure the data is still valid after parsing
                    paper = ResearchPaper(**data)
                    
                    # Clean title and abstract
                    paper.title = self.clean_text(paper.title)
                    paper.abstract = self.clean_text(paper.abstract)
                    
                    f_out.write(paper.model_dump_json() + "\n")
                    processed_count += 1
                except Exception as e:
                    logger.error(f"Error processing line: {e}")

        logger.info(f"Preprocessing complete. Processed {processed_count} papers. Saved to {self.output_path}")

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.process()
