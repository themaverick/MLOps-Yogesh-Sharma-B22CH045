import json
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_training_data(input_path: str = "data/processed/papers_cleaned.jsonl", 
                          output_dir: str = "data/training"):
    """
    Prepare (Title, Abstract) pairs for fine-tuning.
    Splits into train and validation sets.
    """
    input_file = Path(input_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        logger.error(f"Input file {input_file} not found.")
        return

    data = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                paper = json.loads(line)
                # We use (Title, Abstract) as a positive pair
                data.append({
                    "set": [paper["title"], paper["abstract"]]
                })
            except Exception as e:
                logger.error(f"Error parsing line: {e}")

    logger.info(f"Loaded {len(data)} pairs.")

    # Split data (Increasing validation set to 20% for better stability)
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

    # Save to JSON
    with open(output_path / "train_pairs.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2)
    
    with open(output_path / "val_pairs.json", "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2)

    logger.info(f"Saved {len(train_data)} training and {len(val_data)} validation pairs to {output_path}")

if __name__ == "__main__":
    prepare_training_data()
