"""Constants and shared utilities for Goodreads genre classification."""
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Model and training constants (CPU-optimized)
MODEL_NAME = "distilbert-base-cased"
DEVICE = "cpu"
MAX_LENGTH = 512
CACHED_MODEL_DIR = "distilbert-reviews-genres"
HF_REPO_ID = "themaverick1/goodreads-genre-classifier"

# Genre URLs - UCSD Goodreads by genre
# Source: https://mengtingwan.github.io/data/goodreads.html#datasets
GENRE_URL_DICT = {
    "poetry": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz",
    "children": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz",
    "comics_graphic": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz",
    "fantasy_paranormal": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz",
    "history_biography": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz",
    "mystery_thriller_crime": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz",
    "romance": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz",
    "young_adult": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz",
}


def get_label_mappings(unique_labels):
    """Build label2id and id2label from sorted unique labels."""
    labels = sorted(unique_labels)
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label


def compute_metrics(pred):
    """Compute accuracy and F1 for Trainer evaluation."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    return {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec}


class ReviewDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for tokenized reviews and labels."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
