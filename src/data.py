"""Data loading and train/test split for Goodreads genre classification."""
import gzip
import json
import random
import requests
from .utils import GENRE_URL_DICT


def load_reviews(url, head=10000, sample_size=2000):
    """Stream reviews from URL and return a random sample."""
    reviews = []
    count = 0
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with gzip.open(response.raw, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                reviews.append(d["review_text"])
            except (KeyError, json.JSONDecodeError):
                continue
            count += 1
            if head is not None and count >= head:
                break
    return random.sample(reviews, min(sample_size, len(reviews)))


def load_all_genre_reviews(genre_url_dict=None, head=10000, sample_size=2000):
    """Load and sample reviews for each genre. Returns dict genre -> list of review texts."""
    genre_url_dict = genre_url_dict or GENRE_URL_DICT
    out = {}
    for genre, url in genre_url_dict.items():
        print(f"Loading reviews for genre: {genre}")
        out[genre] = load_reviews(url, head=head, sample_size=sample_size)
    return out


def train_test_split(genre_reviews_dict, train_per_genre=800, test_per_genre=200, seed=42):
    """
    Split per genre into train/test. Each genre is sampled to train_per_genre + test_per_genre.
    Returns (train_texts, train_labels, test_texts, test_labels).
    """
    random.seed(seed)
    train_texts, train_labels = [], []
    test_texts, test_labels = [], []
    n_per_genre = train_per_genre + test_per_genre
    for genre, reviews in genre_reviews_dict.items():
        sampled = random.sample(reviews, min(n_per_genre, len(reviews)))
        for t in sampled[:train_per_genre]:
            train_texts.append(t)
            train_labels.append(genre)
        for t in sampled[train_per_genre : train_per_genre + test_per_genre]:
            test_texts.append(t)
            test_labels.append(genre)
    return train_texts, train_labels, test_texts, test_labels
