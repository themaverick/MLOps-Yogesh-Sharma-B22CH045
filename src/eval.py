"""Evaluation script: load model (local or from HF), run evaluation, save results."""
import json
import os
import argparse
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from .data import load_all_genre_reviews, train_test_split
from .utils import (
    MAX_LENGTH,
    HF_REPO_ID,
    get_label_mappings,
    compute_metrics,
    ReviewDataset,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default=None,
        help="Local path or HF repo id (e.g. themaverick1/goodreads-genre-classifier)",
    )
    parser.add_argument(
        "--output",
        default="evaluation_results.json",
        help="Path to save evaluation JSON",
    )
    parser.add_argument("--head", type=int, default=5000)
    parser.add_argument("--sample", type=int, default=1000)
    parser.add_argument("--train-per-genre", type=int, default=800)
    parser.add_argument("--test-per-genre", type=int, default=200)
    parser.add_argument("--max-test-samples", type=int, default=None, help="Cap test set size for quick eval (e.g. in Docker)")
    args = parser.parse_args()

    model_path = args.model_path or HF_REPO_ID
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if token:
        login(token=token)

    # Load test data (same split as training for comparison)
    genre_reviews = load_all_genre_reviews(head=args.head, sample_size=args.sample)
    _, _, test_texts, test_labels = train_test_split(
        genre_reviews,
        train_per_genre=args.train_per_genre,
        test_per_genre=args.test_per_genre,
    )
    unique_labels = sorted(set(test_labels))
    label2id, id2label = get_label_mappings(unique_labels)
    if args.max_test_samples is not None:
        n = min(args.max_test_samples, len(test_texts))
        test_texts, test_labels = test_texts[:n], test_labels[:n]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    test_enc = tokenizer(
        test_texts, truncation=True, padding=True, max_length=MAX_LENGTH
    )
    test_labels_enc = [label2id[y] for y in test_labels]
    test_dataset = ReviewDataset(test_enc, test_labels_enc)

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    training_args = TrainingArguments(
        output_dir="./eval_tmp",
        per_device_eval_batch_size=8,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    results = trainer.evaluate()
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print("Evaluation results:", results)
    print(f"Saved to {args.output}")
    return results


if __name__ == "__main__":
    main()
