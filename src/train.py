"""Training script: load data, train with Trainer API, save and push to Hugging Face."""
import json
import os
import argparse
from dotenv import load_dotenv
from huggingface_hub import login, HfApi
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from .data import load_all_genre_reviews, train_test_split
from .utils import (
    MODEL_NAME,
    DEVICE,
    MAX_LENGTH,
    CACHED_MODEL_DIR,
    HF_REPO_ID,
    get_label_mappings,
    compute_metrics,
    ReviewDataset,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", type=int, default=5000, help="Max reviews per genre to fetch")
    parser.add_argument("--sample", type=int, default=1000, help="Sample size per genre")
    parser.add_argument("--train-per-genre", type=int, default=800)
    parser.add_argument("--test-per-genre", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--push-to-hub", action="store_true", help="Push model to HF after training")
    parser.add_argument("--output-dir", default="./results")
    args = parser.parse_args()

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("Logged in to Hugging Face Hub")

    # Data
    genre_reviews = load_all_genre_reviews(head=args.head, sample_size=args.sample)
    train_texts, train_labels, test_texts, test_labels = train_test_split(
        genre_reviews,
        train_per_genre=args.train_per_genre,
        test_per_genre=args.test_per_genre,
    )
    unique_labels = sorted(set(train_labels))
    label2id, id2label = get_label_mappings(unique_labels)
    num_labels = len(unique_labels)

    # Tokenizer and encodings
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    train_enc = tokenizer(
        train_texts, truncation=True, padding=True, max_length=MAX_LENGTH
    )
    test_enc = tokenizer(
        test_texts, truncation=True, padding=True, max_length=MAX_LENGTH
    )
    train_labels_enc = [label2id[y] for y in train_labels]
    test_labels_enc = [label2id[y] for y in test_labels]
    train_dataset = ReviewDataset(train_enc, train_labels_enc)
    test_dataset = ReviewDataset(test_enc, test_labels_enc)

    # Model
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels
    ).to(DEVICE)
    model.config.id2label = id2label
    model.config.label2id = label2id

    # Training (CPU-optimized)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(CACHED_MODEL_DIR)
    tokenizer.save_pretrained(CACHED_MODEL_DIR)

    # Save training config (Task 7: push model, tokenizer, and training config)
    training_config_path = os.path.join(CACHED_MODEL_DIR, "training_config.json")
    with open(training_config_path, "w") as f:
        json.dump(
            {
                "num_train_epochs": args.epochs,
                "per_device_train_batch_size": args.batch_size,
                "learning_rate": 5e-5,
                "eval_steps": args.eval_steps,
            },
            f,
            indent=2,
        )

    if args.push_to_hub and hf_token:
        tokenizer.push_to_hub(HF_REPO_ID)
        model.push_to_hub(HF_REPO_ID)
        api = HfApi()
        api.upload_file(
            path_or_fileobj=training_config_path,
            path_in_repo="training_config.json",
            repo_id=HF_REPO_ID,
        )
        print(f"Pushed to https://huggingface.co/{HF_REPO_ID}")

    # Log final eval
    metrics = trainer.evaluate()
    print("Training complete. Eval metrics:", metrics)
    return metrics


if __name__ == "__main__":
    main()
