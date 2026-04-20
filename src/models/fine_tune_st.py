import json
import os
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

def run_fine_tuning(train_path="train_pairs.json", val_path="val_pairs.json", model_name="all-MiniLM-L6-v2", output_path="fine_tuned_model"):
    # 1. Load Data
    with open(train_path, "r") as f:
        train_raw = json.load(f)
    with open(val_path, "r") as f:
        val_raw = json.load(f)

    # 2. Convert to InputExamples
    train_examples = [InputExample(texts=item["set"]) for item in train_raw]
    val_examples = [InputExample(texts=item["set"]) for item in val_raw]

    # 3. Initialize Model
    model = SentenceTransformer(model_name)

    # 4. Define Loss
    # MultipleNegativesRankingLoss is great for (Query, Positive) pairs
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # 5. Define Evaluator
    # We can use InformationRetrievalEvaluator if we have queries and corpus, 
    # but for simplicity here we use the loss on the validation set.
    # Actually, let's use a simpler evaluator for Colab visibility
    val_loader = DataLoader(val_examples, batch_size=16)
    
    # 6. Training Arguments
    args = SentenceTransformerTrainingArguments(
        output_dir=output_path,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        warmup_steps=100,
        fp16=True, # Use mixed precision for speed on Colab (GPU)
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        logging_steps=10,
        load_best_model_at_end=True,
    )

    # 7. Initialize Trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_examples,
        eval_dataset=val_examples,
        loss=train_loss,
    )

    # 8. Train
    print("Starting training...")
    trainer.train()

    # 9. Plot Results
    # SentenceTransformerTrainer saves logs in trainer.state.log_history
    history = trainer.state.log_history
    
    train_loss_history = [log["loss"] for log in history if "loss" in log]
    eval_loss_history = [log["eval_loss"] for log in history if "eval_loss" in log]
    steps = [log["step"] for log in history if "loss" in log]
    eval_steps = [log["step"] for log in history if "eval_loss" in log]

    plt.figure(figsize=(10, 5))
    plt.plot(steps, train_loss_history, label="Train Loss")
    plt.plot(eval_steps, eval_loss_history, label="Eval Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Fine-tuning Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_plot.png")
    plt.show()
    print("Training complete. Loss plot saved as loss_plot.png")

    # 10. Save final model
    model.save(output_path)
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    # In Colab, you would first upload train_pairs.json and val_pairs.json
    # and run: !pip install sentence-transformers
    if os.path.exists("train_pairs.json"):
        run_fine_tuning()
    else:
        print("Please ensure train_pairs.json and val_pairs.json are in the current directory.")
