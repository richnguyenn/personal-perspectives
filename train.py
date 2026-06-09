"""Fine-tune a transformer model on PANDORA for Big Five personality prediction.

This script loads PANDORA data, trains a model (roberta-base, bert-base-uncased,
etc.) with MSE loss, and saves the model locally for use with main.py.
"""

import argparse
import os

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

import load_pandora
from personality_model import BIG_FIVE_TRAITS


class BigFiveDataset(Dataset):
    """Dataset for Big Five regression: text -> 5 float labels."""

    def __init__(self, samples, tokenizer, max_length=512):
        """Create dataset from list of sample dicts.

        Parameters
        ----------
        samples : list of dict
            Each dict has "text" and the five trait keys.
        tokenizer : PreTrainedTokenizer
            Tokenizer for the model.
        max_length : int
            Max token length per sample.
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample["text"]
        labels = [sample[trait] for trait in BIG_FIVE_TRAITS]

        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.float32),
        }


class RegressionTrainer(Trainer):
    """Trainer that uses MSE loss for regression instead of cross-entropy."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.MSELoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def main():
    """Run the training pipeline."""
    parser = argparse.ArgumentParser(
        description="Fine-tune a model on PANDORA for Big Five personality prediction"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="roberta-base",
        help="Hugging Face model name (e.g., roberta-base, bert-base-uncased)",
    )
    parser.add_argument(
        "--pandora_dir",
        type=str,
        default="data/pandora",
        help="Path to PANDORA data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/roberta-base-pandora",
        help="Directory to save the trained model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Training batch size",
    )
    parser.add_argument(
        "--max_comments_per_author",
        type=int,
        default=100,
        help="Max comments per author when aggregating (controls text length)",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Fraction of data for validation (0-1)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Max token length per sample",
    )
    parser.add_argument(
        "--use_cpu",
        action="store_true",
        help="Force CPU (skip GPU/MPS). Use if you hit out-of-memory errors.",
    )
    parser.add_argument(
        "--limit_samples",
        type=int,
        default=None,
        help="Limit number of training samples (for testing)",
    )
    args = parser.parse_args()

    # Load PANDORA data
    samples = load_pandora.load_pandora_training_data(
        pandora_dir=args.pandora_dir,
        max_comments_per_author=args.max_comments_per_author,
        min_comments_per_author=1,
        limit_samples=args.limit_samples,
    )

    if len(samples) == 0:
        print("No training samples. Exiting.")
        return

    # Train/val split (by sample, not by author - for simplicity; author-level split would need more logic)
    train_samples, val_samples = train_test_split(
        samples, test_size=args.val_split, random_state=42
    )
    print(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")

    # Load tokenizer and model
    print(f"Loading model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=5,
    )

    # Create datasets
    train_dataset = BigFiveDataset(train_samples, tokenizer, max_length=args.max_length)
    val_dataset = BigFiveDataset(val_samples, tokenizer, max_length=args.max_length)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        seed=42,
        use_cpu=args.use_cpu,
    )

    trainer = RegressionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    print("Starting training...")
    trainer.train()

    # Save final model and tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to: {args.output_dir}")
    print("Run predictions with: python main.py --model", args.output_dir)


if __name__ == "__main__":
    main()
