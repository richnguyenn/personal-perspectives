"""Compare roberta-base before and after fine-tuning on PANDORA.

This script evaluates the untrained roberta-base on a held-out test set,
trains it on the training set, then evaluates again to see if training improved
performance. Uses MSE (Mean Squared Error) as the metric.
"""

import argparse
import os

from sklearn.model_selection import train_test_split
from tqdm import tqdm

import numpy as np

import load_pandora
import personality_model
from train import BigFiveDataset, RegressionTrainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments


def compute_mse(predictor, samples, batch_size=16):
    """Compute MSE for each trait and overall across samples.

    Parameters
    ----------
    predictor : PersonalityPredictor
        The model to evaluate.
    samples : list of dict
        List of samples with "text" and trait keys.
    batch_size : int
        Batch size for prediction.

    Returns
    -------
    dict
        Keys: trait names + "overall". Values: MSE (float).
    """
    texts = [s["text"] for s in samples]
    labels = [[s[t] for t in personality_model.BIG_FIVE_TRAITS] for s in samples]

    all_preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        preds = predictor.predict_batch(batch)
        for p in preds:
            all_preds.append([p[t] for t in personality_model.BIG_FIVE_TRAITS])

    preds_arr = np.array(all_preds)
    labels_arr = np.array(labels)

    mse_per_trait = {}
    for j, trait in enumerate(personality_model.BIG_FIVE_TRAITS):
        mse = float(((preds_arr[:, j] - labels_arr[:, j]) ** 2).mean())
        mse_per_trait[trait] = mse

    mse_per_trait["overall"] = float(((preds_arr - labels_arr) ** 2).mean())
    return mse_per_trait


def main():
    parser = argparse.ArgumentParser(
        description="Compare roberta-base before and after PANDORA fine-tuning"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="roberta-base",
        help="Base model to evaluate and train",
    )
    parser.add_argument(
        "--pandora_dir",
        type=str,
        default="data/pandora",
        help="Path to PANDORA data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/roberta-base-pandora-eval",
        help="Where to save the trained model",
    )
    parser.add_argument(
        "--max_comments_per_author",
        type=int,
        default=50,
        help="Max comments per author",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction for test set (0.2 = 20%%)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training and eval",
    )
    parser.add_argument(
        "--use_cpu",
        action="store_true",
        help="Force CPU",
    )
    parser.add_argument(
        "--limit_samples",
        type=int,
        default=None,
        help="Limit samples (for quick testing)",
    )
    args = parser.parse_args()

    # Load PANDORA data
    print("Loading PANDORA data...")
    samples = load_pandora.load_pandora_training_data(
        pandora_dir=args.pandora_dir,
        max_comments_per_author=args.max_comments_per_author,
        min_comments_per_author=1,
        limit_samples=args.limit_samples,
    )

    if len(samples) < 20:
        print("Need at least 20 samples. Exiting.")
        return

    # Split train/test
    train_samples, test_samples = train_test_split(
        samples, test_size=args.test_size, random_state=42
    )
    print(f"Train: {len(train_samples)}, Test: {len(test_samples)}")

    # --- BEFORE: Evaluate untrained roberta-base ---
    print("\n" + "=" * 50)
    print("BEFORE TRAINING (untrained roberta-base)")
    print("=" * 50)
    predictor_before = personality_model.PersonalityPredictor(
        model_name=args.base_model, device="cpu" if args.use_cpu else None
    )
    mse_before = compute_mse(predictor_before, test_samples, batch_size=args.batch_size)
    for trait in personality_model.BIG_FIVE_TRAITS:
        print(f"  {trait}: MSE = {mse_before[trait]:.4f}")
    print(f"  overall: MSE = {mse_before['overall']:.4f}")

    # --- TRAIN ---
    print("\n" + "=" * 50)
    print("TRAINING...")
    print("=" * 50)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=5
    )

    train_dataset = BigFiveDataset(train_samples, tokenizer, max_length=512)
    val_dataset = BigFiveDataset(test_samples, tokenizer, max_length=512)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_ratio=0.1,
        weight_decay=0.01,
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
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to: {args.output_dir}")

    # --- AFTER: Evaluate trained model ---
    print("\n" + "=" * 50)
    print("AFTER TRAINING (fine-tuned on PANDORA)")
    print("=" * 50)
    predictor_after = personality_model.PersonalityPredictor(
        model_name=args.output_dir, device="cpu" if args.use_cpu else None
    )
    mse_after = compute_mse(predictor_after, test_samples, batch_size=args.batch_size)
    for trait in personality_model.BIG_FIVE_TRAITS:
        print(f"  {trait}: MSE = {mse_after[trait]:.4f}")
    print(f"  overall: MSE = {mse_after['overall']:.4f}")

    # --- COMPARISON ---
    print("\n" + "=" * 50)
    print("COMPARISON (lower MSE = better)")
    print("=" * 50)
    print(f"{'Trait':<20} {'Before':>10} {'After':>10} {'Change':>10}")
    print("-" * 52)
    for trait in personality_model.BIG_FIVE_TRAITS:
        change = mse_after[trait] - mse_before[trait]
        sign = "+" if change > 0 else ""
        print(f"{trait:<20} {mse_before[trait]:>10.4f} {mse_after[trait]:>10.4f} {sign}{change:>9.4f}")
    change_overall = mse_after["overall"] - mse_before["overall"]
    sign = "+" if change_overall > 0 else ""
    print("-" * 52)
    print(f"{'overall':<20} {mse_before['overall']:>10.4f} {mse_after['overall']:>10.4f} {sign}{change_overall:>9.4f}")

    if mse_after["overall"] < mse_before["overall"]:
        print("\nTraining IMPROVED the model (lower MSE after training).")
    else:
        print("\nTraining did not improve overall MSE. Try more epochs or different hyperparameters.")


if __name__ == "__main__":
    main()
