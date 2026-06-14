"""Evaluate and compare personality models on the PANDORA validation set.

Loads the same val split used during training (random_state=42, 20% hold-out),
runs each model, computes per-trait MSE against ground truth Big Five labels,
and saves a comparison report + charts.

Usage:
    python evaluate_models.py
    python evaluate_models.py --models vladinc/bigfive-regression-model models/roberta-pandora
    python evaluate_models.py --pandora_dir data/pandora --output_dir results/evaluation
"""

import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import load_pandora
import personality_model


DEFAULT_MODELS = [
    "vladinc/bigfive-regression-model",
    "models/roberta-pandora",
]


def run_predictions(model_name, texts, batch_size=32):
    predictor = personality_model.PersonalityPredictor(model_name=model_name)
    predictions = []
    for i in tqdm(range(0, len(texts), batch_size), desc="  Predicting", leave=False):
        predictions.extend(predictor.predict_batch(texts[i : i + batch_size]))
    return predictions


def compute_metrics(predictions, val_samples):
    traits = personality_model.BIG_FIVE_TRAITS
    metrics = {}
    for trait in traits:
        preds = np.array([p[trait] for p in predictions])
        labels = np.array([s[trait] for s in val_samples])
        metrics[trait] = {
            "mse": float(np.mean((preds - labels) ** 2)),
            "mae": float(np.mean(np.abs(preds - labels))),
            "pred_mean": float(np.mean(preds)),
            "pred_std": float(np.std(preds)),
            "label_mean": float(np.mean(labels)),
            "label_std": float(np.std(labels)),
        }
    metrics["overall_mse"] = float(np.mean([metrics[t]["mse"] for t in traits]))
    metrics["overall_mae"] = float(np.mean([metrics[t]["mae"] for t in traits]))
    return metrics


def save_mse_chart(all_metrics, output_path):
    traits = personality_model.BIG_FIVE_TRAITS
    model_names = list(all_metrics.keys())
    x = np.arange(len(traits))
    width = 0.8 / len(model_names)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for i, model_name in enumerate(model_names):
        metrics = all_metrics[model_name]
        label = os.path.basename(model_name)
        offset = (i - len(model_names) / 2 + 0.5) * width

        mse_vals = [metrics[t]["mse"] for t in traits]
        std_vals = [metrics[t]["pred_std"] for t in traits]

        axes[0].bar(x + offset, mse_vals, width, label=label)
        axes[1].bar(x + offset, std_vals, width, label=label)

    trait_labels = [t.capitalize() for t in traits]
    for ax, title, ylabel in [
        (axes[0], "MSE per Trait (lower = better)", "MSE"),
        (axes[1], "Prediction Std Dev per Trait (higher = better spread)", "Std Dev"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels(trait_labels)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend()

    plt.suptitle("Model Evaluation on PANDORA Validation Set", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_pred_vs_label_chart(all_predictions, val_samples, output_path):
    traits = personality_model.BIG_FIVE_TRAITS
    model_names = list(all_predictions.keys())
    labels = {t: np.array([s[t] for s in val_samples]) for t in traits}

    fig, axes = plt.subplots(1, len(traits), figsize=(4 * len(traits), 4))

    for j, trait in enumerate(traits):
        ax = axes[j]
        for model_name, preds in all_predictions.items():
            pred_vals = np.array([p[trait] for p in preds])
            ax.scatter(labels[trait], pred_vals, alpha=0.3, s=10,
                       label=os.path.basename(model_name))
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect")
        ax.set_xlabel("True Score")
        ax.set_ylabel("Predicted Score")
        ax.set_title(trait.capitalize())
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        if j == 0:
            ax.legend(fontsize=7)

    plt.suptitle("Predicted vs True Scores (PANDORA Val Set)", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate models on PANDORA validation set")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--pandora_dir", type=str, default="data/pandora")
    parser.add_argument("--output_dir", type=str, default="results/evaluation")
    parser.add_argument("--max_comments_per_author", type=int, default=50)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    # Reproduce exact val split from training
    print("Loading PANDORA data...")
    samples = load_pandora.load_pandora_training_data(
        pandora_dir=args.pandora_dir,
        max_comments_per_author=args.max_comments_per_author,
        min_comments_per_author=1,
    )
    _, val_samples = train_test_split(samples, test_size=args.val_split, random_state=42)
    print(f"Validation set: {len(val_samples)} authors\n")

    texts = [s["text"] for s in val_samples]

    all_metrics = {}
    all_predictions = {}

    for model_name in args.models:
        print(f"Running: {model_name}")
        preds = run_predictions(model_name, texts, batch_size=args.batch_size)
        all_predictions[model_name] = preds
        all_metrics[model_name] = compute_metrics(preds, val_samples)

    # Print summary
    traits = personality_model.BIG_FIVE_TRAITS
    col_w = 20
    model_labels = [os.path.basename(m) for m in args.models]

    print(f"\n=== MSE per Trait (lower = better) ===")
    print(f"{'Trait':<22}" + "".join(f"{l:>{col_w}}" for l in model_labels))
    print("-" * (22 + col_w * len(args.models)))
    for trait in traits:
        row = f"{trait.capitalize():<22}"
        row += "".join(f"{all_metrics[m][trait]['mse']:>{col_w}.4f}" for m in args.models)
        print(row)
    print(f"{'Overall MSE':<22}" + "".join(f"{all_metrics[m]['overall_mse']:>{col_w}.4f}" for m in args.models))

    print(f"\n=== Prediction Std Dev per Trait ===")
    print(f"{'Trait':<22}" + "".join(f"{l:>{col_w}}" for l in model_labels))
    print("-" * (22 + col_w * len(args.models)))
    for trait in traits:
        row = f"{trait.capitalize():<22}"
        row += "".join(f"{all_metrics[m][trait]['pred_std']:>{col_w}.4f}" for m in args.models)
        print(row)

    # Save results
    rows = []
    for model_name, metrics in all_metrics.items():
        for trait in traits:
            rows.append({
                "model": model_name,
                "trait": trait,
                **metrics[trait],
            })
        rows.append({
            "model": model_name,
            "trait": "overall",
            "mse": metrics["overall_mse"],
            "mae": metrics["overall_mae"],
        })
    csv_path = os.path.join(run_dir, "evaluation_results.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    json_path = os.path.join(run_dir, "evaluation_results.json")
    with open(json_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    chart_path = os.path.join(run_dir, "evaluation_chart.png")
    save_mse_chart(all_metrics, chart_path)

    scatter_path = os.path.join(run_dir, "pred_vs_true.png")
    save_pred_vs_label_chart(all_predictions, val_samples, scatter_path)

    print(f"\nSaved to: {run_dir}")
    print(f"  evaluation_results.csv")
    print(f"  evaluation_results.json")
    print(f"  evaluation_chart.png  (MSE + std dev bar chart)")
    print(f"  pred_vs_true.png      (scatter: predicted vs ground truth per trait)")


if __name__ == "__main__":
    main()
