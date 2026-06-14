"""Benchmark and compare multiple personality models on a Reddit comment database.

Runs each model against the same dataset, computes per-trait avg and std dev,
saves a comparison CSV, and generates a side-by-side bar chart.

Usage:
    .venv/bin/python benchmark.py
    .venv/bin/python benchmark.py --models vladinc/bigfive-regression-model models/roberta-pandora
    .venv/bin/python benchmark.py --db_path data/small_reddit.db --output_dir results/benchmark
"""

import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import load_database
import personality_model


DEFAULT_MODELS = [
    "vladinc/bigfive-regression-model",
    "models/roberta-pandora",
]


def run_model(model_name, comments, batch_size=32):
    predictor = personality_model.PersonalityPredictor(model_name=model_name)
    texts = [c.get("body", "") or "" for c in comments]

    predictions = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"  Predicting", leave=False):
        predictions.extend(predictor.predict_batch(texts[i : i + batch_size]))
    return predictions


def compute_stats(predictions):
    stats = {}
    for trait in personality_model.BIG_FIVE_TRAITS:
        vals = [p[trait] for p in predictions]
        stats[trait] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return stats


def _bar_chart(all_stats, key, title, ylabel, output_path, ylim_min=None):
    traits = [t.capitalize() for t in personality_model.BIG_FIVE_TRAITS]
    model_names = list(all_stats.keys())
    x = np.arange(len(traits))
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, model_name in enumerate(model_names):
        vals = [all_stats[model_name][t][key] for t in personality_model.BIG_FIVE_TRAITS]
        offset = (i - len(model_names) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=os.path.basename(model_name))

    ax.set_xticks(x)
    ax.set_xticklabels(traits)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()
    if ylim_min is not None:
        ax.set_ylim(0, max(ylim_min, ax.get_ylim()[1]))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_avg_score_chart(all_stats, output_path):
    _bar_chart(all_stats, "mean", "Average Sigmoid Score per Trait", "Mean Score (0-1)", output_path, ylim_min=1.0)


def save_std_chart(all_stats, output_path):
    _bar_chart(all_stats, "std", "Prediction Std Dev per Trait", "Std Dev", output_path)


def main():
    parser = argparse.ArgumentParser(description="Benchmark multiple personality models")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--db_path", type=str, default="data/small_reddit.db")
    parser.add_argument("--output_dir", type=str, default="results/benchmark")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    print(f"Loading comments from {args.db_path}...")
    comments = load_database.load_comments(args.db_path, limit=args.limit)
    print(f"Loaded {len(comments)} comments.\n")

    all_stats = {}
    all_predictions = {}

    for model_name in args.models:
        print(f"Running: {model_name}")
        preds = run_model(model_name, comments, batch_size=args.batch_size)
        all_stats[model_name] = compute_stats(preds)
        all_predictions[model_name] = preds

    # Print summary table
    print("\n=== Average Scores ===")
    header = f"{'Trait':<22}" + "".join(f"{os.path.basename(m):>18}" for m in args.models)
    print(header)
    print("-" * len(header))
    for trait in personality_model.BIG_FIVE_TRAITS:
        row = f"{trait.capitalize():<22}"
        row += "".join(f"{all_stats[m][trait]['mean']:>18.3f}" for m in args.models)
        print(row)

    print("\n=== Std Dev ===")
    print(header)
    print("-" * len(header))
    for trait in personality_model.BIG_FIVE_TRAITS:
        row = f"{trait.capitalize():<22}"
        row += "".join(f"{all_stats[m][trait]['std']:>18.3f}" for m in args.models)
        print(row)

    # Save CSV
    rows = []
    for model_name, stats in all_stats.items():
        for trait in personality_model.BIG_FIVE_TRAITS:
            rows.append({
                "model": model_name,
                "trait": trait,
                "mean": stats[trait]["mean"],
                "std": stats[trait]["std"],
            })
    csv_path = os.path.join(run_dir, "benchmark_results.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nSaved results to: {csv_path}")

    avg_path = os.path.join(run_dir, "avg_score_per_trait.png")
    save_avg_score_chart(all_stats, avg_path)
    print(f"Saved chart to:   {avg_path}")

    std_path = os.path.join(run_dir, "std_dev_per_trait.png")
    save_std_chart(all_stats, std_path)
    print(f"Saved chart to:   {std_path}")


if __name__ == "__main__":
    main()
