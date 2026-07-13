"""Compare trained models against a trivial no-model baseline.

Uses the same val split as evaluate_models.py (random_state=42, 20% hold-out) so
results are directly comparable.

Usage:
    python evaluate_mean_baseline.py
    python evaluate_mean_baseline.py --models vladinc/bigfive-regression-model models/roberta-pandora
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import load_pandora
import personality_model
from evaluate_models import (
    compute_metrics,
    run_predictions,
    run_predictions_comment_level,
    save_mse_chart,
    save_pred_vs_label_chart,
    save_std_chart,
)

MEAN_BASELINE_NAME = "mean-baseline (no model)"

DEFAULT_MODELS = [
    "vladinc/bigfive-regression-model",
    "models/roberta-pandora",
    "models/roberta-pandora-comment-level",
]


def mean_baseline_predictions(train_samples, n_val):
    """Predict each trait's train-set mean for every val sample (no model)."""
    traits = personality_model.BIG_FIVE_TRAITS
    means = {t: float(np.mean([s[t] for s in train_samples])) for t in traits}
    return [dict(means) for _ in range(n_val)]


def main():
    parser = argparse.ArgumentParser(description="Compare models against a mean-only (no-model) baseline")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--comment_level_models", nargs="+", default=["models/roberta-pandora-comment-level"],
                        help="Models trained at comment level — evaluated by predicting per comment then averaging per author")
    parser.add_argument("--pandora_dir", type=str, default="data/pandora")
    parser.add_argument("--output_dir", type=str, default="results/evaluation")
    parser.add_argument("--max_comments_per_author", type=int, default=50)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    comment_level_set = set(args.comment_level_models or [])

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(args.output_dir, f"{timestamp}_mean_baseline")
    os.makedirs(run_dir, exist_ok=True)

    print("Loading PANDORA data (user-level)...")
    samples = load_pandora.load_pandora_training_data(
        pandora_dir=args.pandora_dir,
        max_comments_per_author=args.max_comments_per_author,
        min_comments_per_author=1,
    )
    train_samples, val_samples = train_test_split(samples, test_size=args.val_split, random_state=42)
    print(f"Train: {len(train_samples)} authors, Val: {len(val_samples)} authors")

    val_author_data = None
    if any(m in comment_level_set for m in args.models):
        print("Loading PANDORA data (comment-level for comment-level models)...")
        all_author_data = load_pandora.load_pandora_by_author(
            pandora_dir=args.pandora_dir,
            max_comments_per_author=args.max_comments_per_author,
            min_comments_per_author=1,
        )
        authors = list(all_author_data.keys())
        _, val_authors = train_test_split(authors, test_size=args.val_split, random_state=42)
        val_author_data = {a: all_author_data[a] for a in val_authors}
        val_samples = []
        for author in val_authors:
            d = val_author_data[author]
            sample = {"author": author, "text": " ".join(d["comments"])}
            sample.update(d["labels"])
            val_samples.append(sample)
        print(f"Comment-level val: {len(val_samples)} authors\n")

    texts = [s["text"] for s in val_samples]

    all_metrics = {}
    all_predictions = {}

    print(f"Running: {MEAN_BASELINE_NAME}")
    baseline_preds = mean_baseline_predictions(train_samples, len(val_samples))
    all_predictions[MEAN_BASELINE_NAME] = baseline_preds
    all_metrics[MEAN_BASELINE_NAME] = compute_metrics(baseline_preds, val_samples)

    for model_name in args.models:
        print(f"Running: {model_name}")
        if model_name in comment_level_set and val_author_data is not None:
            print("  (comment-level evaluation: predict per comment, average per author)")
            preds = run_predictions_comment_level(model_name, val_author_data, val_samples, batch_size=args.batch_size)
        else:
            preds = run_predictions(model_name, texts, batch_size=args.batch_size)
        all_predictions[model_name] = preds
        all_metrics[model_name] = compute_metrics(preds, val_samples)

    names = [MEAN_BASELINE_NAME] + args.models
    labels = [MEAN_BASELINE_NAME] + [os.path.basename(m) for m in args.models]
    traits = personality_model.BIG_FIVE_TRAITS
    col_w = 24

    print(f"\n=== MSE per Trait (lower = better) ===")
    print(f"{'Trait':<22}" + "".join(f"{l:>{col_w}}" for l in labels))
    print("-" * (22 + col_w * len(names)))
    for trait in traits:
        row = f"{trait.capitalize():<22}"
        row += "".join(f"{all_metrics[m][trait]['mse']:>{col_w}.4f}" for m in names)
        print(row)
    print(f"{'Overall MSE':<22}" + "".join(f"{all_metrics[m]['overall_mse']:>{col_w}.4f}" for m in names))

    rows = []
    for model_name, metrics in all_metrics.items():
        for trait in traits:
            rows.append({"model": model_name, "trait": trait, **metrics[trait]})
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

    save_mse_chart(all_metrics, os.path.join(run_dir, "mse_per_trait.png"))
    save_std_chart(all_metrics, os.path.join(run_dir, "std_dev_per_trait.png"))
    save_pred_vs_label_chart(all_predictions, val_samples, os.path.join(run_dir, "pred_vs_true.png"))

    print(f"\nSaved to: {run_dir}")
    print(f"  evaluation_results.csv")
    print(f"  evaluation_results.json")
    print(f"  mse_per_trait.png     (mean-baseline vs. models, per trait)")
    print(f"  std_dev_per_trait.png (prediction spread per trait)")
    print(f"  pred_vs_true.png      (scatter: predicted vs ground truth per trait)")


if __name__ == "__main__":
    main()
