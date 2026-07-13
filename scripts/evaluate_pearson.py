"""Pearson/R2 comparison with the sigmoid-mismatch fix (inference-side).

Reproduces the June-23 comparison (mean baseline + pretrained baseline + both
PANDORA fine-tunes) but adds two things:

1. Pearson r and R2 per trait, alongside MSE/MAE. R2 <= 0 means "no better than
   predicting the mean"; Pearson r shows whether the model tracks per-author
   variation at all.

2. The sigmoid-mismatch fix, applied at INFERENCE (no retraining). train.py
   regresses raw logits directly onto [0, 1] targets (MSE with no sigmoid), but
   inference was applying sigmoid, inflating every trait toward ~0.6. Here each
   fine-tune is scored BOTH ways from a single forward pass:
     - "[sigmoid OLD]" : the previous (buggy) behavior
     - "[clip FIX]"    : raw logits clamped to [0, 1] (matches how it trained)
   The vladinc baseline was trained with a sigmoid, so it is scored sigmoid-only.

Results (CSV/JSON/charts) are written to results/Pearson-Coefficient/<timestamp>/.

Usage:
    python evaluate_pearson.py
    python evaluate_pearson.py --batch_size 32 --max_comments_per_author 50
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Allow running as `python scripts/evaluate_pearson.py` from the repo root:
# put the repo root on the path so the project modules import cleanly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import load_pandora
import personality_model
from evaluate_models import (
    compute_metrics,
    save_mse_chart,
    save_pred_vs_label_chart,
    save_std_chart,
)

TRAITS = personality_model.BIG_FIVE_TRAITS

# Entry labels (also used as chart/legend names).
MEAN_BASELINE = "mean-baseline (no model)"


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _clip01(x):
    return np.clip(x, 0.0, 1.0)


ACTIVATIONS = {"sigmoid": _sigmoid, "clip": _clip01}


def apply_activation(raw_preds, fn):
    """Turn raw-logit prediction dicts into activated 0-1 prediction dicts."""
    return [{t: float(fn(p[t])) for t in TRAITS} for p in raw_preds]


def mean_baseline_predictions(train_samples, n_val):
    """Predict each trait's train-set mean for every val sample (no model)."""
    means = {t: float(np.mean([s[t] for s in train_samples])) for t in TRAITS}
    return [dict(means) for _ in range(n_val)]


def predict_user_level_raw(model_name, texts, batch_size):
    """Run one forward pass per author, returning RAW-logit prediction dicts."""
    predictor = personality_model.PersonalityPredictor(model_name, output_activation="raw")
    preds = []
    n_batches = (len(texts) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(texts), batch_size), total=n_batches,
                  desc=f"  {os.path.basename(model_name)} (user-level)", unit="batch"):
        preds.extend(predictor.predict_batch(texts[i : i + batch_size]))
    return preds


def predict_comment_level_raw(model_name, val_author_data, author_order, batch_size):
    """Run one forward pass per COMMENT, returning raw preds + their author map.

    The caller applies an activation per comment, then averages per author, so a
    single forward pass yields both the sigmoid and clip author-level scores.
    """
    predictor = personality_model.PersonalityPredictor(model_name, output_activation="raw")
    all_texts, text_author = [], []
    for author in author_order:
        comments = val_author_data[author]["comments"]
        all_texts.extend(comments)
        text_author.extend([author] * len(comments))

    flat_raw = []
    n_batches = (len(all_texts) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(all_texts), batch_size), total=n_batches,
                  desc=f"  {os.path.basename(model_name)} (comment-level, {len(all_texts)} comments)",
                  unit="batch"):
        flat_raw.extend(predictor.predict_batch(all_texts[i : i + batch_size]))
    return flat_raw, text_author


def average_comment_preds(flat_raw, text_author, author_order, fn):
    """Apply activation per comment, then average per author -> author-level preds."""
    by_author = defaultdict(list)
    for author, raw in zip(text_author, flat_raw):
        by_author[author].append({t: float(fn(raw[t])) for t in TRAITS})
    out = []
    for author in author_order:
        ps = by_author[author]
        out.append({t: float(np.mean([q[t] for q in ps])) for t in TRAITS})
    return out


def print_metric_table(all_metrics, order, labels, key, title, overall_key):
    col_w = 26
    print(f"\n=== {title} ===")
    print(f"{'Trait':<20}" + "".join(f"{l:>{col_w}}" for l in labels))
    print("-" * (20 + col_w * len(order)))
    for trait in TRAITS:
        row = f"{trait.capitalize():<20}"
        row += "".join(f"{all_metrics[m][trait][key]:>{col_w}.4f}" for m in order)
        print(row)
    overall_label = {"mse": "Overall MSE", "pearson": "Overall Pearson", "r2": "Overall R2"}[key]
    print(f"{overall_label:<20}" + "".join(f"{all_metrics[m][overall_key]:>{col_w}.4f}" for m in order))


def main():
    parser = argparse.ArgumentParser(description="Pearson/R2 comparison with sigmoid-mismatch fix")
    parser.add_argument("--pandora_dir", type=str, default="data/pandora")
    parser.add_argument("--output_dir", type=str, default="results/Pearson-Coefficient")
    parser.add_argument("--baseline_model", type=str, default="vladinc/bigfive-regression-model")
    parser.add_argument("--user_model", type=str, default="models/roberta-pandora")
    parser.add_argument("--comment_model", type=str, default="models/roberta-pandora-comment-level")
    parser.add_argument("--max_comments_per_author", type=int, default=50)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    # --- Data: reproduce the val split, using the author-level split (same seed)
    # so mean baseline, user-level, and comment-level models all score against
    # the same held-out authors (matches evaluate_mean_baseline.py). ---
    print("Loading PANDORA data (user-level, for train means + user-level model)...")
    samples = load_pandora.load_pandora_training_data(
        pandora_dir=args.pandora_dir,
        max_comments_per_author=args.max_comments_per_author,
        min_comments_per_author=1,
    )
    train_samples, _ = train_test_split(samples, test_size=args.val_split, random_state=42)

    print("Loading PANDORA data (by author, for comment-level model)...")
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
        s = {"author": author, "text": " ".join(d["comments"])}
        s.update(d["labels"])
        val_samples.append(s)
    print(f"Validation set: {len(val_samples)} authors\n")

    texts = [s["text"] for s in val_samples]

    all_metrics = {}
    all_predictions = {}

    def record(name, preds):
        all_predictions[name] = preds
        all_metrics[name] = compute_metrics(preds, val_samples)

    # Ordered list of (name, label) entries the tables/CSV will use.
    order, labels = [], []

    def add(name, label):
        order.append(name)
        labels.append(label)

    # 1) Mean baseline (no model)
    print("[1/4] mean baseline")
    record(MEAN_BASELINE, mean_baseline_predictions(train_samples, len(val_samples)))
    add(MEAN_BASELINE, "mean-baseline")

    # 2) Pretrained baseline (trained WITH sigmoid) -> sigmoid only
    print("[2/4] pretrained baseline (sigmoid)")
    base_raw = predict_user_level_raw(args.baseline_model, texts, args.batch_size)
    name = f"{os.path.basename(args.baseline_model)} [sigmoid]"
    record(name, apply_activation(base_raw, ACTIVATIONS["sigmoid"]))
    add(name, "vladinc")

    # 3) User-level fine-tune: one forward pass, scored sigmoid (OLD) and clip (FIX)
    print("[3/4] user-level fine-tune (sigmoid OLD vs clip FIX)")
    user_raw = predict_user_level_raw(args.user_model, texts, args.batch_size)
    for act, tag in (("sigmoid", "sigmoid OLD"), ("clip", "clip FIX")):
        name = f"{os.path.basename(args.user_model)} [{tag}]"
        record(name, apply_activation(user_raw, ACTIVATIONS[act]))
        add(name, f"user:{act}")

    # 4) Comment-level fine-tune: one forward pass, scored sigmoid (OLD) and clip (FIX)
    print("[4/4] comment-level fine-tune (sigmoid OLD vs clip FIX)")
    flat_raw, text_author = predict_comment_level_raw(
        args.comment_model, val_author_data, val_authors, args.batch_size)
    for act, tag in (("sigmoid", "sigmoid OLD"), ("clip", "clip FIX")):
        name = f"{os.path.basename(args.comment_model)} [{tag}]"
        record(name, average_comment_preds(flat_raw, text_author, val_authors, ACTIVATIONS[act]))
        add(name, f"comment:{act}")

    # --- Report ---
    print_metric_table(all_metrics, order, labels, "mse",
                       "MSE per Trait (lower = better)", "overall_mse")
    print_metric_table(all_metrics, order, labels, "pearson",
                       "Pearson r per Trait (higher = better; 0 = no signal)", "overall_pearson")
    print_metric_table(all_metrics, order, labels, "r2",
                       "R2 per Trait (<= 0 = no better than the mean)", "overall_r2")

    # --- Save ---
    rows = []
    for name in order:
        m = all_metrics[name]
        for trait in TRAITS:
            rows.append({"model": name, "trait": trait, **m[trait]})
        rows.append({
            "model": name, "trait": "overall",
            "mse": m["overall_mse"], "mae": m["overall_mae"],
            "pearson": m["overall_pearson"], "r2": m["overall_r2"],
        })
    pd.DataFrame(rows).to_csv(os.path.join(run_dir, "evaluation_results.csv"), index=False)
    with open(os.path.join(run_dir, "evaluation_results.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    save_mse_chart(all_metrics, os.path.join(run_dir, "mse_per_trait.png"))
    save_std_chart(all_metrics, os.path.join(run_dir, "std_dev_per_trait.png"))
    save_pred_vs_label_chart(all_predictions, val_samples, os.path.join(run_dir, "pred_vs_true.png"))

    print(f"\nSaved to: {run_dir}")
    print("  evaluation_results.csv / .json")
    print("  mse_per_trait.png, std_dev_per_trait.png, pred_vs_true.png")


if __name__ == "__main__":
    main()
