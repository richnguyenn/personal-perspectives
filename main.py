"""Main entry point for the BigPersonalities experiment.

This script loads Reddit comments from a database, runs Big Five personality
prediction, saves results to CSV and XLSX, and creates visualizations.
"""

import argparse
import os
from datetime import datetime

import pandas as pd
from tqdm import tqdm

import load_database
import personality_model
import visualize


def main():
    """Run the full pipeline: load data, predict, save, and visualize."""
    parser = argparse.ArgumentParser(
        description="Big Five personality prediction from Reddit comments"
    )
    parser.add_argument(
        "--db_path",
        type=str,
        default="data/small_reddit.db",
        help="Path to the SQLite database file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="vladinc/bigfive-regression-model",
        help="Hugging Face model name for personality prediction",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save results and graphs",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of comments to process",
    )
    parser.add_argument(
        "--where",
        type=str,
        default=None,
        dest="where_clause",
        help="Optional WHERE clause to filter comments",
    )
    args = parser.parse_args()

    # Create output directory if it does not exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Timestamped subdirectory for this run
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    # Step 1: Load comments from the database
    print("Loading comments from database...")
    comments = load_database.load_comments(
        db_path=args.db_path,
        limit=args.limit,
        where_clause=args.where_clause,
    )
    print("Loaded", len(comments), "comments.")

    if len(comments) == 0:
        print("No comments to process. Exiting.")
        return

    # Step 2: Initialize the personality predictor
    predictor = personality_model.PersonalityPredictor(model_name=args.model)

    # Step 3: Run predictions
    texts = []
    for comment in comments:
        body = comment.get("body", "")
        texts.append(body if body else "")

    print("Running personality predictions...")
    batch_size = 32
    all_predictions = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
        batch = texts[i : i + batch_size]
        predictions = predictor.predict_batch(batch)
        all_predictions.extend(predictions)

    # Step 4: Build results and save to CSV and XLSX
    results = []
    for i, comment in enumerate(comments):
        row = dict(comment)
        row.update(all_predictions[i])
        results.append(row)

    df = pd.DataFrame(results)

    csv_path = os.path.join(run_dir, "personality_results.csv")
    xlsx_path = os.path.join(run_dir, "personality_results.xlsx")

    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)

    print("Saved results to:", csv_path)
    print("Saved results to:", xlsx_path)

    # Step 5: Create visualizations
    trait_scores = [all_predictions[i] for i in range(len(all_predictions))]

    avg_scores = {}
    for trait in personality_model.BIG_FIVE_TRAITS:
        total = 0.0
        for pred in trait_scores:
            total += pred.get(trait, 0.0)
        avg_scores[trait] = total / len(trait_scores) if trait_scores else 0.0

    summary_path = os.path.join(run_dir, "summary_bar_chart.png")
    visualize.create_summary_chart(avg_scores, summary_path)
    print("Saved summary chart to:", summary_path)

    dist_path = os.path.join(run_dir, "distribution_per_trait.png")
    visualize.create_distribution_plots(trait_scores, dist_path)
    print("Saved distribution plot to:", dist_path)

    if len(trait_scores) > 0:
        sample_path = os.path.join(run_dir, "sample_radar.png")
        visualize.create_radar_chart(trait_scores[0], sample_path)
        print("Saved sample radar chart to:", sample_path)

    # Step 6: Print summary
    print("\n--- Summary ---")
    print("Average scores:")
    for trait in personality_model.BIG_FIVE_TRAITS:
        print(f"  {trait}: {avg_scores[trait]:.3f}")
    print("\nDone!")


if __name__ == "__main__":
    main()
