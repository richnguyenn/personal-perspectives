"""Load PANDORA dataset for Big Five personality training.

This module loads author profiles and comments from the PANDORA dataset,
aggregates comments per author, and returns training samples with normalized
Big Five labels (0-1 scale).
"""

import os
from collections import defaultdict

import pandas as pd

# Big Five trait columns in author_profiles.csv
BIG_FIVE_COLS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]


def load_pandora_training_data(
    pandora_dir,
    max_comments_per_author=None,
    min_comments_per_author=1,
    limit_samples=None,
):
    """Load PANDORA training data: aggregated comments per author with Big Five labels.

    Parameters
    ----------
    pandora_dir : str
        Path to the PANDORA data directory (contains author_profiles.csv
        and all_comments_since_2015.csv).
    max_comments_per_author : int, optional
        Maximum number of comments to use per author. If None, use all comments.
        Capping helps control text length and memory.
    min_comments_per_author : int, optional
        Minimum number of comments required per author. Authors with fewer
        comments are excluded (default: 1).
    limit_samples : int, optional
        Max number of samples to return. Useful for testing (default: None).

    Returns
    -------
    list of dict
        Each dict has keys: "text", "openness", "conscientiousness",
        "extraversion", "agreeableness", "neuroticism". Labels are normalized
        to 0-1 (PANDORA uses 0-100).
    """
    profiles_path = os.path.join(pandora_dir, "author_profiles.csv")
    comments_path = os.path.join(pandora_dir, "all_comments_since_2015.csv")

    if not os.path.exists(profiles_path):
        raise FileNotFoundError(f"Author profiles not found: {profiles_path}")
    if not os.path.exists(comments_path):
        raise FileNotFoundError(f"Comments file not found: {comments_path}")

    # Load author profiles and filter to authors with all Big Five traits
    print("Loading author profiles...")
    profiles = pd.read_csv(profiles_path)
    valid_profiles = profiles.dropna(subset=BIG_FIVE_COLS)
    valid_authors = set(valid_profiles["author"].tolist())
    print(f"Found {len(valid_authors)} authors with Big Five labels.")

    # Build author -> list of comment bodies via chunked reading
    print("Loading comments (chunked)...")
    author_comments = defaultdict(list)
    chunk_size = 50000

    for chunk in pd.read_csv(comments_path, chunksize=chunk_size, usecols=["author", "body"]):
        chunk = chunk[chunk["author"].isin(valid_authors)]
        for _, row in chunk.iterrows():
            author = row["author"]
            body = row["body"]
            if pd.isna(body) or str(body).strip() == "":
                continue
            if max_comments_per_author is not None:
                if len(author_comments[author]) >= max_comments_per_author:
                    continue
            author_comments[author].append(str(body).strip())

    # Build training samples: aggregate text, get labels from profiles
    samples = []
    profiles_by_author = valid_profiles.set_index("author")

    for author in valid_authors:
        comments = author_comments.get(author, [])
        if len(comments) < min_comments_per_author:
            continue

        text = " ".join(comments)
        if not text.strip():
            continue

        row = profiles_by_author.loc[author]
        sample = {
            "text": text,
            "openness": float(row["openness"]) / 100.0,
            "conscientiousness": float(row["conscientiousness"]) / 100.0,
            "extraversion": float(row["extraversion"]) / 100.0,
            "agreeableness": float(row["agreeableness"]) / 100.0,
            "neuroticism": float(row["neuroticism"]) / 100.0,
        }
        samples.append(sample)
        if limit_samples is not None and len(samples) >= limit_samples:
            break

    print(f"Built {len(samples)} training samples.")
    return samples
