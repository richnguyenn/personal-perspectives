# BigPersonalities

Big Five personality prediction from Reddit comments using transformer models.

## Overview

This project loads Reddit comments from a SQLite database, predicts the Big Five personality traits (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism) for each comment using a transformer model, and saves the results along with visualizations.

## Components

- **load_database.py** — Loads comments from a SQLite database. Provides `load_comments()` with options for table name, limit, and filtering.
- **load_pandora.py** — Loads PANDORA training data (author profiles + comments). Supports user-level aggregation (`load_pandora_training_data`) and author-grouped comment-level loading (`load_pandora_by_author` + `expand_to_comment_samples`) for comment-level training without data leakage.
- **train.py** — Fine-tunes any Hugging Face base model (roberta-base, bert-base-uncased, etc.) on PANDORA and saves the model locally. Supports both user-level (aggregated) and comment-level training.
- **evaluate_pandora.py** — Compares model performance before and after fine-tuning on PANDORA (MSE on held-out test set).
- **evaluate_models.py** — Compares multiple trained/pre-trained models side-by-side on the same PANDORA validation split, reporting per-trait MSE/MAE and saving comparison charts.
- **benchmark.py** — Runs multiple models against the same Reddit comment database and compares prediction distributions (mean/std per trait) without ground-truth labels.
- **personality_model.py** — Loads a Hugging Face model and predicts Big Five scores. Supports pre-trained models like `vladinc/bigfive-regression-model` and base models like `roberta-base` with a custom regression head.
- **visualize.py** — Creates graphs: summary bar chart, distribution histograms, and a sample radar chart.
- **main.py** — Entry point that runs the full pipeline: load data, predict, save to CSV/XLSX, and generate visualizations.

## How to Run

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run with default settings (uses `data/small_reddit.db` and `vladinc/bigfive-regression-model`):
   ```
   python main.py
   ```

3. Run with custom database and model:
   ```
   python main.py --db_path data/small_reddit.db --model vladinc/bigfive-regression-model
   ```

4. Run with roberta-base (experimental; uses a randomly initialized regression head):
   ```
   python main.py --model roberta-base
   ```

5. Limit the number of comments (e.g., for testing):
   ```
   python main.py --limit 100
   ```

6. Filter by subreddit:
   ```
   python main.py --where "subreddit = 'AskReddit'"
   ```

Each run creates a timestamped subdirectory under `--output_dir` (e.g. `results/2026-06-15_18-06-22/`) containing that run's CSV, XLSX, and charts.

## Training on PANDORA

You can fine-tune a model on the PANDORA dataset (Reddit comments with Big Five labels) and then use it for predictions:

1. Place PANDORA data in `data/pandora/`:
   - `author_profiles.csv`
   - `all_comments_since_2015.csv`

2. Train a model (e.g., roberta-base):
   ```
   python train.py --base_model roberta-base --output_dir models/roberta-base-pandora
   ```

3. Train with different base models:
   ```
   python train.py --base_model bert-base-uncased --output_dir models/bert-base-uncased-pandora
   python train.py --base_model distilbert-base-uncased --output_dir models/distilbert-pandora
   ```

4. Run predictions with the trained model:
   ```
   python main.py --model models/roberta-base-pandora --db_path data/small_reddit.db
   ```

5. Training options:
   ```
   python train.py --base_model roberta-base --output_dir models/roberta-pandora --epochs 3 --batch_size 8 --max_comments_per_author 100
   ```

6. If you hit out-of-memory errors (e.g., on Apple Silicon), use `--use_cpu` or reduce `--batch_size`:
   ```
   python train.py --base_model roberta-base --use_cpu --batch_size 2
   ```

7. Train on individual comments instead of aggregated per-author text (split is still done by author, to avoid data leakage between train/val):
   ```
   python train.py --base_model roberta-base --output_dir models/roberta-pandora-comment-level --comment_level
   ```

8. Compare before vs after training (evaluates untrained model, trains, then evaluates again):
   ```
   python evaluate_pandora.py --base_model roberta-base --use_cpu
   python evaluate_pandora.py --limit_samples 200 --epochs 2  # Quick test
   ```

## Evaluating and Benchmarking Models

Two scripts compare multiple models side-by-side:

- **evaluate_models.py** — Scores models against PANDORA ground-truth labels (MSE/MAE), using the same val split as training (`random_state=42`, 20% hold-out). Pass user-level trained models via `--models` and comment-level trained models via `--comment_level_models` (these are evaluated by predicting per comment, then averaging per author).
  ```
  python evaluate_models.py --models vladinc/bigfive-regression-model models/roberta-pandora
  python evaluate_models.py --comment_level_models models/roberta-pandora-comment-level
  ```

- **benchmark.py** — Runs models against a Reddit comment database (no ground truth) and compares the distribution of predictions (mean/std per trait).
  ```
  python benchmark.py --models vladinc/bigfive-regression-model models/roberta-pandora
  python benchmark.py --db_path data/small_reddit.db --output_dir results/benchmark
  ```

Both scripts save a timestamped subdirectory under `--output_dir` with a results CSV/JSON and comparison charts.

## Parameters

### main.py

| Parameter    | Default                         | Description                          |
|-------------|----------------------------------|--------------------------------------|
| `--db_path` | `data/small_reddit.db`           | Path to the SQLite database file     |
| `--model`   | `vladinc/bigfive-regression-model` | Hugging Face model for prediction |
| `--output_dir` | `results`                    | Directory to save outputs            |
| `--limit`   | None                            | Max number of comments to process    |
| `--where`   | None                            | Optional WHERE clause for filtering  |

### train.py

| Parameter    | Default                         | Description                          |
|-------------|----------------------------------|--------------------------------------|
| `--base_model` | `roberta-base`                | Hugging Face base model to fine-tune |
| `--pandora_dir` | `data/pandora`                | Path to PANDORA data directory       |
| `--output_dir`  | `models/roberta-base-pandora` | Directory to save the trained model  |
| `--epochs`      | `3`                            | Number of training epochs            |
| `--batch_size`  | `8`                            | Training batch size                  |
| `--max_comments_per_author` | `100`             | Max comments per author when aggregating |
| `--val_split`   | `0.2`                          | Fraction of data for validation      |
| `--max_length`  | `512`                          | Max token length per sample          |
| `--use_cpu`     | off                            | Force CPU instead of GPU/MPS         |
| `--limit_samples` | None                         | Limit number of training samples (for testing) |
| `--comment_level` | off                          | Train on individual comments instead of aggregated user text |

### evaluate_models.py

| Parameter    | Default                         | Description                          |
|-------------|----------------------------------|--------------------------------------|
| `--models`  | `vladinc/bigfive-regression-model`, `models/roberta-pandora` | User-level models to evaluate |
| `--comment_level_models` | `models/roberta-pandora-comment-level` | Comment-level models to evaluate |
| `--pandora_dir` | `data/pandora`                | Path to PANDORA data directory       |
| `--output_dir`  | `results/evaluation`          | Directory to save outputs            |
| `--max_comments_per_author` | `50`              | Max comments per author when aggregating |
| `--val_split`   | `0.2`                          | Fraction of data for validation      |
| `--batch_size`  | `32`                           | Inference batch size                 |

### benchmark.py

| Parameter    | Default                         | Description                          |
|-------------|----------------------------------|--------------------------------------|
| `--models`  | `vladinc/bigfive-regression-model`, `models/roberta-pandora` | Models to compare |
| `--db_path` | `data/small_reddit.db`           | Path to the SQLite database file     |
| `--output_dir` | `results/benchmark`           | Directory to save outputs            |
| `--limit`   | None                            | Max number of comments to process    |
| `--batch_size` | `32`                          | Inference batch size                 |

## Output

All outputs are saved under the `results/` folder (or the path given by `--output_dir`), inside a timestamped subdirectory per run, e.g. `results/2026-06-15_18-06-22/`:

- **personality_results.csv** — Full results in comma-separated format (from `main.py`)
- **personality_results.xlsx** — Same data in Excel format (from `main.py`)
- **summary_bar_chart.png** — Bar chart of average Big Five scores (from `main.py`)
- **distribution_per_trait.png** — Histograms for each trait (from `main.py`)
- **sample_radar.png** — Radar chart for a sample comment (from `main.py`)
- **evaluation_results.csv / evaluation_results.json** — Per-model MSE/MAE metrics (from `evaluate_models.py`, under `results/evaluation/`)
- **mse_per_trait.png / std_dev_per_trait.png / pred_vs_true.png** — Comparison charts (from `evaluate_models.py`)
- **benchmark_results.csv** — Per-model prediction distribution stats (from `benchmark.py`, under `results/benchmark/`)
- **avg_score_per_trait.png / std_dev_per_trait.png** — Comparison charts (from `benchmark.py`)

## Note on Models

The default model `vladinc/bigfive-regression-model` is pre-trained on essay data and gives reasonable predictions. For short Reddit comments, fine-tuning on PANDORA (see "Training on PANDORA" above) typically yields better results. Use `train.py` to fine-tune roberta-base, bert-base-uncased, or other models on the PANDORA dataset.
