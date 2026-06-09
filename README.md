# BigPersonalities

Big Five personality prediction from Reddit comments using transformer models.

## Overview

This project loads Reddit comments from a SQLite database, predicts the Big Five personality traits (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism) for each comment using a transformer model, and saves the results along with visualizations.

## Components

- **load_database.py** — Loads comments from a SQLite database. Provides `load_comments()` with options for table name, limit, and filtering.
- **load_pandora.py** — Loads PANDORA training data (author profiles + comments), aggregates comments per author, and returns samples with normalized Big Five labels.
- **train.py** — Fine-tunes any Hugging Face base model (roberta-base, bert-base-uncased, etc.) on PANDORA and saves the model locally.
- **evaluate_pandora.py** — Compares model performance before and after fine-tuning on PANDORA (MSE on held-out test set).
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

7. Compare before vs after training (evaluates untrained model, trains, then evaluates again):
   ```
   python evaluate_pandora.py --base_model roberta-base --use_cpu
   python evaluate_pandora.py --limit_samples 200 --epochs 2  # Quick test
   ```

## Parameters

| Parameter    | Default                         | Description                          |
|-------------|----------------------------------|--------------------------------------|
| `--db_path` | `data/small_reddit.db`           | Path to the SQLite database file     |
| `--model`   | `vladinc/bigfive-regression-model` | Hugging Face model for prediction |
| `--output_dir` | `results`                    | Directory to save outputs            |
| `--limit`   | None                            | Max number of comments to process    |
| `--where`   | None                            | Optional WHERE clause for filtering  |

## Output

All outputs are saved to the `results/` folder (or the path given by `--output_dir`):

- **CSV**: `YYYY-MM-DD_HH-MM-SS_personality_results.csv` — Full results in comma-separated format
- **XLSX**: `YYYY-MM-DD_HH-MM-SS_personality_results.xlsx` — Same data in Excel format
- **summary_bar_chart.png** — Bar chart of average Big Five scores
- **distribution_per_trait.png** — Histograms for each trait
- **sample_radar.png** — Radar chart for a sample comment

All files use the same timestamp prefix for easy matching.

## Note on Models

The default model `vladinc/bigfive-regression-model` is pre-trained on essay data and gives reasonable predictions. For short Reddit comments, fine-tuning on PANDORA (see "Training on PANDORA" above) typically yields better results. Use `train.py` to fine-tune roberta-base, bert-base-uncased, or other models on the PANDORA dataset.
