"""Create graphs and visualizations for Big Five personality results.

This module creates bar charts, distribution plots, and radar charts
to help visualize the personality prediction results.
"""

import matplotlib.pyplot as plt
import numpy as np

from personality_model import BIG_FIVE_TRAITS


def create_summary_chart(avg_scores, output_path):
    """Create a bar chart showing the average Big Five scores.

    Parameters
    ----------
    avg_scores : dict
        Dictionary with trait names as keys and average scores as values.
    output_path : str
        Path to save the PNG image.
    """
    traits = []
    scores = []
    for trait in BIG_FIVE_TRAITS:
        traits.append(trait.capitalize())
        scores.append(avg_scores.get(trait, 0.0))

    plt.figure(figsize=(10, 6))
    bars = plt.bar(traits, scores, color="steelblue", edgecolor="black")
    plt.xlabel("Personality Trait")
    plt.ylabel("Average Score (0 to 1)")
    plt.title("Average Big Five Personality Scores Across All Comments")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def create_distribution_plots(results_list, output_path):
    """Create histograms showing the distribution of each trait.

    Parameters
    ----------
    results_list : list of dict
        List of prediction results. Each dict has trait names as keys.
    output_path : str
        Path to save the PNG image.
    """
    num_traits = len(BIG_FIVE_TRAITS)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # Flatten axes for easy looping (we have 5 traits, 6 subplot spots)
    axes_flat = axes.flatten()

    for i, trait in enumerate(BIG_FIVE_TRAITS):
        scores = []
        for result in results_list:
            scores.append(result.get(trait, 0.0))

        ax = axes_flat[i]
        ax.hist(scores, bins=20, color="steelblue", edgecolor="black")
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.set_title(trait.capitalize())
        ax.set_xlim(0, 1)

    # Hide the extra subplot (we have 5 traits, 6 subplots)
    axes_flat[5].axis("off")

    plt.suptitle("Distribution of Big Five Scores Across Comments", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def create_radar_chart(scores_dict, output_path):
    """Create a radar (spider) chart for one comment's Big Five scores.

    Parameters
    ----------
    scores_dict : dict
        Dictionary with trait names as keys and scores as values.
    output_path : str
        Path to save the PNG image.
    """
    traits = [t.capitalize() for t in BIG_FIVE_TRAITS]
    scores = [scores_dict.get(t, 0.0) for t in BIG_FIVE_TRAITS]

    # Close the polygon for radar chart
    angles = np.linspace(0, 2 * np.pi, len(traits), endpoint=False).tolist()
    angles += angles[:1]
    scores_plot = scores + scores[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))
    ax.plot(angles, scores_plot, "o-", linewidth=2, color="steelblue")
    ax.fill(angles, scores_plot, alpha=0.25, color="steelblue")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(traits)
    ax.set_ylim(0, 1)
    ax.set_title("Big Five Personality Profile (Sample)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
