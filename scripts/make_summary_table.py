"""Render the per-trait Pearson comparison as a paper-style (booktabs) table
image + a markdown summary, saved next to the evaluation results."""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

RUN_DIR = sys.argv[1] if len(sys.argv) > 1 else "results/Pearson-Coefficient/2026-07-07_21-08-08"

# Columns: label -> model key in the results CSV. Fine-tunes use the clip FIX read
# (Pearson is identical to the sigmoid read; clip is the corrected inference).
COLS = [
    ("Mean baseline", "mean-baseline (no model)"),
    ("vladinc (pretrained)", "bigfive-regression-model [sigmoid]"),
    ("Comment-level", "roberta-pandora-comment-level [clip FIX]"),
    ("User-level (fine-tune)", "roberta-pandora [clip FIX]"),
]
# Trait row order (paper convention), plus an average row.
ROWS = ["agreeableness", "openness", "conscientiousness", "neuroticism", "extraversion"]


def fmt(v):
    s = f"{v:.3f}"
    if s.startswith("0."):
        return "." + s[2:]
    if s.startswith("-0."):
        return "-." + s[3:]
    return s


def main():
    df = pd.read_csv(os.path.join(RUN_DIR, "evaluation_results.csv"))
    piv = df.pivot(index="trait", columns="model", values="pearson")

    col_labels = ["Mean\nbaseline", "vladinc\n(pretrained)", "Comment-\nlevel", "User-level\n(fine-tune)"]
    keys = [c[1] for c in COLS]

    # Build value matrix (traits + Average row).
    data_rows = []
    display_rows = ROWS + ["overall"]
    for r in display_rows:
        data_rows.append([float(piv.loc[r, k]) for k in keys])

    # --- Render booktabs-style table ---
    n_rows, n_cols = len(display_rows), len(COLS)
    fig, ax = plt.subplots(figsize=(11, 0.9 + 0.5 * (n_rows + 1)))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    x_label = 0.03
    x_cols = [0.45, 0.62, 0.79, 0.955]  # centers for the 4 value columns
    title_y = 0.955
    top_rule_y = 0.88
    header_y = 0.825          # header text center
    header_rule_y = 0.755
    first_row_y = 0.685       # first data row center
    row_h = 0.095
    row_ys = [first_row_y - i * row_h for i in range(n_rows)]
    avg_rule_y = (row_ys[-2] + row_ys[-1]) / 2  # between Extraversion and Average
    bottom_rule_y = row_ys[-1] - 0.06

    def rule(y, lw=1.4):
        ax.plot([0.01, 0.99], [y, y], color="black", lw=lw)

    ax.text(0.5, title_y, "Regression (Pearson correlation coefficient)",
            ha="center", va="center", fontsize=12.5, style="italic")
    rule(top_rule_y, 1.6)
    for x, lab in zip(x_cols, col_labels):
        ax.text(x, header_y, lab, ha="center", va="center", fontsize=9, weight="bold")
    rule(header_rule_y, 1.0)

    for ri, r in enumerate(display_rows):
        y = row_ys[ri]
        name = "Average" if r == "overall" else r.capitalize()
        ax.text(x_label, y, name, ha="left", va="center", fontsize=9.5,
                weight="bold" if r == "overall" else "normal")
        vals = data_rows[ri]
        best = max(vals)
        for x, v in zip(x_cols, vals):
            is_best = abs(v - best) < 1e-9
            ax.text(x, y, fmt(v), ha="center", va="center", fontsize=9.5,
                    weight="bold" if is_best else "normal",
                    color="#0a7d2c" if is_best else "black")

    rule(avg_rule_y, 1.0)
    rule(bottom_rule_y, 1.6)

    out_png = os.path.join(RUN_DIR, "pearson_table.png")
    plt.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print("wrote", out_png)

    # --- Markdown summary ---
    md = []
    md.append("# Big5 / PANDORA — Pearson comparison (sigmoid-mismatch fix)\n")
    md.append(f"_Run: `{RUN_DIR}`_\n")
    md.append("## Per-trait Pearson r (best per row in **bold**)\n")
    header = "| Trait | " + " | ".join(col_labels) + " |"
    sep = "|" + "---|" * (n_cols + 1)
    md.append(header)
    md.append(sep)
    for ri, r in enumerate(display_rows):
        name = "**Average**" if r == "overall" else r.capitalize()
        vals = data_rows[ri]
        best = max(vals)
        cells = " | ".join((f"**{fmt(v)}**" if abs(v - best) < 1e-9 else fmt(v)) for v in vals)
        md.append(f"| {name} | {cells} |")

    # Overall MSE / R2 mini-table
    ov = df[df.trait == "overall"].set_index("model")
    md.append("\n## Overall (user-level fine-tune: sigmoid bug vs fix)\n")
    md.append("| | MSE ↓ | Pearson ↑ | R² ↑ |")
    md.append("|---|---|---|---|")
    for key, label in [
        ("mean-baseline (no model)", "mean baseline"),
        ("roberta-pandora [sigmoid OLD]", "fine-tune (old / sigmoid)"),
        ("roberta-pandora [clip FIX]", "fine-tune (fixed / clip)"),
    ]:
        row = ov.loc[key]
        md.append(f"| {label} | {row['mse']:.3f} | {row['pearson']:.3f} | {row['r2']:+.3f} |")
    md.append("\n**Notes:** Fixing the inference activation (clip logits to [0,1] instead of sigmoid) "
              "flips R² from −0.29 to +0.05, so the fine-tune now beats the mean baseline. "
              "Pearson is unchanged (0.266) — the model always tracked personality; the sigmoid "
              "only inflated the absolute values that MSE penalizes. Comment-level shows Pearson ≈ 0 "
              "(no per-author signal). User-level r's (.15–.33) match the published PANDORA range (~.18–.39).\n")

    out_md = os.path.join(RUN_DIR, "SUMMARY.md")
    with open(out_md, "w") as f:
        f.write("\n".join(md))
    print("wrote", out_md)

    # --- LaTeX (booktabs) table ---
    tex_labels = ["Mean baseline", "vladinc", "Comment-level", "User-level"]
    tex = []
    tex.append("% requires \\usepackage{booktabs}")
    tex.append("\\begin{table}[t]")
    tex.append("\\centering")
    tex.append("\\caption{Regression (Pearson correlation coefficient) on the PANDORA "
               "validation set. Best per row in \\textbf{bold}.}")
    tex.append("\\label{tab:pearson}")
    tex.append("\\begin{tabular}{l" + "c" * n_cols + "}")
    tex.append("\\toprule")
    tex.append("Trait & " + " & ".join(tex_labels) + " \\\\")
    tex.append("\\midrule")
    for ri, r in enumerate(display_rows):
        vals = data_rows[ri]
        best = max(vals)
        cells = " & ".join(("\\textbf{%s}" % fmt(v) if abs(v - best) < 1e-9 else fmt(v))
                           for v in vals)
        if r == "overall":
            tex.append("\\midrule")
            tex.append("Average & " + cells + " \\\\")
        else:
            tex.append(f"{r.capitalize()} & " + cells + " \\\\")
    tex.append("\\bottomrule")
    tex.append("\\end{tabular}")
    tex.append("\\end{table}")

    out_tex = os.path.join(RUN_DIR, "pearson_table.tex")
    with open(out_tex, "w") as f:
        f.write("\n".join(tex) + "\n")
    print("wrote", out_tex)


if __name__ == "__main__":
    main()
