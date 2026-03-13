#!/usr/bin/env python3
"""
Model Comparison Script
======================
Compares and contrasts model results across different architectures and hyperparameters.
Each folder = model, each CSV = hyperparameters from filename (lrh, lrf, wd).
Generates visualizations and prints detailed comparison statistics.
"""

import re
import os
from pathlib import Path
from collections import defaultdict

try:
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError as e:
    print("Required packages not found. Install with: pip install pandas matplotlib")
    raise SystemExit(1) from e

# Configuration
RESULTS_DIR = Path(__file__).parent
OUTPUT_DIR = RESULTS_DIR / "visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)

# Filename pattern: {model}__lrh-{lr_head}__lrf-{lr_final}__wd-{weight_decay}__ran-{timestamp}.csv
FILENAME_PATTERN = re.compile(
    r"^(.+?)__lrh-([\de\-]+)__lrf-([\de\-]+)__wd-([\de\-]+)__ran-.+\.csv$",
    re.IGNORECASE
)


def parse_filename(filepath: Path) -> dict | None:
    """Extract model name and hyperparameters from CSV filename."""
    match = FILENAME_PATTERN.match(filepath.name)
    if not match:
        return None
    return {
        "model": match.group(1),
        "lrh": match.group(2),
        "lrf": match.group(3),
        "wd": match.group(4),
    }


def load_all_results() -> pd.DataFrame:
    """Load all CSV files, parse hyperparameters, and combine into one DataFrame."""
    records = []
    for folder in RESULTS_DIR.iterdir():
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        for csv_path in folder.glob("*.csv"):
            if csv_path.name.startswith(".~lock"):
                continue
            params = parse_filename(csv_path)
            if params is None:
                continue
            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"  Warning: Could not read {csv_path}: {e}")
                continue
            df["folder_model"] = folder.name
            df["lrh"] = params["lrh"]
            df["lrf"] = params["lrf"]
            df["wd"] = params["wd"]
            df["config_id"] = f"{params['lrh']}_{params['lrf']}_{params['wd']}"
            records.append(df)
    if not records:
        raise FileNotFoundError(f"No valid CSV files found in {RESULTS_DIR}")
    return pd.concat(records, ignore_index=True)


def get_best_epoch_per_run(df: pd.DataFrame) -> pd.DataFrame:
    """For each run, get the epoch with best mean label AUC on validation."""
    val_labels = df[(df["split"] == "val") & (df["group_type"] == "label")]
    mean_auc = val_labels.groupby(["folder_model", "config_id", "epoch"])["auc"].mean().reset_index()
    best_epochs = mean_auc.loc[mean_auc.groupby(["folder_model", "config_id"])["auc"].idxmax()]
    return best_epochs


def plot_model_comparison_by_label(df: pd.DataFrame) -> None:
    """Bar chart: mean test AUC per label, comparing models (best config each)."""
    best = get_best_epoch_per_run(df)
    test_labels = df[(df["split"] == "test") & (df["group_type"] == "label")]
    merged = test_labels.merge(
        best[["folder_model", "config_id", "epoch"]],
        on=["folder_model", "config_id", "epoch"],
        how="inner"
    )
    pivot = merged.pivot_table(
        index="group_name",
        columns="folder_model",
        values="auc",
        aggfunc="mean"
    )
    pivot = pivot.sort_values(by=pivot.columns[0], ascending=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    pivot.plot(kind="barh", ax=ax, width=0.8)
    ax.set_xlabel("Test AUC")
    ax.set_title("Model Comparison: Test AUC by Label (Best Config per Model)")
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_comparison_by_label.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'model_comparison_by_label.png'}")


def plot_learning_curves(df: pd.DataFrame) -> None:
    """Line plot: validation AUC over epochs for each model (best config)."""
    val_labels = df[(df["split"] == "val") & (df["group_type"] == "label")]
    mean_auc = val_labels.groupby(["folder_model", "config_id", "epoch"])["auc"].mean().reset_index()
    best_configs = mean_auc.groupby("folder_model").apply(
        lambda g: g.loc[g["auc"].idxmax()]
    ).reset_index(drop=True)
    best_config_ids = best_configs[["folder_model", "config_id"]].drop_duplicates()
    plot_data = mean_auc.merge(best_config_ids, on=["folder_model", "config_id"], how="inner")
    fig, ax = plt.subplots(figsize=(10, 6))
    for model in plot_data["folder_model"].unique():
        sub = plot_data[plot_data["folder_model"] == model]
        ax.plot(sub["epoch"], sub["auc"], marker="o", markersize=4, label=model)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Validation AUC (labels)")
    ax.set_title("Learning Curves: Best Config per Model")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "learning_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'learning_curves.png'}")


def plot_hyperparameter_heatmap(df: pd.DataFrame, model_name: str) -> None:
    """Heatmap: mean test AUC vs lrh/lrf for a given model, across WD."""
    model_df = df[(df["folder_model"] == model_name) & (df["split"] == "test") & (df["group_type"] == "label")]
    if model_df.empty:
        return
    best_epoch = get_best_epoch_per_run(df)
    best_model = best_epoch[best_epoch["folder_model"] == model_name]
    merged = model_df.merge(
        best_model[["config_id", "epoch"]],
        left_on=["config_id", "epoch"],
        right_on=["config_id", "epoch"],
        how="inner"
    )
    mean_auc = merged.groupby(["lrh", "lrf", "wd"])["auc"].mean().reset_index()
    pivot = mean_auc.pivot_table(index="lrf", columns="lrh", values="auc", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=0.5, vmax=0.9)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Learning Rate Head (lrh)")
    ax.set_ylabel("Learning Rate Final (lrf)")
    ax.set_title(f"Test AUC by Hyperparameters: {model_name}")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.3f}" if not pd.isna(val) else "N/A", ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax, label="Mean Test AUC")
    plt.tight_layout()
    safe_name = model_name.replace(" ", "_")
    plt.savefig(OUTPUT_DIR / f"heatmap_{safe_name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / f'heatmap_{safe_name}.png'}")


def plot_fairness_comparison(df: pd.DataFrame) -> None:
    """Compare AUC across gender and age bins for best config of each model."""
    best = get_best_epoch_per_run(df)
    test_fairness = df[(df["split"] == "test") & (df["group_type"].isin(["gender", "age_bin"]))]
    merged = test_fairness.merge(
        best[["folder_model", "config_id", "epoch"]],
        on=["folder_model", "config_id", "epoch"],
        how="inner"
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # Gender
    gender_data = merged[merged["group_type"] == "gender"]
    pivot_g = gender_data.pivot_table(index="folder_model", columns="group_name", values="auc", aggfunc="mean")
    pivot_g.plot(kind="bar", ax=axes[0], rot=45)
    axes[0].set_title("Test AUC by Gender")
    axes[0].set_ylabel("AUC")
    axes[0].legend(title="Gender")
    # Age
    age_data = merged[merged["group_type"] == "age_bin"]
    pivot_a = age_data.pivot_table(index="group_name", columns="folder_model", values="auc", aggfunc="mean")
    pivot_a.plot(ax=axes[1], marker="o")
    axes[1].set_title("Test AUC by Age Bin (Best Config per Model)")
    axes[1].set_xlabel("Age Bin")
    axes[1].set_ylabel("AUC")
    axes[1].legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    axes[1].tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fairness_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'fairness_comparison.png'}")


def plot_overall_ranking(df: pd.DataFrame) -> None:
    """Bar chart: overall mean test AUC per model (best config)."""
    best = get_best_epoch_per_run(df)
    test_labels = df[(df["split"] == "test") & (df["group_type"] == "label")]
    merged = test_labels.merge(
        best[["folder_model", "config_id", "epoch"]],
        on=["folder_model", "config_id", "epoch"],
        how="inner"
    )
    ranking = merged.groupby("folder_model")["auc"].mean().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ranking.plot(kind="barh", ax=ax, color="steelblue", edgecolor="navy")
    ax.set_xlabel("Mean Test AUC (labels)")
    ax.set_title("Model Ranking: Best Config per Model")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_ranking.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'model_ranking.png'}")


def print_summary_report(df: pd.DataFrame) -> None:
    """Print detailed comparison statistics to console."""
    best = get_best_epoch_per_run(df)
    test_labels = df[(df["split"] == "test") & (df["group_type"] == "label")]
    merged = test_labels.merge(
        best[["folder_model", "config_id", "epoch"]],
        on=["folder_model", "config_id", "epoch"],
        how="inner"
    )

    print("\n" + "=" * 70)
    print("MODEL COMPARISON REPORT")
    print("=" * 70)

    # Overall ranking
    overall = merged.groupby("folder_model")["auc"].agg(["mean", "std", "min", "max"]).round(4)
    overall = overall.sort_values("mean", ascending=False)
    print("\n--- Overall Mean Test AUC (labels, best config) ---")
    print(overall.to_string())

    # Best config per model
    print("\n--- Best Hyperparameter Config per Model ---")
    for model in df["folder_model"].unique():
        model_best = best[best["folder_model"] == model]
        if model_best.empty:
            continue
        top = model_best.nlargest(1, "auc").iloc[0]
        cfg = merged[(merged["folder_model"] == model) & (merged["config_id"] == top["config_id"])]
        mean_auc = cfg["auc"].mean()
        config_parts = top["config_id"].split("_")
        print(f"  {model}: lrh={config_parts[0]}, lrf={config_parts[1]}, wd={config_parts[2]} -> mean AUC {mean_auc:.4f}")

    # Per-label best model
    print("\n--- Best Model per Label ---")
    label_best = merged.groupby("group_name").apply(
        lambda g: g.loc[g["auc"].idxmax(), ["folder_model", "auc"]]
    )
    for label, row in label_best.iterrows():
        print(f"  {label}: {row['folder_model']} (AUC {row['auc']:.4f})")

    # Hyperparameter summary
    print("\n--- Hyperparameter Grid ---")
    configs = df[["folder_model", "lrh", "lrf", "wd"]].drop_duplicates()
    for model in configs["folder_model"].unique():
        mcfg = configs[configs["folder_model"] == model]
        print(f"  {model}: {len(mcfg)} configs (lrh: {mcfg['lrh'].unique().tolist()}, "
              f"lrf: {mcfg['lrf'].unique().tolist()}, wd: {mcfg['wd'].unique().tolist()})")

    print("\n" + "=" * 70)


def main():
    print("Loading results...")
    df = load_all_results()
    models = df["folder_model"].unique().tolist()
    print(f"Found {len(models)} models: {models}")
    print(f"Total runs: {df.groupby(['folder_model','config_id']).ngroups}")

    print("\nGenerating visualizations...")
    plot_model_comparison_by_label(df)
    plot_learning_curves(df)
    plot_overall_ranking(df)
    plot_fairness_comparison(df)
    for model in models:
        plot_hyperparameter_heatmap(df, model)

    print_summary_report(df)
    print(f"\nAll visualizations saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
