"""
Phase 3g: Final model comparison — summarise all 5 models in one plot and table.

Produces:
  - reports/model_comparison.png  (bar chart + heatmap)
  - reports/final_comparison.csv  (consolidated metrics table)

Run: python src/models/compare_models.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

REPORTS_DIR = Path("reports")

# ── Load all individual metrics CSVs ──────────────────────────────────────
# Each model script saved its own metrics CSV — we combine them here
# rather than re-running models, keeping comparison fast and reproducible

def load_all_metrics() -> pd.DataFrame:
    """Load and combine metrics from all 5 model runs."""

    # ARIMA and Prophet: per-state metrics
    arima   = pd.read_csv(REPORTS_DIR / "arima_metrics.csv")[["state","model","rmse","mae","mape"]]
    prophet = pd.read_csv(REPORTS_DIR / "prophet_metrics.csv")[["state","model","rmse","mae","mape"]]
    rf      = pd.read_csv(REPORTS_DIR / "rf_metrics.csv")[["state","model","rmse","mae","mape"]]
    xgb     = pd.read_csv(REPORTS_DIR / "xgboost_metrics.csv")[["state","model","rmse","mae","mape"]]

    # TFT: single overall metric (not per-state) — broadcast to all states
    # so it appears in the comparison table alongside the others
    tft_raw = pd.read_csv(REPORTS_DIR / "tft_metrics.csv")
    tft_rows = []
    for state in ["NSW", "VIC", "QLD", "SA", "WA"]:
        tft_rows.append({
            "state": state,
            "model": "TFT",
            "rmse":  tft_raw["rmse"].values[0],
            "mae":   tft_raw["mae"].values[0],
            "mape":  tft_raw["mape"].values[0],
        })
    tft = pd.DataFrame(tft_rows)

    # Combine all into one long-format DataFrame
    combined = pd.concat([arima, prophet, rf, xgb, tft], ignore_index=True)

    # Standardise model name for TFT (csv has 'TFT', others have full names)
    combined["model"] = combined["model"].replace({
        "RandomForest": "Random Forest",
    })

    return combined


def plot_mape_heatmap(df: pd.DataFrame):
    """Heatmap of MAPE by state and model — clearest way to compare all 25 combos."""
    # Pivot to state × model matrix
    pivot = df.pivot(index="state", columns="model", values="mape")

    # Order models from simplest to most complex
    model_order = ["ARIMA", "Prophet", "Random Forest", "XGBoost", "TFT"]
    state_order = ["NSW", "VIC", "QLD", "SA", "WA"]
    pivot = pivot.reindex(index=state_order, columns=model_order)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # ── Heatmap ────────────────────────────────────────────────────────────
    ax = axes[0]
    # Green = good (low MAPE), Red = bad (high MAPE)
    im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto",
                   vmin=0, vmax=pivot.values.max())

    # Labels
    ax.set_xticks(range(len(model_order)))
    ax.set_xticklabels(model_order, rotation=20, ha="right", fontsize=10)
    ax.set_yticks(range(len(state_order)))
    ax.set_yticklabels(state_order, fontsize=10)
    ax.set_title("MAPE (%) by State and Model\n(Green = better, Red = worse)",
                 fontsize=12, fontweight="bold")

    # Annotate each cell with the MAPE value
    for i, state in enumerate(state_order):
        for j, model in enumerate(model_order):
            val = pivot.loc[state, model]
            # White text on dark cells, black on light
            color = "white" if val > pivot.values.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.2f}%",
                    ha="center", va="center",
                    fontsize=9, fontweight="bold", color=color)

    plt.colorbar(im, ax=ax, label="MAPE (%)")

    # ── Bar chart: average MAPE per model ─────────────────────────────────
    ax2 = axes[1]
    avg_mape = df.groupby("model")["mape"].mean().reindex(model_order)
    bar_colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336"]

    bars = ax2.bar(model_order, avg_mape.values, color=bar_colors, width=0.6)

    # Annotate bars with average MAPE
    for bar, val in zip(bars, avg_mape.values):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.1,
                 f"{val:.2f}%",
                 ha="center", va="bottom",
                 fontsize=10, fontweight="bold")

    ax2.set_title("Average MAPE Across All States\n(lower = better)",
                  fontsize=12, fontweight="bold")
    ax2.set_ylabel("Average MAPE (%)")
    ax2.set_ylim(0, avg_mape.max() * 1.2)
    ax2.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = REPORTS_DIR / "model_comparison.png"
    plt.savefig(path, dpi=120)
    print(f"Saved comparison plot to {path}")
    plt.close()

    return pivot, avg_mape


def print_final_summary(df: pd.DataFrame, pivot: pd.DataFrame, avg_mape: pd.Series):
    """Print the final comparison table for the report."""
    model_order = ["ARIMA", "Prophet", "Random Forest", "XGBoost", "TFT"]
    state_order = ["NSW", "VIC", "QLD", "SA", "WA"]

    print("\n" + "="*65)
    print("FINAL MODEL COMPARISON — MAPE (%) BY STATE")
    print("="*65)
    print(pivot.reindex(index=state_order, columns=model_order).round(2).to_string())

    print("\n" + "="*65)
    print("AVERAGE MAPE ACROSS ALL STATES (lower = better)")
    print("="*65)
    for model in model_order:
        marker = " ← best overall" if model == avg_mape.idxmin() else ""
        print(f"  {model:<15}: {avg_mape[model]:.2f}%{marker}")

    print("\n" + "="*65)
    print("BEST MODEL PER STATE")
    print("="*65)
    for state in state_order:
        best_model = pivot.loc[state].idxmin()
        best_mape  = pivot.loc[state].min()
        print(f"  {state}: {best_model} ({best_mape:.2f}%)")

    print("\n" + "="*65)
    print("KEY FINDINGS FOR REPORT")
    print("="*65)
    print("  1. ARIMA is best for stable markets (NSW: 0.86%, VIC: 1.63%)")
    print("  2. Random Forest wins for fast-growing markets (QLD: 1.37%)")
    print("  3. XGBoost edges RF on WA's volatile market (2.67% vs 2.94%)")
    print("  4. Prophet underperforms on quarterly data (designed for daily/weekly)")
    print("  5. TFT requires more data than 59 quarters to outperform classical ML")
    print("     — consistent with literature: deep learning needs 1000s of time steps")


def run_comparison():
    """Main: load all metrics, plot comparison, print summary."""
    print("Loading metrics from all model runs...")
    df = load_all_metrics()

    print("Generating comparison plot...")
    pivot, avg_mape = plot_mape_heatmap(df)

    print_final_summary(df, pivot, avg_mape)

    # Save consolidated CSV
    output_path = REPORTS_DIR / "final_comparison.csv"
    df.to_csv(output_path, index=False)
    print(f"\nConsolidated metrics saved to {output_path}")


if __name__ == "__main__":
    run_comparison()