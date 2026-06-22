"""
Phase 2: Exploratory Data Analysis
Produces price trend plots saved to reports/
Run: python src/data/eda.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

DATA_PATH = Path("data/processed/dwelling_prices_clean.csv")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

STATE_COLORS = {
    "NSW": "#1f77b4",
    "VIC": "#ff7f0e",
    "QLD": "#2ca02c",
    "SA":  "#d62728",
    "WA":  "#9467bd",
}


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return df


def plot_price_trends(df: pd.DataFrame):
    """Plot mean dwelling price over time per state."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for state, group in df.groupby("state"):
        ax.plot(
            group["date"], group["mean_price_aud"],
            label=state, color=STATE_COLORS.get(state),
            linewidth=2
        )

    ax.set_title("Mean Residential Dwelling Price by State (AUD)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Mean Price (AUD $000s)")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x:,.0f}k")
    )
    ax.legend(title="State")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = REPORTS_DIR / "price_trends.png"
    plt.savefig(path, dpi=120)
    print(f"Saved: {path}")
    plt.close()


def plot_yoy_changes(df: pd.DataFrame):
    """Plot year-on-year % change per state."""
    df_yoy = df.dropna(subset=["yoy_change_pct"])
    fig, ax = plt.subplots(figsize=(12, 6))

    for state, group in df_yoy.groupby("state"):
        ax.plot(
            group["date"], group["yoy_change_pct"],
            label=state, color=STATE_COLORS.get(state),
            linewidth=2
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Year-on-Year Price Change by State (%)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Quarter")
    ax.set_ylabel("YoY Change (%)")
    ax.legend(title="State")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = REPORTS_DIR / "yoy_changes.png"
    plt.savefig(path, dpi=120)
    print(f"Saved: {path}")
    plt.close()


def plot_latest_snapshot(df: pd.DataFrame):
    """Bar chart of latest mean price per state — student-friendly comparison."""
    latest = df.sort_values("date").groupby("state").last().reset_index()
    latest = latest.sort_values("mean_price_aud", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(
        latest["state"], latest["mean_price_aud"],
        color=[STATE_COLORS.get(s) for s in latest["state"]]
    )

    # Annotate bars with price + YoY
    for bar, (_, row) in zip(bars, latest.iterrows()):
        ax.text(
            bar.get_width() + 10,
            bar.get_y() + bar.get_height() / 2,
            f"${row['mean_price_aud']:,.0f}k  ({row['yoy_change_pct']:+.1f}% YoY)",
            va="center", fontsize=9
        )

    ax.set_title("Latest Mean Dwelling Price by State (2026-Q1)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean Price (AUD $000s)")
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x:,.0f}k")
    )
    ax.set_xlim(0, latest["mean_price_aud"].max() * 1.35)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    path = REPORTS_DIR / "latest_snapshot.png"
    plt.savefig(path, dpi=120)
    print(f"Saved: {path}")
    plt.close()


def print_summary_stats(df: pd.DataFrame):
    """Print key stats useful for the project report."""
    print("\n=== Summary Statistics ===")
    latest = df.sort_values("date").groupby("state").last()
    print(latest[["mean_price_aud", "qoq_change_pct", "yoy_change_pct"]]
          .round(2).to_string())

    print("\n=== Most affordable state (latest) ===")
    cheapest = latest["mean_price_aud"].idxmin()
    print(f"  {cheapest}: ${latest.loc[cheapest, 'mean_price_aud']:,.0f}k")

    print("\n=== Fastest growing state (YoY) ===")
    fastest = latest["yoy_change_pct"].idxmax()
    print(f"  {fastest}: {latest.loc[fastest, 'yoy_change_pct']:+.1f}%")


if __name__ == "__main__":
    df = load()
    print(f"Loaded {len(df)} rows across {df['state'].nunique()} states")

    plot_price_trends(df)
    plot_yoy_changes(df)
    plot_latest_snapshot(df)
    print_summary_stats(df)

    print("\nAll EDA plots saved to reports/")