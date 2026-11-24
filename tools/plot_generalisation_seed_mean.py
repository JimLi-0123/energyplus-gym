from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_CONFIGS = {
    "q_transformer": {
        "label": "Q-transformer",
        "color": "tab:orange",
        "prefix": "eval_generalisation_q_transformer_summary",
        "output_prefix": "eval_generalisation_q_transformer_summary_seed_mean",
    },
    "ddqn_mlp": {
        "label": "MLP",
        "color": "tab:blue",
        "prefix": "eval_generalisation_ddqn_mlp_summary",
        "output_prefix": "eval_generalisation_ddqn_mlp_summary_seed_mean",
    },
    "ddqn_bilstm": {
        "label": "Bi-LSTM",
        "color": "tab:green",
        "prefix": "eval_generalisation_ddqn_bilstm_summary",
        "output_prefix": "eval_generalisation_ddqn_bilstm_summary_seed_mean",
    },
}


def find_repo_root(start_file: Path) -> Path:
    here = start_file.resolve()
    for path in [here.parent, *here.parents]:
        if (path / "runs").exists() and (path / "scripts").exists():
            return path
    raise FileNotFoundError(f"Could not find repo root from: {start_file}")


def annotate_bars(ax, bars, fmt: str) -> None:
    for bar in bars:
        height = bar.get_height()
        offset = 3 if height >= 0 else -12
        va = "bottom" if height >= 0 else "top"
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=9,
        )


def load_seed_summaries(metrics_dir: Path, prefix: str, seeds: list[int]) -> pd.DataFrame:
    frames = []
    for seed in seeds:
        path = metrics_dir / f"{prefix}_seed{seed}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing seed summary CSV: {path}")

        df = pd.read_csv(path)
        df["seed"] = seed
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def aggregate_seed_summaries(raw_df: pd.DataFrame) -> pd.DataFrame:
    excluded_columns = {"seed"}
    metric_columns = [
        column
        for column in raw_df.select_dtypes(include=[np.number]).columns
        if column not in excluded_columns
    ]
    grouped = raw_df.groupby("location", sort=False)[metric_columns]
    mean_df = grouped.mean().reset_index()
    std_df = grouped.std(ddof=1).reset_index()

    for column in metric_columns:
        mean_df[f"{column}_std"] = std_df[column]

    return mean_df


def save_plot(summary_df: pd.DataFrame, plots_dir: Path, output_prefix: str, label: str, color: str) -> Path:
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / f"{output_prefix}.png"

    metric_specs = [
        ("average_score", "(c) Average Score", "Average score", "{:.2f}"),
        ("indoor_violations", "(d) Indoor Violation (occupied periods)", "Indoor violation", "{:.0f}"),
        ("average_energy_mj", "(e) Average Energy Consumption [MJ]", "Average energy consumption", "{:.3f}"),
    ]

    locations = summary_df["location"].tolist()
    x = np.arange(len(locations))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    for ax, (column, title, ylabel, fmt) in zip(axes, metric_specs):
        values = summary_df[column].to_numpy(dtype=float)
        bars = ax.bar(x, values, color=color, label=f"{label} mean")
        annotate_bars(ax, bars, fmt)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(locations, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Average seed1-3 generalisation summaries and plot them.")
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_CONFIGS),
        default="q_transformer",
        help="Which model's seed summaries to average.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Seed numbers to average.",
    )
    parser.add_argument(
        "--input-prefix",
        help="Override the seed CSV prefix under runs/metrics.",
    )
    parser.add_argument(
        "--output-prefix",
        help="Override the output CSV/plot prefix.",
    )
    args = parser.parse_args()

    repo = find_repo_root(Path(__file__))
    metrics_dir = repo / "runs" / "metrics"
    plots_dir = repo / "runs" / "plots"
    config = MODEL_CONFIGS[args.model]
    input_prefix = args.input_prefix or config["prefix"]
    output_prefix = args.output_prefix or config["output_prefix"]

    raw_df = load_seed_summaries(metrics_dir, input_prefix, args.seeds)
    summary_df = aggregate_seed_summaries(raw_df)

    csv_path = metrics_dir / f"{output_prefix}.csv"
    summary_df.to_csv(csv_path, index=False)

    plot_path = save_plot(
        summary_df=summary_df,
        plots_dir=plots_dir,
        output_prefix=output_prefix,
        label=config["label"],
        color=config["color"],
    )

    print(f"Saved CSV: {csv_path}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
