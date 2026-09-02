from __future__ import annotations

from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


AGENTS = {
    "MLP": {
        "color": "tab:blue",
        "train": "generalisation_ddqn_mlp_metrics.pkl",
        "eval": "eval_generalisation_ddqn_mlp_summary.csv",
    },
    "Q-transformer": {
        "color": "tab:orange",
        "train": "generalisation_q_transformer_metrics.pkl",
        "eval": "eval_generalisation_q_transformer_summary.csv",
    },
    "Bi-LSTM": {
        "color": "tab:green",
        "train": "generalisation_ddqn_bilstm_metrics.pkl",
        "eval": "eval_generalisation_ddqn_bilstm_summary.csv",
    },
}


def find_repo_root(start_file: Path) -> Path:
    here = start_file.resolve()
    for p in [here.parent, *here.parents]:
        if (p / "runs").exists() and (p / "scripts").exists():
            return p
    raise FileNotFoundError(f"Could not find repo root from: {start_file}")


def load_train_metrics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    scores = np.asarray(data.get("scores", []), dtype=float)
    eps = np.asarray(data.get("eps_history", []), dtype=float)
    if scores.size == 0:
        raise ValueError(f"{path} has no scores data.")
    return scores, eps


def load_eval_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"location", "average_score", "indoor_violations", "average_energy_mj"}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df


def annotate_bars(ax, bars, fmt: str):
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
            fontsize=8,
        )


def check_inputs(metrics_dir: Path):
    missing = []
    for info in AGENTS.values():
        for key in ("train", "eval"):
            path = metrics_dir / info[key]
            if not path.exists():
                missing.append(path)
    if missing:
        pretty = "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(
            "Missing generalisation metrics. Run the three train scripts and three eval scripts first:\n"
            f"{pretty}"
        )


def main() -> None:
    repo = find_repo_root(Path(__file__))
    metrics_dir = repo / "runs" / "metrics"
    plots_dir = repo / "runs" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    check_inputs(metrics_dir)

    train_data = {}
    eval_data = {}
    for name, info in AGENTS.items():
        train_data[name] = load_train_metrics(metrics_dir / info["train"])
        eval_data[name] = load_eval_summary(metrics_dir / info["eval"])

    locations = eval_data["MLP"]["location"].tolist()

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1], hspace=0.42, wspace=0.28)

    ax_train = fig.add_subplot(gs[0, :])
    ax_eps = ax_train.twinx()

    eps_plotted = False
    for name, (scores, eps) in train_data.items():
        x = np.arange(1, len(scores) + 1)
        ax_train.plot(x, scores, label=name, color=AGENTS[name]["color"], linewidth=1.8)
        if not eps_plotted and eps.size:
            ax_eps.plot(
                np.arange(1, len(eps) + 1),
                eps,
                color="black",
                linestyle="--",
                linewidth=2,
                label="Epsilon",
            )
            eps_plotted = True

    ax_train.set_title("(b) Scores and Epsilon Decay Over Episodes in Generalisation Test")
    ax_train.set_xlabel("Episode")
    ax_train.set_ylabel("Score")
    ax_eps.set_ylabel("Epsilon")
    ax_train.grid(alpha=0.3)
    lines, labels = ax_train.get_legend_handles_labels()
    eps_lines, eps_labels = ax_eps.get_legend_handles_labels()
    ax_train.legend(lines + eps_lines, labels + eps_labels, loc="upper center", ncol=4)

    metric_specs = [
        ("average_score", "(c) Average Score", "Average score", "{:.2f}"),
        ("indoor_violations", "(d) Indoor Violation (occupied periods)", "Indoor violation", "{:.0f}"),
        ("average_energy_mj", "(e) Average Energy Consumption [MJ]", "Average energy consumption", "{:.3f}"),
    ]

    x = np.arange(len(locations))
    width = 0.24
    offsets = [-width, 0, width]

    for ax_idx, (column, title, ylabel, fmt) in enumerate(metric_specs):
        ax = fig.add_subplot(gs[1, ax_idx])
        for offset, (name, df) in zip(offsets, eval_data.items()):
            values = df.set_index("location").loc[locations, column].to_numpy(dtype=float)
            bars = ax.bar(
                x + offset,
                values,
                width=width,
                color=AGENTS[name]["color"],
                label=name,
            )
            annotate_bars(ax, bars, fmt)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(locations, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()

    out = plots_dir / "generalisation_compare_paper_style.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
