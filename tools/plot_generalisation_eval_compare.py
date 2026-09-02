from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = ["MLP", "Q-transformer", "Bi-LSTM"]

MODEL_COLORS = {
    "MLP": "tab:blue",
    "Q-transformer": "tab:orange",
    "Bi-LSTM": "tab:green",
}

DEFAULT_AGENTS = {
    "MLP": {
        "color": MODEL_COLORS["MLP"],
        "eval": "eval_generalisation_ddqn_mlp_summary.csv",
    },
    "Q-transformer": {
        "color": MODEL_COLORS["Q-transformer"],
        "eval": "eval_generalisation_q_transformer_summary.csv",
    },
    "Bi-LSTM": {
        "color": MODEL_COLORS["Bi-LSTM"],
        "eval": "eval_generalisation_ddqn_bilstm_summary.csv",
    },
}

COLORS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
]


def canonical_model_name(name: str) -> str:
    normalized = name.strip().lower().replace("_", "-").replace(" ", "")
    if "mlp" in normalized:
        return "MLP"
    if "q-transformer" in normalized or "qtransformer" in normalized:
        return "Q-transformer"
    if "bilstm" in normalized or "bi-lstm" in normalized:
        return "Bi-LSTM"
    return name.strip()


def model_sort_key(name: str) -> tuple[int, str]:
    if name in MODEL_ORDER:
        return MODEL_ORDER.index(name), name
    return len(MODEL_ORDER), name


def find_repo_root(start_file: Path) -> Path:
    here = start_file.resolve()
    for p in [here.parent, *here.parents]:
        if (p / "runs").exists() and (p / "scripts").exists():
            return p
    raise FileNotFoundError(f"Could not find repo root from: {start_file}")


def load_eval_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = {"location", "average_score", "indoor_violations", "average_energy_mj"}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df


def parse_agent_arg(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(
            f"Invalid --agent value '{raw}'. Use the form: Model Name=path/to/summary.csv"
        )
    name, path = raw.split("=", 1)
    name = name.strip()
    path = Path(path.strip().strip('"'))
    if not name:
        raise ValueError(f"Invalid --agent value '{raw}': empty model name.")
    return name, path


def collect_agents(
    repo: Path,
    extra_agents: list[str],
    include_defaults: bool = True,
) -> dict[str, dict[str, object]]:
    metrics_dir = repo / "runs" / "metrics"
    agents: dict[str, dict[str, object]] = {}

    if include_defaults:
        for name, info in DEFAULT_AGENTS.items():
            path = metrics_dir / str(info["eval"])
            if path.exists():
                agents[name] = {
                    "path": path,
                    "color": MODEL_COLORS.get(name, info["color"]),
                }

    for raw in extra_agents:
        name, path = parse_agent_arg(raw)
        name = canonical_model_name(name)
        if not path.is_absolute():
            path = (repo / path).resolve()
        agents[name] = {
            "path": path,
            "color": MODEL_COLORS.get(name, COLORS[len(agents) % len(COLORS)]),
        }

    if not agents:
        if include_defaults:
            raise FileNotFoundError(
                f"No eval summary CSV files found under {metrics_dir}, and no --agent CSV was provided."
            )
        raise ValueError("--only-agent requires at least one --agent entry.")

    missing = [info["path"] for info in agents.values() if not Path(info["path"]).exists()]
    if missing:
        pretty = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing eval summary CSV file(s):\n{pretty}")

    return dict(sorted(agents.items(), key=lambda item: model_sort_key(item[0])))


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot generalisation score, indoor violation, energy, and control metrics."
    )
    parser.add_argument(
        "--agent",
        action="append",
        default=[],
        help='Add or override a model summary CSV: "Model Name=path/to/eval_summary.csv"',
    )
    parser.add_argument(
        "--only-agent",
        action="store_true",
        help="Plot only the CSV files specified with --agent; skip default models.",
    )
    parser.add_argument(
        "--summary-output",
        default="generalisation_eval_compare_bars.png",
        help="Score/indoor/energy output filename under runs/plots.",
    )
    parser.add_argument(
        "--control-output",
        default="generalisation_eval_control_compare_bars.png",
        help="Control-metric output filename under runs/plots.",
    )
    args = parser.parse_args()

    repo = find_repo_root(Path(__file__))
    plots_dir = repo / "runs" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    agents = collect_agents(repo, args.agent, include_defaults=not args.only_agent)
    eval_data = {
        name: load_eval_summary(Path(info["path"]))
        for name, info in agents.items()
    }
    locations = next(iter(eval_data.values()))["location"].tolist()

    metric_specs = [
        ("average_score", "(c) Average Score", "Average score", "{:.2f}"),
        ("indoor_violations", "(d) Indoor Violation (occupied periods)", "Indoor violation", "{:.0f}"),
        ("average_energy_mj", "(e) Average Energy Consumption [MJ]", "Average energy consumption", "{:.3f}"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    x = np.arange(len(locations))
    width = min(0.8 / max(len(eval_data), 1), 0.24)
    center = (len(eval_data) - 1) / 2

    for ax, (column, title, ylabel, fmt) in zip(axes, metric_specs):
        for i, (name, df) in enumerate(eval_data.items()):
            values = df.set_index("location").loc[locations, column].to_numpy(dtype=float)
            bars = ax.bar(
                x + (i - center) * width,
                values,
                width=width,
                color=str(agents[name]["color"]),
                label=name,
            )
            annotate_bars(ax, bars, fmt)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(locations, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()

    fig.tight_layout()
    out = plots_dir / args.summary_output
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    control_specs = [
        ("ahu_switching_count_command", "AHU Switching (command)", "Switch count", "{:.0f}"),
        ("ahu_switching_count_actual", "AHU Switching (actual state)", "Switch count", "{:.0f}"),
        ("short_cycle_count", "Short Cycles", "Count", "{:.0f}"),
        ("sat_avg_abs_change_occupied", "SAT Avg Abs Change (occupied)", "C / step", "{:.3f}"),
        ("oa_flow_avg_abs_change_occupied", "OA Flow Avg Abs Change (occupied)", "kg/s / step", "{:.3f}"),
        ("action_proxy_energy_cost", "Action Proxy Energy Cost", "Proxy cost", "{:.3f}"),
    ]
    required_control_columns = {column for column, *_ in control_specs}
    available_columns = set().union(*(set(df.columns) for df in eval_data.values()))
    if not required_control_columns.issubset(available_columns):
        missing = sorted(required_control_columns.difference(available_columns))
        print(
            "Control comparison plot skipped because summary CSV files are missing columns: "
            + ", ".join(missing)
        )
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.2))
    for ax, (column, title, ylabel, fmt) in zip(axes.ravel(), control_specs):
        for i, (name, df) in enumerate(eval_data.items()):
            values = df.set_index("location").loc[locations, column].to_numpy(dtype=float)
            bars = ax.bar(
                x + (i - center) * width,
                values,
                width=width,
                color=str(agents[name]["color"]),
                label=name,
            )
            annotate_bars(ax, bars, fmt)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(locations, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()

    fig.tight_layout()
    out = plots_dir / args.control_output
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
