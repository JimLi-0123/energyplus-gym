from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


SERIES = [
    (
        "MLP baseline",
        "tab:blue",
        "baseline_refresh_seed101_103/"
        "eval_generalisation_ddqn_mlp_summary_seed101_103_mean.csv",
    ),
    (
        "Bi-LSTM baseline",
        "tab:green",
        "baseline_refresh_seed101_103/"
        "eval_generalisation_ddqn_bilstm_summary_seed101_103_mean.csv",
    ),
    (
        "Ver.7\nCO2/IAT 0.5/0.5 + aux",
        "tab:orange",
        "eval_generalisation_q_transformer_summary_"
        "co2_iat_gate_05_05_aux_control_penalty_off_seed_mean_from_docx.csv",
    ),
    (
        "Ver.8\nCO2 1.0 + aux",
        "tab:red",
        "eval_generalisation_q_transformer_summary_"
        "co2_iat_gate_10_00_aux_control_penalty_off_seed_mean_from_docx.csv",
    ),
    (
        "Ver.9\nIAT 1.0 no aux",
        "tab:purple",
        "eval_generalisation_q_transformer_summary_"
        "co2_iat_gate_00_10_aux_control_penalty_off_seed_mean_from_docx.csv",
    ),
    (
        "Ver.10\nCO2 1.0 no aux",
        "tab:brown",
        "eval_generalisation_q_transformer_summary_"
        "co2_iat_gate_10_00_no_aux_control_penalty_off_seed_mean_from_docx.csv",
    ),
    (
        "Ver.11\nQ-Transformer baseline",
        "tab:gray",
        "eval_generalisation_q_transformer_summary_"
        "baseline_control_penalty_off_seed_mean.csv",
    ),
]


METRICS = [
    (
        "total_co2_violations",
        "Total CO2 Violations (lower is better)",
        "CO2 Violations",
        "{:.0f}",
        "total_co2_violations_seed101_103.png",
    ),
    (
        "average_score",
        "Average Score (higher is better)",
        "Average Score",
        "{:.2f}",
        "average_score_seed101_103.png",
    ),
    (
        "total_iat_violations",
        "Total IAT Violations (lower is better)",
        "IAT Violations",
        "{:.0f}",
        "total_iat_violations_seed101_103.png",
    ),
    (
        "average_energy_mj",
        "Average Energy Consumption [MJ] (lower is better)",
        "Energy [MJ]",
        "{:.2f}",
        "average_energy_consumption_seed101_103.png",
    ),
    (
        "indoor_violations",
        "Indoor Violations (occupied periods, lower is better)",
        "Indoor Violations",
        "{:.0f}",
        "indoor_violations_seed101_103.png",
    ),
]


def find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for path in [here.parent, *here.parents]:
        if (path / "runs" / "metrics").exists():
            return path
    raise FileNotFoundError("Could not locate the repository root.")


def load_five_city_means(metrics_dir: Path) -> pd.DataFrame:
    records = []
    for label, color, relative_path in SERIES:
        path = metrics_dir / relative_path
        frame = pd.read_csv(path)
        if len(frame) != 5:
            raise ValueError(f"Expected five cities in {path}, found {len(frame)}")
        record = {"label": label, "color": color}
        for column, *_ in METRICS:
            record[column] = float(frame[column].mean())
        records.append(record)
    return pd.DataFrame(records)


def save_metric_plot(
    summary: pd.DataFrame,
    output_dir: Path,
    column: str,
    title: str,
    ylabel: str,
    value_format: str,
    filename: str,
) -> None:
    fig, ax = plt.subplots(figsize=(14.2, 6.7))
    bars = ax.bar(summary["label"], summary[column], color=summary["color"])
    ax.set_title(title, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, summary[column]):
        if value >= 0:
            xytext = (0, 4)
            va = "bottom"
        else:
            xytext = (0, -16)
            va = "top"
        ax.annotate(
            value_format.format(value),
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=xytext,
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=11,
        )

    ax.margins(x=0.05)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    repo = find_repo_root()
    metrics_dir = repo / "runs" / "metrics"
    output_dir = repo / "runs" / "plots" / "version_overview_refresh_seed101_103"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = load_five_city_means(metrics_dir)
    summary.to_csv(output_dir / "five_city_means_seed101_103.csv", index=False)

    for metric in METRICS:
        save_metric_plot(summary, output_dir, *metric)

    print(summary.to_string(index=False))
    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    main()
