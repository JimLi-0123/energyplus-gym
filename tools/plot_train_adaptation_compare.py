# tools/plot_train_adaptation_compare.py
from __future__ import annotations

from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt


def find_repo_root(start_file: Path) -> Path:
    """
    Walk upwards from this script location until we find a folder containing 'runs/'.
    This makes the script runnable no matter where it lives (tools/, scripts/, etc.).
    """
    here = start_file.resolve()
    for p in [here.parent, *here.parents]:
        if (p / "runs").exists():
            return p
    raise FileNotFoundError(
        f"Could not find repo root containing 'runs/' when searching from: {start_file}"
    )


def load_metrics(pkl_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # Expect dict-like: {"scores": [...], "eps_history": [...], "steps_array": [...]}
    scores = np.asarray(data.get("scores", []), dtype=float)
    eps = np.asarray(data.get("eps_history", []), dtype=float)
    steps = np.asarray(data.get("steps_array", []), dtype=float)

    if scores.size == 0:
        raise ValueError(f"{pkl_path} has no 'scores' data (empty or missing key).")

    # eps/steps might be empty depending on training script; handle gracefully
    return scores, eps, steps


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    w = min(w, x.size)
    return np.convolve(x, np.ones(w) / w, mode="valid")


def main() -> None:
    repo = find_repo_root(Path(__file__))
    metrics_dir = repo / "runs" / "metrics"
    plots_dir = repo / "runs" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "DDQN-MLP": metrics_dir / "adaptation_ddqn_mlp_metrics.pkl",
        "DDQN-BiLSTM": metrics_dir / "adaptation_ddqn_bilstm_metrics.pkl",
        "Q-Transformer": metrics_dir / "adaptation_q_transformer_metrics.pkl",
    }

    # Sanity check
    missing = [name for name, p in files.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing metrics file(s): "
            + ", ".join(missing)
            + f"\nExpected under: {metrics_dir}"
        )

    # -------------------------
    # Plot 1: Score comparison
    # -------------------------
    plt.figure()
    for name, path in files.items():
        scores, eps, steps = load_metrics(path)
        x = np.arange(1, len(scores) + 1)
        plt.plot(x, scores, label=f"{name} (raw)")

        # Smoothed curve (tune window if you run longer training)
        w = 5
        ma = moving_average(scores, w)
        x_ma = np.arange(w, len(scores) + 1)
        plt.plot(x_ma, ma, linestyle="--", label=f"{name} (MA{w})")

        # Quick numeric summary
        last_n = min(10, len(scores))
        final_mean = float(np.mean(scores[-last_n:]))
        best_ma = float(np.max(moving_average(scores, min(10, len(scores)))))
        print(
            f"[{name}] episodes={len(scores)}  "
            f"final_mean(last{last_n})={final_mean:.2f}  best_MA10={best_ma:.2f}  "
            f"file={path.name}"
        )

    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("Train Adaptation: Score vs Episode (MLP vs BiLSTM vs Q-Transformer)")
    plt.legend()
    plt.tight_layout()
    out_scores = plots_dir / "adaptation_compare_scores.png"
    plt.savefig(out_scores, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_scores}")

    # ----------------------------
    # Plot 2: Epsilon comparison
    # ----------------------------
    plt.figure()
    any_eps = False
    for name, path in files.items():
        scores, eps, steps = load_metrics(path)
        if eps.size == 0:
            print(f"[Warn] {name}: eps_history is empty in {path.name} (skip epsilon plot).")
            continue
        any_eps = True
        x = np.arange(1, len(eps) + 1)
        plt.plot(x, eps, label=name)

    if any_eps:
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")
        plt.title("Train Adaptation: Epsilon vs Episode")
        plt.legend()
        plt.tight_layout()
        out_eps = plots_dir / "adaptation_compare_epsilon.png"
        plt.savefig(out_eps, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_eps}")
    else:
        plt.close()
        print("No epsilon data found in any metrics file; epsilon plot was not generated.")

    print("Done.")


if __name__ == "__main__":
    main()