"""Train and evaluate the configured Q-Transformer across the default seeds."""

import os
import subprocess
import sys
from pathlib import Path

from eplus_gym.seeding import SEED_ENV_VAR


def main():
    scripts_dir = Path(__file__).resolve().parent
    env = os.environ.copy()
    env.pop(SEED_ENV_VAR, None)

    for script in (
        "train_generalisation_q_transformer.py",
        "eval_generalisation_q_transformer.py",
    ):
        subprocess.run(
            [sys.executable, str(scripts_dir / script)],
            cwd=scripts_dir.parent,
            env=env,
            check=True,
        )


if __name__ == "__main__":
    main()
