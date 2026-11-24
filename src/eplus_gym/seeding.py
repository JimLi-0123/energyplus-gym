import os
import random
import subprocess
import sys

import numpy as np

DEFAULT_SEEDS = (101, 102, 103)
SEED_ENV_VAR = "EPLUS_GYM_SEED"


def seed_suffix(seed: int) -> str:
    return f"_seed{seed}"


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_seeded_script(seeds=DEFAULT_SEEDS) -> int:
    raw_seed = os.environ.get(SEED_ENV_VAR)
    if raw_seed is None:
        for seed in seeds:
            env = os.environ.copy()
            env[SEED_ENV_VAR] = str(seed)
            print(f"\n===== Running seed {seed} =====", flush=True)
            subprocess.run([sys.executable, *sys.argv], check=True, env=env)
        raise SystemExit(0)

    seed = int(raw_seed)
    set_global_seed(seed)
    print(f"[Seed] {seed}", flush=True)
    return seed
