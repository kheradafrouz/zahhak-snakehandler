"""All tunable hyperparameters in one place, plus shared CLI helpers."""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, replace
from typing import Tuple

MODEL_NAME = "Zahhak-snakehandler"


@dataclass
class Config:
    model_name: str = MODEL_NAME

    # --- Environment ---
    grid_w: int = 15
    grid_h: int = 15
    cell_size: int = 40                 # pixels (GUI only)
    starvation_factor: int = 100        # episode ends after factor*len steps without food
    reward_food: float = 10.0
    reward_death: float = -10.0         # also used for starvation timeout
    reward_closer: float = 0.1          # shaping: step toward food
    reward_farther: float = -0.15       # asymmetric so circling is never profitable

    # --- Network / DQN ---
    hidden_sizes: Tuple[int, ...] = (128, 128)
    lr: float = 1e-3
    gamma: float = 0.95
    batch_size: int = 64
    buffer_size: int = 50_000
    warmup_steps: int = 1_000           # no learning until buffer has this many samples
    learn_every: int = 2                # one gradient step per N env steps
    target_update: int = 500            # hard target sync every N gradient steps
    grad_clip: float = 10.0

    # --- Exploration (epsilon decays once per episode) ---
    eps_start: float = 1.0
    eps_end: float = 0.02
    eps_decay: float = 0.997

    # --- Run control ---
    seed: int = 42
    device: str = "auto"                # auto | cpu | cuda | mps
    num_threads: int = 1                # CPU threads (small nets run fastest on 1)
    max_episodes: int = 3000
    save_every_episodes: int = 100
    checkpoint_dir: str = "checkpoints"
    log_every: int = 50
    eval_every: int = 250
    eval_episodes: int = 20

    # --- GUI ---
    default_speed: int = 30             # visual mode: env steps per second
    fast_budget: float = 0.025          # fast mode: seconds of simulation per frame

    def checkpoint_path(self, kind: str = "latest") -> str:
        """e.g. checkpoints/Zahhak-snakehandler_latest.pt"""
        return os.path.join(self.checkpoint_dir, f"{self.model_name}_{kind}.pt")

    @classmethod
    def gpu_preset(cls) -> "Config":
        """Larger network and batches: this is where a GPU actually pays off.
        A 128x128 MLP is so small that CPU is often just as fast."""
        return cls(hidden_sizes=(256, 256), batch_size=512, buffer_size=200_000,
                   learn_every=4, target_update=250, device="auto")


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--hidden", type=int, nargs="+", default=None, help="e.g. --hidden 64 64")
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=None)
    p.add_argument("--preset", choices=["cpu", "gpu"], default="cpu",
                   help="hyperparameter preset (gpu = bigger net/batches)")


def config_from_args(args: argparse.Namespace) -> Config:
    cfg = Config.gpu_preset() if getattr(args, "preset", "cpu") == "gpu" else Config()
    over = {}
    if args.seed is not None:
        over["seed"] = args.seed
    if args.hidden:
        over["hidden_sizes"] = tuple(args.hidden)
    if args.device:
        over["device"] = args.device
    return replace(cfg, **over)
