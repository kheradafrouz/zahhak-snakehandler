"""Greedy evaluation and random-baseline comparison (separate env, never touches training state)."""
import argparse
import random
from dataclasses import replace
from typing import Callable, Dict

import numpy as np

from zahhak_snakehandler.agent import DQNAgent, load_agent
from zahhak_snakehandler.config import Config
from zahhak_snakehandler.snake_env import N_ACTIONS, SnakeEnv

Policy = Callable[[np.ndarray], int]


def run_episodes(cfg: Config, policy: Policy, episodes: int, seed: int) -> Dict[str, float]:
    env = SnakeEnv(cfg, seed=seed)
    scores, steps = [], []
    for _ in range(episodes):
        s, done, info = env.reset(), False, {}
        while not done:
            s, _, done, info = env.step(policy(s))
        scores.append(info["score"])
        steps.append(info["steps"])
    return {"avg_score": float(np.mean(scores)), "max_score": int(np.max(scores)),
            "avg_steps": float(np.mean(steps)), "episodes": episodes}


def compare(agent: DQNAgent, cfg: Config, episodes: int = 100, seed: int = 12345) -> Dict[str, dict]:
    rng = random.Random(seed)
    return {
        "trained (greedy, eps=0)": run_episodes(cfg, lambda s: agent.act(s, greedy=True), episodes, seed),
        "random baseline": run_episodes(cfg, lambda s: rng.randrange(N_ACTIONS), episodes, seed),
    }


def print_report(res: Dict[str, dict]) -> None:
    print("\n=== EVALUATION (no exploration; separate from training metrics) ===")
    print(f"{'policy':26s} {'avg score':>10s} {'max score':>10s} {'avg survival':>13s}  episodes")
    for name, r in res.items():
        print(f"{name:26s} {r['avg_score']:10.2f} {r['max_score']:10d} {r['avg_steps']:13.1f}  {r['episodes']}")


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a Zahhak-snakehandler checkpoint vs. a random baseline")
    p.add_argument("--model", required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    args = p.parse_args()
    agent, _ = load_agent(args.model, replace(Config(), device=args.device))
    print_report(compare(agent, agent.cfg, args.episodes, args.seed))


if __name__ == "__main__":
    main()
