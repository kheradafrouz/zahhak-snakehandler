"""Trainer (one env step at a time) + headless fast-training CLI."""
import argparse
import os
import time
from dataclasses import asdict
from typing import List

import numpy as np
import torch

from zahhak_snakehandler.agent import DQNAgent, read_checkpoint, set_seed
from zahhak_snakehandler.config import Config, add_common_args, config_from_args
from zahhak_snakehandler.snake_env import SnakeEnv


def recent_avg(xs: List[float], n: int = 100) -> float:
    return float(np.mean(xs[-n:])) if xs else 0.0


class Trainer:
    """Owns env + agent. step() advances ONE environment step, so any front-end
    (CLI loop or GUI) controls pacing; there is no rendering or sleeping in here."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        set_seed(cfg.seed)
        torch.set_num_threads(cfg.num_threads)
        self.env = SnakeEnv(cfg, seed=cfg.seed)
        self.agent = DQNAgent(cfg)
        self.episode = 0
        self.total_steps = 0
        self.best_score = 0
        self.scores: List[float] = []
        self.rewards: List[float] = []
        self.lengths: List[float] = []
        self.losses: List[float] = []
        self._new_episode()

    def _new_episode(self) -> None:
        self.state = self.env.reset()
        self.ep_reward = 0.0
        self.ep_steps = 0
        self.ep_losses: List[float] = []

    def reset_episode(self) -> None:
        """Abandon the current episode without recording it."""
        self._new_episode()

    def step(self) -> bool:
        """Advance one step. Returns True if an episode just finished."""
        a = self.agent.act(self.state)
        ns, r, done, info = self.env.step(a)
        self.agent.remember(self.state, a, r, ns, done)
        self.total_steps += 1
        self.ep_reward += r
        self.ep_steps += 1
        if self.total_steps % self.cfg.learn_every == 0:
            loss = self.agent.learn()
            if loss is not None:
                self.ep_losses.append(loss)
        self.state = ns
        if not done:
            return False
        self.episode += 1
        self.agent.decay_epsilon()
        self.scores.append(info["score"])
        self.rewards.append(self.ep_reward)
        self.lengths.append(info["steps"])
        self.losses.append(float(np.mean(self.ep_losses)) if self.ep_losses else 0.0)
        self.best_score = max(self.best_score, info["score"])
        self._new_episode()
        return True

    # ----------------------------------------------------------- checkpointing
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        ckpt = {
            "model_name": self.cfg.model_name,
            "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self.cfg).items()},
            "agent": self.agent.state_dict(),   # weights, target, optimizer, epsilon, replay buffer
            "trainer": {"episode": self.episode, "total_steps": self.total_steps,
                        "best_score": self.best_score, "scores": self.scores,
                        "rewards": self.rewards, "lengths": self.lengths, "losses": self.losses},
        }
        tmp = path + ".tmp"
        torch.save(ckpt, tmp)
        os.replace(tmp, path)  # atomic: never leaves a half-written checkpoint

    def load(self, path: str) -> None:
        ckpt = read_checkpoint(path)
        if tuple(ckpt["config"]["hidden_sizes"]) != tuple(self.cfg.hidden_sizes):
            raise ValueError(f"Checkpoint hidden_sizes {ckpt['config']['hidden_sizes']} "
                             f"!= current {list(self.cfg.hidden_sizes)}. Use --hidden to match.")
        self.agent.load_state_dict(ckpt["agent"])
        t = ckpt["trainer"]
        self.episode, self.total_steps, self.best_score = t["episode"], t["total_steps"], t["best_score"]
        self.scores, self.rewards = list(t["scores"]), list(t["rewards"])
        self.lengths, self.losses = list(t["lengths"]), list(t["losses"])
        self._new_episode()


def log_line(tr: Trainer, sps: float) -> str:
    return (f"ep {tr.episode:6d} | steps {tr.total_steps:9d} | eps {tr.agent.epsilon:.3f} | "
            f"avg score {recent_avg(tr.scores):6.2f} | best {tr.best_score:3d} | "
            f"avg reward {recent_avg(tr.rewards):8.2f} | avg len {recent_avg(tr.lengths):7.1f} | "
            f"loss {recent_avg(tr.losses):.4f} | {sps:,.0f} steps/s")


def main() -> None:
    from zahhak_snakehandler.evaluate import compare, print_report, run_episodes

    p = argparse.ArgumentParser(description="Fast headless Zahhak-snakehandler training")
    p.add_argument("--episodes", type=int, default=None, help="episodes to run this session")
    p.add_argument("--resume", type=str, default=None, help="checkpoint to resume from")
    add_common_args(p)
    args = p.parse_args()

    cfg = config_from_args(args)
    n_episodes = args.episodes if args.episodes is not None else cfg.max_episodes

    trainer = Trainer(cfg)
    print(f"{cfg.model_name} | device: {trainer.agent.device} | hidden: {cfg.hidden_sizes} | "
          f"batch: {cfg.batch_size} | seed: {cfg.seed}")
    if args.resume:
        trainer.load(args.resume)
        print(f"Resumed from {args.resume}: episode {trainer.episode}, eps {trainer.agent.epsilon:.3f}")
    latest, best = cfg.checkpoint_path("latest"), cfg.checkpoint_path("best")
    target = trainer.episode + n_episodes
    best_eval = -1.0
    t0, s0 = time.perf_counter(), trainer.total_steps

    try:
        while trainer.episode < target:
            if not trainer.step():
                continue
            ep = trainer.episode
            if ep % cfg.log_every == 0:
                now = time.perf_counter()
                print(log_line(trainer, (trainer.total_steps - s0) / max(now - t0, 1e-9)), flush=True)
                t0, s0 = now, trainer.total_steps
            if ep % cfg.eval_every == 0:
                ev = run_episodes(cfg, lambda s: trainer.agent.act(s, greedy=True),
                                  cfg.eval_episodes, cfg.seed + 999)
                print(f"  [EVAL greedy, {cfg.eval_episodes} eps] avg score {ev['avg_score']:.2f} | "
                      f"max {ev['max_score']} | avg survival {ev['avg_steps']:.1f} steps", flush=True)
                if ev["avg_score"] > best_eval:
                    best_eval = ev["avg_score"]
                    trainer.save(best)
            if ep % cfg.save_every_episodes == 0:
                trainer.save(latest)
    except KeyboardInterrupt:
        print("\nInterrupted; saving checkpoint...")

    trainer.save(latest)
    print(f"Saved {latest}")
    print_report(compare(trainer.agent, cfg, episodes=100, seed=cfg.seed + 2024))


if __name__ == "__main__":
    main()
