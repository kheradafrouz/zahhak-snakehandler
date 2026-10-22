import random

from zahhak_snakehandler.config import Config
from zahhak_snakehandler.evaluate import compare, run_episodes
from zahhak_snakehandler.snake_env import N_ACTIONS
from zahhak_snakehandler.train import Trainer


def tiny_cfg(tmp_path) -> Config:
    return Config(hidden_sizes=(16,), buffer_size=500, warmup_steps=50, batch_size=16,
                  learn_every=1, device="cpu", checkpoint_dir=str(tmp_path))


def test_run_episodes_is_deterministic():
    cfg = Config()
    a = run_episodes(cfg, lambda s: random.Random(0).randrange(N_ACTIONS), 5, seed=3)
    b = run_episodes(cfg, lambda s: random.Random(0).randrange(N_ACTIONS), 5, seed=3)
    assert a == b


def test_compare_reports_both_policies(tmp_path):
    tr = Trainer(tiny_cfg(tmp_path))
    res = compare(tr.agent, tr.cfg, episodes=3, seed=1)
    assert set(res) == {"trained (greedy, eps=0)", "random baseline"}
    for r in res.values():
        assert r["episodes"] == 3 and r["avg_steps"] > 0
