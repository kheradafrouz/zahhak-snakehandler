import torch

from zahhak_snakehandler.config import Config
from zahhak_snakehandler.train import Trainer


def tiny_cfg(tmp_path) -> Config:
    return Config(hidden_sizes=(16,), buffer_size=500, warmup_steps=50, batch_size=16,
                  learn_every=1, device="cpu", checkpoint_dir=str(tmp_path))


def test_training_runs_and_learns_steps(tmp_path):
    tr = Trainer(tiny_cfg(tmp_path))
    for _ in range(400):
        tr.step()
    assert tr.agent.learn_steps > 0
    assert tr.total_steps == 400


def test_checkpoint_roundtrip(tmp_path):
    cfg = tiny_cfg(tmp_path)
    a = Trainer(cfg)
    for _ in range(400):
        a.step()
    path = cfg.checkpoint_path("latest")
    assert path.endswith("Zahhak-snakehandler_latest.pt")
    a.save(path)

    b = Trainer(cfg)
    b.load(path)
    assert b.episode == a.episode and b.total_steps == a.total_steps
    assert b.agent.epsilon == a.agent.epsilon
    assert len(b.agent.buffer) == len(a.agent.buffer)
    for p, q in zip(a.agent.policy.parameters(), b.agent.policy.parameters()):
        assert torch.equal(p, q)
