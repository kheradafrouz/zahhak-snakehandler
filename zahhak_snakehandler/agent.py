"""Q-network, replay buffer and Double-DQN agent with CPU/GPU support and full checkpointing."""
import copy
import random
from dataclasses import replace
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from zahhak_snakehandler.config import Config
from zahhak_snakehandler.snake_env import N_ACTIONS, STATE_SIZE


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str = "auto") -> torch.device:
    """'auto' picks CUDA, then Apple MPS, then CPU. Explicit requests fail loudly if unavailable."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Install a CUDA build of PyTorch "
                           "(see requirements-gpu.txt) or use --device cpu.")
    if name == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS requested but not available on this machine.")
    return torch.device(name)


class QNetwork(nn.Module):
    """MLP: STATE_SIZE -> hidden... -> N_ACTIONS (Q-value per action)."""

    def __init__(self, in_dim: int, hidden: Sequence[int], out_dim: int) -> None:
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Fixed-size circular buffer on preallocated numpy arrays (always host memory)."""

    def __init__(self, capacity: int, state_dim: int, seed: int) -> None:
        self.cap = capacity
        self.s = np.zeros((capacity, state_dim), np.float32)
        self.ns = np.zeros((capacity, state_dim), np.float32)
        self.a = np.zeros((capacity, 1), np.int64)
        self.r = np.zeros((capacity, 1), np.float32)
        self.d = np.zeros((capacity, 1), np.float32)
        self.ptr = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def add(self, s, a, r, ns, d) -> None:
        i = self.ptr
        self.s[i], self.a[i], self.r[i], self.ns[i], self.d[i] = s, a, r, ns, float(d)
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, n: int):
        idx = self.rng.integers(0, self.size, n)
        return (torch.from_numpy(self.s[idx]), torch.from_numpy(self.a[idx]),
                torch.from_numpy(self.r[idx]), torch.from_numpy(self.ns[idx]),
                torch.from_numpy(self.d[idx]))

    def state_dict(self) -> dict:
        n = self.size
        out = {k: torch.from_numpy(getattr(self, k)[:n].copy()) for k in ("s", "ns", "a", "r", "d")}
        out["ptr"] = self.ptr
        return out

    def load_state_dict(self, d: dict) -> None:
        n = min(len(d["s"]), self.cap)
        for k in ("s", "ns", "a", "r", "d"):
            getattr(self, k)[:n] = d[k][:n].numpy()
        self.size = n
        self.ptr = int(d["ptr"]) % self.cap if n == self.cap else n % self.cap


class DQNAgent:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.policy = QNetwork(STATE_SIZE, cfg.hidden_sizes, N_ACTIONS).to(self.device)
        self.target = copy.deepcopy(self.policy)
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(cfg.buffer_size, STATE_SIZE, cfg.seed)
        self.rng = random.Random(cfg.seed + 1)   # exploration RNG
        self.epsilon = cfg.eps_start
        self.learn_steps = 0

    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        """Epsilon-greedy; greedy=True disables exploration (used for evaluation)."""
        if not greedy and self.rng.random() < self.epsilon:
            return self.rng.randrange(N_ACTIONS)
        with torch.no_grad():
            q = self.policy(torch.from_numpy(state).unsqueeze(0).to(self.device))
        return int(q.argmax(dim=1).item())

    def remember(self, s, a, r, ns, done) -> None:
        self.buffer.add(s, a, r, ns, done)

    def learn(self) -> Optional[float]:
        cfg = self.cfg
        if len(self.buffer) < max(cfg.batch_size, cfg.warmup_steps):
            return None
        s, a, r, ns, d = (t.to(self.device) for t in self.buffer.sample(cfg.batch_size))
        q = self.policy(s).gather(1, a)
        with torch.no_grad():
            # Double DQN: policy net picks the action, target net evaluates it (less overestimation).
            best = self.policy(ns).argmax(dim=1, keepdim=True)
            nq = self.target(ns).gather(1, best)
            y = r + cfg.gamma * nq * (1.0 - d)
        loss = F.smooth_l1_loss(q, y)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.grad_clip)
        self.optimizer.step()
        self.learn_steps += 1
        if self.learn_steps % cfg.target_update == 0:
            self.target.load_state_dict(self.policy.state_dict())
        return float(loss.item())

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.cfg.eps_end, self.epsilon * self.cfg.eps_decay)

    # ----------------------------------------------------------- checkpointing
    def state_dict(self) -> dict:
        return {"policy": self.policy.state_dict(), "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(), "epsilon": self.epsilon,
                "learn_steps": self.learn_steps, "buffer": self.buffer.state_dict()}

    def load_state_dict(self, d: dict) -> None:
        # Checkpoints are read onto CPU; load_state_dict copies into tensors on self.device.
        self.policy.load_state_dict(d["policy"])
        self.target.load_state_dict(d["target"])
        self.optimizer.load_state_dict(d["optimizer"])
        self.epsilon = float(d["epsilon"])
        self.learn_steps = int(d["learn_steps"])
        if "buffer" in d:
            self.buffer.load_state_dict(d["buffer"])


def read_checkpoint(path: str) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"Checkpoint not found: {path}") from None


def load_agent(path: str, cfg: Config) -> Tuple[DQNAgent, dict]:
    """Build an agent whose architecture matches the checkpoint, then load it.
    Checkpoints are device-portable: train on GPU, evaluate on CPU (or vice versa)."""
    ckpt = read_checkpoint(path)
    cfg = replace(cfg, hidden_sizes=tuple(ckpt["config"]["hidden_sizes"]))
    agent = DQNAgent(cfg)
    agent.load_state_dict(ckpt["agent"])
    return agent, ckpt
