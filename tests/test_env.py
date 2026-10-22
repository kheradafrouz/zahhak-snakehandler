import numpy as np
import pytest

from zahhak_snakehandler.config import Config
from zahhak_snakehandler.snake_env import N_ACTIONS, STATE_SIZE, SnakeEnv


def test_reset_state_shape():
    env = SnakeEnv(Config(), seed=1)
    s = env.reset()
    assert s.shape == (STATE_SIZE,) and s.dtype == np.float32
    assert len(env.snake) == 3 and env.food not in env.occupied


def test_invalid_action_raises():
    env = SnakeEnv(Config(), seed=1)
    with pytest.raises(ValueError):
        env.step(3)


def test_step_after_done_raises():
    env = SnakeEnv(Config(), seed=1)
    done = False
    while not done:
        _, _, done, _ = env.step(0)  # straight until a wall (or starvation) ends it
    with pytest.raises(RuntimeError):
        env.step(0)


def test_wall_collision_ends_episode_with_death_reward():
    cfg = Config()
    env = SnakeEnv(cfg, seed=1)
    env.food = (0, 0)  # keep food out of the straight path
    r, done = 0.0, False
    for _ in range(cfg.grid_w):
        _, r, done, _ = env.step(0)
        if done:
            break
    assert done and r == cfg.reward_death


def test_determinism_with_seed():
    a, b = SnakeEnv(Config(), seed=7), SnakeEnv(Config(), seed=7)
    for act in [0, 1, 0, 2, 0, 0, 1, 1, 2, 0]:
        if a.done or b.done:
            break
        sa, ra, da, _ = a.step(act)
        sb, rb, db, _ = b.step(act)
        assert np.array_equal(sa, sb) and ra == rb and da == db


def test_actions_constant():
    assert N_ACTIONS == 3
