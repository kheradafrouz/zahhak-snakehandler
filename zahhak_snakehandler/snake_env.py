"""Snake game rules. No rendering, no ML dependencies; deterministic given a seed."""
import random
from collections import deque
from typing import Deque, Dict, Optional, Set, Tuple

import numpy as np

from zahhak_snakehandler.config import Config

# Directions clockwise: 0=up, 1=right, 2=down, 3=left (screen coords, y grows downward)
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
N_ACTIONS = 3        # 0 = straight, 1 = turn right, 2 = turn left (relative to heading)
STATE_SIZE = 14

Point = Tuple[int, int]


class SnakeEnv:
    def __init__(self, cfg: Config, seed: Optional[int] = None) -> None:
        if cfg.grid_w < 5 or cfg.grid_h < 5:
            raise ValueError("Grid must be at least 5x5.")
        self.cfg = cfg
        self.w, self.h = cfg.grid_w, cfg.grid_h
        self.rng = random.Random(seed)
        self.reset()

    # ------------------------------------------------------------------ API
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng.seed(seed)
        cx, cy = self.w // 2, self.h // 2
        self.direction = 1  # heading right
        self.snake: Deque[Point] = deque([(cx, cy), (cx - 1, cy), (cx - 2, cy)])  # head first
        self.occupied: Set[Point] = set(self.snake)
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.done = False
        self.food: Point = self._place_food()
        return self._state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, int]]:
        if self.done:
            raise RuntimeError("step() called on a finished episode; call reset().")
        if action not in (0, 1, 2):
            raise ValueError(f"Invalid action {action}; expected 0, 1 or 2.")

        if action == 1:
            self.direction = (self.direction + 1) % 4
        elif action == 2:
            self.direction = (self.direction - 1) % 4

        dx, dy = DIRS[self.direction]
        head = self.snake[0]
        new = (head[0] + dx, head[1] + dy)
        eating = new == self.food
        tail = self.snake[-1]
        self.steps += 1
        self.steps_since_food += 1

        hit_wall = not (0 <= new[0] < self.w and 0 <= new[1] < self.h)
        # Moving into the tail cell is legal when not eating (the tail moves away).
        hit_self = new in self.occupied and not (new == tail and not eating)
        if hit_wall or hit_self:
            self.done = True
            return self._state(), self.cfg.reward_death, True, self._info()

        old_dist = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])

        if not eating:  # pop tail BEFORE adding head so tail-chasing keeps occupancy correct
            self.occupied.discard(self.snake.pop())
        self.snake.appendleft(new)
        self.occupied.add(new)

        if eating:
            self.score += 1
            self.steps_since_food = 0
            reward = self.cfg.reward_food
            if len(self.snake) == self.w * self.h:  # board full: perfect game
                self.done = True
            else:
                self.food = self._place_food()
        else:
            new_dist = abs(new[0] - self.food[0]) + abs(new[1] - self.food[1])
            reward = self.cfg.reward_closer if new_dist < old_dist else self.cfg.reward_farther

        if not self.done and self.steps_since_food > self.cfg.starvation_factor * len(self.snake):
            self.done = True  # starvation: prevents endless loops
            reward = self.cfg.reward_death

        return self._state(), reward, self.done, self._info()

    # -------------------------------------------------------------- helpers
    def _info(self) -> Dict[str, int]:
        return {"score": self.score, "steps": self.steps, "length": len(self.snake)}

    def _place_food(self) -> Point:
        free = [(x, y) for x in range(self.w) for y in range(self.h) if (x, y) not in self.occupied]
        return self.rng.choice(free)

    def _blocked(self, p: Point) -> bool:
        x, y = p
        if not (0 <= x < self.w and 0 <= y < self.h):
            return True
        return p in self.occupied and p != self.snake[-1]

    def _state(self) -> np.ndarray:
        hx, hy = self.snake[0]
        d = self.direction
        feats = []
        for rel in (0, 1, 3):  # straight, right, left
            dx, dy = DIRS[(d + rel) % 4]
            feats.append(float(self._blocked((hx + dx, hy + dy))))          # danger 1 step away
            feats.append(float(self._blocked((hx + 2 * dx, hy + 2 * dy))))  # danger 2 steps away
        feats += [float(d == i) for i in range(4)]                          # heading one-hot
        fx, fy = self.food
        feats += [float(fx < hx), float(fx > hx), float(fy < hy), float(fy > hy)]  # food L/R/U/D
        return np.asarray(feats, dtype=np.float32)
