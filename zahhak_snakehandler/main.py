"""Pygame dashboard. Visual mode renders the board at an adjustable steps/sec;
fast mode runs simulation in a time budget per frame and only draws the stats panel."""
import argparse
import time
from typing import Callable, List, Optional, Tuple

import numpy as np
import pygame

from zahhak_snakehandler.agent import load_agent
from zahhak_snakehandler.config import Config, add_common_args, config_from_args
from zahhak_snakehandler.evaluate import compare
from zahhak_snakehandler.snake_env import SnakeEnv
from zahhak_snakehandler.train import Trainer, recent_avg

TRAINING, PAUSED, EVALUATING, FINISHED = "Training", "Paused", "Evaluating", "Finished"
BG, PANEL, PANEL2 = (18, 20, 26), (28, 31, 40), (22, 24, 31)
GRID, TEXT, DIM = (30, 33, 42), (225, 228, 235), (140, 146, 160)
GREEN, HEAD, RED, BLUE, ORANGE = (60, 190, 110), (130, 240, 160), (230, 70, 80), (90, 160, 255), (240, 170, 60)
STATUS_COLOR = {TRAINING: GREEN, PAUSED: ORANGE, EVALUATING: BLUE, FINISHED: DIM}


class App:
    def __init__(self, cfg: Config, args: argparse.Namespace) -> None:
        self.cfg = cfg
        self.running = True
        self.trainer = Trainer(cfg)
        self.ckpt_path = cfg.checkpoint_path("latest")
        self.message = f"Ready. Device: {self.trainer.agent.device}"
        self.bench_text = "Benchmark: press B to compare greedy policy vs random."
        self.fast = args.fast
        self.speed = cfg.default_speed
        self.accum = 0.0
        self.episode_limit = cfg.max_episodes
        self.eval_env: Optional[SnakeEnv] = None
        self.eval_state: Optional[np.ndarray] = None
        self.eval_count = 0
        self.loop_eval = False
        self.status = TRAINING if args.autostart else PAUSED
        self.sps, self._sps_t, self._sps_s = 0.0, time.perf_counter(), 0

        if args.load:
            self.load(args.load)
        if args.watch:
            self.load(args.watch)
            self.fast = False
            self.start_eval(loop=True)

        pygame.init()
        pygame.display.set_caption(f"{cfg.model_name} - Snake DQN")
        self.board_px = cfg.cell_size * cfg.grid_w
        self.panel_w = 400
        self.bar_h = 44
        self.W = self.board_px + self.panel_w
        self.H = cfg.cell_size * cfg.grid_h + self.bar_h
        self.screen = pygame.display.set_mode((self.W, self.H))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas,menlo,dejavusansmono,monospace", 16)
        self.big = pygame.font.SysFont("consolas,menlo,dejavusansmono,monospace", 22, bold=True)
        self.buttons = self._build_buttons()

    # ----------------------------------------------------------------- actions
    def toggle(self) -> None:
        if self.status == TRAINING:
            self.status = PAUSED
        elif self.status == PAUSED:
            self.status = TRAINING
        elif self.status == EVALUATING:
            self.loop_eval = False
            self.status = PAUSED
        elif self.status == FINISHED:
            self.episode_limit = self.trainer.episode + self.cfg.max_episodes
            self.status = TRAINING

    def reset_episode(self) -> None:
        self.trainer.reset_episode()
        self.message = "Current training episode reset (not recorded)."

    def toggle_fast(self) -> None:
        self.fast = not self.fast
        self.accum = 0.0

    def faster(self) -> None:
        self.speed = min(2048, self.speed * 2)

    def slower(self) -> None:
        self.speed = max(1, self.speed // 2)

    def save(self) -> None:
        try:
            self.trainer.save(self.ckpt_path)
            self.message = f"Saved {self.ckpt_path} (episode {self.trainer.episode})."
        except Exception as e:  # noqa: BLE001 - show any I/O error in the UI
            self.message = f"Save failed: {e}"

    def load(self, path: Optional[str] = None) -> None:
        path = path or self.ckpt_path
        try:
            self.trainer.load(path)
            self.eval_env = None
            if self.status == EVALUATING:
                self.status = PAUSED
            self.message = f"Loaded {path} (episode {self.trainer.episode})."
        except Exception as e:  # noqa: BLE001
            self.message = f"Load failed: {e}"

    def start_eval(self, loop: bool = False) -> None:
        self.eval_count += 1
        self.eval_env = SnakeEnv(self.cfg, seed=self.cfg.seed + 777 + self.eval_count)
        self.eval_state = self.eval_env.reset()
        self.loop_eval = loop
        self.status = EVALUATING
        self.accum = 0.0

    def benchmark(self) -> None:
        self.message = "Running benchmark..."
        self._draw()
        pygame.display.flip()
        res = compare(self.trainer.agent, self.cfg, episodes=self.cfg.eval_episodes * 2, seed=self.cfg.seed + 4242)
        t, r = res["trained (greedy, eps=0)"], res["random baseline"]
        self.bench_text = (f"EVAL greedy: avg {t['avg_score']:.1f} max {t['max_score']} len {t['avg_steps']:.0f} | "
                           f"RANDOM: avg {r['avg_score']:.1f} max {r['max_score']} len {r['avg_steps']:.0f}")
        self.message = "Benchmark done (evaluation, not training, numbers)."

    def _build_buttons(self) -> List[Tuple[pygame.Rect, Callable[[], str], Callable[[], None]]]:
        specs = [
            (lambda: {TRAINING: "Pause [SPACE]", PAUSED: "Start/Resume [SPACE]",
                      EVALUATING: "Stop eval [SPACE]", FINISHED: "Continue [SPACE]"}[self.status], self.toggle),
            (lambda: "Reset ep [R]", self.reset_episode),
            (lambda: "Mode: FAST [F]" if self.fast else "Mode: VISUAL [F]", self.toggle_fast),
            (lambda: "Watch greedy [E]", lambda: self.start_eval(False)),
            (lambda: "Speed + [UP]", self.faster),
            (lambda: "Speed - [DOWN]", self.slower),
            (lambda: "Save [S]", self.save),
            (lambda: "Load [L]", self.load),
            (lambda: "Benchmark [B]", self.benchmark),
            (lambda: "Quit [ESC]", self.quit),
        ]
        pad, bw, bh = 10, (self.panel_w - 30) // 2, 32
        out = []
        for i, (label, fn) in enumerate(specs):
            r = pygame.Rect(self.board_px + pad + (i % 2) * (bw + pad), 190 + (i // 2) * (bh + 6), bw, bh)
            out.append((r, label, fn))
        return out

    def quit(self) -> None:
        self.running = False

    # -------------------------------------------------------------- simulation
    def _train_steps(self) -> None:
        tr = self.trainer
        if self.fast:
            t0 = time.perf_counter()
            n = 0
            while self.status == TRAINING:
                tr.step()
                n += 1
                if tr.episode >= self.episode_limit:
                    self._finish()
                elif n % 32 == 0 and time.perf_counter() - t0 > self.cfg.fast_budget:
                    break
        else:
            n = min(int(self.accum), 500)
            self.accum = 0.0 if n == 500 else self.accum - n
            for _ in range(n):
                tr.step()
                if tr.episode >= self.episode_limit:
                    self._finish()
                    break

    def _finish(self) -> None:
        self.status = FINISHED
        self.save()
        self.message = f"Finished {self.trainer.episode} episodes. Checkpoint saved."

    def _eval_steps(self) -> None:
        for _ in range(min(int(self.accum), 500)):
            self.accum -= 1
            a = self.trainer.agent.act(self.eval_state, greedy=True)
            self.eval_state, _, done, info = self.eval_env.step(a)
            if done:
                self.message = f"Eval episode (greedy): score {info['score']}, survived {info['steps']} steps."
                if self.loop_eval:
                    self.start_eval(loop=True)
                else:
                    self.status = PAUSED
                break
        self.accum = min(self.accum, 500)

    def advance(self, dt: float) -> None:
        self.accum += self.speed * dt
        if self.status == TRAINING:
            self._train_steps()
        elif self.status == EVALUATING:
            self._eval_steps()
        now = time.perf_counter()
        if now - self._sps_t >= 1.0:
            self.sps = (self.trainer.total_steps - self._sps_s) / (now - self._sps_t)
            self._sps_t, self._sps_s = now, self.trainer.total_steps

    # ---------------------------------------------------------------- drawing
    def _text(self, s: str, x: int, y: int, color=TEXT, font=None) -> None:
        self.screen.blit((font or self.font).render(s, True, color), (x, y))

    def _draw_board(self, env: SnakeEnv) -> None:
        c = self.cfg.cell_size
        pygame.draw.rect(self.screen, BG, (0, 0, self.board_px, self.board_px))
        for i in range(self.cfg.grid_w + 1):
            pygame.draw.line(self.screen, GRID, (i * c, 0), (i * c, self.board_px))
            pygame.draw.line(self.screen, GRID, (0, i * c), (self.board_px, i * c))
        fx, fy = env.food
        pygame.draw.circle(self.screen, RED, (fx * c + c // 2, fy * c + c // 2), c // 2 - 4)
        for i, (x, y) in enumerate(env.snake):
            col = HEAD if i == 0 else GREEN
            pygame.draw.rect(self.screen, col, (x * c + 2, y * c + 2, c - 4, c - 4), border_radius=6)

    def _draw_chart(self, rect: pygame.Rect) -> None:
        pygame.draw.rect(self.screen, PANEL2, rect, border_radius=6)
        data = np.asarray(self.trainer.scores[-200:], dtype=np.float32)
        self._text("Training score (last 200 eps)", rect.x + 8, rect.y + 4, DIM)
        if len(data) < 2:
            return
        top, bot = rect.y + 26, rect.bottom - 8
        ymax = max(float(data.max()), 1.0)
        xs = np.linspace(rect.x + 8, rect.right - 8, len(data))

        def pts(v: np.ndarray):
            return [(float(x), float(bot - (y / ymax) * (bot - top))) for x, y in zip(xs[-len(v):], v)]

        pygame.draw.lines(self.screen, (70, 90, 130), False, pts(data), 1)
        if len(data) >= 20:
            ma = np.convolve(data, np.ones(20) / 20, mode="valid")
            pygame.draw.lines(self.screen, ORANGE, False, pts(ma), 2)
        self._text(f"max {ymax:.0f}", rect.right - 80, rect.y + 4, DIM)

    def _draw(self) -> None:
        tr, ag = self.trainer, self.trainer.agent
        self.screen.fill(BG)
        if self.status == EVALUATING and self.eval_env is not None:
            self._draw_board(self.eval_env)
        elif self.fast and self.status == TRAINING:
            pygame.draw.rect(self.screen, BG, (0, 0, self.board_px, self.board_px))
            self._text("FAST MODE", 20, self.board_px // 2 - 30, GREEN, self.big)
            self._text("board rendering off - simulating at full speed", 20, self.board_px // 2 + 4, DIM)
        else:
            self._draw_board(tr.env)
        px = self.board_px
        pygame.draw.rect(self.screen, PANEL, (px, 0, self.panel_w, self.board_px))
        self._text(self.status.upper(), px + 12, 8, STATUS_COLOR[self.status], self.big)
        if self.status == EVALUATING and self.eval_env is not None:
            score_line = f"Eval score {self.eval_env.score}  (greedy)"
        else:
            score_line = f"Score {tr.env.score}   Train best {tr.best_score}"
        mode = "FAST" if self.fast else f"VISUAL {self.speed}/s"
        loss = recent_avg(tr.losses, 50)
        lines = [
            f"Episode {tr.episode}/{self.episode_limit}   Steps {tr.total_steps}",
            score_line,
            f"Reward {tr.ep_reward:7.1f}   Avg100 {recent_avg(tr.rewards):7.1f}",
            f"Avg score(100) {recent_avg(tr.scores):.2f}",
            f"Epsilon {ag.epsilon:.3f}   Loss {loss:.4f}",
            f"Mode {mode}   Dev {ag.device.type}",
            f"Sim speed {self.sps:,.0f} steps/s",
        ]
        for i, ln in enumerate(lines):
            self._text(ln, px + 12, 40 + i * 20)
        mouse = pygame.mouse.get_pos()
        for rect, label, _ in self.buttons:
            pygame.draw.rect(self.screen, (55, 60, 78) if rect.collidepoint(mouse) else (42, 46, 60),
                             rect, border_radius=6)
            self._text(label(), rect.x + 8, rect.y + 7)
        self._draw_chart(pygame.Rect(px + 10, 190 + 5 * 38 + 6, self.panel_w - 20, self.board_px - (190 + 5 * 38 + 16)))
        pygame.draw.rect(self.screen, PANEL2, (0, self.board_px, self.W, self.bar_h))
        self._text(self.message[:110], 10, self.board_px + 4, DIM)
        self._text(self.bench_text[:120], 10, self.board_px + 24, ORANGE)

    # ------------------------------------------------------------------- loop
    def run(self) -> None:
        keymap = {pygame.K_SPACE: self.toggle, pygame.K_r: self.reset_episode, pygame.K_f: self.toggle_fast,
                  pygame.K_UP: self.faster, pygame.K_DOWN: self.slower, pygame.K_s: self.save,
                  pygame.K_l: self.load, pygame.K_e: lambda: self.start_eval(False),
                  pygame.K_b: self.benchmark, pygame.K_ESCAPE: self.quit}
        try:
            while self.running:
                dt = self.clock.tick(30 if (self.fast and self.status == TRAINING) else 60) / 1000.0
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        self.running = False
                    elif ev.type == pygame.KEYDOWN and ev.key in keymap:
                        keymap[ev.key]()
                    elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                        for rect, _, fn in self.buttons:
                            if rect.collidepoint(ev.pos):
                                fn()
                                break
                self.advance(min(dt, 0.1))
                self._draw()
                pygame.display.flip()   # once per FRAME, never per simulation step
        finally:
            if self.trainer.total_steps > 0:
                try:
                    self.trainer.save(self.ckpt_path)
                except Exception:  # noqa: BLE001
                    pass
            pygame.quit()


def main() -> None:
    p = argparse.ArgumentParser(description="Zahhak-snakehandler dashboard")
    p.add_argument("--fast", action="store_true", help="start in fast (no board rendering) mode")
    p.add_argument("--no-autostart", dest="autostart", action="store_false", help="start paused")
    p.add_argument("--load", type=str, default=None, help="checkpoint to resume training from")
    p.add_argument("--watch", type=str, default=None, help="checkpoint: watch greedy policy play in a loop")
    add_common_args(p)
    args = p.parse_args()
    cfg = config_from_args(args)
    if args.watch:  # match architecture to the checkpoint
        cfg = load_agent(args.watch, cfg)[0].cfg
    App(cfg, args).run()


if __name__ == "__main__":
    main()
