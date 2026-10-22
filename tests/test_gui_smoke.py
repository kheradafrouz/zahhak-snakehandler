"""Headless smoke test of the Pygame dashboard (SDL dummy video driver)."""
import argparse
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pytest  # noqa: E402

pygame = pytest.importorskip("pygame")

from zahhak_snakehandler.config import Config  # noqa: E402
from zahhak_snakehandler.main import App  # noqa: E402


def test_dashboard_renders_frames(tmp_path):
    cfg = Config(hidden_sizes=(16,), buffer_size=500, warmup_steps=50, batch_size=16,
                 learn_every=1, device="cpu", checkpoint_dir=str(tmp_path))
    args = argparse.Namespace(fast=True, autostart=True, load=None, watch=None)
    app = App(cfg, args)
    try:
        for _ in range(3):
            app.advance(0.05)
            app._draw()
            pygame.display.flip()
        assert app.trainer.total_steps > 0

        app.toggle_fast()          # visual mode
        app.start_eval()           # greedy evaluation episode
        app.advance(0.2)
        app._draw()
        pygame.display.flip()
    finally:
        pygame.quit()
