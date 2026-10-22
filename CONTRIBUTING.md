# Contributing to Zahhak-snakehandler

Thanks for your interest. Bug reports, fixes, experiments and docs improvements are all welcome.
By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Ways to contribute
- **Report a bug** using the bug template. Include the exact command, seed, device and versions.
- **Propose a feature** with the feature template. For anything large, open an issue first so we can agree on direction.
- **Improve docs or tests.** These are always appreciated.
- **Run experiments** (new state features, reward shaping, architectures) and share evaluation results.

## Development setup

```bash
git clone https://github.com/<you>/Zahhak-snakehandler.git
cd Zahhak-snakehandler
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt                    # GPU: make install-gpu
pre-commit install                                     # optional but recommended
```

Common tasks (`make help` lists them all):

| Command | Purpose |
|---|---|
| `make test` | Run the test suite |
| `make cov` | Tests with coverage report |
| `make lint` | Ruff lint |
| `make smoke` | 30-episode train + evaluate round trip |
| `make build` | Build and verify the package |

## Workflow

1. Fork, then branch from `main`: `git checkout -b feat/short-description`.
2. Make focused changes. One logical change per pull request.
3. Add or update tests. Bug fixes should include a regression test.
4. Run `make lint test` locally. CI runs the same checks plus multi-OS tests and a smoke run.
5. Update `README.md` and `CHANGELOG.md` (under "Unreleased") when behaviour changes.
6. Open a pull request using the template and keep it up to date with `main`.

### Commit messages
Use [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `perf:`, `ci:`, `chore:`.
Example: `fix(env): allow moving into the tail cell when not eating`.

## Architecture rules

These keep the project easy to reason about. Reviews will enforce them.

- `snake_env.py` must **not** import pygame or torch. Game rules stay pure and deterministic given a seed.
- `Trainer.step()` advances exactly **one** environment step. No rendering, no sleeping, no `pygame.display` calls in training code.
- Rendering and UI live only in `main.py`. The display is updated once per frame, never per simulation step.
- All tunable values belong in `config.py`, not as magic numbers elsewhere.
- Device handling goes through `resolve_device`. Never hardcode `"cuda"`.
- Checkpoints must remain loadable with `weights_only=True` (plain tensors, numbers, strings, lists, dicts only) and portable across devices.
- Don't hardcode solutions or add path-finding. The agent must learn from experience.

## Code style
- Python 3.9+ compatible. Type hints on public functions where they help.
- Lint is enforced by `ruff` (see `pyproject.toml`); line length is 120.
- Comment the *why* for algorithmic decisions, not the *what*.
- Prefer small, readable functions over clever abstractions.

## Tests
- Framework: `pytest`. Tests live in `tests/` and must run on **CPU in seconds**.
- Use tiny configs (small buffer and network) so CI stays fast. See `tests/test_agent.py` for the pattern.
- GUI tests use SDL's dummy video driver (see `tests/test_gui_smoke.py`); no display is required.
- Tests that need CUDA must skip when it is unavailable.

## Changes that affect learning

State, reward, network and hyperparameter changes can look fine and still make the agent worse.
For these, the pull request must include evidence:

1. Train with at least **3 seeds** before and after (`--seed 1`, `--seed 2`, `--seed 3`) using the same episode budget.
2. Report **greedy evaluation** numbers (average score, max score, average survival) from `evaluate.py`, not training metrics.
3. Include the random baseline for context, and note the device, preset and library versions.

## Review and merging
- At least one maintainer approval and a green **CI passed** check are required.
- Maintainers squash-merge, so write a clear PR title in Conventional Commit form.

## Releasing (maintainers)
1. Move "Unreleased" entries in `CHANGELOG.md` under a new version heading.
2. Bump `version` in `pyproject.toml` and `__version__` in `zahhak_snakehandler/__init__.py`.
3. Merge to `main`, then tag: `git tag vX.Y.Z && git push origin vX.Y.Z`.
4. The **Release** workflow builds the package and publishes the GitHub release. It fails if the tag and `pyproject.toml` disagree.

## Security
Do not report vulnerabilities in public issues. See [SECURITY.md](SECURITY.md).
