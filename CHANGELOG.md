# Changelog

All notable changes to this project are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning: [SemVer](https://semver.org/).

## [Unreleased]

## [1.1.0]
### Added
- Logo and banner (`assets/`), redesigned README, detailed CONTRIBUTING guide.
- CODE_OF_CONDUCT, SECURITY policy, issue forms, pull request template, `.editorconfig`, pre-commit config.
- CI: ruff lint, multi-OS/multi-Python test matrix with coverage, package build + `twine check`,
  end-to-end smoke job, aggregate `CI passed` gate, CodeQL, tag-driven release workflow, manual GPU workflow, Dependabot.
- Tests for evaluation, config/device handling and a headless GUI smoke test.
- `make help`, `lint`, `cov`, `build`, `smoke`, `clean` targets.

## [1.0.0]
### Added
- Double-DQN Snake agent, Pygame dashboard, fast headless training, greedy evaluation vs random baseline,
  CPU and GPU (CUDA / MPS) support, resumable checkpoints.

[Unreleased]: https://github.com/YOUR_USERNAME/Zahhak-snakehandler/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/YOUR_USERNAME/Zahhak-snakehandler/releases/tag/v1.1.0
[1.0.0]: https://github.com/YOUR_USERNAME/Zahhak-snakehandler/releases/tag/v1.0.0
