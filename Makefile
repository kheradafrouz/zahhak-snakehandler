.PHONY: help install install-gpu hooks test cov lint build smoke gui train train-gpu watch eval clean
.DEFAULT_GOAL := help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install CPU deps + dev tools
	pip install -r requirements-dev.txt

install-gpu: ## Install CUDA PyTorch + dev tools
	pip install -r requirements-gpu.txt pytest pytest-cov ruff build twine pre-commit

hooks: ## Install pre-commit git hooks
	pre-commit install

test: ## Run the test suite
	python -m pytest -q

cov: ## Run tests with coverage report
	python -m pytest --cov=zahhak_snakehandler --cov-report=term-missing -q

lint: ## Lint with ruff
	ruff check .

build: ## Build and verify sdist + wheel
	python -m build && twine check dist/*

smoke: ## 30-episode headless training + evaluation round trip
	python -m zahhak_snakehandler.train --episodes 30 --seed 1 --device cpu
	python -m zahhak_snakehandler.evaluate --model checkpoints/Zahhak-snakehandler_latest.pt --episodes 10

gui: ## Launch the dashboard (trains live)
	python -m zahhak_snakehandler.main

train: ## Headless training on CPU
	python -m zahhak_snakehandler.train --episodes 3000

train-gpu: ## Headless training on CUDA with the GPU preset
	python -m zahhak_snakehandler.train --preset gpu --device cuda --episodes 3000

watch: ## Watch the best checkpoint play
	python -m zahhak_snakehandler.main --watch checkpoints/Zahhak-snakehandler_best.pt

eval: ## Evaluate best checkpoint vs random baseline
	python -m zahhak_snakehandler.evaluate --model checkpoints/Zahhak-snakehandler_best.pt --episodes 200

clean: ## Remove build and cache artifacts
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .coverage coverage.xml
