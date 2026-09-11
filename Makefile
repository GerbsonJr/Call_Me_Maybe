SHELL := /bin/bash

BLUE := \033[1;34m
GREEN := \033[1;32m
YELLOW := \033[1;33m
RED := \033[1;31m
RESET := \033[0m

.PHONY: install run debug clean fclean lint lint-strict help re

install:
	@printf "$(BLUE)▶ Installing dependencies...$(RESET)\n"
	@uv sync
	@printf "$(GREEN)✔ Installation complete$(RESET)\n"

run:
	@printf "$(BLUE)▶ Running project...$(RESET)\n"
	@uv run python -m src

debug:
	@printf "$(YELLOW)▶ Starting debugger...$(RESET)\n"
	@uv run python -m pdb src/__main__.py

clean:
	@printf "$(YELLOW)▶ Cleaning caches...$(RESET)\n"
	@find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	@find . -type d -name ".mypy_cache" -prune -exec rm -rf {} +
	@find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
	@printf "$(GREEN)✔ Cache cleanup done$(RESET)\n"

fclean: clean
	@printf "$(YELLOW)▶ Removing virtual environment and outputs...$(RESET)\n"
	@rm -rf .venv
	@rm -f data/output/function_calling_results.json
	@printf "$(GREEN)✔ Full cleanup done$(RESET)\n"

lint:
	@printf "$(BLUE)▶ Running lint checks...$(RESET)\n"
	@uv run flake8 .
	@uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
	@printf "$(GREEN)✔ Lint passed$(RESET)\n"

re: fclean
	@printf "$(BLUE)▶ Installing dependencies...$(RESET)\n"
	@uv sync
	@printf "$(GREEN)✔ Installation complete$(RESET)\n"

help:
	@printf "$(BLUE)Available targets:$(RESET)\n"
	@printf "  make install      - install dependencies\n"
	@printf "  make run          - run the project\n"
	@printf "  make debug        - run in debugger\n"
	@printf "  make clean        - remove caches\n"
	@printf "  make fclean       - remove caches, .venv and output\n"
	@printf "  make lint         - run flake8 and mypy\n"
	@printf "  make lint-strict  - run strict checks\n"
	@printf "  make help         - show this help\n"