.DEFAULT_GOAL := help
UV ?= uv

.PHONY: help install lint format types test cov check eval seed serve docker clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies including dev and panel extras
	$(UV) sync --extra dev --extra panel

lint:  ## Lint
	$(UV) run ruff check src tests evals scripts
	$(UV) run ruff format --check src tests evals scripts

format:  ## Auto-format and auto-fix
	$(UV) run ruff format src tests evals scripts
	$(UV) run ruff check --fix src tests evals scripts

types:  ## Type-check (strict)
	$(UV) run mypy src

test:  ## Run the test suite
	$(UV) run pytest

cov:  ## Run tests with a coverage report
	$(UV) run pytest --cov --cov-report=term-missing

check: lint types test  ## Everything CI runs, except the eval gate

eval:  ## Run the eval suite against thresholds (non-zero exit on regression)
	$(UV) run python evals/run_eval.py

seed:  ## Regenerate the synthetic demo cassettes
	$(UV) run python scripts/seed_cassettes.py

serve:  ## Run the HTTP API locally
	$(UV) run scholar-graph serve --reload

docker:  ## Build the container image
	docker build -t scholar-graph .

clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist build .scholar-graph
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
