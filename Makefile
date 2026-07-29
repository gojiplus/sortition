.PHONY: install lint format test test-full check clean

install:
	uv sync --all-extras

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest

test-full:
	SORTITION_FULL_SIMS=1 uv run pytest

check: lint test

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +
