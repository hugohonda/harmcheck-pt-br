.PHONY: lint fmt type-check check test install

install:
	uv sync --all-groups

lint:
	uv run ruff check .

fmt:
	uv run ruff format .
	uv run ruff check --fix .

type-check:
	uv run mypy src/ --ignore-missing-imports

check: lint type-check

test:
	uv run pytest
