.PHONY: build test test-all format format-check lint typecheck release

build:
	.venv/bin/python -m build --wheel --no-isolation

test:
	.venv/bin/python -m pytest -q -m "not slow"

test-all:
	.venv/bin/python -m pytest -q

format:
	.venv/bin/ruff format src tests examples scripts
	.venv/bin/ruff check --fix src tests examples scripts

format-check:
	.venv/bin/ruff format --check src tests examples scripts
	.venv/bin/ruff check src tests examples scripts

lint:
	.venv/bin/ruff check src tests examples scripts

typecheck:
	.venv/bin/mypy src

release: build test-all format-check lint typecheck

