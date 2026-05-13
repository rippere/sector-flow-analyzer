.PHONY: test lint format typecheck coverage install

install:
	uv pip install -e ".[dev]"

test:
	pytest

coverage:
	pytest --cov=sector_flow --cov-report=html

lint:
	ruff check src/ tests/

format:
	black src/ tests/

typecheck:
	mypy src/
