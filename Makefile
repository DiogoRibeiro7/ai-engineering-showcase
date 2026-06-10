.PHONY: install test lint typecheck quality demo api docker-build docker-run clean

install:
	poetry install

test:
	poetry run pytest

lint:
	poetry run ruff check src tests

typecheck:
	poetry run mypy src

quality: lint typecheck test

demo:
	poetry run python scripts/run_demo.py

api:
	poetry run uvicorn ai_engineering_showcase.api:create_app --factory --reload

docker-build:
	docker build -t ai-engineering-showcase .

docker-run:
	docker run --rm -p 8000:8000 ai-engineering-showcase

clean:
	rm -rf .artifacts .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage dist build
