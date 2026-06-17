.PHONY: install test lint format-check typecheck coverage build quality ci demo api docker-build docker-run compose-prod-up compose-prod-down clean

COVERAGE_THRESHOLD := 63

install:
	poetry install

test:
	poetry run pytest

lint:
	poetry run ruff check .

format-check:
	poetry run ruff format --check .

typecheck:
	poetry run mypy src

coverage:
	poetry run pytest --cov=feedback_intelligence_agent --cov-report=term-missing --cov-fail-under=$(COVERAGE_THRESHOLD)

build:
	poetry build

quality: lint typecheck test

ci: lint format-check typecheck coverage build

demo:
	poetry run python scripts/run_demo.py

api:
	poetry run uvicorn feedback_intelligence_agent.api:create_app --factory --reload

docker-build:
	docker build -t feedback-intelligence-agent .

docker-run:
	docker run --rm -p 8000:8000 feedback-intelligence-agent

# Production-like Docker Compose (built image, gunicorn workers, healthcheck).
# Requires a deploy/.env.prod file (copy from .env.example) and the 'latest'
# image tag, e.g. `make docker-build && docker tag feedback-intelligence-agent feedback-intelligence-agent:latest`.
compose-prod-up:
	docker compose -f deploy/docker-compose.prod.yml up -d

compose-prod-down:
	docker compose -f deploy/docker-compose.prod.yml down

clean:
	rm -rf .artifacts .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage dist build
