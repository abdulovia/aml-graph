# AML-Graph developer tasks.
# Local dev is Intel Homebrew Python under Rosetta; the venv holds the
# local-only workarounds (torch==2.2.2, polars[rtcompat]). Docker/CI use the
# clean pins in requirements.txt.

VENV    := .venv
PY      := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip
RUFF    := $(VENV)/bin/ruff
PYTEST  := $(VENV)/bin/pytest

.PHONY: help setup lint test experiment demo api figures clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Create the venv and install dependencies
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

lint: ## Ruff lint + format check
	$(RUFF) check .
	$(RUFF) format --check .

test: ## Run the test suite
	$(PYTEST) tests -q

experiment: ## Run the full MVP pipeline (data -> models -> metrics -> figures)
	$(PY) -c "from src import pipeline; pipeline.run_mvp(use_llm=False)"

figures: ## Regenerate figures + metrics from the pipeline
	$(PY) -c "from src import pipeline; pipeline.run_mvp(use_llm=False)"

demo: ## Launch the Streamlit demo (Docker if available, else local)
	@if command -v docker >/dev/null 2>&1; then \
		docker compose up demo; \
	else \
		$(VENV)/bin/streamlit run app/streamlit_app.py; \
	fi

api: ## Launch the FastAPI service on :8000
	$(VENV)/bin/uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload

clean: ## Remove caches and generated artifacts (keeps committed figures/metrics)
	rm -rf .pytest_cache .ruff_cache mlruns
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -f data/processed/edges_scored.parquet
