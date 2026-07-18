PYTHON ?= python
PIP ?= $(PYTHON) -m pip
DATA ?= submission/router/training_data.jsonl
SEEDS ?= 7,17,27
GENERATIONS ?= 90

.PHONY: help install fixtures fetch-fixtures validate-data train eval eval-fast demos feedback test manifest reproduce verify quality ci format format-check clean

help:
	@echo "Available targets:"
	@echo "  make install        Install runtime dependencies"
	@echo "  make fixtures       Build deterministic local fixtures"
	@echo "  make fetch-fixtures Download real Wikimedia fixtures, then rebuild"
	@echo "  make validate-data  Validate routing data and leakage checks"
	@echo "  make train          Train Task2 router weights"
	@echo "  make eval           Run full Task2 evaluation"
	@echo "  make eval-fast      Run Task2 evaluation without ablations"
	@echo "  make demos          Regenerate Task3 demo outputs"
	@echo "  make feedback       Run Task4 feedback-loop experiment"
	@echo "  make test           Run unit tests"
	@echo "  make manifest       Build submission/MANIFEST.sha256"
	@echo "  make reproduce      Install deps and run README five-step reproduction"
	@echo "  make verify         Run delivery validation without dev-only lint tools"
	@echo "  make quality        Run ruff check and diff whitespace checks"
	@echo "  make ci             Run quality + verify"

install:
	$(PIP) install -r requirements.txt

fixtures:
	$(PYTHON) -m submission.router.build_fixtures

fetch-fixtures:
	$(PYTHON) -m submission.router.fetch_real_fixtures
	$(PYTHON) -m submission.router.build_fixtures

validate-data:
	$(PYTHON) -c 'from submission.router.data_validation import validate_file; import json; print(json.dumps(validate_file("$(DATA)"), ensure_ascii=False, indent=2))'

train: validate-data
	$(PYTHON) -m submission.router.mm_router --mode train

eval: validate-data
	$(PYTHON) -m submission.router.evaluate --seeds "$(SEEDS)" --generations "$(GENERATIONS)"

eval-fast: validate-data
	$(PYTHON) -m submission.router.evaluate --seeds "$(SEEDS)" --generations 30 --skip-ablations

demos:
	$(PYTHON) -m submission.pipeline.demos.run_demos

feedback:
	$(PYTHON) -m submission.innovation.run_feedback_experiment --reset-demo-state

test:
	$(PYTHON) -m unittest discover -v

manifest:
	$(PYTHON) scripts/build_submission_manifest.py

reproduce: install fixtures train demos test

verify: validate-data eval-fast demos feedback test manifest

quality:
	$(PYTHON) -m ruff check .
	git diff --check

ci: quality verify

format:
	$(PYTHON) -m ruff format .

format-check:
	$(PYTHON) -m ruff format --check .

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .ruff_cache
	rm -f submission/MANIFEST.sha256
	rm -f submission/innovation/router_incremental.npy
