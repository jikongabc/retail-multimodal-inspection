PYTHON ?= python
PIP ?= $(PYTHON) -m pip
DATA ?= submission/router/training_data.jsonl
SEEDS ?= 7,17,27
GENERATIONS ?= 90
FAST_EVAL_OUTPUT ?= /tmp/retail-router-eval-fast.json

-include .env

REMOTE_HOST ?= root@connect.nmb2.seetacloud.com
REMOTE_PORT ?= 15020
REMOTE_BASE ?= /root/autodl-tmp/workspace
REMOTE_DIR ?= $(REMOTE_BASE)/retail-multimodal-inspection
REMOTE_PYTHON ?= python
REMOTE_VENV ?= $(REMOTE_DIR)/.venv
REMOTE_GPU_VENV ?= $(REMOTE_DIR)/.venv-gpu
REMOTE_MODEL_PATH ?= /root/autodl-tmp/models/Ostrakon-VL-8B
OSTRAKON_PORT ?= 8000
REMOTE_SSH_KEY ?=
REMOTE_SSH := ssh $(if $(REMOTE_SSH_KEY),-i $(REMOTE_SSH_KEY) )-p $(REMOTE_PORT)

.PHONY: help install fixtures fetch-fixtures validate-data train eval eval-fast demos feedback test manifest reproduce verify quality ci format format-check sync-cloud init-cloud init-gpu-cloud run-cloud test-cloud feedback-cloud real-e2e-cloud gpu-cloud shell-cloud clean

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
	@echo "  make sync-cloud     Sync local workspace to the SSH GPU host"
	@echo "  make init-cloud     Create cloud venv and install dependencies"
	@echo "  make init-gpu-cloud Create cloud GPU venv for Ostrakon"
	@echo "  make test-cloud     Sync, then run tests on the SSH GPU host"
	@echo "  make feedback-cloud Sync, then run Task4 feedback experiment on the SSH GPU host"
	@echo "  make real-e2e-cloud Run Task2 -> Task3 -> Ostrakon on the cloud GPU"
	@echo "  make gpu-cloud      Show cloud GPU status"
	@echo "  make shell-cloud    Open a shell in the cloud workspace"

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
	$(PYTHON) -m submission.router.evaluate --seeds "$(SEEDS)" --generations 30 --skip-ablations --output "$(FAST_EVAL_OUTPUT)"

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

sync-cloud:
	$(REMOTE_SSH) $(REMOTE_HOST) "mkdir -p $(REMOTE_DIR)"
	rsync -az --delete \
		--include='.env.example' \
		--exclude='.env*' \
		--exclude='.git/' \
		--exclude='.venv/' \
		--exclude='.venv-gpu/' \
		--exclude='.codex/' \
		--exclude='__pycache__/' \
		--exclude='.pytest_cache/' \
		--exclude='.ruff_cache/' \
		--exclude='*.pyc' \
		-e "$(REMOTE_SSH)" \
		./ $(REMOTE_HOST):$(REMOTE_DIR)/

init-cloud: sync-cloud
	$(REMOTE_SSH) $(REMOTE_HOST) "cd $(REMOTE_DIR) && $(REMOTE_PYTHON) -m venv .venv && . $(REMOTE_VENV)/bin/activate && python -m pip install -r requirements.txt"

init-gpu-cloud: sync-cloud
	$(REMOTE_SSH) $(REMOTE_HOST) "cd $(REMOTE_DIR) && $(REMOTE_PYTHON) -m venv --system-site-packages .venv-gpu && . $(REMOTE_GPU_VENV)/bin/activate && python -m pip install -r requirements-gpu.txt"

run-cloud: sync-cloud
	$(REMOTE_SSH) $(REMOTE_HOST) "cd $(REMOTE_DIR) && . $(REMOTE_VENV)/bin/activate && make verify"

test-cloud: sync-cloud
	$(REMOTE_SSH) $(REMOTE_HOST) "cd $(REMOTE_DIR) && . $(REMOTE_VENV)/bin/activate && make test"

feedback-cloud: sync-cloud
	$(REMOTE_SSH) $(REMOTE_HOST) "cd $(REMOTE_DIR) && . $(REMOTE_VENV)/bin/activate && make feedback"

real-e2e-cloud: sync-cloud
	$(REMOTE_SSH) $(REMOTE_HOST) "cd $(REMOTE_DIR) && . $(REMOTE_GPU_VENV)/bin/activate && python scripts/run_real_e2e.py --model-path $(REMOTE_MODEL_PATH) --port $(OSTRAKON_PORT)"
	mkdir -p submission/pipeline/demos/real_cloud
	rsync -az -e "$(REMOTE_SSH)" $(REMOTE_HOST):$(REMOTE_DIR)/submission/pipeline/demos/real_cloud/ submission/pipeline/demos/real_cloud/

gpu-cloud:
	$(REMOTE_SSH) $(REMOTE_HOST) "nvidia-smi"

shell-cloud:
	$(REMOTE_SSH) $(REMOTE_HOST) "cd $(REMOTE_DIR) && exec bash"

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .ruff_cache
	rm -f submission/MANIFEST.sha256
	rm -f submission/innovation/router_incremental.npy
