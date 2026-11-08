# Shortcuts for the pipeline, notebooks and quality checks.
# Recipes only call Python, so they work from cmd, PowerShell, Git Bash and Unix shells.
# Run `make` or `make help` to list the targets.

ifeq ($(OS),Windows_NT)
VENV_PYTHON := .venv/Scripts/python.exe
else
VENV_PYTHON := .venv/bin/python
endif

# Interpreter used to create the virtual environment.
BASE_PYTHON ?= python
# Project interpreter: the virtual environment if it exists, otherwise `python` on PATH.
PYTHON ?= $(if $(wildcard $(VENV_PYTHON)),"$(VENV_PYTHON)",python)
# Package installer. With a uv environment use: make install PIP="uv pip"
PIP ?= $(PYTHON) -m pip
# Notebook for `make notebook`.
NB ?= notebooks/03_mlp_experiments.ipynb

export PYTHONIOENCODING := utf-8

.DEFAULT_GOAL := help
.PHONY: help venv install audit eda preprocess train features evaluate all \
	notebooks notebook test lint format check clean

help: ## List the available targets
	@$(PYTHON) -c "import re; [print(f'  make {m[1]:<12} {m[2]}') for m in re.finditer(r'(?m)^([a-z-]+):.*?## (.*)', open('Makefile', encoding='utf-8').read())]"

venv: ## Create the .venv virtual environment
	$(BASE_PYTHON) -m venv .venv

install: ## Install the package with dev dependencies
	$(PIP) install -e ".[dev]"

audit: ## Schema and quality audit, feature contract, battery split
	$(PYTHON) -m ev_battery audit

eda: ## Train-only exploratory analysis
	$(PYTHON) -m ev_battery eda

preprocess: ## Fit preprocessing recipes on train and write model inputs
	$(PYTHON) -m ev_battery preprocess

train: ## Staged MLP study on train and validation
	$(PYTHON) -m ev_battery train

features: ## Feature-engineering ablation on validation
	$(PYTHON) -m ev_battery features

evaluate: ## Score the locked checkpoints once on the test split
	$(PYTHON) -m ev_battery evaluate

all: ## Run every pipeline stage in order
	$(PYTHON) -m ev_battery all

notebooks: ## Execute notebooks 00 to 05 in order
	$(PYTHON) scripts/run_notebook.py --all

notebook: ## Execute one notebook, e.g. make notebook NB=notebooks/01_eda.ipynb
	$(PYTHON) scripts/run_notebook.py $(NB)

test: ## Run the test suite
	$(PYTHON) -m pytest

lint: ## Check linting and formatting without changing files
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format: ## Format the code and apply safe lint fixes
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

check: lint test ## Run lint and tests

clean: ## Remove Python and tool caches (results and models are kept)
	@$(PYTHON) -c "import pathlib, shutil; names = {'__pycache__', '.pytest_cache', '.ruff_cache', '.ipynb_checkpoints'}; dirs = [p for p in pathlib.Path('.').rglob('*') if p.is_dir() and p.name in names and '.venv' not in p.parts]; [shutil.rmtree(p, ignore_errors=True) for p in dirs]; print(f'Removed {len(dirs)} cache directories.')"
