# Keel: fresh-clone demo and day-to-day targets (Linux, macOS, Git Bash). Windows users run demo.ps1.
#
#   make demo          create .venv, install, start llama-server, ingest the fixture corpus, start the web app
#   make up / down     start / stop the native stack (deploy/onprem/run.sh, stop.sh)
#   make test          full pytest run           make airgap-test   the egress guard tests only
#   make lint          ruff                      make compose-check parse the compose files

SHELL := /bin/bash
PYTHON ?= python3

ifeq ($(OS),Windows_NT)
VENV_PY := .venv/Scripts/python.exe
else
VENV_PY := .venv/bin/python
endif
export KEEL_PYTHON := $(VENV_PY)

.PHONY: demo venv install up down test airgap-test lint compose-check

demo: install
	@echo "==> Starting llama-server"
	bash deploy/onprem/run.sh --skip-web
	@echo "==> Ingesting the fixture corpus (fixtures/corpus.yaml)"
	$(VENV_PY) -m keel.cli ingest --manifest fixtures/corpus.yaml
	@echo "==> Starting the Keel web app"
	bash deploy/onprem/run.sh
	@echo ""
	@echo "Keel is running: http://127.0.0.1:8400"
	@echo "Try asking: How many quotes does a \$$20,000 purchase need at Northbank Council?"
	@echo "Stop everything with: make down"

venv: $(VENV_PY)

$(VENV_PY):
	@echo "==> Creating .venv with $(PYTHON)"
	$(PYTHON) -m venv .venv

install: $(VENV_PY)
	@echo "==> Installing Keel into .venv (pip install -e .[dev])"
	$(VENV_PY) -m pip install --disable-pip-version-check -q -e ".[dev]"

up:
	bash deploy/onprem/run.sh

down:
	bash deploy/onprem/stop.sh

test:
	$(VENV_PY) -m pytest -q

airgap-test:
	$(VENV_PY) -m pytest -q tests/test_airgap.py

lint:
	$(VENV_PY) -m ruff check keel tests

compose-check:
	$(VENV_PY) -c "import sys, yaml; [yaml.safe_load(open(f)) for f in sys.argv[1:]]; print('compose YAML ok')" \
		deploy/onprem/docker-compose.yml deploy/onprem/docker-compose.gpu.yml
