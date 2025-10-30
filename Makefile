.PHONY: clean-pyc clean-build doc clean build

# NOTE: Take care not to use tabs in any programming flow outside the
# make target

# Use uv (if it is installed) to run all python related commands,
# and prefere the active environment over .venv in a parent folder
ifeq ($(OS),Windows_NT)
  HAS_UV := $(if $(shell where uv 2>NUL),true,false)
else
  HAS_UV := $(if $(shell command -v uv 2>/dev/null),true,false)
endif

ifeq ($(HAS_UV),true)
  PYTHON ?= uv run --active python
  PIP ?= uv pip
  UVRUN ?= uv run --active
else
  PYTHON ?= python
  PIP ?= pip
  UVRUN ?=
endif

BROWSER := $(PYTHON) -mwebbrowser

all:
	@echo "Using Python: $(PYTHON)"

help:
	@echo "clean - remove all build, test, coverage and Python artifacts"
	@echo "clean-build - remove build artifacts"
	@echo "clean-pyc - remove Python file artifacts"
	@echo "clean-test - remove test and coverage artifacts"
	@echo "lint - check style with flake8"
	@echo "test - run tests quickly with the default Python"
	@echo "coverage - check code coverage quickly with the default Python"
	@echo "doc - generate Sphinx HTML documentation, including API docs"
	@echo "release - package and upload a release"
	@echo "dist - package"
	@echo "install - install the package to the active Python's site-packages"

clean: clean-build clean-cache clean-test

clean-build:
	rm -fr build/
	rm -fr dist/
	find . -name '*.egg-info' -exec rm -fr {} +

clean-cache:
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	$(UVRUN) coverage erase
	rm -f .coverage
	rm -f coverage.xml
	rm -fr htmlcov/

ruff:
	ruff check . $(args)

format:
	$(UVRUN) ruff format --check .

format-fix:
	$(UVRUN) ruff format .

lint:
	$(UVRUN) ruff check .

lint-fix:
	$(UVRUN) ruff check --fix .

fix: format-fix lint-fix

typecheck:
	$(UVRUN) pyright

test: clean-test
	$(UVRUN) pytest --runslow

test-fast: clean-test
	$(UVRUN) pytest

coverage:
	$(UVRUN) coverage report -m
	$(UVRUN) coverage html
	$(BROWSER) htmlcov/index.html


doc:
	$(MAKE) -C doc clean
	$(MAKE) -C doc html
	$(BROWSER) doc/_build/html/index.html

release-major:
	@$(PYTHON) ./tools/release-checklist.py major

release-minor:
	@$(PYTHON) ./tools/release-checklist.py minor

release-patch:
	@$(PYTHON) ./tools/release-checklist.py patch

dist: clean-build
	$(PYTHON) -m build
	ls -l dist

build: dist

install: clean
	ls -l dist
	$(PIP) install .

develop: clean-cache
	$(PIP) install -e ".[all]"

develop-update: clean-cache
	$(PIP) install --upgrade -e ".[all]"
	$(UVRUN) pre-commit autoupdate
