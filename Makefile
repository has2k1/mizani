.PHONY: clean-pyc clean-build doc clean build
BROWSER := python -mwebbrowser

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

clean: clean-build clean-pyc clean-test

clean-build:
	rm -fr build/
	rm -fr dist/
	find . -name '*.egg-info' -exec rm -fr {} +

clean-pyc:
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -f .coverage
	rm -f coverage.xml
	rm -fr htmlcov/

ruff:
	ruff check . $(args)

format:
	ruff format . --check

format-fix:
	ruff format .

lint: ruff

lint-fix:
	make lint args="--fix"

fix: format-fix lint-fix

typecheck:
	pyright

test: clean-test
	pytest --runslow

test-fast: clean-test
	pytest

coverage:
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

doc:
	$(MAKE) -C doc clean
	$(MAKE) -C doc html
	$(BROWSER) doc/_build/html/index.html

release-major:
	@python ./tools/release-checklist.py major

release-minor:
	@python ./tools/release-checklist.py minor

release-patch:
	@python ./tools/release-checklist.py patch

build: clean
	python -m build

dist: build
	ls -l dist

develop: clean-pyc
	pip install -e ".[all]"

install: clean
	ls -l dist
	pip install .
