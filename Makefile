.PHONY: all setup check lint typecheck test stest docs docs-build docs-serve loc clean help

UV=uv
UVX=uvx
PYTEST=pytest
LINTER=ruff
TYPECHECKER=ty
MKDOCS_CONFIG=.github/mkdocs.yml

all: clean setup check test

setup:
	@echo '=== Setup ==='
	$(UV) pip uninstall .
	$(UV) sync
	$(UV) pip install -e .

check: lint typecheck

lint:
	@echo '=== Linter ==='
	$(UV) run $(LINTER) format

typecheck:
	@echo '=== Type checker ==='
	$(UVX) $(TYPECHECKER) check

test:
	@echo '=== Tests ==='
	$(UV) run $(PYTEST) --cov=hyperbench --cov-report=term-missing

stest:
	@echo '=== Test for $(FILE) ==='
	$(UV) run $(PYTEST) hyperbench/tests/$(FILE) -v -s

docs: docs-build docs-serve

docs-build:
	@echo '=== Building docs ==='
	$(UV) run mkdocs build -f $(MKDOCS_CONFIG)

docs-serve:
	@echo '=== Serving docs at http://127.0.0.1:8000 ==='
	$(UV) run mkdocs serve -f $(MKDOCS_CONFIG)

loc:
	@echo '=== Counting lines of code ==='
	find . -type f -name "*.py" -not -path "*/.venv/*" -exec cat {} + | wc -l

clean:
	@echo '=== Cleaning up ==='
	rm -rf **/__pycache__ **/*.pyc hyperbench.egg-info .pytest_cache .coverage .github/site

help:
    @echo "Usage: make [target]"
    @echo "Targets:"
	@echo "  all        - Setup, lint, typecheck, test"
	@echo "  setup      - Install dependencies"
	@echo "  lint       - Run linter"
	@echo "  typecheck  - Run type checker"
	@echo "  test       - Run tests"
	@echo "  stest      - Run single test"
	@echo "  check      - Run lint and typecheck"
	@echo "  docs       - Build and serve documentation"
	@echo "  docs-build - Build documentation without serving"
	@echo "  docs-serve - Serve built documentation locally at http://127.0.0.1:8000"
	@echo "  loc        - Count lines of code"
	@echo "  clean      - Remove build/test artifacts"
