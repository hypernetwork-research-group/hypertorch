.PHONY: all setup check lint typecheck test docs docs-serve clean help

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

docs:
	@echo '=== Building docs ==='
	$(UV) run mkdocs build -f $(MKDOCS_CONFIG)

docs-serve:
	@echo '=== Serving docs at http://127.0.0.1:8000 ==='
	$(UV) run mkdocs serve -f $(MKDOCS_CONFIG)

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
	@echo "  check      - Run lint and typecheck"
	@echo "  docs       - Build documentation"
	@echo "  docs-serve - Serve docs locally at http://127.0.0.1:8000"
	@echo "  clean      - Remove build/test artifacts"
