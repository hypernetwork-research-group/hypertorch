.PHONY: all setup setup-tensorboard check lint typecheck test stest run docs docs-build docs-serve loc clean help

PROJECT_NAME=hyperbench
UV=uv
UVX=uvx
PYTEST=pytest
LINTER=ruff
TYPECHECKER=ty
MKDOCS_CONFIG=.github/mkdocs.yml
MKDOCS_URL=http://127.0.0.1:8000

all: clean setup check test

setup:
	@echo '=== Setup ==='
	$(UV) pip uninstall .
	$(UV) sync
	$(UV) pip install -e .

setup-tensorboard:
	@echo '=== Setup TensorBoard ==='
	$(UV) pip install -e ".[tensorboard]"

check: lint typecheck

lint:
	@echo '=== Linter ==='
	$(UV) run $(LINTER) format

typecheck:
	@echo '=== Type checker ==='
	$(UVX) $(TYPECHECKER) check

test:
	@echo '=== Tests ==='
	$(UV) run $(PYTEST) --cov=$(PROJECT_NAME) --cov-report=term-missing

# If the first argument is stest...
ifeq ($(firstword $(MAKECMDGOALS)),stest)
  # use the rest as arguments for stest...
  STEST_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  # ...and turn them into do-nothing targets
  $(eval .PHONY: $(STEST_ARGS))
  $(eval $(STEST_ARGS): ;@true)
endif

stest:
	@echo '=== Test for $(filter-out $@,$(MAKECMDGOALS)) ==='
	$(UV) run $(PYTEST) -s $(PROJECT_NAME)/tests/$(filter-out $@,$(MAKECMDGOALS))

# If the first argument is run...
ifeq ($(firstword $(MAKECMDGOALS)),run)
  # use the rest as arguments for run...
  RUN_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  # ...and turn them into do-nothing targets
  $(eval .PHONY: $(RUN_ARGS))
  $(eval $(RUN_ARGS): ;@true)
endif

run:
	@echo '=== Run $(filter-out $@,$(MAKECMDGOALS)) ==='
	$(UV) run python3 $(filter-out $@,$(MAKECMDGOALS))

docs: docs-build docs-serve

docs-build:
	@echo '=== Building docs ==='
	$(UV) run mkdocs build -f $(MKDOCS_CONFIG)

docs-serve:
	@echo '=== Serving docs at $(MKDOCS_URL) ==='
	$(UV) run mkdocs serve -f $(MKDOCS_CONFIG)

loc:
	@echo '=== Counting lines of code ==='
	find . -type f -name "*.py" -not -path "*/.venv/*" -exec cat {} + | wc -l

clean:
	@echo '=== Cleaning up ==='
	$(UV) pip uninstall .
	rm -rf **/__pycache__ **/*.pyc $(PROJECT_NAME).egg-info .pytest_cache .coverage .github/site

help:
	@echo "Usage: make [target]"
	@echo "Targets:"
	@echo "  all        	    - Clean, setup, lint, typecheck, test"
	@echo "  setup              - Install dependencies"
	@echo "  setup-tensorboard  - Install optional TensorBoard dependency"
	@echo "  lint               - Run linter"
	@echo "  typecheck          - Run type checker"
	@echo "  test               - Run all tests"
	@echo "  stest <test_name>  - Run a single test"
	@echo "  run <file.py>      - Run a single file"
	@echo "  check              - Run lint and typecheck"
	@echo "  docs               - Build and serve documentation"
	@echo "  docs-build         - Build documentation without serving"
	@echo "  docs-serve         - Serve built documentation locally at $(MKDOCS_URL)"
	@echo "  loc                - Count lines of code"
	@echo "  clean              - Remove build/test artifacts"
