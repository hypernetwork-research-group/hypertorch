.PHONY: all build setup setup-tensorboard check format typecheck test stest run docs docs-build docs-serve loc clean destroy help

PROJECT_NAME=hyperbench
UV=uv
PYTEST=pytest
LINTER=ruff
TYPECHECKER=ty
MKDOCS_CONFIG=.github/mkdocs.yml
MKDOCS_URL=http://127.0.0.1:8000

all: clean setup check test

build: clean setup

setup:
	@echo '=== Setup ==='
	$(UV) sync
	$(UV) pip install -e .

setup-tensorboard:
	@echo '=== Setup TensorBoard ==='
	$(UV) pip install -e ".[tensorboard]"

check: format typecheck

format:
	@echo '=== Linter and formatter ==='
	$(UV) run $(LINTER) format

typecheck:
	@echo '=== Type checker ==='
	$(UV) run $(TYPECHECKER) check

test:
	@echo '=== Tests ==='
	$(UV) run $(PYTEST) --cov=$(PROJECT_NAME) --cov-report=term-missing

stest:
	@echo '=== Test for $(T) ==='
	$(UV) run $(PYTEST) -s $(PROJECT_NAME)/tests/$(T)

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
	find . -type f -name "*.py" -not -path "*/.venv/*" | xargs wc -l

clean:
	@echo '=== Cleaning up ==='
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf $(PROJECT_NAME).egg-info .pytest_cache .coverage .$(LINTER)_cache .github/site

destroy: clean
	@echo '=== Destroying environment ==='
	rm -rf .venv $(UV).lock

help:
	@echo "Usage: make [target]"
	@echo "Targets:"
	@echo "  all                  - Clean, setup, lint, typecheck, test"
	@echo "  build                - Clean and setup"
	@echo "  setup                - Install dependencies"
	@echo "  setup-tensorboard    - Install optional TensorBoard dependency"
	@echo "  format               - Run linter and formatter"
	@echo "  typecheck            - Run type checker"
	@echo "  test                 - Run all tests"
	@echo "  stest T=<test_name>  - Run a single test"
	@echo "  run <file.py>        - Run a single file"
	@echo "  check                - Run lint and typecheck"
	@echo "  docs                 - Build and serve documentation"
	@echo "  docs-build           - Build documentation without serving"
	@echo "  docs-serve           - Serve built documentation locally at $(MKDOCS_URL)"
	@echo "  loc                  - Count lines of code"
	@echo "  clean                - Remove build/test artifacts"
	@echo "  destroy              - Destroy the environment"
