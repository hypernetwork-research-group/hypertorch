.PHONY: all build setup setup-tensorboard clean destroy \
		test stest run \
		check format typecheck lint lint-fix lint-rule lint-rule-fix \
		docs docs-build docs-serve \
		loc help

PROJECT_NAME=hyperbench
UV=uv
PYTEST=pytest
LINTER=ruff
TYPECHECKER=ty
ZENSICAL_CONFIG=zensical.toml
DOCS_ADDR=127.0.0.1:8000

all: clean setup check test

build: clean setup

setup:
	@echo '=== Setup ==='
	$(UV) sync
	$(UV) pip install -e .

setup-tensorboard:
	@echo '=== Setup TensorBoard ==='
	$(UV) pip install -e ".[tensorboard]"

check: lint format typecheck

format:
	@echo '=== Formatting ==='
	$(UV) run $(LINTER) format

typecheck:
	@echo '=== Type checking ==='
	$(UV) run $(TYPECHECKER) check

lint:
	@echo '=== Linting ==='
	$(UV) run $(LINTER) check

lint-fix:
	@echo '=== Linting with fix ==='
	$(UV) run $(LINTER) check --fix

lint-rule:
	@echo '=== Linting a single rule ==='
	$(UV) run $(LINTER) check --select $(R)

lint-rule-fix:
	@echo '=== Linting a single rule with fix ==='
	$(UV) run $(LINTER) check --select $(R) --fix

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
	@echo '=== Running $(filter-out $@,$(MAKECMDGOALS)) ==='
	$(UV) run python3 $(filter-out $@,$(MAKECMDGOALS))

docs: docs-build docs-serve

docs-build:
	@echo '=== Building docs ==='
	$(UV) run zensical build --clean -f $(ZENSICAL_CONFIG)

docs-serve:
	@echo '=== Serving docs at http://$(DOCS_ADDR) ==='
	$(UV) run zensical serve -f $(ZENSICAL_CONFIG) -a $(DOCS_ADDR)

loc:
	@echo '=== Counting lines of code ==='
	find . -type f -name "*.py" -not -path "*/.venv/*" | xargs wc -l

clean:
	@echo '=== Cleaning up ==='
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf $(PROJECT_NAME).egg-info .pytest_cache .coverage .$(LINTER)_cache site docs/site

destroy: clean
	@echo '=== Destroying environment ==='
	rm -rf .venv $(UV).lock hyperbench_logs

help:
	@echo "Usage: make [target]"
	@echo "Targets:"
	@echo "  all                     - Clean, setup, lint, typecheck, test"
	@echo "  build                   - Clean and setup"
	@echo "  setup                   - Install dependencies"
	@echo "  setup-tensorboard       - Install optional TensorBoard dependency"
	@echo "  check                   - Run lint and typecheck"
	@echo "  format                  - Run formatting"
	@echo "  typecheck               - Run type checking"
	@echo "  lint                    - Run linting"
	@echo "  lint-fix                - Run linting and fix issues"
	@echo "  lint-rule R=<rule>      - Run linting for a specific rule (e.g., R=E501)"
	@echo "  lint-rule-fix R=<rule>  - Run linting for a specific rule and fix issues"
	@echo "  test                    - Run all tests"
	@echo "  stest T=<test_name>     - Run a single test"
	@echo "  run <file.py>           - Run a single file"
	@echo "  docs                    - Build and serve documentation"
	@echo "  docs-build              - Build documentation without serving"
	@echo "  docs-serve              - Serve built documentation locally at $(MKDOCS_URL)"
	@echo "  loc                     - Count lines of code"
	@echo "  clean                   - Remove build/test artifacts"
	@echo "  destroy                 - Destroy the environment"
