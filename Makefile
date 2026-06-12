.PHONY: all release build setup setup-tensorboard clean destroy \
		test stest i-test si-test run \
		check format typecheck lint lint-fix lint-rule lint-rule-fix \
		docs docs-build docs-serve \
		loc help

PROJECT_NAME=hyperbench
UV=uv
PYTEST=pytest
PYTEST_WORKERS?=auto
PYTEST_INTEGRATION_WORKERS?=auto
LINTER=ruff
TYPECHECKER=ty
ZENSICAL_CONFIG=zensical.toml
DOCS_ADDR=127.0.0.1:8000

all: clean setup check test

release: clean setup check test i-test

build: clean setup

setup:
	@echo '=== Setup ==='
	$(UV) sync
	$(UV) pip install -e .

setup-tensorboard:
	@echo '=== Setup TensorBoard ==='
	$(UV) pip install -e ".[tensorboard]"

check: typecheck format lint docstring-check

format:
	@echo '=== Formatting ==='
	$(UV) run $(LINTER) format

typecheck:
	@echo '=== Type checking ==='
	$(UV) run $(TYPECHECKER) check

docstring-check:
	@echo '=== Docstring checking ==='
	$(UV) run ./scripts/validate_docstrings.py

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
	@echo '=== Running unit tests in parallel ==='
	$(UV) run $(PYTEST) -n $(PYTEST_WORKERS) --cov=$(PROJECT_NAME) --cov-report=term-missing -m "not integration"

stest:
	@echo '=== Running single unit test for $(T) ==='
	$(UV) run $(PYTEST) -n $(PYTEST_WORKERS) -s $(PROJECT_NAME)/tests/$(T)

i-test:
	@echo '=== Running integration tests in parallel ==='
	$(UV) run $(PYTEST) -n $(PYTEST_INTEGRATION_WORKERS) -m "integration"

si-test:
	@echo '=== Running single integration test for $(T) ==='
	$(UV) run $(PYTEST) -n $(PYTEST_INTEGRATION_WORKERS) -s $(PROJECT_NAME)/integration_tests/$(T) -m "integration"

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
	./scripts/count_loc.sh

clean:
	@echo '=== Cleaning up ==='
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf $(PROJECT_NAME).egg-info .pytest_cache .coverage .$(LINTER)_cache site docs/site .python-version

destroy: clean
	@echo '=== Destroying environment ==='
	rm -rf .venv $(UV).lock hyperbench_logs .hyperbench_cache .$(UV)-cache

help:
	@echo "Usage: make [target]"
	@echo "Targets:"
	@echo "  all                     - Clean, setup, lint, typecheck, test"
	@echo "  release                 - Clean, setup, lint, typecheck, test, i-test"
	@echo "  build                   - Clean and setup"
	@echo "  setup                   - Install dependencies"
	@echo "  setup-tensorboard       - Install optional TensorBoard dependency"
	@echo "  check                   - Run lint and typecheck"
	@echo "  format                  - Run formatting"
	@echo "  typecheck               - Run type checking"
	@echo "  docstring-check         - Check docstring formatting"
	@echo "  lint                    - Run linting"
	@echo "  lint-fix                - Run linting and fix issues"
	@echo "  lint-rule R=<rule>      - Run linting for a specific rule (e.g., R=E501)"
	@echo "  lint-rule-fix R=<rule>  - Run linting for a specific rule and fix issues"
	@echo "  test                    - Run all unit tests in parallel"
	@echo "  stest T=<test_name>     - Run a single unit test"
	@echo "  i-test                  - Run integration tests in parallel"
	@echo "  si-test T=<test_name>   - Run a single integration test"
	@echo "  run <file.py>           - Run a single file"
	@echo "  docs                    - Build and serve documentation"
	@echo "  docs-build              - Build documentation without serving"
	@echo "  docs-serve              - Serve built documentation locally at http://$(DOCS_ADDR)"
	@echo "  loc                     - Count lines of code"
	@echo "  clean                   - Remove build/test artifacts"
	@echo "  destroy                 - Destroy the environment"
