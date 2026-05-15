# AGENTS instructions

## Naming

Write hyperbench lowercase when in hyperbench core (writing functions or classes) and HyperBench when referring to the organization or project as a whole (e.g. "HyperBench is an open-source project for benchmarking hypergraph learning algorithms").

## Environment Setup

- install `uv` and `make`
- run `make` to set up the environment and install dependencies

## Commands

- `make test` to run the full test suite
- `make stest T=<test_file_or_path>` to run a specific test or folder
- `uv run pytest` to run pytest directly (not recommended for regular use)

- `make lint` to run linter
- `make format` to run formatter
- `make typecheck` to run type checker
- `make docs` to build documentation
- `make build` to build the package
- `make clean` to clean up generated files

- `make run T=<example_file>` to run a Python script (e.g. `make run T=examples/gcn.py`)

## Repository Structure

- `hyperbench/`: core codebase
  - `data/`: data loading and processing utilities
  - `hlp/`: hyperlink prediction modules
  - `models/`: actual models like GCN, Node2Vec, etc.
  - `nn/`: layers, enrichers, aggregators, and losses
  - `train/`: training utilities and modules
  - `tests/`: unit test
  - `types/`: type definitions and type system utilities
  - `utils/`: general utilities and helpers
- `docs/`: documentation
- `examples/`: example scripts for training and evaluation
- `agents/`: agent code and references
- `.github/`: GitHub configuration (workflows, issue templates, etc.)
- `CONTRIBUTING.md`: contribution guidelines
- `README.md`: project overview and quickstart guide
- `Makefile`: commands for setup, testing, linting, etc.
- `pyproject.toml`: package configuration and dependencies


## Coding Standards

- **Formatting:** Run the formatter and linter on changed files before committing. Prefer the repository Makefile targets:
  - `make format` (format code)
  - `make lint` (lint and style checks)
  - Use `ruff` for fast linting and formatting checks.
- **Typing:** Add and preserve type annotations. Use the project's typecheck target (`make typecheck`) during development.
- **Imports:** Keep imports at the top of the file. Use `TYPE_CHECKING` guards for heavy, optional, or type-only imports.
- **No asserts in production code:** Avoid `assert` statements for runtime checks in library code — use explicit error handling and exceptions instead.
- **Keep public APIs stable:** Avoid changing public-facing function/class signatures without a clear, documented migration path.
- **Small, focused commits:** Keep commits minimal and focused; prefer multiple small PRs to one large refactor.

## Testing Standards

- **Tests location:** Tests live under `hyperbench/tests/` and should mirror module locations.
- **Run tests:** Prefer running tests via Makefile targets:
  - `make test` for the full suite
  - `make stest T=<test_file_or_path>` for a single test or folder
  - For one-off runs use `uv run pytest`.
- **Pytest style:** Use pytest functions and fixtures (avoid `unittest.TestCase`). Use `@pytest.mark.parametrize` for multiple inputs; prefer `pytest.param(..., id=...)` for readable case ids.
- **Fixtures:** Put shared fixtures in `conftest.py` and keep fixture scope minimal. Use `autouse` sparingly.
- **Deterministic tests:** Tests must be deterministic — avoid relying on timing, network, or external services. Use mocks and fixtures for isolation.
- **Property tests & strategies:** Use Hypothesis for property-based tests where appropriate; keep generated example sizes small in CI.
- **Mocking:** Use `unittest.mock` with `spec`/`autospec` for safety; prefer `AsyncMock` for async code.
- **Time-dependent tests:** Use time-freezing helpers (e.g. `time_machine`) rather than sleeping or real clock checks.
- **Coverage and CI:** Add tests for new behavior, edge cases, and failure modes. Ensure CI stays green before requesting a review.


## Conventions

Git remote, commits and PRs should follow the repository's contribution guidelines:
  - CONTRIBUTING.md
  - docs/contributing.md

## Boundaries

- Ask first
  - Large cross-package refactors.
- New dependencies with broad impact.
- Destructive data or migration changes.
- Never
  - Commit secrets, credentials, or tokens.
  - Edit generated files by hand when a generation workflow exists.
  - Use destructive git operations unless explicitly requested.

## Specific agents references

- [Testing](docs/agents/references/testing.md)
- [Package](docs/agents/references/package.md)
- [Type System](docs/agents/references/type-system.md)
- [Standard Library](docs/agents/references/standard-lib.md)
