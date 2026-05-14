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

- `make lint` to run linters and formatters
- `make typecheck` to run type checkers
- `make docs` to build documentation
- `make build` to build the package
- `make clean` to clean up generated files

- `make run T=<example_file>` to run a training script (e.g. `make run T=examples/gcn.py`)

## Repository Structure

- `hyperbench/` - core codebase
  - `data/` - data loading and processing utilities
  - `hlp/` - hyperlink prediction modules
  - `models/` - actual models like GCN, Node2Vec, etc.
  - `nn/` - layers, enrichers, aggregators, and losses
  - `train/` - training utilities and modules
  - `tests/` - unit test
  - `types/` - type definitions and type system utilities
  - `utils/` - general utilities and helpers
- `docs/` - documentation
- `examples/` - example scripts for training and evaluation
- `agents/` - agent code and references
- `.github/` - GitHub configuration (workflows, issue templates, etc.)
- `CONTRIBUTING.md` - contribution guidelines
- `README.md` - project overview and quickstart guide
- `Makefile` - commands for setup, testing, linting, etc.
- `pyproject.toml` - package configuration and dependencies


## Architecture Boundaries

## Security Model

## Shared libraries

## Coding Standards

## Testing Standards

## Conventions

Git remote, commits and PRs should follow the repository's contribution guidelines:
  - CONTRIBUTING.md
  - docs/contributing.md

## Creating Pull Requests

## Tracking issues for deferred work

## GitHub messages drafted by agents

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

- docs/agents/references/testing.md
- docs/agents/references/package.md
- docs/agents/references/type-system.md
- docs/agents/references/standard-lib.md
