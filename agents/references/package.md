# Packaging and project setup

## HyperBench specifics

- HyperBench uses `uv` for environments and execution; prefer running commands via `make` targets.
- Common targets:
  - `make setup` (or `uv sync`)
  - `make check` (lint/typecheck)
  - `make test` / `make stest T=<path>`
  - `make docs-build` / `make docs-serve`
- Package metadata lives in `pyproject.toml` and the package is discovered via setuptools (`[tool.setuptools.packages.find]`).
- Optional extras are in `pyproject.toml` under `[project.optional-dependencies]` (e.g. `tensorboard`).
- For a one-off command, prefer `uv run ...` (example: `uv run pytest`).

## Repository structure

HyperBench uses a “flat” package layout: the `hyperbench/` package lives at the repo root (there is no `src/` directory).

## Project structure

```bash
.
├── .github                 # GitHub workflows and templates
├── Makefile                # convenience build/run targets
├── agents                  # agent docs and references
│   ├── SKILLS.md
│   └── references
├── configs                 # project configuration files
├── docs                    # documentation sources and site output
├── examples                # runnable examples and demos
│   ├── hgnn.py
|   ├── ...
│   └── villain.py
├── hyperbench              # core Python package
│   ├── data                # datasets, loaders, and sampling
│   ├── hlp                 # HLP task helpers and pipelines
│   ├── models              # model implementations
│   ├── nn                  # neural network building blocks
│   ├── tests               # test utilities
│   ├── train               # training loops and loggers
│   ├── types               # shared type definitions
│   └── utils               # reusable helpers
├── hyperbench_logs         # local experiment outputs
│   └── experiment_0
│       ├── common_neighbors
│       ├── comparison
│       └── mlp
├── pyproject.toml          # package metadata and dependencies
├── uv.lock                 # pinned dependency lockfile
└── zensical.toml           # zensical config for docs
```

## Project metadata

Look at [pyproject.toml](../../pyproject.toml) for the actual dependencies, optional extras, and setuptools package discovery.

## Common commands

- Setup (editable install): `make setup`
- Setup with TensorBoard extra: `make setup-tensorboard`
- Lint/format/typecheck: `make check`
- Tests: `make test` (single test/folder: `make stest T=<path>`)
- Run a script: `make run path/to/script.py`

If you need a one-off command without Makefile sugar, prefer `uv run ...` (example: `uv run ruff check`).

## Docs

- Build: `make docs-build`
- Serve: `make docs-serve` (defaults to `127.0.0.1:8000`)

## Pre-commit (optional)

The pre-commit configuration is stored at [.github/hooks/.pre-commit-config.yaml](../../.github/hooks/.pre-commit-config.yaml).

```bash
uv run pre-commit install -c .github/hooks/.pre-commit-config.yaml
uv run pre-commit run -a -c .github/hooks/.pre-commit-config.yaml
```
