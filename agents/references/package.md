# Packaging and Project Setup

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

```
.
├── CITATION.cff
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
├── Makefile
├── README.md
├── SECURITY.md
├── agents
│   ├── SKILLS.md
│   └── references
│       ├── package.md
│       ├── standard-lib.md
│       ├── testing.md
│       └── type-system.md
├── configs
├── docs
│   ├── api
│   │   ├── data.md
│   │   ├── hlp.md
│   │   ├── models.md
│   │   ├── nn.md
│   │   ├── reference.md
│   │   ├── train.md
│   │   ├── types.md
│   │   └── utils.md
│   ├── assets
│   │   ├── data_design.excalidraw
│   │   ├── deprecated_design.excalidraw
│   │   └── design.png
│   ├── getting-started
│   │   └── installation.md
│   ├── index.md
│   ├── stylesheets
│   │   └── extra.css
│   └── user-guide
│       └── user.md
├── examples
│   ├── early_stopping.py
|   ├── ...
│   └── villain.py
├── hyperbench
│   ├── __init__.py
│   ├── data
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   ├── dataset.py
│   │   ├── datasets
│   │   ├── hif.py
│   │   ├── loader.py
│   │   ├── sampling.py
│   │   └── supported_datasets.py
│   ├── hlp
│   │   ├── __init__.py
│   │   ├── common.py
│   │   ├── common_neighbors_hlp.py
│   │   ├── gcn_hlp.py
│   │   ├── hgnn_hlp.py
│   │   ├── hgnnp_hlp.py
│   │   ├── hnhn_hlp.py
│   │   ├── hypergcn_hlp.py
│   │   ├── mlp_hlp.py
│   │   ├── nhp_hlp.py
│   │   ├── node2vec_common.py
│   │   ├── node2vecgcn_hlp.py
│   │   ├── node2vecslp_hlp.py
│   │   └── villain_hlp.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── common_neighbors.py
│   │   ├── gcn.py
│   │   ├── hgnn.py
│   │   ├── hgnnp.py
│   │   ├── hnhn.py
│   │   ├── hypergcn.py
│   │   ├── mlp.py
│   │   ├── nhp.py
│   │   ├── node2vec.py
│   │   └── villain.py
│   ├── nn
│   │   ├── __init__.py
│   │   ├── aggregator.py
│   │   ├── conv.py
│   │   ├── enricher.py
│   │   ├── loss.py
│   │   └── scorer.py
│   ├── tests
│   │   ├── __init__.py
│   │   ├── data
│   │   ├── mock
│   │   ├── train
│   │   ├── types
│   │   └── utils
│   ├── train
│   │   ├── __init__.py
│   │   ├── latex_logger.py
│   │   ├── markdown_logger.py
│   │   ├── negative_sampler.py
│   │   ├── negative_sampling_scheduler.py
│   │   └── trainer.py
│   ├── types
│   │   ├── __init__.py
│   │   ├── graph.py
│   │   ├── hdata.py
│   │   ├── hypergraph.py
│   │   └── model.py
│   └── utils
│       ├── __init__.py
│       ├── data_utils.py
│       ├── file_utils.py
│       ├── hif_utils.py
│       ├── nn_utils.py
│       ├── node_utils.py
│       ├── schema
│       ├── sparse_utils.py
│       └── url_utils.py
├── hyperbench_logs
│   └── experiment_0
│       ├── common_neighbors
│       ├── comparison
│       └── mlp
├── pyproject.toml
├── uv.lock
└── zensical.toml
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
