# Packaging and project setup

## HyperTorch specifics

- HyperTorch uses `uv` for environment management and command execution.
- Prefer Makefile targets over ad hoc commands.
- The package uses a flat layout: `hypertorch/` lives at the repository root.
- Package metadata, dependencies, and tool configuration live in `pyproject.toml`.
- Optional extras are defined in `pyproject.toml` under `[project.optional-dependencies]`, currently including `tensorboard`.

## Primary commands

- Setup:
  - `make setup`
  - `make setup-tensorboard` for the optional TensorBoard extra
- Quality:
  - `make lint`
  - `make format`
  - `make typecheck`
  - `make check`
- Tests:
  - `make test`
  - `make stest T=<path-within-hypertorch/tests>`
- Docs:
  - `make docs-build`
  - `make docs-serve`
  - `make docs`
- One-off commands:
  - `uv run pytest`
  - `uv run ruff check`
  - `uv run ty check`

## Repository shape

```bash
.
├── .github                   # GitHub workflows and templates
├── Makefile                  # convenience build/run targets
├── agents                    # agent docs and references
│   └── references
├── configs                   # project configuration files
├── docs                      # documentation sources and site output
├── examples                  # runnable examples and demos
│   ├── hgnn.py
|   ├── ...
│   └── villain.py
├── hypertorch                # core Python package
│   ├── data                  # datasets, loaders, and sampling
│   ├── hyperlink_prediction  # Hyperlink prediction (HLP) task helpers and pipelines
│   ├── models                # model implementations
│   ├── nn                    # neural network building blocks
│   ├── tests                 # unit tests
│   ├── train                 # training loops and loggers
│   ├── types                 # shared type definitions
│   └── utils                 # reusable helpers
├── hypertorch_logs           # local experiment outputs
│   └── experiment_0
│       ├── comparison
│       ├── hgnn
│       ├── ...
│       └── villain
├── pyproject.toml            # package metadata and dependencies
├── uv.lock                   # exact dependency lockfile for development and CI
└── zensical.toml             # zensical config for docs
```


## Packaging notes

- `make setup` runs `uv sync` and then installs HyperTorch in editable mode with `uv pip install -e .`.
- `make build` is a convenience target for `clean` followed by `setup`.
- The project is discovered via setuptools in `pyproject.toml` under `[tool.setuptools.packages.find]`.
- If you change dependencies, extras, or tool configuration, update `pyproject.toml` and any affected docs together.

## Documentation notes

- Docs are built with `zensical` using `zensical.toml`.
- Generated site output lives under `docs/site/`; treat it as generated content.
- When updating contributor-facing instructions, keep `AGENTS.md`, `CONTRIBUTING.md`, and `docs/development/` aligned.

## Pre-commit

The pre-commit configuration lives at `.github/hooks/.pre-commit-config.yaml`.

```bash
uv run pre-commit install -c .github/hooks/.pre-commit-config.yaml
uv run pre-commit run -a -c .github/hooks/.pre-commit-config.yaml
```
