---
name: hyperbench
description: Use when contributing to HyperBench (a research-focused hypergraph learning benchmark toolkit). Generates small, reviewable changes and validates them with the repo tooling (uv + Makefile, ruff, ty, pytest, docs).
license: MIT
metadata:
  author: https://github.com/hypernetwork-research-group
  version: "0.1.0"
  domain: project
  triggers: HyperBench, hypergraph, benchmarking, uv, make, ruff, ty, pytest, mkdocs, mkdocstrings
  role: specialist
  scope: implementation
  output-format: code
---

# HyperBench

Based on [Python-pro skill](https://github.com/Jeffallan/claude-skills/tree/main/skills/python-pro).

Repo-specific guidance for making safe changes in HyperBench.

## When to use this skill

- Editing or adding HyperBench Python code (models, loaders, training)
- Fixing bugs with minimal API surface change
- Adding/adjusting tests under `hyperbench/tests`
- Updating docs under `docs/` (mkdocs + mkdocstrings)

## Core workflow

1. **Orient** — Find the smallest relevant module(s) under `hyperbench/` and the matching tests.
2. **Implement** — Keep changes scoped; preserve public APIs unless explicitly requested.
3. **Test** — Add/adjust pytest coverage for behavior changes.
4. **Validate** — Prefer repo targets:
   - `make check` (lint/typecheck)
   - `make test` (full suite)
   - `make stest T=<path>` (single test)
5. **Docs (if needed)** — Build/serve docs with `make docs-build` / `make docs-serve`.

## Tooling expectations

- Python requirement: `>=3.10`
- Environment runner: `uv` (via `make` targets)
- Linting/formatting: `ruff`
- Type checking: `ty`
- Tests: `pytest`

Docs deployment is handled by GitHub Actions in [.github/workflows/docs.yaml](.github/workflows/docs.yaml).
If you need a one-off command, prefer `uv run ...` (e.g., `uv run pytest`).

## Constraints

### MUST DO

- Use straightforward, explicit code and typing (prefer builtin generics like `list[str]`).
- Keep changes reviewable; avoid drive-by refactors.
- Add or update pytest tests for behavior changes.
- Run the relevant `make` targets before considering work complete.

### MUST NOT DO

- Introduce new dependencies without a clear need.
- Change public APIs silently.
- Add non-deterministic tests (timing- or network-dependent).

## Reference guide

Load these when you need extra detail:

- Follow the repository's contributor guidance:
  - [CONTRIBUTING.md](CONTRIBUTING.md)
  - [docs/contributing.md](docs/contributing.md)

| Topic | Reference | Load When |
|-------|-----------|-----------|
| Packaging & tooling | `references/package.md` | `uv`, dependency groups, `pyproject.toml`, make targets |
| Testing | `references/testing.md` | pytest patterns, fixtures, selecting/running tests |
| Type system | `references/type-system.md` | typing patterns, Protocols, TypedDict, structural typing |
| Standard library | `references/standard-lib.md` | pathlib, dataclasses, functools, itertools |

## Additional links
- Code: https://github.com/hypernetwork-research-group/hyperbench
- Docs: https://hypernetwork-research-group.github.io/hyperbench/
- Issues: https://github.com/hypernetwork-research-group/hyperbench/issues
