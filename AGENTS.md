# HyperBench Agent Instructions

## Project Overview
HyperBench is a research-focused benchmarking toolkit for hypergraph learning.

## Purpose
- Assist contributors by suggesting code changes, tests, and documentation edits that fit the existing style and tooling.
- Keep changes easy to review and safe to merge.

## Persona & Tone
- Concise, neutral, code-focused.
- Prioritize correctness, readability, and reproducibility.

## Project Guidelines
- Follow the repository's contributor guidance:
  - [CONTRIBUTING.md](CONTRIBUTING.md)
  - [docs/contributing.md](docs/contributing.md)
- Prefer repository tooling and workflows:
  - [Makefile](Makefile) targets (they run via `uv`)
  - Docs configuration in [zensical.toml](zensical.toml)

## Decision heuristics
- Favor small, scoped changes with tests.
- Avoid drive-by refactors unless explicitly requested.
- Treat public API changes as high-risk: prefer backward-compatible additions or a clear migration path.
- Prefer readability over micro-optimizations unless benchmarks are requested.
- Add tests for behavioral changes; update docs only after code change is final.

## Tooling & Validation (summary)
HyperBench uses `uv` for environments and execution. Prefer `make` targets when available.

- Setup (first time): `make setup` (or `uv sync` then `uv pip install -e .`)
- Format + typecheck: `make check`
- Tests: `make test` (single test: `make stest T=<test_file_or_path>`)
- Docs build/serve: `make docs-build` / `make docs-serve`

Docs deployment is handled by GitHub Actions in [.github/workflows/docs.yaml](.github/workflows/docs.yaml).

If you need a one-off command, prefer `uv run ...` (e.g., `uv run pytest`).

## Type hints guidance (summary)
- Prefer straightforward PEP 484 typing.
- Use builtin generics (`list`, `dict`, `tuple`) where possible.
- Avoid `typing.cast` unless necessary; prefer refactors that make types obvious.
- Keep `ty` happy: if you change types, run `make check`.

## Docstring guidance (summary)
- Use consistent Google-style docstrings (as rendered by mkdocstrings/griffe in the docs build).
- Keep docstrings consistent with function signatures (griffe warns when documented parameters don't exist).
- Prefer an `Examples:` section with fenced code blocks.
- Keep examples minimal and deterministic.

## Pull Requests (summary)
- Keep PR descriptions short and specific; link relevant issues.
- Follow the repository conventions for commit messages / branch naming (see [CONTRIBUTING.md](CONTRIBUTING.md)).
- For new issues, ensure at least one "type" label exists (automation expects one of: `chore`, `docs`, `feature`, `fix`, `refactoring`).
  - Enforcement lives in [.github/workflows/management.yaml](.github/workflows/management.yaml)

## When to Avoid AI
- Handling credentials/secrets.
- Security-sensitive reports in public issues (use private reporting; see [SECURITY.md](SECURITY.md)).

## Additional Links
- Code: https://github.com/hypernetwork-research-group/hyperbench
- Docs: https://hypernetwork-research-group.github.io/hyperbench/
- Issues: https://github.com/hypernetwork-research-group/hyperbench/issues
