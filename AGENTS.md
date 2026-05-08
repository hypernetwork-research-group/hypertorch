
# HyperBench Agent Instructions

## Project Overview
HyperBench is a research-focused benchmarking toolkit for hypergraph learning.

## Purpose
- Help contributors propose code, tests, and documentation edits that fit the existing style and tooling.
- Keep changes easy to review and safe to merge.

## Persona & Tone
- Concise, neutral, code-focused.
- Prioritize correctness, readability, and reproducibility.

## Project Guidelines
- Follow the repository’s contributor guidance:
	- `CONTRIBUTING.md`
	- `docs/contributing.md`
- Prefer repository tooling and workflows:
	- `Makefile` targets (which use `uv`)
	- Docs configuration in `zensical.toml`

## Decision heuristics
- Favor small, scoped changes with tests.
- Avoid drive-by refactors unless explicitly requested.
- Treat public API changes as high-risk: prefer backward-compatible additions or a clear migration path.
- If behavior changes, update docs after the code is final and validated.
- Prefer readability over micro-optimizations unless benchmarks are requested.
- Add tests for behavioral changes; update docs only after code change is final.

## Tooling & Validation (summary)
HyperBench uses `uv` for environments and execution. Prefer `make` targets when available.

- Setup (first time): `uv sync` then `uv pip install -e .`
- Format + typecheck: `make check`
- Tests: `make test`
- Docs build/serve: `make docs-build` / `make docs-serve`

If you need a one-off command, prefer `uv run ...` (e.g., `uv run pytest`).

## Type hints guidance (summary)
- Prefer straightforward PEP 484 typing.
- Use builtin generics (`list`, `dict`, `tuple`) where possible.
- Avoid `typing.cast` unless necessary; prefer refactors that make types obvious.
- Keep `ty` happy: if you change types, run `make check`.

## Docstring guidance (summary)
- Use consistent Google-style docstrings (as rendered by mkdocstrings/griffe in the docs build).
- Prefer an `Examples:` section with fenced code blocks.
- Keep examples minimal and deterministic.

## Pull Requests (summary)
- Keep PR descriptions short and specific; link relevant issues.
- Follow the repository conventions for commit messages / branch naming (see `CONTRIBUTING.md`).
- For new issues, ensure at least one “type” label exists (automation expects one of: `chore`, `docs`, `feature`, `fix`, `refactoring`).

## When to Avoid AI
- Handling credentials/secrets.
- Security-sensitive reports in public issues (use private reporting; see `SECURITY.md`).

## Additional Links
- Code: https://github.com/hypernetwork-research-group/hyperbench
- Docs: https://hypernetwork-research-group.github.io/hyperbench/
- Issues: https://github.com/hypernetwork-research-group/hyperbench/issues
