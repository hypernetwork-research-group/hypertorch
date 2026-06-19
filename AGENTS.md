# AGENTS instructions

## Naming

Write `hypertorch` in lowercase when referring to the Python package, modules, functions, classes, or CLI/package internals.

Write `HyperTorch` when referring to the project, repository, organization, or published package in prose.

## Environment setup

- Install `uv` and `make`.
- Run `make setup` to install dependencies and the editable package.
- Run `make` when you want the full local quality flow (`clean`, `setup`, `check`, `test`).

## Commands

- `make test` to run the full test suite.
- `make stest T=<path-within-hypertorch/tests>` to run a specific test file or folder.
- `uv run pytest` for one-off pytest commands (not recommended for regular use).

- `make lint` to run Ruff checks.
- `make format` to run Ruff formatting.
- `make typecheck` to run `ty check`.
- `make check` to run lint, format, and typecheck together.

- `make docs-build` to build documentation.
- `make docs-serve` to serve documentation locally.
- `make docs` to build and then serve the docs.
- `make build` to clean and set up the package.
- `make clean` to remove generated artifacts.

- `make run <python_file>` to run a Python script, for example `make run examples/gcn.py`.

## Repository structure

- `hypertorch/`: core package
  - `data/`: dataset loading, sampling, and HIF integration
  - `hlp/`: hyperlink prediction modules
  - `models/`: model implementations (e.g., GCN, Node2Vec, etc.)
  - `nn/`: layers, enrichers, aggregators, and losses
  - `train/`: training utilities and trainers
  - `tests/`: test suite
  - `types/`: shared type definitions
  - `utils/`: reusable helpers
- `docs/`: documentation sources
- `examples/`: runnable training and evaluation examples
- `.github/`: workflows, templates, and hooks
- `CONTRIBUTING.md`: contributor quickstart and workflow expectations
- `docs/development/contribution.md`: detailed contribution process
- `README.md`: project overview and installation
- `Makefile`: supported development commands
- `pyproject.toml`: package metadata, dependencies, tool configuration

## Coding standards

- **Formatting:** Prefer the Makefile targets:
    - `make format`
    - `make lint`
    - `make check`
- **Typing:** Add and preserve type annotations. Run `make typecheck` for changes that touch typed code.
- **Imports:** Keep imports at module top level unless a delayed import is necessary. Use `TYPE_CHECKING` guards for type-only or heavyweight imports.
- **Validation:** Use explicit validation functions for argument checks. Avoid `assert` and do not validate types at runtime.
- **Runtime checks:** Do not use `assert` for library-facing validation. Raise explicit exceptions instead.
- **Public APIs:** Avoid changing public signatures without a clear reason and matching docs/tests updates.
- **Scope:** Keep changes narrow. Do not mix behavioral edits with unrelated refactors.
- **Documentation:** Update docstrings and public documentation for any behavior changes, new features, or edge cases.
    - Docstrings should follow the Google style guide and include in this order: function/method description, usage examples if helpful (Examples), parameter descriptions (Args), return values (Returns), and raised exceptions (Raises).

## Testing standards

- **Location:** Tests live under `hypertorch/tests/` and should mirror the package layout they exercise.
- **Execution:** Prefer:
    - `make test`
    - `make stest T=<path-within-hypertorch/tests>`
    - `uv run pytest ...` only for targeted one-off invocations
- **Style:** Use pytest function tests and fixtures. Prefer `pytest.mark.parametrize` with readable `id=` values in `pytest.param(..., id=...)`.
- **Determinism:** Avoid network access, sleeps, and external services. Patch HTTP calls, filesystem state, and subprocesses as needed.
- **Fixtures:** Keep fixture scope as small as practical. Put shared fixtures in `conftest.py`.
- **Coverage:** Add tests for new behavior, edge cases, and failure paths when code changes.
    - Always ensure 100% coverage for new or refactored code. Run `make test` to see coverage reports and identify untested lines.

## Security model

When flagging security concerns, distinguish between:

1. **Actual vulnerabilities**: code that violates the security model and can be exploited in practice.
   These should be reported immediately and treated as high priority for fixes.
2. **Known limitations**: documented gaps where the current implementation doesn't provide full
   isolation. These are tracked for improvement in future versions and should not be reported as new findings.

## Conventions

Git remotes, branch names, commit messages, and PRs should follow:

- `CONTRIBUTING.md`
- `docs/development/contribution.md`

### GitHub messages drafted by agents

Anything an agent drafts that ends up posted to GitHub on the user's
account, including PR comments, reviews, line comments, issue comments,
or discussion replies, must end with an attribution footer.

Place the footer in its own paragraph at the end of the message,
separated from the body by a blank line and a horizontal rule. Use the
same agent name string used in `Generated-by:` on PR bodies.

- Agent draft posted without prior human review:

```text
---
Drafted-by: <Agent Name and Version> (no human review before posting)
```

- Agent draft reviewed and approved by a human maintainer before posting:

```text
---
Drafted-by: <Agent Name and Version>; reviewed by @<github-handle> before posting
```

The `@<github-handle>` is the human who actually read the draft.

Do not skip the footer to shorten a message.

## Boundaries

- Ask first:
  - Large cross-package refactors.
  - New dependencies with broad impact.
  - Destructive data or migration changes.
- Never:
  - Commit secrets, credentials, or tokens.
  - Edit generated files by hand when a generation workflow exists.
  - Use destructive git operations unless explicitly requested.

## Specific references

- [Testing](docs/agents/references/testing.md)
- [Package](docs/agents/references/package.md)
- [Type System](docs/agents/references/type-system.md)
- [Standard Library](docs/agents/references/standard-lib.md)
