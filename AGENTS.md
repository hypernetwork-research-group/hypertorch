# Agents instructions

## Naming

Write `hypertorch` in lowercase when referring to the Python package, modules,
functions, classes, or CLI/package internals.

Write `HyperTorch` when referring to the project, repository,
organization, or published package in prose.

## Environment setup and commands

- Install `uv` and `make`.
- Run `make` to build and test the project.
- Run `make release` to also run integration tests on top of `make`.

See `Makefile` for supported commands. Use `make help` to list all available commands.

## Repository structure

- `hypertorch/`: core package
  - `data/`: dataset loading, sampling, and HIF integration
  - `hyperlink_prediction/`: hyperlink prediction modules
  - `integration_tests/`: integration tests for the package
  - `models/`: model implementations (e.g., GCN, Node2Vec, etc.)
  - `node_classification/`: node classification modules
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

- **Formatting**: Prefer the Makefile targets: `make format` and `make lint`.
- **Typing**: Add and preserve type annotations.
    Run `make typecheck` for changes that touch typed code.
- **Imports**: Keep imports at module top level unless a delayed import is necessary.
    Use `TYPE_CHECKING` guards for type-only or heavyweight imports.
- **Validation**: Use explicit validation functions for argument checks.
    Avoid `assert` and do not validate types at runtime.
- **Runtime checks**: Do not use `assert` for library-facing validation.
    Raise explicit exceptions instead.
- **Public APIs**: Avoid changing public signatures without a clear reason
    and matching docs/tests updates.
- **Scope**: Keep changes narrow. Do not mix behavioral edits with unrelated refactors.
- **Documentation**: Update docstrings and public documentation
    for any behavior changes, new features, or edge cases.
    - Docstrings should follow the Google style guide and include in this order:
    function/method description, usage examples if helpful (Examples),
    parameter descriptions (Args), return values (Returns), and raised exceptions (Raises).

## Unit tests

- **Location**: Tests live under `hypertorch/tests/` and should mirror the package layout.
- **Execution**: Prefer:
    - `make test`
    - `make stest T=<path-within-hypertorch/tests>`
    - `uv run pytest ...` only for targeted one-off invocations
- **Style**: Use pytest function tests and fixtures. Prefer `pytest.mark.parametrize`
    with readable `id=` values in `pytest.param(..., id=...)`.
- **Determinism**: Avoid network access, sleeps, and external services.
    Patch HTTP calls, filesystem state, and subprocesses as needed.
- **Fixtures**: Keep fixture scope as small as practical. Put shared fixtures in `conftest.py`.
- **Coverage**: Add tests for new behavior, edge cases, and failure paths when code changes.
    - Always ensure 100% coverage for new or refactored code. Run `make test` to see coverage
    reports and identify untested lines.
- **Never add for**: `hyperlink_prediction` (HLP), `nc`, `models`, and `nn` modules.

## Integration tests

- **Location**: Integration tests live under `hypertorch/integration_tests/`
    and should test real workflows.
- **Execution**: Prefer:
    - `make i-test`
    - `make si-test T=<path-within-hypertorch/integration_tests>`
    - `uv run pytest hypertorch/integration_tests/...` only for targeted one-off invocations
- **Style**: Use pytest function tests. Prefer `pytest.mark.parametrize` with readable `id=` values
    in `pytest.param(..., id=...)`.
- **Always add for**: `hyperlink_prediction` (HLP), `nc`, and `train` modules.

## Security

When flagging security concerns, distinguish between:

1. **Actual vulnerabilities**: code that violates the security model and can be exploited
    in practice. These should be reported immediately and treated as high priority for fixes.
2. **Known limitations**: documented gaps where the current implementation does not provide full
    isolation. These are tracked for improvement in future versions and
    should not be reported as new findings.

## Conventions

Git remotes, branch names, commit messages, and PRs should follow the guidelines
in `CONTRIBUTING.md` and `docs/development/contribution.md`.

### GitHub messages drafted by agents

Anything an agent drafts that ends up posted to GitHub on the user's
account, including PR comments, reviews, line comments, issue comments,
or discussion replies, must end with an attribution footer:

```text
---
Drafted-by: <Agent Name and Version>
```

Place the footer in its own paragraph at the end of the message,
separated from the body by a blank line and a horizontal rule. Use the
same agent name string used in `Generated-by:` on PR bodies.

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
- [Type system](docs/agents/references/type-system.md)
- [Standard library](docs/agents/references/standard-lib.md)
