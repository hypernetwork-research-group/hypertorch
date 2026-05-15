# CI

HyperBench uses GitHub Actions for continuous integration (CI).

This page documents the CI checks that **build, test, and validate** the codebase.
Check [Troubleshooting](../development/ci.md#troubleshooting) for common failure modes and how to fix them locally.

## Checks

| Check | Workflow | When it runs | Platforms | What it does | Local equivalent |
|---|---|---|---|---|---|
| Lint (Ruff) | `.github/workflows/ci.yaml` | PRs; pushes to `main` | Ubuntu, macOS, Windows; Python 3.10–3.14 | `ruff check --output-format=github` | `make lint` |
| Format (Ruff) | `.github/workflows/ci.yaml` | PRs; pushes to `main` | Ubuntu, macOS, Windows; Python 3.10–3.14 | `ruff format` | `make format` |
| Tests (+ coverage in console) | `.github/workflows/ci.yaml` | PRs; pushes to `main` | Ubuntu, macOS, Windows; Python 3.10–3.14 | `pytest --cov=hyperbench --cov-report=term-missing` | `make test` |
| Coverage upload (Codecov) | `.github/workflows/coverage.yaml` | pushes to `main` | Ubuntu; Python 3.14 | Generates `coverage.xml` and uploads coverage + JUnit test results to Codecov | `uv run pytest --cov --cov-branch --cov-report=xml` |
| Docs build (deploy) | `.github/workflows/docs.yaml` | pushes to `main`; manual dispatch | Ubuntu; Python 3.14 | Installs docs deps and runs `zensical build --clean`, then deploys `docs/site` to GitHub Pages | `make docs-build` |

## Notes

- CI uses `uv` to create/manage environments.
- The recommended local “all-in-one” pre-PR run is:

	```bash
	make check
	make test
	```

## Troubleshooting

- If a CI failure is in Ruff formatting, run:

	```bash
	make format
	```

- If a CI failure is in Ruff linting, run:

	```bash
	make lint
	```

- If you want to fix lint error, you can try:

	```bash
	make lint-fix
	```

- If you have any issue with type checking, run:

	```bash
	make typecheck
	```

- If a CI failure is in tests, run:

	```bash
	make test
	```

- If you want to run a specific test file or folder, run:

	```bash
	make stest T=<path>
	```
