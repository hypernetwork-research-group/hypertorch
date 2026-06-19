# CI

HyperTorch uses GitHub Actions for continuous integration (CI).

This page documents the CI checks that **build, test, and validate** the codebase.
Check [Troubleshooting](../development/ci.md#troubleshooting) for common failure modes and how to fix them locally.

## Checks

| Check | Workflow | When it runs | Platforms | What it does | Local equivalent |
|---|---|---|---|---|---|
| Lint (Ruff) | `.github/workflows/ci.yaml` | PRs; pushes to `main` | Ubuntu, macOS, Windows; Python 3.10–3.14 | `ruff check --output-format=github` | `make lint` |
| Format (Ruff) | `.github/workflows/ci.yaml` | PRs; pushes to `main` | Ubuntu, macOS, Windows; Python 3.10–3.14 | `ruff format` | `make format` |
| Unit tests (+ coverage in console) | `.github/workflows/ci.yaml` | PRs; pushes to `main` | Ubuntu, macOS, Windows; Python 3.10–3.14 | `pytest -n auto --cov=hypertorch --cov-report=term-missing -m "not integration"` | `make test` |
| Integration tests | `.github/workflows/ci.yaml`; `.github/workflows/daily_ci.yaml` | PRs; pushes to `main`; daily schedule; manual dispatch | CI: Ubuntu, Python 3.10 and 3.14; Daily CI: Ubuntu, macOS, Windows, Python 3.10–3.14 | `pytest -n auto -m "integration"` | `make i-test` |
| Coverage upload (Codecov) | `.github/workflows/coverage.yaml` | pushes to `main` | Ubuntu; Python 3.14 | Generates `coverage.xml` and uploads coverage + JUnit test results to Codecov | `uv run pytest --cov --cov-branch --cov-report=xml` |
| Docs build (deploy) | `.github/workflows/docs.yaml` | pushes to `main`; manual dispatch | Ubuntu; Python 3.14 | Installs docs deps and runs `zensical build --clean`, then deploys `docs/site` to GitHub Pages | `make docs-build` |

## Notes

- CI uses `uv` to create/manage environments.

- The recommended local “all-in-one” pre-PR run is:

	```bash
	make check
	make test
	```

- Optionally, you can also run integration tests locally before pushing (they will always run in CI):

    ```bash
    make i-test
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

- If a CI failure is in unit tests, run:

	```bash
	make test
	```

- If you want to run a specific unit test file or folder, run:

	```bash
	make stest T=<path>
	```

- If a CI failure is in integration tests, run:

	```bash
	make i-test
	```

- To run a single integration test file or folder, run:

	```bash
	make si-test T=<path>
	```

- To run all checks and tests, run:

    ```bash
    make release
    ```
