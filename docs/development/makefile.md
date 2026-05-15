# Makefile

## Available commands

You can always run `make help` to see the latest list.

> Notes
>
> - These are the explicit CLI commands that each `make` target runs.
> - `make all`, `make build`, `make check`, `make docs` are composites (they run multiple steps).

| Target | Description |
|---|---|
| `make help` | Print available targets |
| `make all` | Clean, setup, check, test |
| `make build` | Clean and setup |
| `make setup` | Install dependencies (via `uv`) and install HyperBench in editable mode |
| `make setup-tensorboard` | Install optional TensorBoard extra |
| `make check` | Run lint + format + typecheck |
| `make lint` | Run the linter (`ruff check`) |
| `make lint-fix` | Run the linter with auto-fix (`ruff check --fix`) |
| `make lint-rule R=<RULE>` | Lint a single Ruff rule (`ruff check --select <RULE>`) |
| `make lint-rule-fix R=<RULE>` | Lint a single Ruff rule with auto-fix |
| `make format` | Run the formatter (`ruff format`) |
| `make typecheck` | Run the type checker (`ty check`) |
| `make docstring-check` | Check docstring formatting |
| `make test` | Run all tests (with coverage) |
| `make stest T=<test_path>` | Run a single test file/folder under `hyperbench/tests/` |
| `make run <file.py>` | Run a single Python file (for example: `make run examples/gcn.py`) |
| `make docs` | Build and serve documentation |
| `make docs-build` | Build documentation without serving |
| `make docs-serve` | Serve built documentation locally (default: `http://127.0.0.1:8000`) |
| `make loc` | Count lines of Python code |
| `make clean` | Remove build/test artifacts |
| `make destroy` | Destroy the environment (removes `.venv`, lockfile, logs) |


### Setup

- `make setup`: Install dependencies (via `uv`) and install HyperBench in editable mode

	CLI:

	```bash
	uv sync
	uv pip install -e .
	```

- `make setup-tensorboard`: Install optional TensorBoard extra

	CLI:

	```bash
	uv pip install -e ".[tensorboard]"
	```

### Lint / format / typecheck

- `make lint`: Run the linter (`ruff check`)

	CLI:

	```bash
	uv run ruff check
	```

- `make lint-fix`: Run the linter with auto-fix

	CLI:

	```bash
	uv run ruff check --fix
	```

- `make lint-rule R=<RULE>`: Lint a single Ruff rule

	CLI:

	```bash
	uv run ruff check --select <RULE>
	```

- `make lint-rule-fix R=<RULE>`: Lint a single Ruff rule with auto-fix

	CLI:

	```bash
	uv run ruff check --select <RULE> --fix
	```

- `make format`: Run the formatter (`ruff format`)

	CLI:

	```bash
	uv run ruff format
	```

- `make typecheck`: Run the type checker (`ty check`)

	CLI:

	```bash
	uv run ty check
	```

- `make docstring-check`: Check docstring formatting

	CLI:

	```bash
	uv run python3 ./scripts/check_docstrings.py
	```

### Tests

- `make test`: Run all tests (with coverage)

	CLI:

	```bash
	uv run pytest --cov=hyperbench --cov-report=term-missing
	```

- `make stest T=<test_name>`: Run a single test file or folder under `hyperbench/tests/`

	CLI:

	```bash
	uv run pytest -s hyperbench/tests/<test_name>
	```

### Run scripts

- `make run <file.py>`: Run a single Python file

	CLI:

	```bash
	uv run python3 <file.py>
	```

### Documentation

- `make docs-build`: Build documentation without serving

	CLI:

	```bash
	uv run zensical build --clean -f zensical.toml
	```

- `make docs-serve`: Serve built documentation locally (default: `http://127.0.0.1:8000`)

	CLI:

	```bash
	uv run zensical serve -f zensical.toml -a 127.0.0.1:8000
	```

### Maintenance

- `make loc`: Count lines of Python code

	CLI:

	```bash
	find . -type f -name "*.py" -not -path "*/.venv/*" | xargs wc -l
	```

- `make clean`: Remove build/test artifacts

	CLI:

	```bash
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf hyperbench.egg-info .pytest_cache .coverage .ruff_cache site docs/site
	```

- `make destroy`: Destroy the environment (removes `.venv`, lockfile, logs)

	CLI:

	```bash
	# clean
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf hyperbench.egg-info .pytest_cache .coverage .ruff_cache site docs/site

	# destroy
	rm -rf .venv uv.lock hyperbench_logs
	```


### Composite targets

Utility targets that run multiple steps in sequence.

- `make all`: Clean, setup, check, test
- `make build`: Clean and setup
- `make check`: Run lint + format + typecheck
- `make docs`: Build and serve documentation
