# How to contribute

The project main language is English.

## Quickstart for contributors

We use different tools to make our life easier.
Although they are not mandatory for every contribution, the repository is configured around:

- [uv](https://github.com/astral-sh/uv)
- [make](https://www.gnu.org/software/make/)
- [pre-commit](https://pre-commit.com)

```bash
git clone https://github.com/hypernetwork-research-group/hyperbench.git
cd hyperbench

# Install dependencies with uv
make build

# Install pre-commit to set up automatic local checks
# This will ensure that your code adheres to the project's coding standards before each commit.
uv run pre-commit install \
    --config .github/hooks/.pre-commit-config.yaml \
    --install-hooks \
    --overwrite

# See all the available commands mapped in the Makefile
make help

# Or run the full suite, which includes setup, checks, and tests
make

# Run an existing example
make run examples/mlp_common_neighbors.py
```

For a full explanation of the project structure, commands, and development guidelines, refer to the [documentation](https://hypernetwork-research-group.github.io/hyperbench/).

## Contribution types and expectations

### Feature

Best for: new capabilities (models, datasets, training features, utilities).

Expectations:
- Prefer opening an issue first to discuss scope and API impact.
- Include tests for the new behavior.
- Update documentation (or docstrings) if user-facing behavior changes.

### Fix

Best for: incorrect outputs, crashes, regressions, or broken docs/examples.

Expectations:
- Include a minimal reproduction (or a failing test) and a clear fix.
- Add a regression test when feasible.
- Keep the PR small and focused on the root cause.

### Documentation

Check [guidelines](#contributing-to-the-documentation) for more details.

Best for: documentation, API docs clarity, examples, READMEs.

Expectations:
- Ensure documentation builds correctly (`make docs-build`).
- Keep examples deterministic and copy-paste runnable when possible.

## Workflow

1. Fork the repo and create a branch (see [branch naming](#branch-naming)).
2. Make changes with tests/docs as needed.
3. Run local quality gates (see [quality gates](#quality-gates)).
4. Open a PR and fill in the PR template.
5. Address review feedback.
6. Ensure your changes are rebased onto the latest main branch before merge.

### Commit message style

Commit messages should follow the [conventional commit specification](https://www.conventionalcommits.org/en/v1.0.0/).

The allowed structural elements are:
- `feat` for new features.
- `fix` for bug fixes.
- `chore` for changes to the build process or auxiliary tools and libraries such as documentation generation.
- `refactor` for code changes that neither fix a bug nor add a feature.
- `docs` for any documentation/README changes.

Commit messages should be structured in a way that can be read as if they were completing the sentence *"If applied, this commit will..."*. For example:

> feat: add new authentication method to API

Reads as *"If applied, this commit will add new authentication method to API"*.

### Branch naming

Branch names should follow the pattern `^(feat|fix|chore|refactor|docs)\/[a-z0-9]+(-[a-z0-9]+)*$`. This means that branch names should:
- Start with the same structural elements as commit messages.
- Be descriptive and contain only lowercase letters and numbers.
- Use hyphens to separate words.

For example:
- `feat/add-user-authentication`
- `fix/issue-with-database-connection`
- `chore/update-dependencies`
- `refactor/improve-code-structure`
- `docs/update-contributing-guidelines`

To verify that your branch name adheres to these guidelines, you can use the following command:

```bash
git rev-parse --abbrev-ref HEAD | grep -Eq '^(feat|fix|chore|refactor|docs)\/[a-z0-9]+(-[a-z0-9]+)*$' && echo "Branch name is compliant" || echo "Invalid branch name"
```

## Quality gates

- If you have pre-commit installed, it will automatically check your code before each commit. You can also run it manually with `pre-commit run --all-files`.
- For manual checks, you can use the following commands:
  - Linting, formatting, and type checking: `make check`.
  - Running tests: `make test` (or `make stest T=<path>` for specific tests under `hyperbench/tests/`).

Maintainers may:
- Request additional explanation, tests, or revisions.
- Ask contributors to rewrite or remove content that does not meet project standards.
- Reject contributions that do not satisfy quality, security, or other expectations.

## AI-assisted contribution policy

AI tools (for example, LLMs/code assistants) may be used to help prepare contributions.

Contributors must:

- **Take full responsibility** for all submitted content, including correctness, clarity, style, tests, licensing, and originality.
- **Review and validate manually** any AI-generated code/docs before submission.

## Contributing to the documentation

```bash
# to build the docs locally
make docs-build
# to serve the docs locally (after building)
make docs-serve
# or for a one-off command
make docs
```

### Configuration

- Docs are built with zensical, which uses a custom theme and configuration (see [zensical.toml](./zensical.toml)).
- API reference is generated with [mkdocstrings](https://mkdocstrings.github.io/) from docstrings in the codebase.

### Docstring conventions

References:
- [Google Python style guidelines](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- [Zensical documentation conventions](https://zensical.org/docs/get-started/)

- Docstrings should follow the Google style guidelines and include in this order: 
    - Function/method description.
    - Usage examples under the *Examples* section, if helpful.
    - Parameter descriptions under the *Args* section.
    - Return values under the *Returns* section.
    - Raised exceptions under the *Raises* section.

- Prefer fenced code blocks with syntax highlighting for examples.
- Use type formatting conventions (e.g., `list[str]` instead of `List[str]`).

### API reference

Each API reference is generated with `mkdocstrings` and is under `docs/api/`. Each module should have a corresponding markdown file (e.g., `data.md` for `hyperbench.data`) with an overview and the `::: module` directive.
