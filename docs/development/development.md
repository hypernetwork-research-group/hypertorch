# Overview

If this is your first time contributing to an open-source project, welcome!
Before you dive into the code, we recommend checking out the [contribution guide](../development/contribution.md) for an introduction to general development workflow, tools, and best practices.

This guide will walk you through the process of setting up a development environment for HyperBench, contributing to the codebase and documentation, and understanding our development policies.

## Prerequisites

- [uv](https://github.com/astral-sh/uv)
- [make](https://www.gnu.org/software/make/)
- [pre-commit](https://pre-commit.com)

## Creating a development environment

### Build

Clone the repository and navigate to the project directory:

```bash
git clone https://www.github.com/hypernetwork-research-group/hyperbench.git
cd hyperbench
```

To simplify development, we provide a Makefile with common targets for building, testing, linting, and more. You can see all available commands with:

```bash
make help
```

or check the [Makefile documentation](../development/makefile.md) for details on each command.

To build the project, run:
```bash
make
```
This will set up the environment, install dependencies, and prepare everything for development.

### TensorBoard support

To set up TensorBoard support, run:

```bash
make setup-tensorboard
```
This will install the optional dependencies required for TensorBoard integration and set up the necessary configuration.


### Linter and type checker

Use [Ruff](https://github.com/charliermarsh/ruff) for linting and formatting:

```bash
make lint
make format
```

Use [Ty](https://docs.astral.sh/ty/) for type checking:

```bash
make typecheck
```

Use the `check` target to run both linter and type checker:

```bash
make check
```

## Contributing to the code base

### Pre-commit hooks

Run the following command to install the pre-commit hook:

```bash
make setup

uv run pre-commit install --config .github/hooks/.pre-commit-config.yaml \
    --hook-type pre-commit \
    --install-hooks --overwrite
```

### Tests

Use [pytest](https://docs.pytest.org/en/latest/) to run the test suite:

```bash
make test

# Run tests with HTML report
uv run pytest --cov=hyperbench --cov-report=html
```

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

This will ensure that your code adheres to the project's coding standards before each commit.

## Contributing to the documentation

The HyperBench documentation consists of two parts: the docstrings in the code and topic pages under `docs/`.

The docstrings provide a clear explanation of the usage of the individual functions, while the documentation in this folder consists of tutorial-like overviews per topic together with some other information (what’s new, installation, etc).

Docstrings follow the [Google style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) and are rendered in the documentation with [mkdocstrings](https://mkdocstrings.github.io/) through [Zensical](https://github.com/zensical/zensical).

When contributing to the documentation, please make sure to update both the docstrings and the relevant documentation pages if necessary.

## Policies

See the [Policies](../development/policies.md) page for information on versioning, Python support, and security policies.
