# How to contribute

The project main language is English.

## Pre-commit hooks

Run the following command to install the pre-commit hook:

```bash
make setup

pre-commit install \
    --config .github/hooks/.pre-commit-config.yaml \
    --hook-type pre-commit \
    --install-hooks \
    --overwrite
```

This will ensure that your code adheres to the project's coding standards before each commit.

## Commit message style

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

## Branch naming

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
