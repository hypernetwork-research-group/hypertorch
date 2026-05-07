# How to contribute

The project main language is English.

## Quickstart for contributors

We use different tools to make our life easier.
Altough they are not mandatory, we have several configuration to help you build the project:

- [uv](https://github.com/astral-sh/uv)
- [make](https://www.gnu.org/software/make/)
- [pre-commit](https://pre-commit.com)

```bash
git clone https://github.com/hypernetwork-research-group/hyperbench.git
cd hyperbench

# install pre-commit to setup automatic local check
# This will ensure that your code adheres to the project's coding standards before each commit.
pre-commit install \
    --config .github/hooks/.pre-commit-config.yaml \
    --install-hooks \
    --overwrite

# too see all the available commands already mapped in Makefile
make help

# install dependencies with uv
make build
# or run the full suite, which include building + test
make
```

## Contribution types + expectations

### Bugfix

### Feature

### Docs
Check [Docs Contribution](#docs-contribution) for more details.

## Workflow

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

## AI-assited contributions policy

## Docs Contribution


<!--  -->
<!--  -->
<!--  -->
<!--  -->
<!--  -->
