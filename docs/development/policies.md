# Policies

## Version policy

HyperBench versions follow semantic versioning (`MAJOR.MINOR.PATCH`).

- The released version is defined in `pyproject.toml`.

## Python support

- Supported Python: `>=3.10` (see `pyproject.toml`).
- CI currently tests multiple Python versions (from 3.10 to 3.14).

If you hit install issues, ensure your `torch` / `torch-geometric` / `torch-cluster` versions are compatible with your Python version.

## Security policy

If you believe you’ve found a security vulnerability, please **do not open a public issue**.

- Follow the reporting instructions in [SECURITY.md](https://github.com/hypernetwork-research-group/hyperbench/blob/main/SECURITY.md).
