# Policies

## Version policy

HyperBench versions follow semantic versioning (`MAJOR.MINOR.PATCH`).

- The released version is defined in `pyproject.toml`.

## Python support

- Supported Python: `>=3.10` (see `pyproject.toml`).
- CI currently tests multiple Python versions (from 3.10 to 3.14).

If you hit install issues, ensure your `torch` / `torch-geometric` / PyG extension versions are compatible with your Python version and platform. HyperBench installs `pyg-lib` where the configured PyG CPU index publishes compatible wheels and `torch-cluster` as the fallback extension elsewhere.

### Supported platforms

For each version we aim to support Linux, macOS, and Windows.

Our CI relies on [GitHub-hosted runners](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).

The current CI matrix tests these runner labels:
- `ubuntu-latest`, `ubuntu-24.04`, and `ubuntu-24.04-arm`.
- `macos-latest` and `macos-26`.
- `windows-latest` and `windows-2025`.

We do not support these platforms:
- `macos-26-intel` for incompatibility with PyTorch.
- `windows-11-arm` for incompatibility with PyTorch.

| Virtual machine / container | Processor (CPU) | Memory (RAM) | Storage (SSD) | Architecture | Workflow label | Supported |
|---|---:|---:|---:|---|---|:--:|
| Linux | 1 | 5 GB | 14 GB | x64 | ubuntu-slim | :x: |
| Linux | 4 | 16 GB | 14 GB | x64 | ubuntu-latest, ubuntu-24.04 | :heavy_check_mark: |
| Windows | 4 | 16 GB | 14 GB | x64 | windows-latest, windows-2025 | :heavy_check_mark: |
| Linux | 4 | 16 GB | 14 GB | arm64 | ubuntu-24.04-arm | :heavy_check_mark: |
| Windows | 4 | 16 GB | 14 GB | arm64 | windows-11-arm | :x: |
| macOS | 4 | 14 GB | 14 GB | Intel | macos-26-intel | :x: |
| macOS | 3 (M1) | 7 GB | 14 GB | arm64 | macos-latest, macos-26 | :heavy_check_mark: |

## Security policy

If you believe you’ve found a security vulnerability, please **do not open a public issue**.

- Follow the reporting instructions in [SECURITY.md](https://github.com/hypernetwork-research-group/hyperbench/blob/main/SECURITY.md).
