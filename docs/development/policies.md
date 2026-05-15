# Policies

## Version policy

HyperBench versions follow semantic versioning (`MAJOR.MINOR.PATCH`).

- The released version is defined in `pyproject.toml`.

## Python support

- Supported Python: `>=3.10` (see `pyproject.toml`).
- CI currently tests multiple Python versions (from 3.10 to 3.14).

If you hit install issues, ensure your `torch` / `torch-geometric` / `torch-cluster` versions are compatible with your Python version.

### Supported platforms

For each version we aim to support Linux, macOS, and Windows.

Our CI relys on [GitHub-hosted runners](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).

We routinely test these platforms in our pipelines:
- macos-latest, ubuntu-latest, windows-latest.
- macos-26, ubuntu-24.04, windows-2025.
- ubuntu-24.04-arm, ubuntu-slim.

We do not support these platforms:
- macos-26-intel for incompatibility with PyTorch.
- Windows 11 ARM for incompatibility with PyTorch.

| Virtual machine / container | Processor (CPU) | Memory (RAM) | Storage (SSD) | Architecture | Workflow label | Supported |
|---|---:|---:|---:|---|---|:--:|
| Linux | 1 | 5 GB | 14 GB | x64 | ubuntu-slim | :heavy_check_mark: |
| Linux | 4 | 16 GB | 14 GB | x64 | ubuntu-latest, ubuntu-24.04, ubuntu-22.04 | :heavy_check_mark: |
| Windows | 4 | 16 GB | 14 GB | x64 | windows-latest, windows-2025, windows-2025-vs2026, windows-2022 | :heavy_check_mark: |
| Linux | 4 | 16 GB | 14 GB | arm64 | ubuntu-24.04-arm, ubuntu-22.04-arm | :heavy_check_mark: |
| Windows | 4 | 16 GB | 14 GB | arm64 | windows-11-arm | :x: |
| macOS | 4 | 14 GB | 14 GB | Intel | macos-15-intel, macos-26-intel | :x: |
| macOS | 3 (M1) | 7 GB | 14 GB | arm64 | macos-latest, macos-14, macos-15, macos-26 | :heavy_check_mark: |


## Security policy

If you believe you’ve found a security vulnerability, please **do not open a public issue**.

- Follow the reporting instructions in [SECURITY.md](https://github.com/hypernetwork-research-group/hyperbench/blob/main/SECURITY.md).
