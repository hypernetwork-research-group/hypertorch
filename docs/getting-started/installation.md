# Installation

## Install with pip or uv

For users working with the pip package manager, hypertorch can be installed from PyPI.

```bash
pip install hypertorch
# if you want to install optional dependencies for tensorboard support:
pip install "hypertorch[tensorboard]"
```

Additionally, it is recommended to install and run hypertorch from a virtual environment, for example, using the Python standard library’s venv.
Internally, we use [uv](https://github.com/astral-sh/uv) as a build and development tool, which also provides a convenient way to manage virtual environments and dependencies.
After installing uv, you can create a project environment and add hypertorch with:

```bash
uv init
uv add hypertorch # or uv pip install hypertorch

# For optional dependencies
uv add "hypertorch[tensorboard]"
```

## Python version support

See Python support policy in [Policies](../development/policies.md#python-support).

## Install from source

Use the development installation for contributing or if you want to use the latest features that haven't been released yet. See the [Development guide](../development/development.md) for instructions on setting up a development environment.

## Required dependencies

HyperTorch declares compatibility ranges for direct dependencies in `pyproject.toml`.

| Dependency | Supported range | Markers / notes |
| --- | --- | --- |
| fastjsonschema | `>=2.21.2,<3.0.0` |  |
| huggingface-hub | `>=1.16.4,<2.0.0` |  |
| lightning | `>=2.6.1,<3.0.0` |  |
| numpy | `>=2.2.6,<3.0.0` | `python_full_version < '3.11'` |
| numpy | `>=2.4.4,<3.0.0` | `python_full_version >= '3.11'` |
| pyg-lib | `>=0.6.0,<1.0.0` | Installed via a custom `uv` index on macOS arm64, Linux x86_64, and Windows x64 |
| requests | `>=2.34.2,<3.0.0` |  |
| torch | `>=2.11.0,<3.0.0` |  |
| torch-cluster | `>=1.6.3,<2.0.0` | Installed via a custom `uv` index on platforms without a compatible `pyg-lib` wheel |
| torch-geometric | `>=2.7.0,<2.8.0` | Capped below 2.8 so Node2Vec can use the `torch-cluster` fallback |
| zstandard | `>=0.25.0,<1.0.0` |  |

## Optional dependencies

| Dependency | Supported range | Notes |
| --- | --- | --- |
| tensorboard | `>=2.20.0,<3.0.0` | See [TensorBoard Integration](../development/development.md#tensorboard-support) |

## Development dependencies

| Dependency | Supported range | Notes |
| --- | --- | --- |
| mkdocstrings\[python\] | `>=1.0.4,<2.0.0` |  |
| pre-commit | `>=4.5.1,<5.0.0` |  |
| pytest | `>=9.0.3,<10.0.0` |  |
| pytest-cov | `>=7.1.0,<8.0.0` |  |
| pytest-rerunfailures | `>=16.3,<17.0.0` |  |
| pytest-xdist | `>=2.5.0,<3.0.0` |  |
| ruff | `>=0.15.11,<1.0.0` |  |
| ty | `>=0.0.34,<1.0.0` |  |
| zensical | `>=0.0.43,<1.0.0` |  |
