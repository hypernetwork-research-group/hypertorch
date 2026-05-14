# Installation


## Install with pip or uv

For users working with the pip package manager, hyperbench can be installed from PyPI.

```bash
pip install hyperbench
# if you want to install optional dependencies for tensorboard support:
pip install "hyperbench[tensorboard]"
```

Additionally, it is recommended to install and run hyperbench from a virtual environment, for example, using the Python standard library’s venv.
Internally, we use [uv](https://github.com/astral-sh/uv) as a build and development tool, which also provides a convenient way to manage virtual environments and dependencies.
After installing uv, you can set up the environment and install hyperbench with:

```bash
uv init
uv add hyperbench # or uv pip install hyperbench
# for optional dependencies:
uv add "hyperbench[tensorboard]"
```

## Python version support

See Python support policy in [Policies](../development/policies.md#python-support).

## Install from source

Use the development installation for contributing or if you want to use the latest features that haven't been released yet. See the [Development guide](../development/development.md) for instructions on setting up a development environment.

## Required dependencies

Hyperbench has the following required dependencies:

| Dependency | Version | Markers / notes |
| --- | --- | --- |
| fastjsonschema | 2.21.2 |  |
| huggingface-hub | 1.11.0 |  |
| lightning | 2.6.1 |  |
| numpy | 2.2.6 | `python_full_version < '3.11'` |
| numpy | 2.4.4 | `python_full_version >= '3.11'` |
| requests | 2.33.1 |  |
| torch | 2.11.0 |  |
| torch-cluster | 1.6.3 | Installed via a custom `uv` index (`pyg-cpu`) |
| torch-geometric | 2.7.0 |  |
| zstandard | 0.25.0 |  |

## Optional dependencies

| Dependency | Version | Notes |
| --- | --- | --- |
| tensorboard | 2.20.0 | See [TensorBoard Integration](../development/development.md#tensorboard-support) |

## Development dependencies

| Dependency | Version | Notes |
| --- | --- | --- |
| mkdocstrings-python | 1.0.4 |  |
| pre-commit | 4.5.1 |  |
| pytest | 9.0.3 |  |
| pytest-cov | 7.1.0 |  |
| ruff | 0.15.11 |  |
| ty | 0.0.34 |  |
| zensical | 0.0.41 |  |
