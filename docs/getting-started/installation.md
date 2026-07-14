# Installation

HyperTorch can be installed from PyPI when you want to use it as a dependency,
or from source when you want to contribute or run the latest repository version.

The examples below use [uv](https://github.com/astral-sh/uv), which is also the
build and development tool used inside the project. If you prefer `pip`, replace
`uv pip install` with `pip install` in the installation commands.

## Install from PyPI

The commands below install the CPU wheels for Python 3.10 through 3.14 on the supported platforms:

- [Linux x86_64](#linux-x86-64).
- [Linux ARM/aarch64](#linux-arm-aarch64).
- [macOS arm64](#macos-arm64).
- [Windows x64](#windows-x64).

If you are using CUDA or different hardware, install the matching PyTorch and PyG wheels first,
while staying within the dependency ranges declared by HyperTorch. Then, install HyperTorch
with the same final command for your platform.

!!! note "Planned removal of PyG and Torch Geometric dependencies"
    HyperTorch currently depends on PyG and Torch Geometric, which is why the
    installation steps include platform-specific PyTorch and PyG wheels. We are
    moving away from these dependencies to make installation easier in future
    releases.

### Linux x86_64 {#linux-x86-64}

```bash
uv pip install "torch>=2.12.0,<2.13.0" --index-url https://download.pytorch.org/whl/cpu
uv pip install pyg-lib --find-links https://data.pyg.org/whl/torch-2.12.0+cpu.html
uv pip install hypertorch
```

### Linux ARM/aarch64 {#linux-arm-aarch64}

```bash
uv pip install "torch>=2.11.0,<2.12.0" --index-url https://download.pytorch.org/whl/cpu
uv pip install torch-cluster --find-links https://data.pyg.org/whl/torch-2.11.0+cpu.html
uv pip install hypertorch
```

### macOS arm64 {#macos-arm64}

```bash
uv pip install "torch>=2.12.0,<2.13.0" --index-url https://download.pytorch.org/whl/cpu
uv pip install pyg-lib --find-links https://data.pyg.org/whl/torch-2.12.0+cpu.html
uv pip install hypertorch
```

### Windows x64 {#windows-x64}

```bash
uv pip install "torch>=2.12.0,<2.13.0" --index-url https://download.pytorch.org/whl/cpu
uv pip install pyg-lib --find-links https://data.pyg.org/whl/torch-2.12.0+cpu.html
uv pip install hypertorch
```

## Install from source

Use the development installation for contributing or if you want to use the
latest features that have not been released yet. The Makefile wraps the standard
project setup commands:

```bash
git clone https://github.com/hypernetwork-research-group/hypertorch.git
cd hypertorch

make setup
```

See the [Development guide](../development/development.md) for the full
development workflow.

## Python version support

See Python support policy in [Policies](../development/policies.md#python-support).

## TensorBoard support

To include TensorBoard support, also run HyperTorch install command with the TensorBoard extra:

```bash
uv pip install "hypertorch[tensorboard]"
```

When installing from source, run the command:

```bash
make setup-tensorboard
```

## Required dependencies

HyperTorch declares compatibility ranges for direct dependencies in `pyproject.toml`.

| Dependency | Supported range | Markers / notes |
| --- | --- | --- |
| fastjsonschema | `>=2.21.2,<3.0.0` |  |
| huggingface-hub | `>=1.16.4,<2.0.0` |  |
| lightning | `>=2.6.1,<3.0.0` |  |
| numpy | `>=2.2.6,<3.0.0` | `python_full_version < '3.11'` |
| numpy | `>=2.4.4,<3.0.0` | `python_full_version >= '3.11'` |
| pyg-lib | `>=0.6.0,<1.0.0` | Linux x86_64, macOS arm64, and Windows x64; install from the PyG CPU wheel index for Torch 2.12 |
| requests | `>=2.34.2,<3.0.0` |  |
| torch | `>=2.12.0,<2.13.0` | Linux x86_64, macOS arm64, and Windows x64 |
| torch | `>=2.11.0,<2.12.0` | Linux aarch64 |
| torch-cluster | `>=1.6.3,<2.0.0` | Linux aarch64 fallback extension; install from the PyG CPU wheel index for Torch 2.11 |
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
| pytest-xdist | `>=3.0.0,<4.0.0` |  |
| ruff | `>=0.15.11,<1.0.0` |  |
| ty | `>=0.0.34,<1.0.0` |  |
| zensical | `>=0.0.44,<1.0.0` |  |
