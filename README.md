<picture align="center">
  <img
    alt="HyperTorch Logo"
    src="https://raw.githubusercontent.com/hypernetwork-research-group/hypertorch/main/docs/assets/horizontal.svg"
  />
</picture>

---

<h1 align="center">HyperTorch: A Python library for<br>hypergraph learning and benchmarking</h1>

<p align="center">
  <a href="https://hypernetwork-research-group.github.io/hypertorch/">Documentation</a>
  |
  <a href="https://hypernetwork-research-group.github.io/hypertorch/getting-started/overview/">Getting started</a>
  |
  <a href="https://hypernetwork-research-group.github.io/hypertorch/getting-started/tutorials/">Tutorials</a>
  |
  <a href="https://hypernetwork-research-group.github.io/hypertorch/development/contribution/">Contributing</a>
</p>

<table align="center">
  <tbody>
    <tr>
      <td>Package</td>
      <td>
        <a href="https://pypi.org/project/hypertorch/"><img src="https://img.shields.io/pypi/v/hypertorch.svg?label=PyPI" alt="PyPI Latest Release"></a>
        <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-yellow.svg" alt="License: Apache 2.0"></a>
        <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg?label=Python" alt="Python"></a>
      </td>
    </tr>
    <tr>
      <td>Testing</td>
      <td>
        <a href="https://github.com/hypernetwork-research-group/hypertorch/actions/workflows/coverage.yaml">
            <img src="https://img.shields.io/github/actions/workflow/status/hypernetwork-research-group/hypertorch/coverage.yaml?branch=main&amp;label=Unit%20tests" alt="Unit tests">
        </a>
        <a href="https://github.com/hypernetwork-research-group/hypertorch/actions/workflows/daily_ci.yaml">
            <img src="https://img.shields.io/github/actions/workflow/status/hypernetwork-research-group/hypertorch/daily_ci.yaml?branch=main&amp;label=Integration%20tests" alt="Integration tests">
        </a>
        <a href="https://github.com/hypernetwork-research-group/hypertorch/actions/workflows/weekly_ci.yaml">
            <img src="https://img.shields.io/github/actions/workflow/status/hypernetwork-research-group/hypertorch/weekly_ci.yaml?branch=main&amp;label=Released%20version%20tests" alt="Released version tests">
        </a>
      </td>
    </tr>
    <tr>
      <td>Quality assurance</td>
      <td>
        <a href="https://codecov.io/github/hypernetwork-research-group/hypertorch">
            <img src="https://codecov.io/github/hypernetwork-research-group/hypertorch/graph/badge.svg?token=XE0TB5JMOS" alt="codecov">
        </a>
        <a href="https://www.codefactor.io/repository/github/hypernetwork-research-group/hypertorch">
            <img src="https://www.codefactor.io/repository/github/hypernetwork-research-group/hypertorch/badge" alt="CodeFactor">
        </a>
      </td>
    </tr>
    <tr>
      <td>Project & community</td>
      <td>
        <a href="https://github.com/hypernetwork-research-group/hypertorch/issues">
            <img src="https://img.shields.io/github/issues/hypernetwork-research-group/hypertorch.svg?style=flat&amp;label=Issues" alt="Issues">
        </a>
        <a href="https://github.com/hypernetwork-research-group/hypertorch/stargazers">
            <img src="https://img.shields.io/github/stars/hypernetwork-research-group/hypertorch.svg?style=flat&amp;label=Stars" alt="Stargazers">
        </a>
        <a href="https://github.com/hypernetwork-research-group/hypertorch/network/members">
            <img src="https://img.shields.io/github/forks/hypernetwork-research-group/hypertorch.svg?style=flat&amp;label=Forks" alt="Forks">
        </a>
        <a href="https://github.com/hypernetwork-research-group/hypertorch/graphs/contributors">
            <img src="https://img.shields.io/github/contributors/hypernetwork-research-group/hypertorch.svg?style=flat&amp;label=Contributors" alt="Contributors">
        </a>
        <a href="https://discord.gg/4krTXCWRzD">
            <img src="https://img.shields.io/discord/693092516286693387?style=flat&amp;label=Discord" alt="Discord">
        </a>
      </td>
    </tr>
  </tbody>
</table>

## About the project

HyperTorch is a library for hypergraph learning and benchmarking. It provides standardized workflows for loading hypergraph datasets, training models, evaluating them under comparable
settings, and reporting results for both hyperlink prediction and node classification.

The library is built around extensibility: datasets are represented in [HIF](https://github.com/HIF-org/HIF-standard) format and converted into typed tensor objects, models can be implemented as standard Lightning modules, and benchmarking is handled through reusable trainers, samplers, metrics, loggers, and result exporters (Markdown/LaTeX). HyperTorch includes preloaded datasets, mini-batch and full-hypergraph data loading, negative sampling utilities, structural feature enrichers, neural components, and many built-in models.

Use HyperTorch to:

- Benchmark existing models across a shared collection of hypergraph datasets.
- Develop custom PyTorch or Lightning models and compare them with built-in baselines.
- Load local or remote `.json` and `.json.zst` HIF datasets and run the same training, evaluation, and reporting pipeline on them.

## Table of contents

- [Main features](#main-features)
- [Getting started](#getting-started)
    - [Installation](#installation)
    - [Source installation](#source-installation)
    - [TensorBoard support](#tensorboard-support)
    - [Run examples](#run-examples)
- [Contributing](#contributing)
- [Documentation](#documentation)
- [License](#license)
- [Support](#support)
- [Cite](#cite)

## Main features

| | What you can do | Public APIs |
| :--- | :--- | :--- |
| **Data and HIF integration** | Load built-in datasets or `.json`/`.json.zst` HIF data from files and URLs and work with typed `HData` objects | `hypertorch.data`, `hypertorch.types` |
| **Preparation and enrichment** | Split datasets, sample nodes or hyperedges, generate negative samples, batch data, and enrich node or hyperedge features | `hypertorch.data` |
| **Hyperlink prediction** | Use ready-to-train hyperlink prediction pipelines | `hypertorch.hyperlink_prediction` |
| **Node classification** | Use ready-to-train node classification pipelines | `hypertorch.node_classification` |
| **Models and neural components** | Reuse model implementations, layers, aggregators, losses, activations, and normalization helpers | `hypertorch.models`, `hypertorch.nn` |
| **Training and benchmarking** | Train and compare multiple models with shared data, callbacks, device settings, checkpoints, and per-model trainer options | `hypertorch.train`, `hypertorch.types` |
| **Logging and visualization** | Write CSV metrics and Markdown/LaTeX comparison tables; optionally log to and auto-start TensorBoard | `hypertorch.train` |

## Getting started

### Installation

HyperTorch requires Python 3.10 or newer. The currently documented and CI-tested range is Python 3.10 through 3.14 on Linux x86_64 and ARM/aarch64, macOS arm64, and Windows x64.

For a CPU installation, follow the platform-specific [installation guide][installation] to install compatible PyTorch and PyG wheels, then install HyperTorch from PyPI:

```bash
uv pip install hypertorch
```

If you use `pip`, replace `uv pip install` with `pip install`. For CUDA or other hardware, you can install the matching PyTorch and PyG wheels within HyperTorch's declared dependency ranges before installing HyperTorch.

### Source installation

```bash
git clone https://github.com/hypernetwork-research-group/hypertorch.git
cd hypertorch

make setup
```

See the [installation guide][installation] for platform notes and dependency ranges.

### TensorBoard support

Install the optional TensorBoard integration from PyPI with:

```bash
uv pip install "hypertorch[tensorboard]"
```

For a source installation, use:

```bash
make setup-tensorboard
```

### Run examples

Run examples from the repository root with `make run`. For example:

```bash
# Hyperlink prediction
make run examples/hyperlink_prediction/nhp.py

# Node classification
make run examples/node_classification/hypergcn.py
```

The [tutorials guide][tutorials] lists examples for dataset loading, feature enrichment, hyperlink prediction, node classification, sampling, splitting, and training customization.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contributor quickstart and the [development guide][development] for the complete workflow.

## Documentation

Read the [documentation][docs] for installation, tutorials, user guides, API references, development guidance, and release notes.

Build or serve it locally with the documented Makefile targets:

```bash
make docs-build
make docs-serve
```

Use `make docs` to build and serve in one command. The local site is available at <http://127.0.0.1:8000>.

## License

This project is released under the Apache License 2.0 license. See [LICENSE](LICENSE).

## Support

- Use [GitHub Discussions][discussions] for questions and ideas.
- Use the [GitHub issue tracker][issues] for bugs and feature requests.
- Use [Discord][discord] for community chat.

Please follow [SECURITY.md](SECURITY.md) instead of opening a public issue for suspected security vulnerabilities.

## Cite

Please cite us if you use HyperTorch in your work:

```bibtex
@software{Citro_HyperTorch_2026,
    author = {Citro, Tiziano and De Vinco, Daniele and Spagnuolo, Carmine},
    title = {{HyperTorch: A Python Library for Hypergraph Learning and Benchmarking}},
    url = {https://github.com/hypernetwork-research-group/hypertorch},
    year = {2026}
}
```

<!-- LINKS -->
[discussions]: https://github.com/hypernetwork-research-group/hypertorch/discussions
[discord]: https://discord.gg/4krTXCWRzD
[docs]: https://hypernetwork-research-group.github.io/hypertorch/
[issues]: https://github.com/hypernetwork-research-group/hypertorch/issues
[installation]: https://hypernetwork-research-group.github.io/hypertorch/getting-started/installation/
[tutorials]: https://hypernetwork-research-group.github.io/hypertorch/getting-started/tutorials/
[development]: https://hypernetwork-research-group.github.io/hypertorch/development/development/
