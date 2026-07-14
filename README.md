# HyperTorch

| | |
| --- | --- |
| Repo | [![Forks][forks-shield]][forks-url] [![Stargazers][stars-shield]][stars-url] [![Contributors][contributors-shield]][contributors-url] [![Issues][issues-shield]][issues-url] |
| Package | [![License: MIT][license-shield]][license-url] [![Python][python-shield]][python-url] [![Documentation][docs-shield]][docs-url] |
| Testing | ![Daily CI][daily-ci-shield] [![codecov][codecov-shield]][codecov-url] [![CodeFactor][codefactor-shield]][codefactor-url] |
| Contact | [![Discord][discord-shield]][discord-url] |

## About the project

HyperTorch is a library for hypergraph learning and benchmarking. It provides a standardized workflow for loading hypergraph datasets, training models, evaluating them under comparable settings, and reporting results. The current release focuses on Hyperlink Prediction, with ready-to-run pipelines for established hypergraph baselines.

The library is built around extensibility: datasets are represented in [HIF](https://github.com/HIF-org/HIF-standard) format and converted into typed tensor objects, models can be implemented as standard Lightning modules, and benchmarking is handled through reusable trainers, samplers, metrics, loggers, and result exporters (Markdown/LaTeX). HyperTorch includes preloaded datasets, mini-batch and full-hypergraph data loading, negative sampling utilities, structural feature enrichers, neural components, and built-in models such as HGNN, HNHN, HyperGCN, GCN, MLP/SLP, NHP, Node2Vec, VilLain, and more.

Use HyperTorch to:
- Benchmark existing models across a shared collection of hypergraph datasets.
- Develop custom PyTorch or PyTorch Lightning models and train and compare them against the built-in baselines.
- Integrate new datasets through the HIF format and run the same training, evaluation, and reporting pipeline on them.

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
- [Discussion](#discussion)

## Main features

| Feature | What you can do | Highlights | Location |
| :--- | :--- | :--- | :--- |
| **Dataset management** | Load, process, and validate hypergraph datasets | HIF loader/processor, built-in datasets such as Algebra, Cora, Pubmed, DBLP, Amazon, and IMDB | `hypertorch.data` |
| **Splitting, sampling, and batching** | Prepare train/validation/test data and mini-batches | Dataset splitters, node and hyperedge samplers, negative samplers, data loaders | `hypertorch.data` |
| **Feature enrichment** | Enrich node and hyperedge features before training | Laplacian positional encodings, Node2Vec features, hyperedge weights and attributes | `hypertorch.data` |
| **Models** | Access hypergraph models | HGNN, HGNNP, HNHN, HyperGCN, GCN, MLP/SLP, NHP, Node2Vec, VilLain, CommonNeighbors | `hypertorch.models` |
| **Neural components** | Build models and pipelines | Layers, aggregators, losses, and activation/normalization helpers | `hypertorch.nn` |
| **HLP pipelines** | Use ready-to-train hyperlink prediction modules | HLP modules with encoders, configs, losses, and stage metrics for multiple models | `hypertorch.hlp` |
| **Training and benchmarking** | Train, compare, checkpoint, and report model runs | Multi-model trainer, schedulers, TensorBoard support, CSV/Markdown/LaTeX result tables | `hypertorch.train` |

## Getting started

### Installation

HyperTorch can be installed from PyPI when you want to use it as a dependency, or from source when you want to contribute or run the latest repository version.

CI pipelines validate CPU installs on Python 3.10 through 3.14 for Linux x86_64, Linux ARM/aarch64, macOS arm64, and Windows x64. Install the matching PyTorch and PyG wheels for your platform (e.g., CUDA) before installing HyperTorch.

For more detailed instructions, see the [installation guide](docs/getting-started/installation.md).

### Source installation

```bash
git clone https://github.com/hypernetwork-research-group/hypertorch.git
cd hypertorch

make setup
```

See the [installation guide](docs/getting-started/installation.md) for platform
notes and dependency ranges.

### TensorBoard support

To include TensorBoard support, also run HyperTorch install command with the TensorBoard extra:

```bash
uv pip install "hypertorch[tensorboard]"
```

When installing from source, run the command:

```bash
make setup-tensorboard
```

### Run examples

You can download the [examples](examples) directory and run the example scripts to get started.

With Python:

```bash
python3 examples/hyperlink_prediction/nhp.py
```

Or with `uv`:

```bash
uv run examples/hyperlink_prediction/nhp.py
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on contributing to the project.

## Documentation

You can find the extensive documentation [here][docs].

Alternatively, you can build the documentation locally with the following commands:

```bash
make docs

# With explicit commands
uv run zensical build --clean -f zensical.toml
uv run zensical serve -f zensical.toml -a 127.0.0.1:8000
```
and open the browser at http://localhost:8000 to access the documentation.

## License

See [LICENSE](LICENSE).

## Discussion

Most development discussions take place on GitHub in this repo, via the [GitHub issue tracker][issues].

![Alt](https://repobeats.axiom.co/api/embed/c168082ecb1f9f843c1b170dcfee93542b576f61.svg "Repobeats analytics image")

<a href="https://www.star-history.com/?repos=hypernetwork-research-group%2Fhypertorch&type=date&logscale=&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=hypernetwork-research-group/hypertorch&type=date&theme=dark&logscale&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=hypernetwork-research-group/hypertorch&type=date&logscale&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=hypernetwork-research-group/hypertorch&type=date&logscale&legend=top-left" />
 </picture>
</a>

<!-- LINKS -->
[codecov-shield]: https://codecov.io/github/hypernetwork-research-group/hypertorch/graph/badge.svg?token=XE0TB5JMOS
[codecov-url]: https://codecov.io/github/hypernetwork-research-group/hypertorch
[codefactor-shield]: https://www.codefactor.io/repository/github/hypernetwork-research-group/hypertorch/badge
[codefactor-url]: https://www.codefactor.io/repository/github/hypernetwork-research-group/hypertorch
[daily-ci-shield]: https://github.com/hypernetwork-research-group/hypertorch/actions/workflows/daily_ci.yaml/badge.svg
[contributors-shield]: https://img.shields.io/github/contributors/hypernetwork-research-group/hypertorch.svg?style=flat
[contributors-url]: https://github.com/hypernetwork-research-group/hypertorch/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/hypernetwork-research-group/hypertorch.svg?style=flat
[forks-url]: https://github.com/hypernetwork-research-group/hypertorch/network/members
[stars-shield]: https://img.shields.io/github/stars/hypernetwork-research-group/hypertorch.svg?style=flat
[stars-url]: https://github.com/hypernetwork-research-group/hypertorch/stargazers
[issues-shield]: https://img.shields.io/github/issues/hypernetwork-research-group/hypertorch.svg?style=flat
[issues-url]: https://github.com/hypernetwork-research-group/hypertorch/issues
[license-shield]: https://img.shields.io/badge/License-MIT-yellow.svg
[license-url]: https://opensource.org/licenses/MIT
[docs]: https://hypernetwork-research-group.github.io/hypertorch/
[issues]: https://github.com/hypernetwork-research-group/hypertorch/issues
[python-shield]: https://img.shields.io/badge/python-3.10%2B-blue.svg
[python-url]: https://www.python.org/downloads/
[docs-shield]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-url]: https://hypernetwork-research-group.github.io/hypertorch/
[discord-shield]: https://img.shields.io/discord/693092516286693387
[discord-url]: https://discord.gg/4krTXCWRzD
