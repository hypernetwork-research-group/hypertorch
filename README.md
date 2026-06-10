# HyperBench

| | |
| --- | --- |
| Repo | [![Forks][forks-shield]][forks-url] [![Stargazers][stars-shield]][stars-url] [![Contributors][contributors-shield]][contributors-url] [![Issues][issues-shield]][issues-url] |
| Package | [![License: MIT][license-shield]][license-url] |
| Testing | [![codecov][codecov-shield]][codecov-url] [![CodeFactor][codefactor-shield]][codefactor-url] |
| Contact | ![Discord](https://badgen.net/discord/members/4krTXCWRzD) |

## About the project

HyperBench is a library for hypergraph learning and benchmarking. It provides a standardized workflow for loading hypergraph datasets, training models, evaluating them under comparable settings, and reporting results. The current release focuses on Hyperlink Prediction, with ready-to-run pipelines for established hypergraph baselines.

The library is built around extensibility: datasets are represented in [HIF](https://github.com/HIF-org/HIF-standard) format and converted into typed tensor objects, models can be implemented as standard Lightning modules, and benchmarking is handled through reusable trainers, samplers, metrics, loggers, and result exporters (Markdown/LaTeX). HyperBench includes preloaded datasets, mini-batch and full-hypergraph data loading, negative sampling utilities, structural feature enrichers, neural components, and built-in models such as HGNN, HNHN, HyperGCN, GCN, MLP/SLP, NHP, Node2Vec, VilLain, and more.

Use HyperBench to:
- Benchmark existing models across a shared collection of hypergraph datasets.
- Develop custom PyTorch or PyTorch Lightning models and train and compare them against the built-in baselines.
- Integrate new datasets through the HIF format and run the same training, evaluation, and reporting pipeline on them.

## Table of contents

- [Main features](#main-features)
- [Getting started](#getting-started)
    - [Run examples](#run-examples)
- [Contributing](#contributing)
- [Documentation](#documentation)
- [License](#license)
- [Discussion](#discussion)

## Main features

| Feature | What you can do | Highlights | Package |
| :--- | :--- | :--- | :--- |
| **Dataset management** | Load, preprocess, and manage hypergraph datasets | HIF loader/processor, built-in datasets such as Algebra, Cora, Pubmed, DBLP, Amazon, and IMDB | `hyperbench.data` |
| **Sampling and batching** | Sample sub-hypergraphs and prepare training batches | DataLoader, node and hyperedge samplers, and full-hypergraph evaluation batches | `hyperbench.data` |
| **Training and benchmarking** | Train and benchmark models out of the box | Multi-model trainer, negative sampling, schedulers, Markdown/LaTeX result tables | `hyperbench.train` |
| **Models** | Access a wide range of hypergraph models | HGNN, HGNNP, HNHN, HyperGCN, GCN, MLP/SLP, NHP, Node2Vec, VilLain, CommonNeighbors | `hyperbench.models` |
| **Neural network components** | Build custom architectures and pipelines | Convolutions, aggregators, losses, scorers, enrichers, positional encodings | `hyperbench.nn` |
| **HLP pipelines** | Use ready-to-run training and evaluation pipelines | HLP modules with encoders, configs, and loss definitions for multiple models | `hyperbench.hlp` |

## Getting started

For users working with the [pip](https://pip.pypa.io/en/stable/) package manager, hyperbench can be installed from PyPI.

```bash
pip install hyperbench

# if you want to install optional dependencies for tensorboard support:
pip install "hyperbench[tensorboard]"
```

or alternatively, using [uv](https://docs.astral.sh/uv/):

```bash
uv add hyperbench # or uv pip install hyperbench

# for optional dependencies for tensorboard support:
uv add "hyperbench[tensorboard]"
```

If you want to build the project from source, see the [documentation](#documentation) for more details.

### Run examples

You can download [examples](examples) directory and run the example scripts to get started.

With Python:

```bash
python3 examples/nhp.py
```

Or with `uv`:

```bash
uv run examples/nhp.py
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

<a href="https://www.star-history.com/?repos=hypernetwork-research-group%2Fhyperbench&type=date&logscale=&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=hypernetwork-research-group/hyperbench&type=date&theme=dark&logscale&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=hypernetwork-research-group/hyperbench&type=date&logscale&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=hypernetwork-research-group/hyperbench&type=date&logscale&legend=top-left" />
 </picture>
</a>

<!-- LINKS -->
[codecov-shield]: https://codecov.io/github/hypernetwork-research-group/hyperbench/graph/badge.svg?token=XE0TB5JMOS
[codecov-url]: https://codecov.io/github/hypernetwork-research-group/hyperbench
[codefactor-shield]: https://www.codefactor.io/repository/github/hypernetwork-research-group/hyperbench/badge
[codefactor-url]: https://www.codefactor.io/repository/github/hypernetwork-research-group/hyperbench
[contributors-shield]: https://img.shields.io/github/contributors/hypernetwork-research-group/hyperbench.svg?style=flat
[contributors-url]: https://github.com/hypernetwork-research-group/hyperbench/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/hypernetwork-research-group/hyperbench.svg?style=flat
[forks-url]: https://github.com/hypernetwork-research-group/hyperbench/network/members
[stars-shield]: https://img.shields.io/github/stars/hypernetwork-research-group/hyperbench.svg?style=flat
[stars-url]: https://github.com/hypernetwork-research-group/hyperbench/stargazers
[issues-shield]: https://img.shields.io/github/issues/hypernetwork-research-group/hyperbench.svg?style=flat
[issues-url]: https://github.com/hypernetwork-research-group/hyperbench/issues
[license-shield]: https://img.shields.io/badge/License-MIT-yellow.svg
[license-url]: https://opensource.org/licenses/MIT
[docs]: https://hypernetwork-research-group.github.io/hyperbench/
[issues]: https://github.com/hypernetwork-research-group/hyperbench/issues
