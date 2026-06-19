# HyperTorch

| | |
| --- | --- |
| Repo | [![Forks][forks-shield]][forks-url] [![Stargazers][stars-shield]][stars-url] [![Contributors][contributors-shield]][contributors-url] [![Issues][issues-shield]][issues-url] |
| Package | [![License: MIT][license-shield]][license-url] [![Python][python-shield]][python-url] [![Documentation][docs-shield]][docs-url] |
| Testing | ![Daily CI][daily-ci-shield] [![codecov][codecov-shield]][codecov-url] [![CodeFactor][codefactor-shield]][codefactor-url] |
| Contact | ![Discord](https://badgen.net/discord/members/4krTXCWRzD) |

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
    - [Run examples](#run-examples)
- [Contributing](#contributing)
- [Documentation](#documentation)
- [License](#license)
- [Discussion](#discussion)

## Main features

| Feature | What you can do | Highlights | Package |
| :--- | :--- | :--- | :--- |
| **Dataset management** | Load, preprocess, and manage hypergraph datasets | HIF loader/processor, built-in datasets such as Algebra, Cora, Pubmed, DBLP, Amazon, and IMDB | `hypertorch.data` |
| **Sampling and batching** | Sample sub-hypergraphs and prepare training batches | DataLoader, node and hyperedge samplers, and full-hypergraph evaluation batches | `hypertorch.data` |
| **Training and benchmarking** | Train and benchmark models out of the box | Multi-model trainer, negative sampling, schedulers, Markdown/LaTeX result tables | `hypertorch.train` |
| **Models** | Access a wide range of hypergraph models | HGNN, HGNNP, HNHN, HyperGCN, GCN, MLP/SLP, NHP, Node2Vec, VilLain, CommonNeighbors | `hypertorch.models` |
| **Neural network components** | Build custom architectures and pipelines | Convolutions, aggregators, losses, scorers, enrichers, positional encodings | `hypertorch.nn` |
| **HLP pipelines** | Use ready-to-run training and evaluation pipelines | HLP modules with encoders, configs, and loss definitions for multiple models | `hypertorch.hlp` |

## Getting started

For users working with the [pip](https://pip.pypa.io/en/stable/) package manager, hypertorch can be installed from PyPI.

```bash
pip install hypertorch

# if you want to install optional dependencies for tensorboard support:
pip install "hypertorch[tensorboard]"
```

or alternatively, using [uv](https://docs.astral.sh/uv/):

```bash
uv add hypertorch # or uv pip install hypertorch

# for optional dependencies for tensorboard support:
uv add "hypertorch[tensorboard]"
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
