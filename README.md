# HyperBench

| | |
| --- | --- |
| Repo | [![Forks][forks-shield]][forks-url] [![Stargazers][stars-shield]][stars-url] [![Contributors][contributors-shield]][contributors-url] [![Issues][issues-shield]][issues-url] |
| Package | [![License: MIT][license-shield]][license-url] |
| Testing | [![codecov][codecov-shield]][codecov-url] [![CodeFactor][codefactor-shield]][codefactor-url] |
| Contact | ![Discord](https://badgen.net/discord/members/4krTXCWRzD) |

## About the project

HyperBench is a benchmarking framework for hypergraphs, designed to provide a standardized and comprehensive evaluation of hypernetwork models across various tasks and datasets.
You can use it as plug-and-play for your dataset with all the models and training pipelines already implemented, or you can easily extend it with your own models and training pipelines.

## Table of contents

- [Main features](#main-features)
- [Getting started](#getting-started)
    - [Run examples](#run-examples)
- [Contributing](#contributing)
- [Documentation](#documentation)
- [License](#license)
- [Discussion](#discussion)

## Main features

| Feature Area                    | What You Can Do                                           | Highlights                                                                         | Main Entry Points   |
| :------------------------------ | :-------------------------------------------------------- | :--------------------------------------------------------------------------------- | :------------------ |
| **Dataset management**       | Load, preprocess, and manage hypergraph datasets          | HIF loader/processor, built-in datasets like Cora, Pubmed, DBLP, Amazon, IMDB      | `hyperbench.data`   |
| **Sampling & batching**       | Efficiently sample subgraphs and prepare training batches | DataLoader, node/hyperedge samplers, customizable sampling strategies              | `hyperbench.data`   |
| **Training & benchmarking** | Train and benchmark models out of the box                 | Multi-model trainer, negative sampling, schedulers, Markdown/LaTeX result tables   | `hyperbench.train`  |
| **Models**                | Access a wide range of hypergraph models                  | HGNN, HGNNP, HNHN, HyperGCN, GCN, MLP/SLP, NHP, Node2Vec, VilLain, CommonNeighbors | `hyperbench.models` |
| **Neural components**        | Build custom architectures and pipelines                  | Convolutions, aggregators, losses, scorers, enrichers, positional encodings        | `hyperbench.nn`     |
| **Task pipelines**           | Use ready-to-run training and evaluation pipelines        | HLP modules with encoders, configs, and loss definitions for multiple models       | `hyperbench.hlp`    |
| **Typed APIs**              | Work with strongly typed graph/data/model objects         | Consistent and extensible APIs for datasets and models                             | `hyperbench.types`  |``


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
# for optional dependencies:
uv add "hyperbench[tensorboard]"
```

If you want to build the project from source, see the [documentation](#documentation) for more details.

### Run examples

You can download [examples](examples) directory and run the example scripts to get started. For instance:

```bash
python3 examples/early_stopping.py
```
or with `uv`:

```bash
uv run examples/early_stopping.py
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on contributing to the project.

## Documentation

You can find the extensive documentation [here][docs].
Alternatively, you can build the documentation locally with the following commands:

```bash
make docs
# or with explicit commands:
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
