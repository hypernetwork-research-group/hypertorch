# HyperBench documentation

HyperBench is a Lightning library for hypergraph learning and benchmarking. It provides a standardized workflow for loading hypergraph datasets, training models, evaluating them under comparable settings, and reporting results. The current release focuses on Hyperlink Prediction, with ready-to-run pipelines for established hypergraph baselines.

The library is built around extensibility: datasets are represented in HIF format and converted into typed tensor objects, models can be implemented as standard Lightning modules, and benchmarking is handled through reusable trainers, samplers, metrics, loggers, and result exporters (Markdown/LaTeX). HyperBench includes preloaded datasets, mini-batch and full-hypergraph data loading, negative sampling utilities, structural feature enrichers, neural components, and built-in models such as HGNN, HNHN, HyperGCN, GCN, MLP/SLP, NHP, Node2Vec, VilLain, and more.

Use HyperBench to:
- Benchmark existing models across a shared collection of hypergraph datasets.
- Develop custom PyTorch or PyTorch Lightning models and train and compare them against the built-in baselines.
- Integrate new datasets through the HIF format and run the same training, evaluation, and reporting pipeline on them.

## Quick links

| Section | What you will find | Link |
| --- | --- | --- |
| Getting started | Install HyperBench and verify your environment | [Installation guide](getting-started/installation.md) |
| User guide | How to use datasets/models/training utilities | [User guide](user-guide/user.md) |
| Development | Contributing, code structure, docstring conventions | [Development guide](development/development.md) |
| API reference | Auto-generated reference from docstrings | [API reference](api/reference.md) |
| Release notes | Version history and changelog | [Release notes](release-notes/release_notes.md) |

<!-- | Contributing | Dev workflow (`uv` + `make`), checks, tests, docs | [CONTRIBUTING.md](https://github.com/hypernetwork-research-group/hyperbench/blob/main/CONTRIBUTING.md) | -->
