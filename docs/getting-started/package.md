# Package overview

HyperBench is organized as a flat Python package: the `hyperbench/` package lives at the repository root rather than under `src/`.

```
.
├── .github                 # GitHub workflows and templates
├── Makefile                # convenience build/run targets
├── docs                    # documentation sources and site output
│   └── agents              # agent references
├── examples                # runnable examples and demos
│   ├── hgnn.py
│   ├── ...                 # more examples
│   └── villain.py
├── hyperbench              # core Python package
│   ├── data                # datasets, loaders, and sampling
│   ├── hlp                 # hyperlink prediction modules
│   ├── models              # model implementations
│   ├── nn                  # neural network building blocks
│   ├── tests               # test suite
│   ├── train               # trainers, negative samplers, and loggers
│   ├── types               # shared tensor/data types
│   └── utils               # reusable helpers
├── hyperbench_logs         # local experiment outputs
│   └── experiment_0
│       ├── comparison
│       ├── hgnn
│       ├── ...             # more experiment outputs
│       └── villain
├── pyproject.toml          # package metadata and dependencies
├── uv.lock                 # pinned dependency lockfile
└── zensical.toml           # zensical config for docs
```

## Getting support

If you need help with HyperBench, use these project channels:

- [GitHub Discussions](https://www.github.com/hypernetwork-research-group/hyperbench/discussions): ask questions, share ideas, and connect with the community.
- [GitHub Issues](https://www.github.com/hypernetwork-research-group/hyperbench/issues): report bugs or request features (please check existing issues first).

## Community

HyperBench is developed as an open-source project with contributions from researchers and practitioners in the field of hypergraph learning. We welcome contributions of all kinds, including code, documentation, examples, and discussions.
If you're interested in contributing, please visit the [contributing guide](https://www.github.com/hypernetwork-research-group/hyperbench/blob/main/CONTRIBUTING.md) for more information on how to get involved.

## Development team

The core development team includes:

| GitHub handle | Mail | Full Name |
| --- | --- | --- |
| @tizianocitro | tcitro@unisa.it | Tiziano Citro |
| @ddevin96 | ddevinco@unisa.it | Daniele De Vinco |

## Institutional partners

<img src="../assets/logo_unisa.png" alt="University of Salerno" width="200" />

## License

This project is under the MIT license. See [LICENSE](https://github.com/hypernetwork-research-group/hyperbench/blob/main/LICENSE).