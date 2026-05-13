# Package structure

Hyperbench organization.

## Package structures

Hyperbench is organized as a Python package with the following structure:
<!-- todo: check other type file -> format # -->
```bash
.
├── .github                 # GitHub workflows and templates
├── Makefile                # convenience build/run targets
├── agents                  # agent docs and references
│   ├── SKILLS.md
│   └── references
├── configs                 # project configuration files
├── docs                    # documentation sources and site output
├── examples                # runnable examples and demos
│   ├── hgnn.py
|   ├── ...
│   └── villain.py
├── hyperbench              # core Python package
│   ├── data                # datasets, loaders, and sampling
│   ├── hlp                 # HLP task helpers and pipelines
│   ├── models              # model implementations
│   ├── nn                  # neural network building blocks
│   ├── tests               # test utilities
│   ├── train               # training loops and loggers
│   ├── types               # shared type definitions
│   └── utils               # reusable helpers
├── hyperbench_logs         # local experiment outputs
│   └── experiment_0
│       ├── common_neighbors
│       ├── comparison
│       └── mlp
├── pyproject.toml          # package metadata and dependencies
├── uv.lock                 # pinned dependency lockfile
└── zensical.toml           # zensical config for docs
```

## Getting support
If you need help with using hyperbench, please check out the following resources:

- [GitHub Discussions](https://www.github.com/hypernetwork-research-group/hyperbench/discussions): ask questions, share ideas, and connect with the community.
- [GitHub Issues](https://www.github.com/hypernetwork-research-group/hyperbench/issues): report bugs or request features (please check existing issues first).

## Community

hyperbench is developed as an open-source project with contributions from researchers and practitioners in the field of hypergraph learning. We welcome contributions of all kinds, including code, documentation, examples, and discussions.
If you’re interested in contributing, please visit the [contributing guide](https://www.github.com/hypernetwork-research-group/hyperbench/blob/main/CONTRIBUTING.md) for more information on how to get involved.

## Development team

The core development team includes:

| GitHub handle | Mail | Full Name |
| --- | --- | --- |
| @ddevin96 | ddevinco@unisa.it | Daniele De Vinco |
| @tizianocitro | tcitro@unisa.it | Tiziano Citro |

## Institutional partners

<img src="../assets/logo_unisa.png" alt="University of Salerno" width="200" />

<!-- TODO logo -->
## License

MIT license. See [LICENSE](https://github.com/hypernetwork-research-group/hyperbench/blob/main/LICENSE)
