# Package overview

HyperTorch is organized as a flat Python package: the `hypertorch/` package lives at the repository root rather than under `src/`.

```bash
.
├── .github                 # CI workflows and repository templates
├── Makefile                # Common development, build, and test commands
├── docs                    # Documentation sources and generated site assets
│   └── agents              # Agent reference documentation
├── examples                # Runnable scripts for training and benchmarking
│   ├── hgnn.py             # HGNN training and benchmarking example
│   ├── ...                 # Additional model examples
│   └── villain.py          # VilLain training and benchmarking example
├── hypertorch              # Main HyperTorch Python package
│   ├── data                # Datasets, data loaders, and sampling
│   ├── hyperlink_prediction # Hyperlink prediction (HLP) pipelines
│   ├── integration_tests   # Integration test suite
│   ├── models              # Model implementations
│   ├── nc                  # Node classification pipelines
│   ├── nn                  # Neural-network layers and reusable modules
│   ├── tests               # Unit test suite
│   ├── train               # Training and benchmarking logic, and loggers
│   ├── types               # Shared type definitions
│   └── utils               # General-purpose utility functions
├── hypertorch_logs         # Local logs and outputs from experiments
│   └── experiment_0        # Outputs produced by experiment_0
│       ├── comparison      # Cross-model metric comparisons
│       ├── hgnn            # HGNN-specific experiment results
│       ├── ...             # Additional model-specific outputs
│       └── villain         # VilLain-specific experiment results
├── pyproject.toml          # Metadata, dependencies, and build settings
├── uv.lock                 # Lockfile for dependencies
└── zensical.toml           # Documentation build and site configuration
```

## Getting support

If you need help with HyperTorch, use these project channels:

- [GitHub Discussions](https://www.github.com/hypernetwork-research-group/hypertorch/discussions): ask questions, share ideas, and connect with the community.
- [GitHub Issues](https://www.github.com/hypernetwork-research-group/hypertorch/issues): report bugs or request features (please check existing issues first).

## Community

HyperTorch is developed as an open-source project with contributions from researchers and practitioners in the field of hypergraph learning. We welcome contributions of all kinds, including code, documentation, examples, and discussions.
If you're interested in contributing, please visit the [contributing guide](https://www.github.com/hypernetwork-research-group/hypertorch/blob/main/CONTRIBUTING.md) for more information on how to get involved.

## Development team

The core development team includes:

| GitHub handle | Mail | Full Name |
| --- | --- | --- |
| @tizianocitro | tcitro@unisa.it | Tiziano Citro |
| @ddevin96 | ddevinco@unisa.it | Daniele De Vinco |

## Institutional partners

<img src="../assets/logo_unisa.png" alt="University of Salerno" width="200" />

## License

This project is under the Apache License 2.0 license. See [LICENSE](https://github.com/hypernetwork-research-group/hypertorch/blob/main/LICENSE).
