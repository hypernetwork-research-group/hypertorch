# Models

HyperTorch provides ready-to-use built-in models inspired by the existing literature.

At a high level:
- `hypertorch.hlp.*` contains ready-to-train hyperlink prediction (HLP) modules.
- `hypertorch.nc.*` contains ready-to-train node classification (NC) modules.
- `hypertorch.models.*` contains actual models like Node2Vec, GCN, etc.
- `hypertorch.nn.*` contains layers, enrichers, aggregators, and losses.

## Built-in HLP modules

Supported models include:

- `GCN`.
- `HGNN`.
- `HGNNP`.
- `HNHN`.
- `HyperGCN`.
- `MLP`.
- `NHP`.
- `Node2VecGCN`.
- `Node2VecSLP`.
- `CommonNeighbors` (non-trainable baseline).
- `VilLain`.

## Built-in NC modules

Supported models include:

- `GCN`.
- `HGNN`.
- `HGNNP`.
- `HNHN`.
- `HyperGCN`.
- `MLP`.

## Minimal HLP example: NHP

```python
from hypertorch.hlp import NHPHlpModule

model = NHPHlpModule(
    encoder_config={
        "in_channels": num_features,
        "hidden_channels": 512,
        "aggregation": "maxmin",
    },
    lr=0.001,
    weight_decay=5e-4,
    metrics=metrics,
)
```

## Minimal example: GCN node classification

```python
from hypertorch.nc import GCNNcModule

model = GCNNcModule(
    classifier_config={
        "in_channels": 32,
        "hidden_channels": 16,
        "out_channels": 3,
        "num_layers": 2,
        "drop_rate": 0.3,
        "graph_reduction_strategy": "clique_expansion",
    },
    lr=0.01,
    weight_decay=5e-4,
)
```

## Next steps

- Training loop (callbacks, devices, etc.): [Training](training.md).
- Comparing multiple models consistently: [Benchmarking](benchmarking.md).
- Outputs and logging: [Loggers](loggers.md).
- Visualizing runs: [TensorBoard](tensorboard.md).
