# Models

HyperTorch provides ready-to-use built-in models inspired by the existing literature.

At a high level:
- `hypertorch.hlp.*` contains ready-to-train hyperlink prediction (HLP) modules.
- `hypertorch.nc.*` contains ready-to-train node classification (NC) modules.
- `hypertorch.models.*` contains actual models like Node2Vec, GCN, etc.
- `hypertorch.nn.*` contains layers, enrichers, aggregators, and losses.

## Built-in HLP modules

Supported models include:

- `MLP`.
- `GCN`.
- `HGNN`.
- `HGNNP`.
- `HNHN`.
- `HyperGCN`.
- `NHP`.
- `Node2VecGCN`.
- `Node2VecSLP`.
- `CommonNeighbors` (non-trainable baseline).
- `VilLain`.

## Built-in NC modules

Supported models include:

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

## Minimal example: HyperGCN node classification

```python
from hypertorch.nc import HyperGCNNcModule

model = HyperGCNNcModule(
    classifier_config={
        "in_channels": 32,
        "hidden_channels": 16,
        "out_channels": 3,
        "drop_rate": 0.5,
        "use_mediator": False,
        "fast": False,
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
