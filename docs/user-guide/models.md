# Models

HyperTorch provides ready-to-use built-in models inspired by the existing literature.

At a high level:
- `hypertorch.hyperlink_prediction.*` contains ready-to-train hyperlink prediction (HLP) modules.
- `hypertorch.nc.*` contains ready-to-train node classification (NC) modules.
- `hypertorch.models.*` contains actual models like Node2Vec, GCN, etc.
- `hypertorch.nn.*` contains layers, enrichers, aggregators, and losses.

## Built-in hyperlink prediction modules

Supported models include:

- `CommonNeighbors` (non-trainable baseline).
- `GCN`.
- `HGNN`.
- `HGNNP`.
- `HNHN`.
- `HyperGCN`.
- `MLP`.
- `NHP`.
- `Node2VecGCN`.
- `Node2Vec`.
- `VilLain`.

## Built-in node classification modules

Supported models include:

- `CommonNeighbors` (non-trainable baseline).
- `GCN`.
- `HGNN`.
- `HGNNP`.
- `HNHN`.
- `HyperGCN`.
- `MLP`.
- `Node2VecGCN`.
- `Node2Vec`.
- `VilLain`.

## Minimal hyperlink prediction example: NHP

```python
from hypertorch.hyperlink_prediction import NHPPredictor

model = NHPPredictor(
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

## Minimal node classification example: HyperGCN

```python
from hypertorch.nc import HyperGCNNcModule

model = HyperGCNNcModule(
    classifier_config={
        "in_channels": 32,
        "hidden_channels": 16,
        "out_channels": 3,
        "drop_rate": 0.3,
        "use_mediator": False,
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
