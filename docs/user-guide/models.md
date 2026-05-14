# Models

HyperBench provides ready-to-use built-in models inspired by the existing literature.
At a high level:

- `hyperbench.hlp.*` contains ready-to-train hyperlink prediction (HLP) modules (recommended starting point).

- `hyperbench.models.*` contains actual models like Node2Vec, GCN, etc.

- `hyperbench.nn.*` contains layers, enrichers, aggregators, and losses.

## Built-in HLP modules

Supported models include:

- `MLP`
- `GCN`
- `HGNN`
- `HGNNP`
- `HNHN`
- `HyperGCN`
- `NHP`
- `Node2VecGCN`
- `Node2VecSLP`
- `CommonNeighbors` (non-trainable baseline)
- `VilLain`

## Minimal example: an MLP baseline

```python
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAUROC

from hyperbench.hlp import MLPHlpModule

metrics = MetricCollection({"auc": BinaryAUROC()})

model = MLPHlpModule(
    encoder_config={
        "in_channels": 32,
        "out_channels": 32,
        "hidden_channels": 64,
        "num_layers": 3,
        "drop_rate": 0.3,
    },
    aggregation="mean",
    metrics=metrics,
)
```

## Minimal example: a GCN baseline

```python
from hyperbench.hlp import GCNHlpModule

model = GCNHlpModule(
    encoder_config={
        "in_channels": 32,
        "hidden_channels": 16,
        "out_channels": 16,
        "num_layers": 2,
        "drop_rate": 0.1,
        "bias": True,
        "improved": False,
        "add_self_loops": True,
        "normalize": True,
        "cached": False,
        "graph_reduction_strategy": "clique_expansion",
    },
    aggregation="mean",
    lr=0.001,
    weight_decay=5e-4,
)
```

## Next steps

- Training loop (callbacks, devices, etc.): [Training](training.md)
- Comparing multiple models consistently: [Benchmarking](benchmarking.md)
- Outputs and logging: [Loggers](loggers.md)
- Visualizing runs: [TensorBoard](tensorboard.md)
