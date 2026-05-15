# Models

HyperBench provides ready-to-use built-in models inspired by the existing literature.

At a high level:
- `hyperbench.hlp.*` contains ready-to-train hyperlink prediction (HLP) modules (recommended starting point).
- `hyperbench.models.*` contains actual models like Node2Vec, GCN, etc.
- `hyperbench.nn.*` contains layers, enrichers, aggregators, and losses.

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

## Minimal example: Node2Vec + GCN

```python
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAUROC

from hyperbench.hlp import Node2VecGCNHlpModule, Node2VecGCNHlpConfig

metrics = MetricCollection({"auc": BinaryAUROC()})

gcn_config: Node2VecGCNHlpConfig = {
    "out_channels": num_features,
    "hidden_channels": num_features,
    "num_layers": 2,
    "drop_rate": 0.1,
    "bias": True,
    "improved": False,
    "add_self_loops": True,
    "normalize": True,
    "cached": False,
    "graph_reduction_strategy": "clique_expansion",
}

node2vecgcn = Node2VecGCNHlpModule(
    encoder_config={
        "mode": "precomputed",
        "num_features": num_features,
        "node2vec_config": {},
        "gcn_config": gcn_config,
    },
    aggregation="mean",
    lr=0.001,
    weight_decay=0.0,
    metrics=metrics,
)
```

## Minimal example: GCN

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

- Training loop (callbacks, devices, etc.): [Training](training.md).
- Comparing multiple models consistently: [Benchmarking](benchmarking.md).
- Outputs and logging: [Loggers](loggers.md).
- Visualizing runs: [TensorBoard](tensorboard.md).
