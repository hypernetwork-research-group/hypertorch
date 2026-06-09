# Training

Training in HyperBench is orchestrated via `MultiModelTrainer` (Lightning under the hood).

This page outlines the typical training pipeline. For a complete runnable script, see:
- [examples/gcn.py](https://github.com/hypernetwork-research-group/hyperbench/blob/main/examples/gcn.py)
- [examples/early_stopping.py](https://github.com/hypernetwork-research-group/hyperbench/blob/main/examples/early_stopping.py)

## Typical pipeline

1. Load a dataset (built-in or from HIF).
2. Split it (train/val/test).
3. Add negative samples.
4. Enrich node features.
5. Create dataloaders.
6. Configure one or more models.
7. Train and evaluate.

## Minimal end-to-end skeleton

```python
from hyperbench.data import (
    AlgebraDataset,
    DataLoader,
    LaplacianPositionalEncodingEnricher,
    RandomNegativeSampler,
    SamplingStrategy
)
from hyperbench.train import MultiModelTrainer
from hyperbench.types import ModelConfig
from hyperbench.hlp import MLPHlpModule

dataset = AlgebraDataset(sampling_strategy=SamplingStrategy.HYPEREDGE)
train_ds, test_ds = dataset.split(
    ratios=[0.8, 0.2],
    shuffle=True,
    seed=42,
    node_space_setting="transductive",
)
train_ds, val_ds = train_ds.split(
    ratios=[0.875, 0.125],
    shuffle=True,
    seed=42,
    node_space_setting="transductive",
)

# Add negatives (example strategy; tune per use-case)
neg = RandomNegativeSampler(
    num_negative_samples=train_ds.hdata.num_hyperedges,
    num_nodes_per_sample=int(train_ds.stats()["avg_degree_hyperedge"]),
)
train_ds = train_ds.add_negative_samples(neg, seed=42)
val_ds = val_ds.add_negative_samples(neg, seed=42)
test_ds = test_ds.add_negative_samples(neg, seed=42)

# Enrich node features
train_ds.enrich_node_features(
    enricher=LaplacianPositionalEncodingEnricher(
        num_features=32,
        num_nodes=train_ds.hdata.num_nodes,
    ),
    enrichment_mode="replace",
)
val_ds.enrich_node_features_from(train_ds)
test_ds.enrich_node_features_from(train_ds)

# Dataloaders
train_loader = DataLoader(train_ds, batch_size=128, shuffle=False)
val_loader = DataLoader(val_ds, sample_full_hypergraph=True, shuffle=False)
test_loader = DataLoader(test_ds, sample_full_hypergraph=True, shuffle=False)

# Model(s)
model = MLPHlpModule(
    encoder_config={
        "in_channels": 32,
        "out_channels": 32,
        "hidden_channels": 64,
        "num_layers": 3,
        "drop_rate": 0.3,
    },
    aggregation="mean",
)

configs = [ModelConfig(name="mlp", version="mean", model=model)]

with MultiModelTrainer(
    model_configs=configs,
    max_epochs=50,
    accelerator="auto",
    enable_checkpointing=False,
) as trainer:
    trainer.fit_all(train_dataloader=train_loader, val_dataloader=val_loader)
    trainer.test_all(dataloader=test_loader)
```

Transductive splits keep the full node feature matrix in the first split, but by
default they do not force the first split's hyperedges to cover every node. Pass
`cover_all_nodes_in_train_split=True` when the training hyperedges themselves
must be incident to every node:

```python
train_ds, test_ds = dataset.split(
    ratios=[0.8, 0.2],
    node_space_setting="transductive",
    cover_all_nodes_in_train_split=True,
)
```

Use `split_with_ratios(...)` instead of `split(...)` when you need the final
hyperedge ratios after optional rebalancing.

## Checkpoint callback options

When checkpointing is enabled and you do not pass your own Lightning
`ModelCheckpoint`, HyperBench creates a default checkpoint callback per model.
Use `checkpoint_callback_kwargs` to configure that default callback:

```python
trainer = MultiModelTrainer(
    model_configs=configs,
    enable_checkpointing=True,
    checkpoint_callback_kwargs={
        "filename": "weights-only-{epoch}",
        "save_weights_only": True,
    },
)
```

Pass `dirpath` in `checkpoint_callback_kwargs` to override the default per-model
checkpoint directory.

## Next steps

- Comparing multiple models consistently: [Benchmarking](benchmarking.md).
- Outputs and logging: [Loggers](loggers.md).
- Visualizing runs: [TensorBoard](tensorboard.md).
