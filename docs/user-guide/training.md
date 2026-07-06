# Training

Training in HyperTorch is orchestrated via `MultiModelTrainer` (Lightning under the hood).

This page outlines the typical training pipeline. For a complete runnable script, see:
- [examples/gcn.py](https://github.com/hypernetwork-research-group/hypertorch/blob/main/examples/gcn.py)
- [examples/early_stopping.py](https://github.com/hypernetwork-research-group/hypertorch/blob/main/examples/early_stopping.py)

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
from hypertorch.data import (
    AlgebraDataset,
    DataLoader,
    LaplacianPositionalEncodingEnricher,
    RandomNegativeSampler,
    SamplingStrategy
)
from hypertorch.train import MultiModelTrainer
from hypertorch.types import ModelConfig
from hypertorch.hlp import MLPHlpModule

dataset = AlgebraDataset(sampling_strategy=SamplingStrategy.HYPEREDGE)
train_dataset, val_dataset, test_dataset = dataset.split(
    ratios=[0.7, 0.1, 0.2],
    node_space_setting="transductive",
    shuffle=True,
)
# Add negatives (example strategy; tune per use-case)
negative_sampler = RandomNegativeSampler(
    num_negative_samples=len(train_dataset),
    num_nodes_per_sample=int(train_dataset.stats()["avg_degree_hyperedge"]),
)
train_dataset = train_dataset.add_negative_samples(negative_sampler)
val_dataset = val_dataset.add_negative_samples(negative_sampler)
test_dataset = test_dataset.add_negative_samples(negative_sampler)

# Enrich node features
train_dataset.enrich_node_features(
    enricher=LaplacianPositionalEncodingEnricher(
        num_features=32,
        num_nodes=train_dataset.hdata.num_nodes,
    ),
    enrichment_mode="replace",
)
val_dataset.enrich_node_features_from(train_dataset)
test_dataset.enrich_node_features_from(train_dataset)

# Dataloaders
train_loader = DataLoader(
    train_dataset,
    sample_full_hypergraph=True,
)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

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

For hyperlink prediction, transductive splits keep the full hypergraph as context in each split and mark the supervised hyperedges with `hdata.target_hyperedge_mask`.

For hyperedge-sampling datasets, `len(dataset)` counts these target hyperedges. Meanwhile, the enrichers use the entire hypergraph to compute node features, using non-target hyperedges as context.

Use `sparse_split_hyperedges=True` to use the sparse split behavior,
where each split contains only its own hyperedges. Sparse splitting also supports `cover_all_nodes_in_train_split=True` when the training hyperedges must be incident to every node:

```python
train_ds, test_ds = dataset.split(
    ratios=[0.8, 0.2],
    node_space_setting="transductive",
    cover_all_nodes_in_train_split=True,
    sparse_split_hyperedges=True,
)
```

Use `split_with_ratios(...)` instead of `split(...)` when you need the final target-hyperedge ratios after optional sparse rebalancing.

## Checkpoint callback options

When checkpointing is enabled and you do not pass your own Lightning
`ModelCheckpoint`, HyperTorch creates a default checkpoint callback per model.
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
