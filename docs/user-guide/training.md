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
from hypertorch.hyperlink_prediction import MLPPredictor

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
model = MLPPredictor(
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

When the train, validation, and test loaders share parameters, create all three through
a Lightning-compatible data module. Use `test_loader_kwargs` for settings that should
apply only to the test loader:

```python
data_module = DataLoader.from_datasets(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    test_dataset=test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=4,
    persistent_workers=True,
    test_loader_kwargs={
        "batch_size": 1,
        "sample_full_hypergraph": True,
    },
)

with MultiModelTrainer(
    model_configs=configs,
    max_epochs=50,
    accelerator="auto",
    enable_checkpointing=False,
) as trainer:
    trainer.fit_all(
        train_dataloader=data_module.train_dataloader(),
        val_dataloader=data_module.val_dataloader(),
    )
    trainer.test_all(dataloader=data_module.test_dataloader())
```

Values in `test_loader_kwargs` override shared values for the test loader without
affecting the train or validation loaders. Datasets can be omitted when a split is not
needed. Its corresponding data module hook then returns `None`.

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

## Distributed experiment directories

When `experiment_name` is omitted, `MultiModelTrainer` reserves the next available
directory under `default_root_dir`, such as `experiment_0`. On a single machine,
Lightning's default DDP launcher starts child processes after the parent constructs
the trainer. HyperTorch exports the selected directory through an indexed environment
variable, allowing every child rank to reuse the parent's directory:

```text
HYPERTORCH_AUTO_EXPERIMENT_DIR_0=/absolute/path/to/experiment_0
```

The index identifies an auto-named `MultiModelTrainer` by construction order. When a
script constructs two trainers, the parent exports two values:

```text
HYPERTORCH_AUTO_EXPERIMENT_DIR_0=/absolute/path/to/experiment_0
HYPERTORCH_AUTO_EXPERIMENT_DIR_1=/absolute/path/to/experiment_1
```

Relaunched processes must construct auto-named trainers in the same order as the
parent. Each trainer and all of its ranks then share one experiment directory.

### External launchers

External launchers such as `torchrun` start all ranks as sibling processes before the
training script runs. An environment variable created later by rank zero cannot
propagate to the other ranks. For these launchers, provide the same explicit
`experiment_name` to every rank:

```python
import os

from hypertorch.train import MultiModelTrainer

run_id = os.environ["HYPERTORCH_RUN_ID"]

with MultiModelTrainer(
    model_configs=configs,
    default_root_dir="hypertorch_logs",
    experiment_name=f"experiment_{run_id}",
    accelerator="gpu",
    devices=4,
    strategy="ddp",
) as trainer:
    trainer.fit_all(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
    )
```

Set the run identifier before starting `torchrun` so every rank inherits it:

```bash
HYPERTORCH_RUN_ID=my-run-001 \
torchrun --nproc-per-node=4 train.py
```

Alternatively, create the directory and configure the indexed value before launch:

```bash
mkdir -p /absolute/path/to/hypertorch_logs/experiment_0
export HYPERTORCH_AUTO_EXPERIMENT_DIR_0=/absolute/path/to/hypertorch_logs/experiment_0
torchrun --nproc-per-node=4 train.py
```

For multi-machine jobs, the resolved directory must be accessible at the same path
from every machine, normally through shared storage. A node-local path produces a
separate set of files on each machine even when its textual path is identical.

Experiment-directory sharing only coordinates artifact locations. Distributed metric
values must also be synchronized before global rank zero writes them. See
[Distributed metric logging](loggers.md#distributed-metric-logging).

## Per-model trainer options

`MultiModelTrainer` values are shared defaults. Set per-model trainer options on an
individual `ModelConfig` through `trainer_kwargs` when one model needs different
training settings:

```python
configs = [
    ModelConfig(
        name="fast_baseline",
        version="mean",
        model=baseline_model,
        trainer_kwargs={
            "max_epochs": 20,
            "log_every_n_steps": 5,
            "enable_checkpointing": False,
        },
    ),
    ModelConfig(
        name="larger_model",
        version="maxmin",
        model=larger_model,
        trainer_kwargs={
            "max_epochs": 100,
            "check_val_every_n_epoch": 2,
            "enable_checkpointing": True,
        },
    ),
    ModelConfig(
        name="shared_defaults",
        version="mean",
        model=shared_model,
    )
]

with MultiModelTrainer(
    model_configs=configs,
    max_epochs=50,
    log_every_n_steps=10,
    enable_checkpointing=True,
) as trainer:
    trainer.fit_all(train_dataloader=train_loader, val_dataloader=val_loader)
```

Per-model values override the shared values used to create that model's trainer.
If a key is not present in `trainer_kwargs`, the shared `MultiModelTrainer` value
is used. In the code above:
- `fast_baseline` will train for 20 epochs, logging every 5 steps, and will not checkpoint.
- `larger_model` will train for 100 epochs, validating every 2 epochs, and will checkpoint.
- `shared_defaults` will train for 50 epochs, logging every 10 steps, and will checkpoint (the shared defaults).

It is possible to use the same keys that `MultiModelTrainer` accepts for Lightning `Trainer` construction.

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

For one model only, put `checkpoint_callback_kwargs` inside that model's
`trainer_kwargs`:

```python
configs = [
    ModelConfig(
        name="weights_only",
        version="mean",
        model=model,
        trainer_kwargs={
            "checkpoint_callback_kwargs": {
                "filename": "weights-only-{epoch}",
                "save_weights_only": True,
            },
        },
    )
]
```

## Next steps

- Comparing multiple models consistently: [Benchmarking](benchmarking.md).
- Outputs and logging: [Loggers](loggers.md).
- Visualizing runs: [TensorBoard](tensorboard.md).
