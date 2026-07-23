# Loggers

HyperTorch has a few convenient loggers to make benchmarking easier.

## Default logging behavior

When you create a `MultiModelTrainer` without specifying `logger=...`, HyperTorch configures:

- A `CSVLogger` that logs training and validation metrics for each model to CSV files.
- A `MarkdownTableLogger` that writes a comparison table.
- A `LaTexTableLogger` that writes a LaTeX comparison table.
- A `TensorBoardLogger` (only if TensorBoard is installed).

## Output locations

By default outputs are stored under `hypertorch_logs/`.

Common files to look for:

- `hypertorch_logs/experiment_*/comparison/results.md`.
- `hypertorch_logs/experiment_*/comparison/results.tex`.
- `hypertorch_logs/experiment_*/<model_name>/version_*/metrics.csv` (CSV logger).

## Using your own logger

You can pass any Lightning logger (or list of loggers) into `MultiModelTrainer`.

```python
from lightning.pytorch.loggers import CSVLogger
from hypertorch.train import MultiModelTrainer

logger = CSVLogger(save_dir="hypertorch_logs", name="my_run")

with MultiModelTrainer(model_configs=configs, logger=logger, max_epochs=10) as trainer:
    trainer.fit_all(train_dataloader=train_loader, val_dataloader=val_loader)
```

## Distributed metric logging

When training in a distributed setting, each process computes metrics separately.
HyperTorch's comparison loggers write files only from global
rank zero to prevent multiple processes from writing the same file.

Metrics must be synchronized before they reach the logger. For built-in hypertorch
models, pass `sync_dist=True` through `metrics_log_kwargs`:

```python
from hypertorch.node_classification import GCNClassifier
from hypertorch.train import MultiModelTrainer
from hypertorch.types import ModelConfig

model = GCNClassifier(
    classifier_config={
        "in_channels": 32,
        "out_channels": 2,
    },
    metrics_log_kwargs={"sync_dist": True},
)

configs = [
    ModelConfig(
        name="gcn",
        version="default",
        model=model,
    )
]

with MultiModelTrainer(
    model_configs=configs,
    accelerator="gpu",
    devices=2,
    strategy="ddp",
    max_epochs=50,
) as trainer:
    trainer.fit_all(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
    )
    trainer.test_all(dataloader=test_loader)
```

`metrics_log_kwargs` is a model option. Do not place it in
`ModelConfig.trainer_kwargs`: `sync_dist` configures `LightningModule.log()`, not
the Lightning `Trainer`.

For a custom Lightning module, enable synchronization directly when logging plain
tensor values:

```python
def training_step(self, batch, batch_idx):
    loss = self.compute_loss(batch)
    batch_size = ...  # Number of samples used to compute the loss.

    self.log(
        "train/loss",
        loss,
        batch_size=batch_size,
        sync_dist=True,
        on_step=False,
        on_epoch=True,
    )
    return loss
```

TorchMetrics objects synchronize their distributed states when computed. Plain tensor
values, including training, validation, and test losses, require `sync_dist=True` to
represent data from every rank.

## Next steps

- Enable/inspect TensorBoard: [TensorBoard](tensorboard.md).
