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

## Next steps

- Enable/inspect TensorBoard: [TensorBoard](tensorboard.md).
