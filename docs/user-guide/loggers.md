# Loggers

HyperBench uses Lightning loggers and adds a few convenience loggers to make benchmarking easier.

## Default logging behavior

When you create a `MultiModelTrainer` without specifying `logger=...`, HyperBench configures:

- A Lightning `CSVLogger` per model.
- A `MarkdownTableLogger` that writes a comparison table.
- A `LaTexTableLogger` that writes a LaTeX comparison table.
- A `TensorBoardLogger` (only if TensorBoard is installed).

## Output locations

By default outputs are stored under `hyperbench_logs/`.

Common files to look for:

- `hyperbench_logs/experiment_*/comparison/results.md`
- `hyperbench_logs/experiment_*/comparison/results.tex`
- `hyperbench_logs/experiment_*/<model_name>/version_*/metrics.csv` (CSV logger)

## Using your own logger

You can pass any Lightning logger (or list of loggers) into `MultiModelTrainer`.

```python
from lightning.pytorch.loggers import CSVLogger
from hyperbench.train import MultiModelTrainer

logger = CSVLogger(save_dir="hyperbench_logs", name="my_run")

with MultiModelTrainer(model_configs=configs, logger=logger, max_epochs=10) as trainer:
    trainer.fit_all(train_dataloader=train_loader, val_dataloader=val_loader)
```

## Next steps

- Enable/inspect TensorBoard: [TensorBoard](tensorboard.md)
