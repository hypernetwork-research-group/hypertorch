# Benchmarking

Benchmarking in HyperTorch typically means:
- Running multiple models on the same dataset split.
- Using the same negative sampling and feature enrichment.
- Producing comparable metrics and summary tables.

## Comparing multiple models

The recommended pattern is to pass multiple `ModelConfig` objects to `MultiModelTrainer`:

```python
from hypertorch.types import ModelConfig
from hypertorch.train import MultiModelTrainer
from hypertorch.hlp import MLPHlpModule, NHPHlpModule

configs = [
    ModelConfig(
        name="nhp",
        version="maxmin",
        model=NHPHlpModule(
            encoder_config={
                "in_channels": 32,
                "hidden_channels": 64,
                "aggregation": "maxmin",
            },
        ),
    ),
    ModelConfig(
        name="mlp",
        version="mean",
        model=MLPHlpModule(
            encoder_config={
                "in_channels": 32,
                "out_channels": 32,
                "hidden_channels": 64,
                "num_layers": 3,
                "drop_rate": 0.3,
            },
            aggregation="mean",
        ),
    ),
]

with MultiModelTrainer(model_configs=configs, max_epochs=200, accelerator="auto") as trainer:
    trainer.fit_all(train_dataloader=train_loader, val_dataloader=val_loader)
    trainer.test_all(dataloader=test_loader)
```

## Where results are saved

By default, runs are saved under `hypertorch_logs/`.

The trainer writes comparison tables to:
- `hypertorch_logs/experiment_*/comparison/results.md`.
- `hypertorch_logs/experiment_*/comparison/results.tex`.

## Next steps

- Outputs and logging: [Loggers](loggers.md).
- Visualizing runs: [TensorBoard](tensorboard.md).
