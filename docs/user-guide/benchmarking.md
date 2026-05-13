# Benchmarking

Benchmarking in HyperBench typically means:

- running multiple models on the same dataset split,
- using the same negative sampling and feature enrichment,
- producing comparable metrics and summary tables.

## Comparing multiple models

The recommended pattern is to pass multiple `ModelConfig` objects to `MultiModelTrainer`:

```python
from hyperbench.types import ModelConfig
from hyperbench.train import MultiModelTrainer
from hyperbench.hlp import CommonNeighborsHlpModule, MLPHlpModule

configs = [
    ModelConfig(
        name="common_neighbors",
        version="mean",
        model=CommonNeighborsHlpModule(
            train_hyperedge_index=train_hyperedge_index,
            aggregation="mean",
        ),
        is_trainable=False,
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

For a complete runnable script, see:
https://github.com/hypernetwork-research-group/hyperbench/blob/main/examples/mlp_common_neighbors.py

## Where results are saved

By default, runs are saved under `hyperbench_logs/`.

The trainer writes comparison tables to:

- `hyperbench_logs/experiment_*/comparison/results.md`
- `hyperbench_logs/experiment_*/comparison/results.tex`

## Next steps

- Outputs and logging: [Loggers](loggers.md)
- Visualizing runs: [TensorBoard](tensorboard.md)
