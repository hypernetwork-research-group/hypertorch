# TensorBoard

HyperTorch can log to TensorBoard via Lightning’s `TensorBoardLogger`.

## Install TensorBoard support

From the repository root:

```bash
make setup-tensorboard
```

Or with `uv` directly:

```bash
uv pip install -e ".[tensorboard]"
```

## Auto-start TensorBoard

Many examples enable automatic TensorBoard startup:

```python
from hypertorch.train import MultiModelTrainer

with MultiModelTrainer(
    model_configs=configs,
    auto_start_tensorboard=True,
    auto_wait=True,
) as trainer:
    trainer.fit_all(train_dataloader=train_loader, val_dataloader=val_loader)
```

Set `auto_wait=True` when you want the TensorBoard server to stay alive after training completes, until you press Enter.

## Start TensorBoard manually

If you prefer to run it yourself, add

```python
with MultiModelTrainer(
    ...
    auto_start_tensorboard=False,
    ...
):
    ...
```

```bash
tensorboard --logdir hypertorch_logs --port 6006
```

Then open `http://127.0.0.1:6006` in your browser.

## Next steps

- Learn where logs and tables are written: [Loggers](loggers.md)
