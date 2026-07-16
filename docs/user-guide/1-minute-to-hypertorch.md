# 1 minute to HyperTorch

This page is a quick, “copy/paste and run” introduction to HyperTorch.

## Prerequisites

- Follow the [installation guide](../getting-started/installation.md).
- Recommended tools: `uv` + `make`.

## Run your first example

From the repository root:

```bash
make setup
make run examples/gcn.py
```

Other good starting points are located in `examples/` (e.g. `hgnn.py`, `villain.py`).

## What happens when you run an example

Most examples follow the same high-level pipeline:

1. Load a dataset (e.g. `AlgebraDataset`).
2. Split into train/val/test.
3. Add negative samples.
4. Enrich node features (e.g. Laplacian positional encoding).
5. Create one or more hyperlink prediction (HLP) or node classification (NC) modules.
6. Train + evaluate with `MultiModelTrainer`.

Artifacts are written under `hypertorch_logs/` (by default). In particular:

- `hypertorch_logs/experiment_*/comparison/results.md` (markdown table).
- `hypertorch_logs/experiment_*/comparison/results.tex` (LaTeX table).

## Next steps

- Dataset ingestion: [HIF integration](hif-integration.md).
- Model selection/customization: [Models](models.md).
- Training loop (callbacks, devices, etc.): [Training](training.md).
- Comparing multiple models consistently: [Benchmarking](benchmarking.md).
- Outputs and logging: [Loggers](loggers.md).
- Visualizing runs: [TensorBoard](tensorboard.md).
