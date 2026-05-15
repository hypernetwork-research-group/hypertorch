# HIF integration

HyperBench uses [**HIF (Hypergraph Interchange Format)**](https://github.com/HIF-org/HIF-standard) to represent hypergraphs.

Supported inputs:
- `.json` (plain HIF).
- `.json.zst` (Zstandard-compressed HIF).

## Load built-in datasets

Many datasets are available as built-ins (downloaded and cached automatically):

```python
from hyperbench.data import AlgebraDataset, SamplingStrategy

dataset = AlgebraDataset(sampling_strategy=SamplingStrategy.HYPEREDGE)
print(dataset.stats())
```

Built-in dataset classes include `AlgebraDataset`, `AmazonDataset`, `CoraDataset`, `CourseraDataset`, `IMDBDataset`, and more. See the [Data API reference](../api/data.md) for the complete list.

## Load a dataset from a local file

```python
from hyperbench.data import Dataset

dataset = Dataset.from_path("path/to/hypergraph.json.zst")
print(dataset.stats())
```

## Load a dataset from a URL

```python
from hyperbench.data import Dataset

dataset = Dataset.from_url("https://example.com/hypergraph.json.zst")
print(dataset.stats())
```

## Validate HIF files

If you have a plain `.json` file and want to validate it against the HIF schema:

```python
from hyperbench.utils import validate_hif_json

is_valid = validate_hif_json("path/to/hypergraph.json")
print(is_valid)
```

## How HIF maps into HyperBench

When loaded, HIF data is processed into an `HData` object (see [HData API reference](../api/types.md#hyperbench.types.HData) for details).

## Next steps

- Model selection/customization: [Models](models.md).
- Training loop (callbacks, devices, etc.): [Training](training.md).
- Comparing multiple models consistently: [Benchmarking](benchmarking.md).
- Outputs and logging: [Loggers](loggers.md).
- Visualizing runs: [TensorBoard](tensorboard.md).
