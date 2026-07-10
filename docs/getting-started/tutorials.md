# Tutorials

This page lists the runnable scripts in the `examples/` folder.

Run examples from the repository root:

```bash
make setup

# Optional setup
make setup-tensorboard

make run examples/hyperlink_prediction/nhp.py
```

Under `examples/`, the scripts are organized by task. The following table lists all the runnable
scripts along with a brief description.


| Example | What it demonstrates | Run |
| --- | --- | --- |
| **Dataset management** |  |  |
| [load_dataset_from_external_source.py](https://github.com/hypernetwork-research-group/hypertorch/blob/main/examples/dataset/load_dataset_from_external_source.py) | Load dataset from external source | `make run examples/dataset/load_dataset_from_external_source.py` |
| **Node and hyperedge enrichment** |  |  |
| [hyperedge_enricher.py](https://github.com/hypernetwork-research-group/hypertorch/blob/main/examples/enrichment/hyperedge_enricher.py) | Hyperedge enrichment: weights (degree) + hyperedge attributes | `make run examples/enrichment/hyperedge_enricher.py` |
| [node_enricher.py](https://github.com/hypernetwork-research-group/hypertorch/blob/main/examples/enrichment/node_enricher.py) | Node feature enrichment: Laplacian positional encoding (LPE) + Node2Vec | `make run examples/enrichment/node_enricher.py` |
| **Hyperlink prediction models** |  |  |
| [common_neighbors.py](https://github.com/hypernetwork-research-group/hypertorch/blob/main/examples/hyperlink_prediction/common_neighbors.py) | Common Neighbors HLP baseline on `AlgebraDataset` (negative sampling) | `make run examples/hyperlink_prediction/common_neighbors.py` |
| [gcn.py](https://github.com/hypernetwork-research-group/hypertorch/blob/main/examples/hyperlink_prediction/gcn.py) | GCN HLP pipeline on `AlgebraDataset` (negative sampling + LPE enricher) | `make run examples/hyperlink_prediction/gcn.py` |
| [hgnn.py](https://github.com/hypernetwork-research-group/hypertorch/blob/main/examples/hyperlink_prediction/hgnn.py) | HGNN HLP pipeline on `AlgebraDataset` (negative sampling + LPE enricher) | `make run examples/hyperlink_prediction/hgnn.py` |
| [hnhn.py](https://github.com/hypernetwork-research-group/hypertorch/blob/main/examples/hyperlink_prediction/hnhn.py) | HNHN HLP pipeline on `AlgebraDataset` (negative sampling + LPE enricher) | `make run examples/hyperlink_prediction/hnhn.py` |
| [hgnnp.py](https://github.com/hypernetwork-research-group/hypertorch/blob/main/examples/hyperlink_prediction/hgnnp.py) | HGNNP HLP pipeline on `AlgebraDataset` (negative sampling + LPE enricher) | `make run examples/hyperlink_prediction/hgnnp.py` |
| [hypergcn.py](https://github.com/hypernetwork-research-group/hypertorch/blob/main/examples/hyperlink_prediction/hypergcn.py) | HyperGCN HLP pipeline on `AlgebraDataset` (hyperedge weights + LPE) | `make run examples/hyperlink_prediction/hypergcn.py` |
| [mlp.py](https://github.com/hypernetwork-research-group/hypertorch/blob/main/examples/hyperlink_prediction/mlp.py) | MLP HLP pipeline on `AlgebraDataset` (negative sampling + LPE enricher) | `make run examples/hyperlink_prediction/mlp.py` |
| [nhp.py](https://github.com/hypernetwork-research-group/hypertorch/blob/main/examples/hyperlink_prediction/nhp.py) | NHP HLP pipeline on `AlgebraDataset` (Node2Vec enricher + negative sampling) | `make run examples/hyperlink_prediction/nhp.py` |
| [node2vecgcn.py](https://github.com/hypernetwork-research-group/hypertorch/blob/main/examples/hyperlink_prediction/node2vecgcn.py) | Compute Node2Vec embeddings then train Node2Vec+GCN HLP | `make run examples/hyperlink_prediction/node2vecgcn.py` |
| [node2vecslp.py](https://github.com/hypernetwork-research-group/hypertorch/blob/main/examples/hyperlink_prediction/node2vecslp.py) | Compute Node2Vec embeddings then train Node2Vec+SLP HLP | `make run examples/hyperlink_prediction/node2vecslp.py` |
| [villain.py](https://github.com/hypernetwork-research-group/hypertorch/blob/main/examples/hyperlink_prediction/villain.py) | VilLain HLP pipeline on `CoraDataset` (negative sampling) | `make run examples/hyperlink_prediction/villain.py` |
| **Node classification models** |  |  |
| [common_neighbors.py](https://github.com/hypernetwork-research-group/hypertorch/blob/main/examples/node_classification/common_neighbors.py) | Common Neighbors NC baseline on `AlgebraDataset` (degree labels) | `make run examples/node_classification/common_neighbors.py` |
| [gcn.py](https://github.com/hypernetwork-research-group/hypertorch/blob/main/examples/node_classification/gcn.py) | GCN NC pipeline on `AlgebraDataset` (clique expansion + LPE enricher) | `make run examples/node_classification/gcn.py` |
| [hgnn.py](https://github.com/hypernetwork-research-group/hypertorch/blob/main/examples/node_classification/hgnn.py) | HGNN NC pipeline on `AlgebraDataset` (LPE enricher) | `make run examples/node_classification/hgnn.py` |
| [hnhn.py](https://github.com/hypernetwork-research-group/hypertorch/blob/main/examples/node_classification/hnhn.py) | HNHN NC pipeline on `AlgebraDataset` (LPE enricher) | `make run examples/node_classification/hnhn.py` |
| [hgnnp.py](https://github.com/hypernetwork-research-group/hypertorch/blob/main/examples/node_classification/hgnnp.py) | HGNNP NC pipeline on `AlgebraDataset` (LPE enricher) | `make run examples/node_classification/hgnnp.py` |
| [hypergcn.py](https://github.com/hypernetwork-research-group/hypertorch/blob/main/examples/node_classification/hypergcn.py) | HyperGCN NC pipeline on `AlgebraDataset` (LPE enricher) | `make run examples/node_classification/hypergcn.py` |
| [mlp.py](https://github.com/hypernetwork-research-group/hypertorch/blob/main/examples/node_classification/mlp.py) | MLP NC pipeline on `AlgebraDataset` (LPE enricher) | `make run examples/node_classification/mlp.py` |
| [node2vecgcn.py](https://github.com/hypernetwork-research-group/hypertorch/blob/main/examples/node_classification/node2vecgcn.py) | Compute Node2Vec embeddings then train Node2Vec+GCN NC | `make run examples/node_classification/node2vecgcn.py` |
| [node2vecslp.py](https://github.com/hypernetwork-research-group/hypertorch/blob/main/examples/node_classification/node2vecslp.py) | Compute Node2Vec embeddings then train Node2Vec+SLP NC | `make run examples/node_classification/node2vecslp.py` |
| [villain.py](https://github.com/hypernetwork-research-group/hypertorch/blob/main/examples/node_classification/villain.py) | VilLain NC pipeline on `AlgebraDataset` (degree labels) | `make run examples/node_classification/villain.py` |
| **Sampling strategies** |  |  |
| [custom_negative_sampler.py](https://github.com/hypernetwork-research-group/hypertorch/blob/main/examples/sampling/custom_negative_sampler.py) | Custom negative sampling for HLP | `make run examples/sampling/custom_negative_sampler.py` |
| [negative_sampler.py](https://github.com/hypernetwork-research-group/hypertorch/blob/main/examples/sampling/negative_sampler.py) | Negative sampling for HLP | `make run examples/sampling/negative_sampler.py` |
| **Splitting strategies** |  |  |
| [custom_splitter.py](https://github.com/hypernetwork-research-group/hypertorch/blob/main/examples/splitting/custom_splitter.py) | Custom splitting of a dataset | `make run examples/splitting/custom_splitter.py` |
| **Training customization** |  |  |
| [multi_model_training.py](https://github.com/hypernetwork-research-group/hypertorch/blob/main/examples/training/multi_model_training.py) | Multi-model training for HLP | `make run examples/training/multi_model_training.py` |
| [train_with_checkpoints.py](https://github.com/hypernetwork-research-group/hypertorch/blob/main/examples/training/train_with_checkpoints.py) | Training with checkpointing and prediction | `make run examples/training/train_with_checkpoints.py` |
| [train_with_custom_logger.py](https://github.com/hypernetwork-research-group/hypertorch/blob/main/examples/training/train_with_custom_logger.py) | Training with a custom logger (MLP HLP, AlgebraDataset, negative sampling) | `make run examples/training/train_with_custom_logger.py` |
| [train_with_early_stopping.py](https://github.com/hypernetwork-research-group/hypertorch/blob/main/examples/training/early_stopping.py) | Training with a Lightning `EarlyStopping` callback (MLP HLP, AlgebraDataset, negative sampling) | `make run examples/training/early_stopping.py` |
