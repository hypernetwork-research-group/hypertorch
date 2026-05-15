# Tutorials

This page lists the runnable scripts in the `examples/` folder.

Run examples from the repository root:

```bash
make setup
# optional
make setup-tensorboard

make run examples/early_stopping.py
```

| Example | What it demonstrates | Run |
| --- | --- | --- |
| [early_stopping.py](https://github.com/hypernetwork-research-group/hyperbench/blob/main/examples/early_stopping.py) | Training with a Lightning `EarlyStopping` callback (MLP HLP, AlgebraDataset, negative sampling) | `make run examples/early_stopping.py` |
| [gcn.py](https://github.com/hypernetwork-research-group/hyperbench/blob/main/examples/gcn.py) | GCN HLP pipeline on `AlgebraDataset` (negative sampling + LPE enricher) | `make run examples/gcn.py` |
| [hgnn.py](https://github.com/hypernetwork-research-group/hyperbench/blob/main/examples/hgnn.py) | HGNN HLP pipeline on `AlgebraDataset` (negative sampling + LPE enricher) | `make run examples/hgnn.py` |
| [hgnnp.py](https://github.com/hypernetwork-research-group/hyperbench/blob/main/examples/hgnnp.py) | HGNNP HLP pipeline on `AlgebraDataset` (negative sampling + LPE enricher) | `make run examples/hgnnp.py` |
| [hnhn.py](https://github.com/hypernetwork-research-group/hyperbench/blob/main/examples/hnhn.py) | HNHN HLP pipeline on `AlgebraDataset` (negative sampling + LPE enricher) | `make run examples/hnhn.py` |
| [hypergcn.py](https://github.com/hypernetwork-research-group/hyperbench/blob/main/examples/hypergcn.py) | HyperGCN HLP pipeline on `AlgebraDataset` (hyperedge weights + LPE) | `make run examples/hypergcn.py` |
| [nhp.py](https://github.com/hypernetwork-research-group/hyperbench/blob/main/examples/nhp.py) | NHP HLP pipeline on `AlgebraDataset` (Node2Vec enricher + negative sampling) | `make run examples/nhp.py` |
| [villain.py](https://github.com/hypernetwork-research-group/hyperbench/blob/main/examples/villain.py) | VilLain HLP pipeline on `CoraDataset` (negative sampling) | `make run examples/villain.py` |
| [mlp_common_neighbors.py](https://github.com/hypernetwork-research-group/hyperbench/blob/main/examples/mlp_common_neighbors.py) | MLP HLP and Common Neighbors HLP on `AlgebraDataset` (negative sampling + LPE) | `make run examples/mlp_common_neighbors.py` |
| [node2vecgcn.py](https://github.com/hypernetwork-research-group/hyperbench/blob/main/examples/node2vecgcn.py) | Compute Node2Vec embeddings then train Node2Vec+GCN HLP | `make run examples/node2vecgcn.py` |
| [node2vecslp.py](https://github.com/hypernetwork-research-group/hyperbench/blob/main/examples/node2vecslp.py) | Compute Node2Vec embeddings then train Node2Vec+SLP HLP | `make run examples/node2vecslp.py` |
| [node_enricher.py](https://github.com/hypernetwork-research-group/hyperbench/blob/main/examples/node_enricher.py) | Node feature enrichment: Laplacian positional encoding (LPE) + Node2Vec | `make run examples/node_enricher.py` |
| [hyperedge_enricher.py](https://github.com/hypernetwork-research-group/hyperbench/blob/main/examples/hyperedge_enricher.py) | Hyperedge enrichment: weights (degree) + hyperedge attributes | `make run examples/hyperedge_enricher.py` |
```
