from hyperbench.nn import LaplacianPositionalEncodingEnricher, Node2VecEnricher
from hyperbench.data import AlgebraDataset, SamplingStrategy


if __name__ == "__main__":
    num_features = 32

    print("Loading and preparing dataset...")

    dataset = AlgebraDataset(sampling_strategy=SamplingStrategy.HYPEREDGE, prepare=True)

    dataset.enrich_node_features(
        enricher=LaplacianPositionalEncodingEnricher(num_features=num_features),
        enrichment_mode="replace",
    )

    print(f"Dataset after Laplacian Positional Encoding (LPE) enrichment:")
    if dataset.hdata.x is not None:
        print(f"- Node features shape: {dataset.hdata.x.shape}")
        print(f"- First 5 node features:\n {dataset.hdata.x[:5]}\n")

    node2vec_enricher = Node2VecEnricher(
        num_features=num_features,
        walk_length=20,
        context_size=10,
        num_walks_per_node=10,
        num_negative_samples=1,
        num_nodes=dataset.hdata.num_nodes,
        num_epochs=10,
        learning_rate=0.01,
        batch_size=128,
        sparse=False,
        verbose=True,
    )
    dataset.enrich_node_features(
        enricher=node2vec_enricher,
        enrichment_mode="replace",
    )

    print(f"Dataset after Node2Vec enrichment:")
    if dataset.hdata.x is not None:
        print(f"- Node features shape: {dataset.hdata.x.shape}")
        print(f"- First 5 node features:\n {dataset.hdata.x[:5]}")
