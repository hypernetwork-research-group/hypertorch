from hyperbench.nn import LaplacianPositionalEncodingEnricher
from hyperbench.data import AlgebraDataset, SamplingStrategy


if __name__ == "__main__":
    verbose = False

    sampling_strategy = SamplingStrategy.HYPEREDGE
    print("Loading and preparing dataset...")

    dataset = AlgebraDataset(sampling_strategy=sampling_strategy, prepare=True)
    # NodeEnricher adds features for each node.
    dataset.enrich_node_features(
        enricher=LaplacianPositionalEncodingEnricher(num_features=32),
        enrichment_mode="replace",
    )

    print(f"Dataset:\n {dataset.hdata}\n")
    if dataset.hdata.x is not None:
        print(f"Node features shape: {dataset.hdata.x.shape}\n")
        print(f"First 5 node features:\n {dataset.hdata.x[:5]}\n")
