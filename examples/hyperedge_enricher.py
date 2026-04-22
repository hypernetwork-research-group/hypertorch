from hyperbench.nn import HyperedgeWeightsEnricher, HyperedgeAttrsEnricher
from hyperbench.data import AlgebraDataset, SamplingStrategy


if __name__ == "__main__":
    verbose = False

    sampling_strategy = SamplingStrategy.HYPEREDGE
    print("Loading and preparing dataset...")

    dataset = AlgebraDataset(sampling_strategy=sampling_strategy, prepare=True)
    # HyperedgeWeightsEnricher enriches hyperedges with their degree (number of nodes in each hyperedge) as weights.
    dataset.enrich_hyperedge_weights(
        enricher=HyperedgeWeightsEnricher(
            alpha=0.9, beta=None
        ),  # No scaling, no additional constant
        enrichment_mode="replace",
    )

    print(f"Dataset:\n {dataset.hdata}\n")
    hyperedge_weights = dataset.hdata.hyperedge_weights
    if hyperedge_weights is not None:
        print(f"First 10 hyperedge weights:\n {hyperedge_weights[:10]}\n")

    dataset = AlgebraDataset(sampling_strategy=sampling_strategy, prepare=True)
    # HyperedgeAttrsEnricher adds a feature of 1.0 for each hyperedge, which can be used as a baseline or for methods that require hyperedge features.
    dataset.enrich_hyperedge_attr(
        enricher=HyperedgeAttrsEnricher(),
        enrichment_mode="replace",
    )

    print(f"Dataset:\n {dataset.hdata}\n")
    hyperedge_attr = dataset.hdata.hyperedge_attr
    if hyperedge_attr is not None:
        print(f"First 10 hyperedge attributes:\n {hyperedge_attr[:10]}\n")
