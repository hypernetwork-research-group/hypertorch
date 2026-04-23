from hyperbench.nn import HyperedgeWeightsEnricher, HyperedgeAttrsEnricher
from hyperbench.data import AlgebraDataset, SamplingStrategy


if __name__ == "__main__":
    print("Loading and preparing dataset...\n")

    dataset = AlgebraDataset(sampling_strategy=SamplingStrategy.HYPEREDGE, prepare=True)

    print("Enriching hyperedge weights...")

    # HyperedgeWeightsEnricher enriches hyperedges with their degree (number of nodes in each hyperedge) as weights.
    # It optionally applies scaling and adds a constant to the weights.
    dataset.enrich_hyperedge_weights(
        enricher=HyperedgeWeightsEnricher(
            alpha=0.9, beta=None
        ),  # No scaling, no additional constant
        enrichment_mode="replace",
    )

    print(f"Dataset after enriching hyperedge weights:")
    hyperedge_weights = dataset.hdata.hyperedge_weights
    if hyperedge_weights is not None:
        print(f"- First 10 hyperedge weights:\n {hyperedge_weights[:10]}\n")

    print("Enriching hyperedge attributes...")

    # HyperedgeAttrsEnricher adds a feature of 1.0 for each hyperedge,
    # which can be used as a baseline or for methods that require hyperedge features.
    dataset.enrich_hyperedge_attr(
        enricher=HyperedgeAttrsEnricher(),
        enrichment_mode="replace",
    )

    print(f"Dataset after enriching hyperedge attributes:")
    hyperedge_attr = dataset.hdata.hyperedge_attr
    if hyperedge_attr is not None:
        print(f"- First 10 hyperedge attributes:\n {hyperedge_attr[:10]}")
