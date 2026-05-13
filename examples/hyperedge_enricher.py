from hyperbench.nn import (
    ABHyperedgeWeightsEnricher,
    FillValueHyperedgeAttrsEnricher,
    VilLainHyperedgeAttrsEnricher,
)
from hyperbench.data import AlgebraDataset, SamplingStrategy


if __name__ == "__main__":
    print("Loading and preparing dataset...\n")

    dataset = AlgebraDataset(sampling_strategy=SamplingStrategy.HYPEREDGE)

    print("Enriching hyperedge weights...")

    dataset.enrich_hyperedge_weights(
        enricher=ABHyperedgeWeightsEnricher(
            alpha=0.9,  # Percentage of beta to add to the scaled degree.
            beta=None,  # If beta were set, it would be added to the scaled degree, scaled by alpha.
        ),
        enrichment_mode="replace",
    )

    print("Dataset after enriching hyperedge weights:")
    hyperedge_weights = dataset.hdata.hyperedge_weights
    if hyperedge_weights is not None:
        print(f"- First 10 hyperedge weights:\n {hyperedge_weights[:10]}\n")

    print("Enriching hyperedge attributes...")

    # Fill hyperedge attributes with a constant value of 1.0
    dataset.enrich_hyperedge_attr(
        enricher=FillValueHyperedgeAttrsEnricher(fill_value=1.0),
        enrichment_mode="replace",
    )

    print("Dataset after enriching hyperedge attributes:")
    hyperedge_attr = dataset.hdata.hyperedge_attr
    if hyperedge_attr is not None:
        print(f"- First 10 hyperedge attributes:\n {hyperedge_attr[:10]}\n")

    print("Enriching hyperedge attributes with VilLain...")

    dataset.enrich_hyperedge_attr(
        enricher=VilLainHyperedgeAttrsEnricher(
            num_features=32,
            num_nodes=dataset.hdata.num_nodes,
            num_hyperedges=dataset.hdata.num_hyperedges,
            labels_per_subspace=8,
            training_steps=4,
            generation_steps=32,
            num_epochs=10,
            learning_rate=0.01,
            verbose=True,
        ),
        enrichment_mode="replace",
    )

    print("Dataset after VilLain hyperedge attribute enrichment:")
    hyperedge_attr = dataset.hdata.hyperedge_attr
    if hyperedge_attr is not None:
        print(f"- Hyperedge attributes shape: {hyperedge_attr.shape}")
        print(f"- First 10 hyperedge attributes:\n {hyperedge_attr[:10]}")
