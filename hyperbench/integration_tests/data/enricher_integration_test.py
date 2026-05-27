import pytest

from hyperbench.data import (
    list_datasets,
    get_dataset_by_name,
    Node2VecEnricher,
    FillValueHyperedgeAttrsEnricher,
    VilLainHyperedgeAttrsEnricher,
    VilLainEnricher,
)
from hyperbench.integration_tests.common import (
    split_dataset,
)
from hyperbench.data import LaplacianPositionalEncodingEnricher


def __find_lowest_limit_to_number_of_nodes_for_50_percent_coverage():
    dataset_names = list_datasets()
    node_counts = []
    map = {}
    for dataset_name in dataset_names:
        dataset = get_dataset_by_name(dataset_name)
        node_counts.append(dataset.hdata.num_nodes)
        map[dataset_name] = dataset.hdata.num_nodes

    node_counts.sort()
    cutoff_index = int(0.75 * len(node_counts))
    cutoff_value = node_counts[cutoff_index]

    list_of_datasets_below_cutoff = [
        dataset_name for dataset_name, node_count in map.items() if node_count <= cutoff_value
    ]
    return list_of_datasets_below_cutoff


small_datasets = __find_lowest_limit_to_number_of_nodes_for_50_percent_coverage()
NUM_FEATURES = 8

#### Why we limit the number of nodes and hyperedges in the tests to 6000? ####
#
#
# Some of the dataset have a very large number of nodes and hyperedges,
# which can lead to very long runtimes for the enrichers, especially the
# more complex ones like VilLain. To ensure that our integration tests run in a
# reasonable amount of time, we limit the number of nodes and hyperedges to 6000
# for the enrichment tests. This allows us to test the functionality of the
# enrichers without running into excessively long test times, while still providing
#  a meaningful test of their behavior on reasonably sized datasets.
# With the threshold of 6000 nodes and hyperedges, we cover the 74% of the datasets.
# The datasets.py in the scripts folder contains a function that calculates the node count cutoff to cover 75% of the datasets.


@pytest.mark.parametrize(
    "dataset_name",
    [pytest.param(dataset_name, id=f"{dataset_name}") for dataset_name in small_datasets],
)
@pytest.mark.integration
def test_lpe_node_enricher(dataset_name):
    dataset = get_dataset_by_name(dataset_name)
    _, _, to_enrich_dataset = split_dataset(dataset=dataset, node_space_setting="inductive")
    to_enrich_dataset.enrich_node_features(
        enricher=LaplacianPositionalEncodingEnricher(
            num_features=NUM_FEATURES,
            num_nodes=to_enrich_dataset.hdata.num_nodes,
        ),
        enrichment_mode="replace",
    )


@pytest.mark.parametrize(
    "dataset_name",
    [pytest.param(dataset_name, id=f"{dataset_name}") for dataset_name in small_datasets],
)
@pytest.mark.integration
def test_n2v_node_enricher(dataset_name):
    dataset = get_dataset_by_name(dataset_name)
    _, _, to_enrich_dataset = split_dataset(dataset=dataset, node_space_setting="inductive")
    to_enrich_dataset.enrich_node_features(
        enricher=Node2VecEnricher(
            num_features=NUM_FEATURES,
            context_size=2,
            walk_length=5,
            num_walks_per_node=2,
            num_negative_samples=1,
            num_nodes=dataset.hdata.num_nodes,
            num_epochs=3,
            learning_rate=0.01,
            batch_size=128,
            sparse=False,
            verbose=False,
        ),
        enrichment_mode="replace",
    )


@pytest.mark.parametrize(
    "dataset_name",
    [pytest.param(dataset_name, id=f"{dataset_name}") for dataset_name in small_datasets],
)
@pytest.mark.integration
def test_fill_value_hyperedge_enricher(dataset_name):
    dataset = get_dataset_by_name(dataset_name)
    _, _, to_enrich_dataset = split_dataset(dataset=dataset, node_space_setting="inductive")
    to_enrich_dataset.enrich_hyperedge_attr(
        enricher=FillValueHyperedgeAttrsEnricher(fill_value=1.0),
        enrichment_mode="replace",
    )


@pytest.mark.parametrize(
    "dataset_name",
    [pytest.param(dataset_name, id=f"{dataset_name}") for dataset_name in small_datasets],
)
@pytest.mark.integration
def test_villain_hyperedge_enricher(dataset_name):
    dataset = get_dataset_by_name(dataset_name)
    _, _, to_enrich_dataset = split_dataset(dataset=dataset, node_space_setting="inductive")
    to_enrich_dataset.enrich_hyperedge_attr(
        enricher=VilLainHyperedgeAttrsEnricher(
            num_features=NUM_FEATURES,
            num_nodes=dataset.hdata.num_nodes,
            num_hyperedges=dataset.hdata.num_hyperedges,
            labels_per_subspace=2,
            training_steps=2,
            generation_steps=4,
            num_epochs=3,
            learning_rate=0.01,
            verbose=False,
        ),
        enrichment_mode="replace",
    )


@pytest.mark.parametrize(
    "dataset_name",
    [pytest.param(dataset_name, id=f"{dataset_name}") for dataset_name in small_datasets],
)
@pytest.mark.integration
def test_villain_node_enricher(dataset_name):
    dataset = get_dataset_by_name(dataset_name)
    _, _, to_enrich_dataset = split_dataset(dataset=dataset, node_space_setting="inductive")
    to_enrich_dataset.enrich_node_features(
        enricher=VilLainEnricher(
            num_features=NUM_FEATURES,
            num_nodes=dataset.hdata.num_nodes,
            num_hyperedges=dataset.hdata.num_hyperedges,
            labels_per_subspace=2,
            training_steps=2,
            generation_steps=4,
            num_epochs=3,
            learning_rate=0.01,
            verbose=False,
        ),
        enrichment_mode="replace",
    )
