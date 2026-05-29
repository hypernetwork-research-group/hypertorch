from hyperbench.data import list_datasets, get_dataset_by_name


def __find_datasets_with_number_of_nodes_smaller_than(threshold=75000):
    dataset_names = list_datasets()
    dataset_node_count = {}
    for dataset_name in dataset_names:
        dataset = get_dataset_by_name(dataset_name)
        dataset_node_count[dataset_name] = dataset.hdata.num_nodes

    filter_on_node_count = [
        dataset_name for dataset_name, count in dataset_node_count.items() if count <= threshold
    ]

    return filter_on_node_count


def __find_datasets_with_number_of_hyperedges_smaller_than(threshold=75000):
    dataset_names = list_datasets()
    dataset_hyperedge_count = {}
    for dataset_name in dataset_names:
        dataset = get_dataset_by_name(dataset_name)
        dataset_hyperedge_count[dataset_name] = dataset.hdata.num_hyperedges

    filter_on_hyperedge_count = [
        dataset_name
        for dataset_name, count in dataset_hyperedge_count.items()
        if count <= threshold
    ]

    return filter_on_hyperedge_count


def __datasets_below_node_and_hyperedge_cutoff(threshold_nodes=75000, threshold_hyperedges=75000):
    datasets_below_node_cutoff = __find_datasets_with_number_of_nodes_smaller_than(
        threshold=threshold_nodes
    )
    datasets_below_hyperedge_cutoff = __find_datasets_with_number_of_hyperedges_smaller_than(
        threshold=threshold_hyperedges
    )

    to_exclude = set(list_datasets()) - set(datasets_below_node_cutoff).intersection(
        set(datasets_below_hyperedge_cutoff)
    )
    sorted_to_exclude = sorted(to_exclude)
    return sorted_to_exclude
