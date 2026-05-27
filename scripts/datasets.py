from hyperbench.data import list_datasets, get_dataset_by_name


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
    filter = [count for count in node_counts if count <= 6500]

    print(
        f"Node counts for datasets with 6000 or fewer nodes: {len(filter)} out of {len(node_counts)} - {len(filter) / len(node_counts) * 100:.2f}% of datasets."
    )
    print(f"To cover 75% of datasets, we need to set the node limit to {cutoff_value} nodes.")
    list_of_datasets_below_cutoff = [
        dataset_name for dataset_name, node_count in map.items() if node_count <= cutoff_value
    ]
    return list_of_datasets_below_cutoff
