import pytest
import json
import torch

from unittest.mock import patch
from hyperbench.types import HIFHypergraph, Hypergraph, HyperedgeIndex
from hyperbench.tests import MOCK_BASE_PATH


@pytest.fixture(autouse=True)
def seed():
    """Fix random seed for deterministic projections."""
    torch.manual_seed(42)


def test_build_HIFHypergraph_instance():
    with open(f"{MOCK_BASE_PATH}/algebra.hif.json", "r") as f:
        hiftext = json.load(f)

    hypergraph = HIFHypergraph.from_hif(hiftext)

    assert isinstance(hypergraph, HIFHypergraph)


def test_empty_hifhypergraph_returns_empty_hifhypergraph():
    hifhypergraph = HIFHypergraph.empty()

    assert hifhypergraph.network_type == "undirected"
    assert hifhypergraph.nodes == []
    assert hifhypergraph.hyperedges == []
    assert hifhypergraph.incidences == []
    assert hifhypergraph.metadata == {}


@pytest.mark.parametrize(
    "edges, expected_edges",
    [
        pytest.param([], [], id="empty_hypergraph"),
        pytest.param([[0]], [[0]], id="single_node_single_edge"),
        pytest.param(
            [[0, 1, 2]],
            [[0, 1, 2]],
            id="single_edge_multiple_nodes",
        ),
        pytest.param(
            [[0, 1], [2, 3, 4], [5]],
            [[0, 1], [2, 3, 4], [5]],
            id="multiple_edges",
        ),
        pytest.param(
            [[0, 1, 2], [1, 2, 3], [2, 3, 4]],
            [[0, 1, 2], [1, 2, 3], [2, 3, 4]],
            id="multiple_overlapping_edges",
        ),
        pytest.param([[0, 0, 1]], [[0, 0, 1]], id="duplicate_node_within_edge"),
        pytest.param([[9, 2, 5, 1]], [[9, 2, 5, 1]], id="unordered_nodes"),
    ],
)
def test_init_preserves_edges(edges, expected_edges):
    hypergraph = Hypergraph(edges)
    assert hypergraph.hyperedges == expected_edges


@pytest.mark.parametrize(
    "edges, expected_num_nodes",
    [
        pytest.param([], 0, id="empty_hypergraph"),
        pytest.param([[0]], 1, id="single_node_single_edge"),
        pytest.param([[0, 1, 2]], 3, id="multiple_nodes_single_edge"),
        pytest.param([[0], [1], [2]], 3, id="three_singleton_edges"),
        pytest.param([[0], [1], [1]], 2, id="three_singleton_edges_two_overlapping"),
        pytest.param([[0, 1], [2, 3]], 4, id="two_disjoint_edges"),
        pytest.param([[0, 1], [1, 2]], 3, id="two_overlapping_edges"),
        pytest.param(
            [[0, 1, 2], [1, 2, 3]],
            4,
            id="overlapping_edges_multiple_nodes",
        ),
        pytest.param(
            [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
            9,
            id="multiple_disjoint_edges",
        ),
        pytest.param([[5, 10, 15]], 3, id="non_contiguous_node_ids"),
        pytest.param([[0, 0, 1]], 2, id="edge_with_duplicate_node"),
        pytest.param([[0, 1], [0, 1, 2]], 3, id="edge_subset_of_another"),
        pytest.param([[9, 2, 5, 1]], 4, id="unordered_node_ids"),
    ],
)
def test_num_nodes(edges, expected_num_nodes):
    hypergraph = Hypergraph(edges)
    assert hypergraph.num_nodes == expected_num_nodes


@pytest.mark.parametrize(
    "edges, expected_num_edges",
    [
        pytest.param([], 0, id="empty_hypergraph"),
        pytest.param([[0]], 1, id="single_edge_one_node"),
        pytest.param([[0, 1, 2]], 1, id="single_edge_multiple_nodes"),
        pytest.param([[0], [1], [2]], 3, id="three_singleton_edges"),
        pytest.param([[0, 1], [2, 3]], 2, id="two_disjoint_edges"),
        pytest.param([[0, 1], [1, 2]], 2, id="two_overlapping_edges"),
        pytest.param(
            [[0, 1, 2], [1, 2, 3], [3, 4]],
            3,
            id="three_edges_with_overlap",
        ),
    ],
)
def test_num_edges(edges, expected_num_edges):
    hypergraph = Hypergraph(edges)
    assert hypergraph.num_hyperedges == expected_num_edges


@pytest.mark.parametrize(
    "edge_index_data, expected_edges",
    [
        pytest.param([[[], []]], [], id="empty_hypergraph"),
        pytest.param([[[0], [0]]], [[0]], id="single_node_single_edge"),
        pytest.param(
            [[[0, 1, 2, 3], [0, 0, 0, 0]]],
            [[0, 1, 2, 3]],
            id="multiple_nodes_single_edge",
        ),
        pytest.param(
            [[[0, 1, 2], [0, 1, 2]]],
            [[0], [1], [2]],
            id="multiple_edges_single_nodes",
        ),
        pytest.param(
            [[[0, 1, 2, 3], [0, 0, 1, 1]]],
            [[0, 1], [2, 3]],
            id="two_edges_multiple_nodes",
        ),
        pytest.param(
            [[[0, 1, 2, 3, 4, 5], [0, 0, 1, 2, 2, 2]]],
            [[0, 1], [2], [3, 4, 5]],
            id="complex_varying_edge_sizes",
        ),
    ],
)
def test_from_edge_index_parametrized(edge_index_data, expected_edges):
    nodes, edges = edge_index_data[0]
    hyperedge_index = torch.tensor([nodes, edges], dtype=torch.long)
    hypergraph = Hypergraph.from_hyperedge_index(hyperedge_index)

    assert hypergraph.hyperedges == expected_edges


@pytest.mark.parametrize(
    "hyperedge_index_tensor",
    [
        pytest.param(torch.tensor([[0, 1, 2], [0, 0, 0]]), id="single_hyperedge"),
        pytest.param(torch.tensor([[0, 1], [0, 1]]), id="two_hyperedges"),
        pytest.param(torch.tensor([[], []]), id="empty"),
    ],
)
def test_hyperedge_index_item_returns_tensor(hyperedge_index_tensor):
    hyperedge_index = HyperedgeIndex(hyperedge_index_tensor)

    assert torch.equal(hyperedge_index.item, hyperedge_index_tensor)


def test_hyperedge_index_all_ids_properties_return_original_rows():
    hyperedge_index_tensor = torch.tensor([[3, 1, 3, 2], [9, 7, 9, 8]], dtype=torch.long)

    hyperedge_index = HyperedgeIndex(hyperedge_index_tensor)

    assert torch.equal(hyperedge_index.all_node_ids, hyperedge_index_tensor[0])
    assert torch.equal(hyperedge_index.all_hyperedge_ids, hyperedge_index_tensor[1])


def test_hyperedge_index_unique_ids_properties_are_sorted():
    hyperedge_index_tensor = torch.tensor([[3, 1, 3, 2], [9, 7, 9, 8]], dtype=torch.long)

    hyperedge_index = HyperedgeIndex(hyperedge_index_tensor)

    assert torch.equal(hyperedge_index.node_ids, torch.tensor([1, 2, 3], dtype=torch.long))
    assert torch.equal(hyperedge_index.hyperedge_ids, torch.tensor([7, 8, 9], dtype=torch.long))


@pytest.mark.parametrize(
    "hyperedge_index_tensor, expected_num_incidences",
    [
        pytest.param(torch.zeros((2, 0), dtype=torch.long), 0, id="empty_hypergraph"),
        pytest.param(
            torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]], dtype=torch.long),
            4,
            id="four_incidences",
        ),
    ],
)
def test_hyperedge_index_num_incidences(hyperedge_index_tensor, expected_num_incidences):
    hyperedge_index = HyperedgeIndex(hyperedge_index_tensor)

    assert hyperedge_index.num_incidences == expected_num_incidences


def test_remove_duplicate_edges_removes_duplicates_and_returns_self():
    hyperedge_index_tensor = torch.tensor(
        [[0, 1, 0, 2, 0], [0, 0, 0, 1, 0]],
        dtype=torch.long,
    )

    hyperedge_index = HyperedgeIndex(hyperedge_index_tensor)
    result = hyperedge_index.remove_duplicate_edges()

    expected_hyperedge_index = torch.tensor([[0, 1, 2], [0, 0, 1]], dtype=torch.long)

    assert result is hyperedge_index
    assert torch.equal(result.item, expected_hyperedge_index)


def test_remove_duplicate_edges_preserves_device_dtype_and_contiguity():
    hyperedge_index_tensor = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=torch.long)

    hyperedge_index = HyperedgeIndex(hyperedge_index_tensor)
    hyperedge_index.remove_duplicate_edges()

    assert hyperedge_index.item.device == hyperedge_index_tensor.device
    assert hyperedge_index.item.dtype == hyperedge_index_tensor.dtype
    assert hyperedge_index.item.is_contiguous()


def test_to_0based_without_explicit_ids_rebases_nodes_and_hyperedges():
    hyperedge_index_tensor = torch.tensor([[5, 3, 5, 8], [9, 10, 9, 11]], dtype=torch.long)

    hyperedge_index = HyperedgeIndex(hyperedge_index_tensor)
    result = hyperedge_index.to_0based()

    expected_hyperedge_index = torch.tensor([[1, 0, 1, 2], [0, 1, 0, 2]], dtype=torch.long)

    assert result is hyperedge_index
    assert torch.equal(result.item, expected_hyperedge_index)


def test_to_0based_with_explicit_ids_rebases_using_provided_spaces():
    hyperedge_index_tensor = torch.tensor([[10, 20, 20, 30], [5, 5, 7, 7]], dtype=torch.long)

    hyperedge_index = HyperedgeIndex(hyperedge_index_tensor)
    result = hyperedge_index.to_0based(
        node_ids_to_rebase=torch.tensor([10, 20, 30], dtype=torch.long),
        hyperedge_ids_to_rebase=torch.tensor([5, 7], dtype=torch.long),
    )

    expected_hyperedge_index = torch.tensor([[0, 1, 1, 2], [0, 0, 1, 1]], dtype=torch.long)

    assert result is hyperedge_index
    assert torch.equal(result.item, expected_hyperedge_index)


@pytest.mark.parametrize(
    "hyperedge_index_tensor, expected_num_hyperedges",
    [
        pytest.param(torch.tensor([[], []]), 0, id="empty_hypergraph"),
        pytest.param(torch.tensor([[0], [0]]), 1, id="single_hyperedge_single_node"),
        pytest.param(torch.tensor([[0, 1, 2], [0, 0, 0]]), 1, id="single_hyperedge_multiple_nodes"),
        pytest.param(torch.tensor([[0, 1], [0, 1]]), 2, id="two_hyperedges"),
        pytest.param(
            torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]]), 2, id="two_hyperedges_multiple_nodes"
        ),
        pytest.param(torch.tensor([[0, 1, 2], [0, 1, 2]]), 3, id="three_singleton_hyperedges"),
        pytest.param(
            torch.tensor([[0, 1, 2, 3, 4], [0, 0, 1, 2, 2]]),
            3,
            id="multiple_hyperedges_varying_sizes",
        ),
    ],
)
def test_hyperedge_index_num_hyperedges(hyperedge_index_tensor, expected_num_hyperedges):
    hyperedge_index = HyperedgeIndex(hyperedge_index_tensor)

    assert hyperedge_index.num_hyperedges == expected_num_hyperedges


@pytest.mark.parametrize(
    "hyperedge_index_tensor, expected_num_nodes",
    [
        pytest.param(torch.tensor([[], []]), 0, id="empty_hypergraph"),
        pytest.param(torch.tensor([[0], [0]]), 1, id="single_node"),
        pytest.param(torch.tensor([[0, 1, 2], [0, 0, 0]]), 3, id="three_nodes_one_hyperedge"),
        pytest.param(torch.tensor([[0, 1], [0, 1]]), 2, id="two_nodes_two_hyperedges"),
        pytest.param(torch.tensor([[0, 5], [0, 0]]), 2, id="non_consecutive_node_indices"),
        pytest.param(
            torch.tensor([[0, 1, 2, 3, 4], [0, 0, 1, 1, 2]]), 5, id="five_nodes_three_hyperedges"
        ),
        pytest.param(torch.tensor([[9, 2, 5], [0, 0, 0]]), 3, id="sparse_node_indices"),
    ],
)
def test_hyperedge_index_num_nodes(hyperedge_index_tensor, expected_num_nodes):
    hyperedge_index = HyperedgeIndex(hyperedge_index_tensor)

    assert hyperedge_index.num_nodes == expected_num_nodes


@pytest.mark.parametrize(
    "hyperedge_index_tensor, num_nodes_arg, expected",
    [
        pytest.param(
            torch.tensor([[0, 1, 2], [0, 0, 1]]),
            10,
            10,
            id="isolated_nodes_exist",
        ),
        pytest.param(
            torch.tensor([[0, 1, 2], [0, 0, 1]]),
            3,
            3,
            id="no_isolated_nodes",
        ),
        pytest.param(
            torch.tensor([[0, 1, 2], [0, 0, 1]]),
            1,
            3,
            id="arg_less_than_unique_nodes",
        ),
        pytest.param(
            torch.zeros((2, 0), dtype=torch.long),
            0,
            0,
            id="empty_index_zero_nodes",
        ),
        pytest.param(
            torch.zeros((2, 0), dtype=torch.long),
            5,
            5,
            id="empty_index_with_isolated_nodes",
        ),
    ],
)
def test_hyperedge_index_num_nodes_if_isolated_exist(
    hyperedge_index_tensor, num_nodes_arg, expected
):
    hyperedge_index = HyperedgeIndex(hyperedge_index_tensor)

    assert hyperedge_index.num_nodes_if_isolated_exist(num_nodes_arg) == expected


@pytest.mark.parametrize(
    "hyperedges, node, expected_neighbors",
    [
        pytest.param([], 0, set(), id="empty_hypergraph"),
        pytest.param([[0, 1, 2]], 0, {1, 2}, id="single_hyperedge_node_in_hyperedge"),
        pytest.param([[0, 1, 2]], 3, set(), id="single_hyperedge_node_not_in_hyperedge"),
        pytest.param([[0, 1], [1, 2]], 1, {0, 2}, id="node_in_multiple_hyperedges"),
        pytest.param([[0, 1], [2, 3]], 0, {1}, id="node_in_one_of_two_disjoint_hyperedges"),
        pytest.param([[0, 1, 2], [2, 3, 4]], 2, {0, 1, 3, 4}, id="node_bridging_two_hyperedges"),
        pytest.param([[0, 1], [0, 2], [0, 3]], 0, {1, 2, 3}, id="node_in_all_hyperedges"),
    ],
)
def test_neighbors_of(hyperedges, node, expected_neighbors):
    hypergraph = Hypergraph(hyperedges)
    assert hypergraph.neighbors_of(node) == expected_neighbors


@pytest.mark.parametrize(
    "hyperedges, expected_neighbors_map",
    [
        pytest.param([], {}, id="empty_hypergraph"),
        pytest.param(
            [[0, 1]],
            {0: {1}, 1: {0}},
            id="single_hyperedge_two_nodes",
        ),
        pytest.param(
            [[0, 1, 2]],
            {0: {1, 2}, 1: {0, 2}, 2: {0, 1}},
            id="single_hyperedge_three_nodes",
        ),
        pytest.param(
            [[0, 1], [2, 3]],
            {0: {1}, 1: {0}, 2: {3}, 3: {2}},
            id="two_disjoint_hyperedges",
        ),
        pytest.param(
            [[0, 1], [1, 2]],
            {0: {1}, 1: {0, 2}, 2: {1}},
            id="two_overlapping_hyperedges",
        ),
    ],
)
def test_neighbors_of_all(hyperedges, expected_neighbors_map):
    hypergraph = Hypergraph(hyperedges)
    assert hypergraph.neighbors_of_all() == expected_neighbors_map


@pytest.mark.parametrize(
    "hyperedges, expected_stats",
    [
        pytest.param(
            [],
            {
                "num_nodes": 0,
                "num_hyperedges": 0,
                "avg_degree_node": 0.0,
                "avg_degree_hyperedge": 0.0,
                "node_degree_max": 0,
                "hyperedge_degree_max": 0,
                "node_degree_median": 0.0,
                "hyperedge_degree_median": 0.0,
                "distribution_node_degree": [],
                "distribution_hyperedge_size": [],
                "distribution_node_degree_hist": {},
                "distribution_hyperedge_size_hist": {},
            },
            id="empty_hypergraph",
        ),
        pytest.param(
            [[0, 1]],
            {
                "num_nodes": 2,
                "num_hyperedges": 1,
                "avg_degree_node": 1.0,
                "avg_degree_hyperedge": 2.0,
                "node_degree_max": 1,
                "hyperedge_degree_max": 2,
                "node_degree_median": 1.0,
                "hyperedge_degree_median": 2.0,
                "distribution_node_degree": [1, 1],
                "distribution_hyperedge_size": [2],
                "distribution_node_degree_hist": {1: 2},
                "distribution_hyperedge_size_hist": {2: 1},
            },
            id="single_hyperedge_two_nodes",
        ),
        pytest.param(
            [[0, 1, 2], [2, 3]],
            {
                "num_nodes": 4,
                "num_hyperedges": 2,
                "avg_degree_node": 1.25,
                "avg_degree_hyperedge": 2.5,
                "node_degree_max": 2,
                "hyperedge_degree_max": 3,
                "node_degree_median": 1.0,
                "hyperedge_degree_median": 2.5,
                "distribution_node_degree": [1, 1, 1, 2],
                "distribution_hyperedge_size": [3, 2],
                "distribution_node_degree_hist": {1: 3, 2: 1},
                "distribution_hyperedge_size_hist": {3: 1, 2: 1},
            },
            id="two_hyperedges_varying_sizes",
        ),
    ],
)
def test_hypergraph_stats_returns_correct_statistics(hyperedges, expected_stats):
    hypergraph = Hypergraph(hyperedges)
    stats = hypergraph.stats()

    assert stats == expected_stats


def test_hifhypergraph_stats_returns_correct_statistics():
    expected_stats = {
        "num_nodes": 4,
        "num_hyperedges": 2,
        "avg_degree_node_raw": 1.25,
        "avg_degree_node": 1,
        "avg_degree_hyperedge_raw": 2.5,
        "avg_degree_hyperedge": 2,
        "node_degree_max": 2,
        "hyperedge_degree_max": 3,
        "node_degree_median": 1.0,
        "hyperedge_degree_median": 2.5,
        "distribution_node_degree": [1, 1, 1, 2],
        "distribution_hyperedge_size": [2, 3],
        "distribution_node_degree_hist": {1: 3, 2: 1},
        "distribution_hyperedge_size_hist": {3: 1, 2: 1},
    }

    with open(f"{MOCK_BASE_PATH}/hif_stats.hif.json", "r") as f:
        hiftext = json.load(f)

    hypergraph = HIFHypergraph.from_hif(hiftext)
    stats = hypergraph.stats()

    assert stats == expected_stats


@pytest.mark.parametrize(
    "hyperedge_index_tensor, hyperedge_id, expected_nodes",
    [
        pytest.param(
            torch.tensor([[0, 1, 2], [0, 0, 0]]),
            0,
            [0, 1, 2],
            id="single_hyperedge_all_nodes",
        ),
        pytest.param(
            torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]]),
            0,
            [0, 1],
            id="first_of_two_hyperedges",
        ),
        pytest.param(
            torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]]),
            1,
            [2, 3],
            id="second_of_two_hyperedges",
        ),
    ],
)
def test_hyperedge_index_nodes_in(hyperedge_index_tensor, hyperedge_id, expected_nodes):
    hyperedge_index = HyperedgeIndex(hyperedge_index_tensor)
    assert hyperedge_index.nodes_in(hyperedge_id) == expected_nodes


@pytest.mark.parametrize(
    "hyperedge_index_tensor, expected_num_edges",
    [
        pytest.param(
            torch.tensor([[0, 1, 2], [0, 0, 0]]),
            9,  # Clique of 3 nodes: 6 directed cross-edges + 3 self-loops
            id="single_hyperedge_3_nodes",
        ),
        pytest.param(
            torch.tensor([[0, 1], [0, 0]]),
            4,  # Clique of 2 nodes: 2 directed cross-edges + 2 self-loops
            id="single_hyperedge_2_nodes",
        ),
        pytest.param(
            torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]]),
            8,  # Two disjoint hyperedges of 2 nodes each: 4 directed cross-edges + 4 self-loops
            id="two_disjoint_hyperedges",
        ),
    ],
)
def test_clique_expansion_edge_count(hyperedge_index_tensor, expected_num_edges):
    result = HyperedgeIndex(hyperedge_index_tensor).reduce_to_edge_index_on_clique_expansion()

    assert result.shape[0] == 2
    assert result.shape[1] == expected_num_edges


def test_clique_expansion_overlapping_hyperedges():
    # Two hyperedges sharing node 1: {0,1} and {1,2}
    hyperedge_index = torch.tensor([[0, 1, 1, 2], [0, 0, 1, 1]])
    result = HyperedgeIndex(hyperedge_index).reduce_to_edge_index_on_clique_expansion()
    edges = set(zip(result[0].tolist(), result[1].tolist()))

    # All pairs connected: 0-1, 1-2, and 0-2 (via node 1 shared in both hyperedges)
    assert (0, 1) in edges
    assert (1, 0) in edges
    assert (1, 2) in edges
    assert (2, 1) in edges


@pytest.mark.parametrize(
    "x, hyperedge_index, with_mediators, expected_num_edges",
    [
        pytest.param(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([[0, 1], [0, 0]]),
            False,
            1,  # One hyperedge, so one graph edge, no mediators to create additional edges
            id="single_hyperedge_2_nodes_no_mediators",
        ),
        pytest.param(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([[0, 1], [0, 0]]),
            True,
            # Only 2 nodes and both are extremes (argmin/argmax)
            # No mediators exist (mediators are nodes that are neither argmin nor argmax)
            # So, with mediators enabled and no mediators -> 0 edges produced
            0,
            id="single_hyperedge_2_nodes_with_mediators_produces_no_edges",
        ),
        pytest.param(
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            torch.tensor([[0, 1, 2], [0, 0, 0]]),
            False,
            1,  # One hyperedge, so one graph edge, no mediators to create additional edges
            id="single_hyperedge_3_nodes_no_mediators",
        ),
        pytest.param(
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            torch.tensor([[0, 1, 2], [0, 0, 0]]),
            True,
            2,  # If argmin = 0 and argmax = 2, mediator 1 creates 2 edges [0,1] and [1,2]
            id="single_hyperedge_3_nodes_with_mediators",
        ),
        pytest.param(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [1.0, 1.0]]),
            torch.tensor([[0, 1, 2, 3], [0, 0, 0, 0]]),
            True,
            # 2 nodes are extremes (argmin/argmax), 2 are mediators
            # Each mediator connects to both extremes: 2 mediators * 2 edges = 4 edges
            4,
            id="single_hyperedge_4_nodes_with_mediators",
        ),
        pytest.param(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [1.0, 1.0]]),
            torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]]),
            False,
            # Two hyperedges, each with 2 nodes -> 2 graph edges,
            # there are no mediators to create additional edges
            2,
            id="two_hyperedges_no_mediators",
        ),
        pytest.param(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.5, 0.5, 0.0],
                    [0.0, 0.5, 0.5],
                ]
            ),
            torch.tensor([[0, 1, 2, 2, 3, 4], [0, 0, 0, 1, 1, 1]]),
            True,
            # Hyperedge 0 has 3 nodes -> 1 mediator -> 2 edges
            # Hyperedge 1 has 3 nodes -> 1 mediator -> 2 edges
            # -> 4 edges, there are no mediators to create additional edges
            4,
            id="two_hyperedges_3_nodes_each_with_mediators",
        ),
    ],
)
def test_reduce_to_graph_edge_count(x, hyperedge_index, with_mediators, expected_num_edges):
    result = HyperedgeIndex(hyperedge_index).reduce_to_edge_index_on_random_direction(
        x, with_mediators=with_mediators, remove_selfloops=False
    )

    assert result.shape[1] == expected_num_edges


@pytest.mark.parametrize(
    "x, hyperedge_index",
    [
        pytest.param(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([[0, 1], [0, 0]]),
            id="2_nodes_1_hyperedge",
        ),
        pytest.param(
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            torch.tensor([[0, 1, 2], [0, 0, 0]]),
            id="3_nodes_1_hyperedge",
        ),
        pytest.param(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [1.0, 1.0]]),
            torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]]),
            id="4_nodes_2_hyperedges",
        ),
    ],
)
def test_reduce_to_graph_output_has_two_rows(x, hyperedge_index):
    result = HyperedgeIndex(hyperedge_index).reduce_to_edge_index_on_random_direction(x)

    assert result.shape[0] == 2


def test_reduce_to_graph_output_dtype_is_long():
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    hyperedge_index = torch.tensor([[0, 1], [0, 0]])

    result = HyperedgeIndex(hyperedge_index).reduce_to_edge_index_on_random_direction(x)

    assert result.dtype == torch.long


def test_reduce_to_graph_output_nodes_are_within_bounds():
    """All node indices in the output are valid indices from the input node set."""
    x = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.5, 0.0]])
    hyperedge_index = torch.tensor([[0, 1, 2, 1, 2, 3], [0, 0, 0, 1, 1, 1]])

    result = HyperedgeIndex(hyperedge_index).reduce_to_edge_index_on_random_direction(x)
    num_nodes = x.shape[0]

    assert result.min() >= 0
    assert result.max() < num_nodes


def test_reduce_to_graph_removes_selfloops():
    # Duplicate node in hyperedge forces a self-loop: projections are identical,
    # so argmax and argmin both select index 0, producing edge [0, 0].
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    hyperedge_index = torch.tensor([[0, 0], [0, 0]])

    result = HyperedgeIndex(hyperedge_index).reduce_to_edge_index_on_random_direction(
        x, remove_selfloops=True
    )

    # Either zero or one edge remains, the reason why one edge may remain is that
    # after removing self-loops, there could be multiple hyperedges projecting to
    # the same graph edge, which would be kept as a single edge
    # Example: hyperedges [[0,1,1],[0,0,2]] both project to graph edge [0,2]
    assert result.shape[1] <= 1

    if result.shape[1] > 0:
        # If any edges remain, check that no self-loops are present
        assert not torch.any(result[0] == result[1]).item()


def test_reduce_to_graph_keeps_selfloops_when_disabled():
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    hyperedge_index = torch.tensor([[0, 0], [0, 0]])

    result = HyperedgeIndex(hyperedge_index).reduce_to_edge_index_on_random_direction(
        x, remove_selfloops=False
    )

    assert result.shape[1] == 1  # One node, one hyperedge
    assert result[0, 0] == result[1, 0]  # Self-loop edge [0, 0] is preserved


def test_reduce_to_graph_raises_on_single_node_hyperedge():
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    hyperedge_index = torch.tensor([[0], [0]])

    with pytest.raises(ValueError, match="The number of vertices in an hyperedge must be >= 2."):
        HyperedgeIndex(hyperedge_index).reduce_to_edge_index_on_random_direction(x)


def test_reduce_to_graph_returns_empty_edge_index_for_empty_hyperedge_index():
    x = torch.empty((0, 2))
    hyperedge_index = torch.empty((2, 0), dtype=torch.long)

    result = HyperedgeIndex(hyperedge_index).reduce_to_edge_index_on_random_direction(x)

    assert result.shape == (2, 0)  # Empty edge index


def test_reduce_to_graph_keeps_duplicate_edges_from_different_hyperedges():
    x = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    # Two identical hyperedges {0,1} and both reduce to the same graph edge
    hyperedge_index = torch.tensor([[0, 1, 0, 1], [0, 0, 1, 1]])

    result = HyperedgeIndex(hyperedge_index).reduce_to_edge_index_on_random_direction(
        x,
        with_mediators=False,
        remove_selfloops=False,
    )

    assert result.shape == (2, 2)

    # hyperedges [[0,1],[0,1]] both project to graph edge [0,1],
    # so we expect two identical edges in the output
    assert torch.equal(result, torch.tensor([[0, 0], [1, 1]]))


@pytest.mark.parametrize(
    "hyperedge_index_tensor, k, expected_tensor",
    [
        pytest.param(
            torch.zeros((2, 0), dtype=torch.long),
            1,
            torch.zeros((2, 0), dtype=torch.long),
            id="empty_index",
        ),
        pytest.param(
            torch.tensor([[0, 1, 2], [0, 0, 0]]),
            1,
            torch.tensor([[0, 1, 2], [0, 0, 0]]),
            id="single_hyperedge_3_nodes_k1_all_kept",
        ),
        pytest.param(
            torch.tensor([[0, 1, 2], [0, 0, 0]]),
            3,
            torch.tensor([[0, 1, 2], [0, 0, 0]]),
            id="single_hyperedge_3_nodes_k3_exact_boundary_kept",
        ),
        pytest.param(
            torch.tensor([[0, 1, 2], [0, 0, 0]]),
            4,
            torch.zeros((2, 0), dtype=torch.long),
            id="single_hyperedge_3_nodes_k4_removed",
        ),
        pytest.param(
            torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]]),
            2,
            torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]]),
            id="two_hyperedges_2_nodes_each_k2_both_kept",
        ),
        pytest.param(
            torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]]),
            3,
            torch.zeros((2, 0), dtype=torch.long),
            id="two_hyperedges_2_nodes_each_k3_both_removed",
        ),
        pytest.param(
            torch.tensor([[0, 1, 2, 3, 4], [0, 0, 0, 1, 1]]),
            3,
            torch.tensor([[0, 1, 2], [0, 0, 0]]),
            id="two_hyperedges_first_3_nodes_second_2_nodes_k3_only_first_kept",
        ),
        pytest.param(
            torch.tensor([[0, 1, 2, 3, 4], [0, 0, 1, 1, 1]]),
            3,
            torch.tensor([[2, 3, 4], [1, 1, 1]]),
            id="two_hyperedges_first_2_nodes_second_3_nodes_k3_only_second_kept",
        ),
        pytest.param(
            torch.tensor([[0, 1, 2, 3, 4, 5], [0, 0, 1, 1, 1, 2]]),
            3,
            torch.tensor([[2, 3, 4], [1, 1, 1]]),
            id="three_hyperedges_sizes_2_3_1_k3_only_middle_kept",
        ),
        pytest.param(
            torch.tensor([[0, 1, 2, 3, 4, 5], [0, 0, 0, 1, 1, 1]]),
            2,
            torch.tensor([[0, 1, 2, 3, 4, 5], [0, 0, 0, 1, 1, 1]]),
            id="two_hyperedges_3_nodes_each_k2_both_kept",
        ),
    ],
)
def test_remove_hyperedges_with_fewer_than_k_nodes(hyperedge_index_tensor, k, expected_tensor):
    result = HyperedgeIndex(hyperedge_index_tensor).remove_hyperedges_with_fewer_than_k_nodes(k)

    assert torch.equal(result.item, expected_tensor)


def test_remove_hyperedges_with_fewer_than_k_nodes_returns_self():
    hyperedge_index = HyperedgeIndex(torch.tensor([[0, 1, 2], [0, 0, 0]]))
    result = hyperedge_index.remove_hyperedges_with_fewer_than_k_nodes(1)

    assert result is hyperedge_index


@pytest.mark.parametrize(
    "dropout",
    [
        pytest.param(None, id="no_dropout"),
        pytest.param(0.0, id="with_dropout"),
    ],
)
def test_hypergraph_smoothing_with_laplacian_does_not_apply_dropout_when_not_provided(dropout):
    x = torch.tensor([[1.0], [2.0]])
    laplacian = torch.tensor([[1.0, 0.0], [0.0, 1.0]]).to_sparse()

    with patch(
        "hyperbench.types.hypergraph.sparse_dropout",
        return_value=torch.zeros_like(laplacian),
    ) as mock_sparse_dropout:
        if dropout is not None:
            smoothed_laplacian = Hypergraph.smoothing_with_laplacian_matrix(
                x, laplacian, drop_rate=dropout
            )
        else:
            smoothed_laplacian = Hypergraph.smoothing_with_laplacian_matrix(x, laplacian)

    mock_sparse_dropout.assert_not_called()

    # It is equal to x as the laplacian is identity and no dropout is applied
    assert smoothed_laplacian.shape == x.shape
    assert torch.equal(smoothed_laplacian, x)


def test_hypergraph_smoothing_with_laplacian_applies_dropout_when_enabled():
    x = torch.tensor([[1.0], [2.0]])
    laplacian = torch.tensor([[1.0, 0.0], [0.0, 1.0]]).to_sparse()

    with patch(
        "hyperbench.types.hypergraph.sparse_dropout",
        return_value=torch.zeros_like(laplacian),
    ) as mock_sparse_dropout:
        smoothed_laplacian = Hypergraph.smoothing_with_laplacian_matrix(x, laplacian, drop_rate=0.7)

    mock_sparse_dropout.assert_called_once()

    called_matrix, called_drop_rate = mock_sparse_dropout.call_args.args

    assert called_matrix is laplacian
    assert called_drop_rate == 0.7
    assert smoothed_laplacian.shape == x.shape
    assert torch.equal(smoothed_laplacian, torch.zeros_like(x))


def test_hyperedge_index_sparse_normalized_node_degree_handles_isolated_nodes():
    # Node 2 is isolated by setting num_nodes=3 while incidences involve only nodes 0 and 1
    num_nodes = 3
    hyperedge_index = HyperedgeIndex(torch.tensor([[0, 0, 1], [0, 1, 1]]))
    incidence_matrix = hyperedge_index.get_sparse_incidence_matrix(
        num_nodes=num_nodes, num_hyperedges=2
    )

    node_degree_matrix = hyperedge_index.get_sparse_normalized_node_degree_matrix(
        incidence_matrix, num_nodes=num_nodes
    )

    expected_node_degree_matrix = torch.diag(
        torch.tensor([1 / torch.sqrt(torch.tensor(2.0)), 1.0, 0.0])
    )
    assert torch.allclose(node_degree_matrix.to_dense(), expected_node_degree_matrix, atol=1e-6)


def test_hyperedge_index_sparse_normalized_hyperedge_degree_handles_empty_hyperedge_slot():
    # Hyperedge 1 is isolated by setting num_hyperedges=2 while incidences involve only hyperedge 0
    num_hyperedges = 2
    hyperedge_index = HyperedgeIndex(torch.tensor([[0, 1], [0, 0]]))
    incidence_matrix = hyperedge_index.get_sparse_incidence_matrix(
        num_nodes=2, num_hyperedges=num_hyperedges
    )

    hyperedge_degree_matrix = hyperedge_index.get_sparse_normalized_hyperedge_degree_matrix(
        incidence_matrix, num_hyperedges=num_hyperedges
    )

    expected_hyperedge_degree_matrix = torch.diag(torch.tensor([0.5, 0.0]))
    assert torch.allclose(
        hyperedge_degree_matrix.to_dense(), expected_hyperedge_degree_matrix, atol=1e-6
    )


def test_hyperedge_index_sparse_hgnn_laplacian_matches_formula():
    hyperedge_index = HyperedgeIndex(torch.tensor([[0, 0, 1], [0, 1, 1]]))

    laplacian = hyperedge_index.get_sparse_hgnn_laplacian(num_nodes=3, num_hyperedges=2)

    incidence_matrix = torch.tensor(
        [
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    )
    node_degree_inv_sqrt = torch.diag(torch.tensor([1 / torch.sqrt(torch.tensor(2.0)), 1.0, 0.0]))
    hyperedge_degree_inv = torch.diag(torch.tensor([1.0, 0.5]))

    expected_laplacian = torch.mm(
        node_degree_inv_sqrt,
        torch.mm(
            incidence_matrix,
            torch.mm(
                hyperedge_degree_inv,
                torch.mm(incidence_matrix.t(), node_degree_inv_sqrt),
            ),
        ),
    )

    assert laplacian.is_sparse
    assert torch.allclose(laplacian.to_dense(), expected_laplacian, atol=1e-6)


def test_get_sparse_hgnn_laplacian_inferred_equals_explicit():
    hyperedge_index = HyperedgeIndex(torch.tensor([[0, 1, 0, 2], [0, 0, 1, 1]]))

    laplacian_inferred = hyperedge_index.get_sparse_hgnn_laplacian()
    laplacian_explicit = hyperedge_index.get_sparse_hgnn_laplacian(
        num_nodes=3,
        num_hyperedges=2,
    )

    assert laplacian_inferred.is_sparse
    assert laplacian_explicit.is_sparse
    assert torch.allclose(
        laplacian_inferred.to_dense(),
        laplacian_explicit.to_dense(),
        atol=1e-6,
    )


def test_get_sparse_incidence_matrix_infers_shape_and_values():
    hyperedge_index = HyperedgeIndex(torch.tensor([[0, 1, 0, 2], [0, 0, 1, 1]]))

    incidence_matrix = hyperedge_index.get_sparse_incidence_matrix()

    expected_incidence_matrix = torch.tensor(
        [
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    assert incidence_matrix.is_sparse
    assert incidence_matrix.shape == (3, 2)
    assert torch.allclose(incidence_matrix.to_dense(), expected_incidence_matrix, atol=1e-6)


def test_get_sparse_incidence_matrix_with_explicit_sizes_adds_isolated_nodes_and_hyperedges():
    hyperedge_index = HyperedgeIndex(torch.tensor([[0], [0]]))

    incidence_matrix = hyperedge_index.get_sparse_incidence_matrix(num_nodes=3, num_hyperedges=2)

    expected_incidence_matrix = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    assert incidence_matrix.shape == (3, 2)
    assert torch.allclose(incidence_matrix.to_dense(), expected_incidence_matrix, atol=1e-6)


def test_get_sparse_incidence_matrix_sums_duplicate_incidences_when_coalesced():
    hyperedge_index = HyperedgeIndex(torch.tensor([[0, 0, 1], [0, 0, 1]], dtype=torch.long))

    incidence_matrix = hyperedge_index.get_sparse_incidence_matrix()

    expected_incidence_matrix = torch.tensor(
        [
            [2.0, 0.0],
            [0.0, 1.0],
        ]
    )

    assert incidence_matrix.is_sparse
    assert incidence_matrix.shape == (2, 2)
    assert torch.allclose(incidence_matrix.to_dense(), expected_incidence_matrix, atol=1e-6)


def test_get_sparse_normalized_node_degree_matrix_is_expected_diagonal():
    hyperedge_index = HyperedgeIndex(torch.tensor([[0, 1, 0, 2], [0, 0, 1, 1]]))
    incidence_matrix = hyperedge_index.get_sparse_incidence_matrix()  # shape (3,2)

    node_degree_matrix = hyperedge_index.get_sparse_normalized_node_degree_matrix(
        incidence_matrix,
        num_nodes=3,
    )

    # Example: node degrees: [2,1,1]
    #          -> inv sqrt: [1/sqrt(2),1,1]
    expected_diagonal = torch.tensor([1 / torch.sqrt(torch.tensor(2.0)), 1.0, 1.0])

    assert node_degree_matrix.is_sparse
    assert node_degree_matrix.shape == (3, 3)
    assert torch.allclose(node_degree_matrix.to_dense(), torch.diag(expected_diagonal), atol=1e-6)


def test_get_sparse_normalized_node_degree_matrix_infers_num_nodes():
    hyperedge_index = HyperedgeIndex(torch.tensor([[0, 1, 0, 2], [0, 0, 1, 1]]))
    incidence_matrix = hyperedge_index.get_sparse_incidence_matrix()  # shape (3,2)

    node_degree_matrix = hyperedge_index.get_sparse_normalized_node_degree_matrix(incidence_matrix)

    assert node_degree_matrix.to_dense().shape == (3, 3)


def test_get_sparse_normalized_hyperedge_degree_matrix_is_expected_diagonal():
    hyperedge_index = HyperedgeIndex(torch.tensor([[0, 1, 0, 2], [0, 0, 1, 1]]))
    incidence_matrix = hyperedge_index.get_sparse_incidence_matrix()  # shape (3,2)

    hyperedge_degree_matrix = hyperedge_index.get_sparse_normalized_hyperedge_degree_matrix(
        incidence_matrix,
        num_hyperedges=2,
    )

    # Example: hyperedge degrees: [2, 2]
    #          -> inv: [0.5, 0.5]
    expected_diagonal = torch.diag(torch.tensor([0.5, 0.5]))

    assert hyperedge_degree_matrix.is_sparse
    assert hyperedge_degree_matrix.shape == (2, 2)
    assert torch.allclose(hyperedge_degree_matrix.to_dense(), expected_diagonal, atol=1e-6)


def test_get_sparse_normalized_hyperedge_degree_matrix_infers_num_hyperedges():
    hyperedge_index = HyperedgeIndex(torch.tensor([[0, 1, 0, 2], [0, 0, 1, 1]]))
    incidence_matrix = hyperedge_index.get_sparse_incidence_matrix()  # shape (3,2)

    hyperedge_degree_matrix = hyperedge_index.get_sparse_normalized_hyperedge_degree_matrix(
        incidence_matrix
    )

    assert hyperedge_degree_matrix.shape == (2, 2)
