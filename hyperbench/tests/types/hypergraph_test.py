import pytest
import json
import torch

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
    assert hifhypergraph.edges == []
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
            6,  # Clique of 3 nodes: 3 undirected edges = 6 directed
            id="single_hyperedge_3_nodes",
        ),
        pytest.param(
            torch.tensor([[0, 1], [0, 0]]),
            2,  # Clique of 2 nodes: 1 undirected edge = 2 directed
            id="single_hyperedge_2_nodes",
        ),
        pytest.param(
            torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]]),
            4,  # Two disjoint hyperedges of 2 nodes each: 2 undirected edges = 4 directed
            id="two_disjoint_hyperedges",
        ),
    ],
)
def test_clique_expansion_edge_count(hyperedge_index_tensor, expected_num_edges):
    result = HyperedgeIndex(hyperedge_index_tensor).reduce_to_edge_index_on_clique_expansion()

    assert result.shape[0] == 2
    assert result.shape[1] == expected_num_edges


def test_clique_expansion_removes_selfloops():
    hyperedge_index = torch.tensor([[0, 1, 2], [0, 0, 0]])
    result = HyperedgeIndex(hyperedge_index).reduce_to_edge_index_on_clique_expansion(
        remove_selfloops=True
    )

    # No self-loops should be present (H @ H^T has zero diagonal)
    assert torch.any(result[0] == result[1]).sum().item() == 0


def test_clique_expansion_keeps_selfloops_when_disabled():
    hyperedge_index = torch.tensor([[0, 1, 2], [0, 0, 0]])
    result = HyperedgeIndex(hyperedge_index).reduce_to_edge_index_on_clique_expansion(
        remove_selfloops=False
    )

    # Self-loops should be present (H @ H^T has non-zero diagonal)
    selfloops = (result[0] == result[1]).sum().item()
    assert selfloops > 0


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
