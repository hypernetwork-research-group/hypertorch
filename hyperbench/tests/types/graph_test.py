import pytest
import torch

from hyperbench.types import EdgeIndex, Graph


@pytest.fixture
def mock_single_edge_graph():
    return Graph([[0, 1]])


@pytest.fixture
def mock_linear_graph():
    # Linear graph: 0-1-2-3
    return Graph([[0, 1], [1, 2], [2, 3]])


@pytest.fixture
def mock_graph_with_only_selfloops():
    return Graph([[0, 0], [1, 1]])


@pytest.fixture
def mock_graph_with_one_selfloop():
    return Graph([[0, 1], [1, 1], [2, 3]])


@pytest.mark.parametrize(
    "graph, expected_edges",
    [
        pytest.param(Graph([]), [], id="empty_graph"),
        pytest.param(Graph([[0, 1]]), [[0, 1]], id="single_edge"),
        pytest.param(
            Graph([[0, 1], [1, 2], [2, 3]]),
            [[0, 1], [1, 2], [2, 3]],
            id="linear_graph",
        ),
    ],
)
def test_init_edges(graph, expected_edges):
    assert graph.edges == expected_edges


@pytest.mark.parametrize(
    "graph, expected_num_nodes",
    [
        pytest.param(Graph([]), 0, id="empty_graph"),
        pytest.param(Graph([[0, 1]]), 2, id="single_edge"),
        pytest.param(Graph([[0, 0]]), 1, id="single_edge_selfloop"),
        pytest.param(Graph([[0, 1], [1, 2], [2, 3]]), 4, id="linear_graph"),
        pytest.param(Graph([[0, 0], [1, 1]]), 2, id="only_selfloops"),
        pytest.param(Graph([[0, 1], [1, 1], [2, 3]]), 4, id="one_selfloop"),
        pytest.param(Graph([[0, 1], [2, 3]]), 4, id="disconnected_graph"),
        pytest.param(
            Graph([[0, 1], [0, 1], [1, 2]]),
            3,
            id="duplicate_edges",
        ),
        pytest.param(Graph([[0, 1], [0, 2], [1, 2]]), 3, id="complete_graph"),
    ],
)
def test_num_nodes(graph, expected_num_nodes):
    assert graph.num_nodes == expected_num_nodes


@pytest.mark.parametrize(
    "graph, expected_num_edges",
    [
        pytest.param(Graph([]), 0, id="empty_graph"),
        pytest.param(Graph([[0, 1]]), 1, id="single_edge"),
        pytest.param(Graph([[0, 0]]), 1, id="single_edge_selfloop"),
        pytest.param(Graph([[0, 1], [1, 2], [2, 3]]), 3, id="linear_graph"),
        pytest.param(Graph([[0, 0], [1, 1]]), 2, id="only_selfloops"),
        pytest.param(Graph([[0, 1], [1, 1], [2, 3]]), 3, id="one_selfloop"),
        pytest.param(Graph([[0, 1], [2, 3]]), 2, id="disconnected_graph"),
        pytest.param(
            Graph([[0, 1], [0, 1], [1, 2]]),
            3,
            id="duplicate_edges",
        ),
        pytest.param(Graph([[0, 1], [0, 2], [1, 2]]), 3, id="complete_graph"),
    ],
)
def test_num_edges(graph, expected_num_edges):
    assert graph.num_edges == expected_num_edges


@pytest.mark.parametrize(
    "graph, expected_edges_after_removal",
    [
        pytest.param(Graph([]), [], id="empty_graph"),
        pytest.param(
            Graph([[0, 1], [2, 3]]),
            [[0, 1], [2, 3]],
            id="no_selfloops",
        ),
        pytest.param(Graph([[0, 0]]), [], id="one_edge_one_selfloop"),
        pytest.param(Graph([[0, 1], [1, 1]]), [[0, 1]], id="one_selfloop"),
        pytest.param(
            Graph([[0, 0], [1, 1], [2, 2]]),
            [],
            id="all_selfloops",
        ),
        pytest.param(
            Graph([[0, 1], [1, 2], [2, 2]]),
            [[0, 1], [1, 2]],
            id="mixed_edges",
        ),
        pytest.param(
            Graph([[0, 0], [0, 1], [1, 1], [1, 2]]),
            [[0, 1], [1, 2]],
            id="mixed_edges_multiple_selfloops",
        ),
        pytest.param(
            Graph([[0, 0], [1, 1], [2, 2], [3, 4]]),
            [[3, 4]],
            id="multiple_consecutive_selfloops",
        ),
    ],
)
def test_remove_selfloops(graph, expected_edges_after_removal):
    graph.remove_selfloops()
    assert graph.edges == expected_edges_after_removal


def test_remove_selfloops_preserves_order():
    graph = Graph([[0, 1], [1, 1], [2, 3], [3, 3], [4, 5]])
    graph.remove_selfloops()
    assert graph.edges == [[0, 1], [2, 3], [4, 5]]


@pytest.mark.parametrize(
    "graph, expected_edge_index",
    [
        pytest.param(
            Graph([]),
            torch.empty((2, 0), dtype=torch.long),
            id="empty_graph",
        ),
        pytest.param(
            Graph([[0, 1]]),
            torch.tensor([[0], [1]], dtype=torch.long),
            id="single_edge",
        ),
        pytest.param(
            Graph([[0, 1], [1, 2]]),
            torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            id="multiple_edges",
        ),
        pytest.param(
            Graph([[0, 1], [1, 2], [2, 3]]),
            torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
            id="linear_graph",
        ),
        pytest.param(
            Graph([[0, 0], [1, 1]]),
            torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
            id="only_selfloops",
        ),
        pytest.param(
            Graph([[0, 1], [0, 1], [1, 2]]),
            torch.tensor([[0, 0, 1], [1, 1, 2]], dtype=torch.long),
            id="duplicate_edges",
        ),
    ],
)
def test_to_edge_index(graph, expected_edge_index):
    edge_index = graph.to_edge_index()
    assert torch.equal(edge_index, expected_edge_index)


def test_to_edge_index_returns_long_dtype(mock_single_edge_graph):
    edge_index = mock_single_edge_graph.to_edge_index()
    assert edge_index.dtype == torch.long


def test_to_edge_index_large_graph():
    edges = [[i, i + 1] for i in range(1000)]
    graph = Graph(edges)

    edge_index = graph.to_edge_index()

    assert edge_index.shape == (2, 1000)
    assert edge_index[0, 0] == 0
    assert edge_index[1, -1] == 1000


def test_to_edge_index_does_not_modify_graph(mock_linear_graph):
    original_edges = [edge[:] for edge in mock_linear_graph.edges]
    _ = mock_linear_graph.to_edge_index()

    assert mock_linear_graph.edges == original_edges


def test_to_edge_index_is_contiguous(mock_single_edge_graph):
    """
    Test that to_edge_index returns a contiguous tensor.

    Example:
        If edges = [[0, 1]], then edge_index = [[0], [1]] should be contiguous.
        If edges = [[0, 1], [1, 2], [2, 3]], then edge_index = [[0, 1, 2], [1, 2, 3]] should be contiguous.
    """
    edge_index = mock_single_edge_graph.to_edge_index()
    assert edge_index.is_contiguous()


def test_to_edge_index_before_and_after_removal_all_selfloops(
    mock_graph_with_only_selfloops,
):
    edge_index_before = mock_graph_with_only_selfloops.to_edge_index()
    assert edge_index_before.shape == (2, 2)

    mock_graph_with_only_selfloops.remove_selfloops()
    edge_index_after = mock_graph_with_only_selfloops.to_edge_index()

    expected = torch.tensor([], dtype=torch.long).reshape(2, 0)

    assert edge_index_after.shape == (2, 0)
    assert torch.equal(edge_index_after, expected)


def test_to_edge_index_before_and_after_removal_one_selfloops(
    mock_graph_with_one_selfloop,
):
    edge_index_before = mock_graph_with_one_selfloop.to_edge_index()
    assert edge_index_before.shape == (2, 3)

    mock_graph_with_one_selfloop.remove_selfloops()
    edge_index_after = mock_graph_with_one_selfloop.to_edge_index()

    expected = torch.tensor([[0, 2], [1, 3]])

    assert edge_index_after.shape == (2, 2)
    assert torch.equal(edge_index_after, expected)


def test_bidirectional_edges():
    graph = Graph([[0, 1], [1, 0]])
    assert graph.num_edges == 2
    assert graph.num_nodes == 2

    edge_index = graph.to_edge_index()

    expected = torch.tensor([[0, 1], [1, 0]])

    assert torch.equal(edge_index, expected)


def test_star_graph():
    """Test star graph (all edges connected to central node)."""
    graph = Graph([[0, 1], [0, 2], [0, 3], [0, 4]])
    assert graph.num_nodes == 5
    assert graph.num_edges == 4

    edge_index = graph.to_edge_index()

    assert edge_index.shape == (2, 4)


def test_cyclic_graph():
    """Test cyclic graph (a closed loop)."""
    graph = Graph([[0, 1], [1, 2], [2, 3], [3, 0]])
    assert graph.num_nodes == 4
    assert graph.num_edges == 4

    edge_index = graph.to_edge_index()

    assert edge_index.shape == (2, 4)


@pytest.mark.parametrize(
    "num_nodes, num_features",
    [
        pytest.param(2, 2, id="2x2"),
        pytest.param(3, 4, id="3x4"),
        pytest.param(5, 1, id="5x1"),
        pytest.param(10, 8, id="10x8"),
    ],
)
def test_smoothing_with_gcn_laplacian_output_shape_matches_x_shape(num_nodes, num_features):
    """Output shape should match input node feature matrix X shape (|V|, C)."""
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.tensor([[i, (i + 1) % num_nodes] for i in range(num_nodes)]).T

    gcn_laplacian = EdgeIndex(edge_index).get_sparse_normalized_gcn_laplacian(num_nodes=num_nodes)

    smoothed_x = Graph.smoothing_with_gcn_laplacian_matrix(x, gcn_laplacian)

    assert smoothed_x.shape == x.shape


def test_smoothing_with_gcn_laplacian_with_identity_laplacian_returns_original_x():
    """Smoothing with identity laplacian should return the original features."""
    num_nodes = 3
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    # Create identity matrix as sparse tensor
    indices = torch.arange(num_nodes).unsqueeze(0).repeat(2, 1)
    values = torch.ones(num_nodes)
    identity_gcn_laplacian = torch.sparse_coo_tensor(indices, values, size=(num_nodes, num_nodes))

    smoothed_x = Graph.smoothing_with_gcn_laplacian_matrix(x, identity_gcn_laplacian)

    assert torch.allclose(smoothed_x, x, atol=1e-6)


def test_smoothing_with_gcn_laplacian_zero_features():
    """Zero features should remain zero after smoothing."""
    x = torch.zeros(3, 2)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    gcn_laplacian = EdgeIndex(edge_index).get_sparse_normalized_gcn_laplacian(num_nodes=3)

    smoothed_x = Graph.smoothing_with_gcn_laplacian_matrix(x, gcn_laplacian)

    assert torch.allclose(smoothed_x, torch.zeros_like(x), atol=1e-6)


def test_smoothing_with_gcn_laplacian_single_node_returns_original_x():
    """Single node with self-loop should return the original features."""
    x = torch.tensor([[1.0, 2.0]])
    edge_index = torch.tensor([[0], [0]])  # Self-loop
    gcn_laplacian = EdgeIndex(edge_index).get_sparse_normalized_gcn_laplacian(num_nodes=1)

    smoothed_x = Graph.smoothing_with_gcn_laplacian_matrix(x, gcn_laplacian)

    # Single node with self-loop has L[0,0] = 1, so smoothed_x = 1 * x = x
    # as the laplacian is [[1.0]], so:
    # smoothed_x = L @ x = [[1.0]] @ [[1.0, 2.0]] = [[1.0 * 1.0, 1.0 * 2.0]] = [[1.0, 2.0]] = x
    assert torch.allclose(smoothed_x, x, atol=1e-6)


def test_smoothing_with_gcn_laplacian_preserves_x_device():
    device = torch.device("cpu")

    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=device)
    edge_index = torch.tensor([[0, 1], [1, 0]], device=device)
    gcn_laplacian = EdgeIndex(edge_index).get_sparse_normalized_gcn_laplacian(num_nodes=2)

    smoothed_x = Graph.smoothing_with_gcn_laplacian_matrix(x, gcn_laplacian)

    assert smoothed_x.device == x.device


def test_smoothing_with_gcn_laplacian_preserves_x_dtype():
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    edge_index = torch.tensor([[0, 1], [1, 0]])
    gcn_laplacian = EdgeIndex(edge_index).get_sparse_normalized_gcn_laplacian(num_nodes=2)

    smoothed_x = Graph.smoothing_with_gcn_laplacian_matrix(x, gcn_laplacian)

    assert smoothed_x.dtype == x.dtype


def test_smoothing_with_gcn_laplacian_no_nan_or_inf():
    x = torch.randn(5, 3)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    gcn_laplacian = EdgeIndex(edge_index).get_sparse_normalized_gcn_laplacian(num_nodes=5)

    smoothed_x = Graph.smoothing_with_gcn_laplacian_matrix(x, gcn_laplacian)

    assert not torch.any(torch.isnan(smoothed_x))
    assert not torch.any(torch.isinf(smoothed_x))


def test_smoothing_with_gcn_laplacian_returns_expected_x():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    edge_index = torch.tensor([[0, 1], [1, 0]])
    gcn_laplacian = EdgeIndex(edge_index).get_sparse_normalized_gcn_laplacian(num_nodes=2)

    smoothed_x = Graph.smoothing_with_gcn_laplacian_matrix(x, gcn_laplacian)

    # For 2 nodes with bidirectional edge, GCN adds self-loops, so each node has degree 2.
    # The GCN Laplacian L = D^-1/2 * A_hat * D^-1/2 = [[0.5, 0.5],
    #                                                  [0.5, 0.5]]
    # L @ x = [[0.5*1 + 0.5*3, 0.5*2 + 0.5*4],
    #          [0.5*1 + 0.5*3, 0.5*2 + 0.5*4]]
    #       = [[2.0, 3.0],
    #          [2.0, 3.0]]
    expected_smoothed_x = torch.tensor([[2.0, 3.0], [2.0, 3.0]])

    assert torch.allclose(smoothed_x, expected_smoothed_x, atol=1e-6)


def test_smoothing_with_gcn_laplacian_is_equal_for_zero_and_no_drop_rate():
    """drop_rate=0 should produce the same result as no dropout."""
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    gcn_laplacian = EdgeIndex(edge_index).get_sparse_normalized_gcn_laplacian(num_nodes=3)

    smoothed_x_no_dropout = Graph.smoothing_with_gcn_laplacian_matrix(x, gcn_laplacian)
    smoothed_x_zero_dropout = Graph.smoothing_with_gcn_laplacian_matrix(
        x, gcn_laplacian, drop_rate=0.0
    )

    assert torch.allclose(smoothed_x_no_dropout, smoothed_x_zero_dropout, atol=1e-6)


def test_smoothing_with_gcn_laplacian_nonzero_drop_rate_changes_output():
    torch.manual_seed(123)
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    gcn_laplacian = EdgeIndex(edge_index).get_sparse_normalized_gcn_laplacian(num_nodes=3)

    smoothed_x_no_dropout = Graph.smoothing_with_gcn_laplacian_matrix(
        x, gcn_laplacian.clone(), drop_rate=0.0
    )
    smoothed_x_with_dropout = Graph.smoothing_with_gcn_laplacian_matrix(
        x, gcn_laplacian.clone(), drop_rate=0.7
    )

    assert not torch.allclose(smoothed_x_no_dropout, smoothed_x_with_dropout, atol=1e-6)


def test_smoothing_with_gcn_laplacian_drop_rate_stochastic():
    """Different seeds should produce different outputs with dropout."""
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    gcn_laplacian = EdgeIndex(edge_index).get_sparse_normalized_gcn_laplacian(num_nodes=3)

    torch.manual_seed(42)
    smoothed_x1 = Graph.smoothing_with_gcn_laplacian_matrix(x, gcn_laplacian.clone(), drop_rate=0.5)

    torch.manual_seed(99)
    smoothed_x2 = Graph.smoothing_with_gcn_laplacian_matrix(x, gcn_laplacian.clone(), drop_rate=0.5)

    # Different random seeds should produce different dropout masks
    assert not torch.allclose(smoothed_x1, smoothed_x2, atol=1e-6)


def test_smoothing_with_gcn_laplacian_influences_connected_nodes():
    """
    Features of connected nodes should be aggregated.
    For a connected graph with GCN normalization, smoothing should mix features from neighbors.
    """
    # Two connected nodes with distinct features
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    edge_index = torch.tensor([[0, 1], [1, 0]])  # Bidirectional edge
    gcn_laplacian = EdgeIndex(edge_index).get_sparse_normalized_gcn_laplacian(num_nodes=2)

    smoothed_x = Graph.smoothing_with_gcn_laplacian_matrix(x, gcn_laplacian)

    # After smoothing, node 0 should have some of node 1's features and vice versa
    # Row sum of GCN laplacian is 1 for connected graphs, so features are mixed
    assert smoothed_x[0, 1] > 0  # Node 0 now has some of feature dimension 1 from node 1
    assert smoothed_x[1, 0] > 0  # Node 1 now has some of feature dimension 0 from node 0


def test_smoothing_with_gcn_laplacian_isolated_nodes_have_zero_features():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    edge_index = torch.tensor([[0], [1]])  # Only nodes 0 and 1 connected
    gcn_laplacian = EdgeIndex(edge_index).get_sparse_normalized_gcn_laplacian(num_nodes=3)

    smoothed_x = Graph.smoothing_with_gcn_laplacian_matrix(x, gcn_laplacian)

    # Node 2 is isolated, so its output should be zero
    assert torch.allclose(smoothed_x[2], torch.zeros(2), atol=1e-6)


def test_edge_index_item_returns_tensor():
    """Test that item property returns the edge index tensor."""
    edge_index_tensor = torch.tensor([[0, 1, 2], [1, 2, 0]])
    edge_index = EdgeIndex(edge_index_tensor)

    assert torch.equal(edge_index.item, edge_index_tensor)


@pytest.mark.parametrize(
    "edge_index_tensor, expected_num_edges",
    [
        pytest.param(torch.tensor([[0], [1]]), 1, id="single_edge"),
        pytest.param(torch.tensor([[0, 1, 2], [1, 2, 3]]), 3, id="multiple_edges"),
        pytest.param(torch.tensor([[], []]), 0, id="empty_edge_index"),
        pytest.param(torch.tensor([[0, 1, 1], [0, 1, 2]]), 3, id="with_selfloops"),
    ],
)
def test_edge_index_num_edges(edge_index_tensor, expected_num_edges):
    edge_index = EdgeIndex(edge_index_tensor)

    assert edge_index.num_edges == expected_num_edges


@pytest.mark.parametrize(
    "edge_index_tensor, expected_max_node_id",
    [
        pytest.param(torch.tensor([[0], [1]]), 1, id="single_edge"),
        pytest.param(torch.tensor([[0, 1, 2], [1, 2, 3]]), 3, id="multiple_edges"),
        pytest.param(torch.tensor([[], []]), -1, id="empty_edge_index"),
        pytest.param(torch.tensor([[0, 5], [3, 7]]), 7, id="non_consecutive_indices"),
        pytest.param(torch.tensor([[0, 1, 1], [0, 1, 2]]), 2, id="with_selfloops"),
        pytest.param(torch.tensor([[10, 20], [5, 15]]), 20, id="large_sparse_ids"),
    ],
)
def test_edge_index_max_node_id(edge_index_tensor, expected_max_node_id):
    edge_index = EdgeIndex(edge_index_tensor)

    assert edge_index.max_node_id == expected_max_node_id


@pytest.mark.parametrize(
    "edge_index_tensor, expected_num_nodes",
    [
        pytest.param(torch.tensor([[0], [1]]), 2, id="single_edge"),
        pytest.param(torch.tensor([[0, 1, 2], [1, 2, 3]]), 4, id="multiple_edges"),
        pytest.param(torch.tensor([[], []]), 0, id="empty_edge_index"),
        pytest.param(torch.tensor([[0, 5], [3, 7]]), 4, id="non_consecutive_indices"),
        pytest.param(torch.tensor([[0, 1, 1], [0, 1, 2]]), 3, id="with_selfloops"),
    ],
)
def test_edge_index_num_nodes(edge_index_tensor, expected_num_nodes):
    edge_index = EdgeIndex(edge_index_tensor)

    assert edge_index.num_nodes == expected_num_nodes


def test_add_selfloops_returns_correct_edges():
    edge_index = EdgeIndex(torch.tensor([[0, 1], [1, 2]]))
    edge_index.add_selfloops()

    edges = set(zip(edge_index.item[0].tolist(), edge_index.item[1].tolist()))

    # Original edges should still be present
    assert (0, 1) in edges
    assert (1, 2) in edges

    # Self-loops for nodes 0, 1, 2 should be added
    assert (0, 0) in edges
    assert (1, 1) in edges
    assert (2, 2) in edges


def test_add_selfloops_raises_on_empty_edge_index():
    edge_index = EdgeIndex(torch.tensor([[], []], dtype=torch.long))

    with pytest.raises(ValueError, match="Edge index must have at least one edge"):
        edge_index.add_selfloops()


def test_add_selfloops_does_not_duplicate_selfloops():
    edge_index = EdgeIndex(torch.tensor([[0, 1, 1], [1, 2, 1]]))
    edge_index.add_selfloops()

    edges = list(zip(edge_index.item[0].tolist(), edge_index.item[1].tolist()))

    # (1, 1) should appear only once after duplicate removal
    assert edges.count((1, 1)) == 1


def test_add_selfloops_without_duplicate_removal():
    edge_index = EdgeIndex(torch.tensor([[0, 1, 0], [1, 2, 0]]))
    edge_index.add_selfloops(with_duplicate_removal=False)

    edges = list(zip(edge_index.item[0].tolist(), edge_index.item[1].tolist()))

    # (0, 0) should appear twice: once from original, once from self-loop addition
    assert edges.count((0, 0)) == 2


def test_add_selfloops_infers_num_nodes():
    edge_index = EdgeIndex(torch.tensor([[0, 1], [2, 3]]))
    edge_index.add_selfloops()

    assert edge_index.num_edges == 2 + 4  # 2 original + 4 self-loops (0,0), (1,1), (2,2), (3,3)


def test_add_selfloops_preserves_device():
    device = torch.device("cpu")

    edge_index = EdgeIndex(torch.tensor([[0, 1], [1, 2]], device=device))
    edge_index.add_selfloops()

    assert edge_index.item.device == device


def test_add_selfloops_preserves_dtype():
    edge_index = EdgeIndex(torch.tensor([[0, 1], [1, 2]], dtype=torch.long))
    edge_index.add_selfloops()

    assert edge_index.item.dtype == torch.long


def test_add_selfloops_modifies_in_place():
    tensor = torch.tensor([[0, 1], [1, 2]])
    edge_index = EdgeIndex(tensor)
    original_shape = edge_index.item.shape

    edge_index.add_selfloops()

    assert edge_index.item.shape != original_shape


def test_get_sparse_adjacency_matrix_returns_sparse_tensor():
    edge_index = torch.tensor([[0, 1], [1, 0]])
    adj_matrix = EdgeIndex(edge_index).get_sparse_adjacency_matrix(num_nodes=2)

    assert adj_matrix.is_sparse


@pytest.mark.parametrize(
    "edge_index, num_nodes",
    [
        pytest.param(torch.tensor([[0, 1], [1, 0]]), 2, id="2_nodes"),
        pytest.param(torch.tensor([[0, 1, 2], [1, 2, 0]]), 4, id="4_nodes_3_edges"),
        pytest.param(torch.tensor([[], []], dtype=torch.long), 5, id="5_nodes_empty"),
    ],
)
def test_get_sparse_adjacency_matrix_shape(edge_index, num_nodes):
    adj_matrix = EdgeIndex(edge_index).get_sparse_adjacency_matrix(num_nodes=num_nodes)

    assert adj_matrix.shape == (num_nodes, num_nodes)


def test_get_sparse_adjacency_matrix_empty_edge_index():
    """Empty edge_index produces all-zero adjacency matrix when converted to dense."""
    edge_index = torch.tensor([[], []], dtype=torch.long)
    adj_matrix = EdgeIndex(edge_index).get_sparse_adjacency_matrix(num_nodes=3)
    dense_adj_matrix = adj_matrix.to_dense()

    assert torch.all(dense_adj_matrix == 0)


@pytest.mark.parametrize(
    "edge_index, num_nodes, expected_entries",
    [
        pytest.param(
            torch.tensor([[0], [2]]),
            3,
            [(0, 2, 1.0)],
            id="single_directed_edge",
        ),
        pytest.param(
            torch.tensor([[0, 1], [1, 0]]),
            2,
            [(0, 1, 1.0), (1, 0, 1.0)],
            id="undirected_edge",
        ),
        pytest.param(
            torch.tensor([[1], [1]]),
            3,
            [(1, 1, 1.0)],
            id="self_loop",
        ),
        pytest.param(
            torch.tensor([[0, 1, 2], [1, 2, 0]]),
            3,
            [(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)],
            id="triangle_directed",
        ),
        pytest.param(
            torch.tensor([[0, 1, 2, 2], [1, 2, 0, 1]]),
            3,
            [(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0), (2, 1, 1.0)],
            id="multiple_edges_between_nodes",
        ),
        pytest.param(
            torch.tensor([[0, 1, 2, 2], [1, 2, 0, 0]]),
            3,
            [(0, 1, 1.0), (1, 2, 1.0), (2, 0, 2.0)],  # Duplicate edges are summed
            id="duplicate_edges_to_same_target",
        ),
    ],
)
def test_get_sparse_adjacency_matrix_entries(edge_index, num_nodes, expected_entries):
    adj_matrix = EdgeIndex(edge_index).get_sparse_adjacency_matrix(num_nodes=num_nodes)
    dense_adj_matrix = adj_matrix.to_dense()

    for row, col, val in expected_entries:
        assert dense_adj_matrix[row, col] == val


def test_get_sparse_adjacency_matrix_preserves_device():
    edge_index = torch.tensor([[0], [1]], device="cpu")

    adj_matrix = EdgeIndex(edge_index).get_sparse_adjacency_matrix(num_nodes=2)

    assert adj_matrix.device == edge_index.device


@pytest.mark.parametrize(
    "edge_index, num_nodes, isolated_nodes",
    [
        pytest.param(
            torch.tensor([[0], [1]]),
            4,
            [2, 3],
            id="two_isolated_nodes",
        ),
        pytest.param(
            torch.tensor([[0, 1], [1, 0]]),
            5,
            [2, 3, 4],
            id="three_isolated_nodes",
        ),
    ],
)
def test_get_sparse_adjacency_matrix_isolated_nodes(edge_index, num_nodes, isolated_nodes):
    """Nodes not in edge_index have zero rows and columns."""
    adj_matrix = EdgeIndex(edge_index).get_sparse_adjacency_matrix(num_nodes=num_nodes)
    dense_adj_matrix = adj_matrix.to_dense()

    for node in isolated_nodes:
        assert torch.all(dense_adj_matrix[node, :] == 0)
        assert torch.all(dense_adj_matrix[:, node] == 0)


def test_get_sparse_normalized_degree_matrix_returns_sparse_tensor():
    edge_index = torch.tensor([[0, 1], [1, 0]])
    degree_matrix = EdgeIndex(edge_index).get_sparse_normalized_degree_matrix(num_nodes=2)

    assert degree_matrix.is_sparse


@pytest.mark.parametrize(
    "edge_index, num_nodes",
    [
        pytest.param(torch.tensor([[0, 1], [1, 0]]), 2, id="2_nodes"),
        pytest.param(torch.tensor([[0, 1, 2], [1, 2, 0]]), 4, id="4_nodes_3_edges"),
        pytest.param(torch.tensor([[], []], dtype=torch.long), 5, id="5_nodes_no_edges"),
    ],
)
def test_get_sparse_normalized_degree_matrix_shape(edge_index, num_nodes):
    degree_matrix = EdgeIndex(edge_index).get_sparse_normalized_degree_matrix(num_nodes=num_nodes)

    assert degree_matrix.shape == (num_nodes, num_nodes)


def test_get_sparse_normalized_degree_matrix_is_diagonal():
    """All non-zero entries are on the diagonal."""
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])

    degree_matrix = EdgeIndex(edge_index).get_sparse_normalized_degree_matrix(num_nodes=3)
    dense_degree_matrix = degree_matrix.to_dense()

    # Off-diagonal entries should be zero
    for i in range(3):
        for j in range(3):
            if i != j:
                assert dense_degree_matrix[i, j] == 0


@pytest.mark.parametrize(
    "edge_index, num_nodes, expected_diagonal",
    [
        pytest.param(
            torch.tensor([[0, 1], [1, 0]]),
            2,
            [1.0, 1.0],  # degree 1 -> 1^-0.5 = 1
            id="degree_1_each",
        ),
        pytest.param(
            torch.tensor([[0, 0, 1], [1, 2, 0]]),
            3,
            # degrees [2, 1, 0] -> [2**-0.5 == 1 / 2**0.5, 1.0, 0] -> [0.707, 1, 0]
            [1 / (2**0.5), 1.0, 0.0],
            id="mixed_degrees",
        ),
        pytest.param(
            torch.tensor([[0, 0, 0, 0], [1, 2, 3, 4]]),
            5,
            [0.5, 0.0, 0.0, 0.0, 0.0],  # degree 4 -> 4^-0.5 = 0.5, others are isolated
            id="single_hub_node",
        ),
    ],
)
def test_get_sparse_normalized_degree_matrix_diagonal_values(
    edge_index, num_nodes, expected_diagonal
):
    degree_matrix = EdgeIndex(edge_index).get_sparse_normalized_degree_matrix(num_nodes=num_nodes)
    dense_degree_matrix = degree_matrix.to_dense()

    for i, expected_val in enumerate(expected_diagonal):
        assert torch.isclose(
            dense_degree_matrix[i, i],
            torch.tensor(expected_val),
            atol=1e-6,
        )


def test_get_sparse_normalized_degree_matrix_isolated_nodes_are_zero():
    """Isolated nodes (degree 0) have 0 on diagonal, not inf."""
    edge_index = torch.tensor([[0], [1]])

    degree_matrix = EdgeIndex(edge_index).get_sparse_normalized_degree_matrix(num_nodes=4)
    dense_degree_matrix = degree_matrix.to_dense()

    # Nodes 2 and 3 are isolated
    assert dense_degree_matrix[2, 2] == 0
    assert dense_degree_matrix[3, 3] == 0
    # No inf values
    assert not torch.any(torch.isinf(dense_degree_matrix))


def test_get_sparse_normalized_degree_matrix_empty_edge_index():
    """Empty edge_index produces all-zero matrix (all nodes isolated)."""
    edge_index = torch.tensor([[], []], dtype=torch.long)

    degree_matrix = EdgeIndex(edge_index).get_sparse_normalized_degree_matrix(num_nodes=3)
    dense_degree_matrix = degree_matrix.to_dense()

    assert torch.all(dense_degree_matrix == 0)


def test_get_sparse_normalized_degree_matrix_preserves_device():
    edge_index = torch.tensor([[0], [1]], device="cpu")

    degree_matrix = EdgeIndex(edge_index).get_sparse_normalized_degree_matrix(num_nodes=2)

    assert degree_matrix.device == edge_index.device


def test_get_sparse_normalized_laplacian_returns_sparse_tensor():
    edge_index = torch.tensor([[0, 1], [1, 0]])

    gcn_laplacian = EdgeIndex(edge_index).get_sparse_normalized_gcn_laplacian()

    assert gcn_laplacian.is_sparse


@pytest.mark.parametrize(
    "edge_index, num_nodes",
    [
        pytest.param(torch.tensor([[0, 1], [1, 0]]), 2, id="2_nodes"),
        pytest.param(torch.tensor([[0, 1, 2], [1, 2, 0]]), 4, id="4_nodes"),
        pytest.param(torch.tensor([[0, 1], [1, 0]]), None, id="2_nodes_inferred"),
    ],
)
def test_get_sparse_normalized_laplacian_shape(edge_index, num_nodes):
    gcn_laplacian = EdgeIndex(edge_index).get_sparse_normalized_gcn_laplacian(num_nodes=num_nodes)
    unique_nodes = torch.unique(edge_index)
    expected_num_nodes = num_nodes if num_nodes else len(unique_nodes)

    assert gcn_laplacian.shape == (expected_num_nodes, expected_num_nodes)


def test_get_sparse_normalized_laplacian_is_symmetric():
    """GCN Laplacian L = D^-1/2 * A * D^-1/2 is symmetric."""
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])

    gcn_laplacian = EdgeIndex(edge_index).get_sparse_normalized_gcn_laplacian()
    dense_gcn_laplacian = gcn_laplacian.to_dense()

    assert torch.allclose(dense_gcn_laplacian, dense_gcn_laplacian.T, atol=1e-6)


def test_get_sparse_normalized_laplacian_self_loop_diagonal():
    """Single node graph has diagonal value 1 (self-loop normalized)."""
    edge_index = torch.tensor([[0], [0]])

    gcn_laplacian = EdgeIndex(edge_index).get_sparse_normalized_gcn_laplacian(num_nodes=1)
    dense_gcn_laplacian = gcn_laplacian.to_dense()

    # Self-loop only: degree = 1, so D^-1/2 * A * D^-1/2 = 1 * 1 * 1 = 1
    assert torch.isclose(dense_gcn_laplacian[0, 0], torch.tensor(1.0), atol=1e-6)


@pytest.mark.parametrize(
    "edge_index, num_nodes, expected_row_sum",
    [
        pytest.param(
            torch.tensor([[0, 1], [1, 0]]),
            2,
            1.0,  # Each node has degree 2 (edge + self-loop), diagonal = 1/2 each
            id="connected_graph",
        ),
        pytest.param(
            torch.tensor([[0, 1, 2], [1, 2, 0]]),
            3,
            1.0,  # Triangle: each node degree 3 (2 edges + self-loop), diag = 1/3 each
            id="triangle_graph",
        ),
    ],
)
def test_get_sparse_normalized_laplacian_row_sum(edge_index, num_nodes, expected_row_sum):
    """
    For connected graphs with self-loops, GCN normalization makes the
    laplacian matrix row-stochastic: every row sums to 1.0.
    """
    gcn_laplacian = EdgeIndex(edge_index).get_sparse_normalized_gcn_laplacian(num_nodes=num_nodes)
    dense_gcn_laplacian = gcn_laplacian.to_dense()

    # Each row should sum to 1 for connected graphs with self-loops
    for i in range(num_nodes):
        assert torch.isclose(
            dense_gcn_laplacian[i].sum(), torch.tensor(expected_row_sum), atol=1e-6
        )


def test_get_sparse_normalized_laplacian_preserves_device():
    edge_index = torch.tensor([[0, 1], [1, 0]], device="cpu")

    gcn_laplacian = EdgeIndex(edge_index).get_sparse_normalized_gcn_laplacian()

    assert gcn_laplacian.device == edge_index.device


def test_get_sparse_normalized_laplacian_no_nan_or_inf():
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])

    gcn_laplacian = EdgeIndex(edge_index).get_sparse_normalized_gcn_laplacian(num_nodes=4)
    dense_gcn_laplacian = gcn_laplacian.to_dense()

    assert not torch.any(torch.isnan(dense_gcn_laplacian))
    assert not torch.any(torch.isinf(dense_gcn_laplacian))


def test_get_sparse_normalized_laplacian_has_0_for_isolated_nodes():
    edge_index = torch.tensor([[0], [1]])

    gcn_laplacian = EdgeIndex(edge_index).get_sparse_normalized_gcn_laplacian(num_nodes=4)
    dense_gcn_laplacian = gcn_laplacian.to_dense()

    assert torch.all(dense_gcn_laplacian[2, :] == 0)
    assert torch.all(dense_gcn_laplacian[:, 2] == 0)
    assert torch.all(dense_gcn_laplacian[3, :] == 0)
    assert torch.all(dense_gcn_laplacian[:, 3] == 0)


def test_get_sparse_identity_matrix_is_sparse():
    edge_index = EdgeIndex(torch.tensor([[0], [1]]))
    identity_matrix = edge_index.get_sparse_identity_matrix(num_nodes=2)

    assert identity_matrix.is_sparse


def test_edge_index_remove_selfloops():
    edge_index = EdgeIndex(torch.tensor([[0, 1, 2, 3], [1, 1, 3, 2]]))
    edge_index.remove_selfloops()
    edges = set(zip(edge_index.item[0].tolist(), edge_index.item[1].tolist()))

    assert (1, 1) not in edges
    assert (0, 1) in edges
    assert (2, 3) in edges
    assert (3, 2) in edges
    assert edge_index.num_edges == 3


def test_edge_index_remove_selfloops_when_all_selfloops():
    edge_index = EdgeIndex(torch.tensor([[0, 1], [0, 1]]))
    edge_index.remove_selfloops()

    assert edge_index.num_edges == 0


def test_edge_index_remove_selfloops_when_no_selfloops():
    edge_index = EdgeIndex(torch.tensor([[0, 1], [1, 0]]))
    edge_index.remove_selfloops()

    assert edge_index.num_edges == 2


def test_get_sparse_identity_matrix():
    edge_index = EdgeIndex(torch.tensor([[0, 1], [1, 0]]))
    identity = edge_index.get_sparse_identity_matrix(num_nodes=3)
    dense_identity = identity.to_dense()

    expected = torch.eye(3)
    assert torch.allclose(dense_identity, expected)


def test_get_sparse_identity_matrix_infers_num_nodes():
    edge_index = EdgeIndex(torch.tensor([[0, 1, 2], [1, 2, 0]]))
    identity = edge_index.get_sparse_identity_matrix()
    dense_identity = identity.to_dense()

    expected = torch.eye(3)
    assert torch.allclose(dense_identity, expected)


def test_get_sparse_normalized_laplacian_returns_sparse():
    edge_index = EdgeIndex(torch.tensor([[0, 1], [1, 0]]))
    laplacian = edge_index.get_sparse_normalized_laplacian()

    assert laplacian.is_sparse


def test_get_sparse_normalized_laplacian_shape():
    edge_index = EdgeIndex(torch.tensor([[0, 1, 2], [1, 2, 0]]))
    laplacian = edge_index.get_sparse_normalized_laplacian(num_nodes=3)

    assert laplacian.shape == (3, 3)


def test_get_sparse_normalized_laplacian_is_symmetric():
    edge_index = EdgeIndex(torch.tensor([[0, 1, 2], [1, 2, 0]]))
    laplacian = edge_index.get_sparse_normalized_laplacian()
    dense_laplacian = laplacian.to_dense()

    assert torch.allclose(dense_laplacian, dense_laplacian.T, atol=1e-6)


def test_get_sparse_normalized_laplacian_diagonal_values():
    """For a connected graph without self-loops, diagonal of the laplacian should be non-negative."""
    edge_index = EdgeIndex(torch.tensor([[0, 1], [1, 0]]))
    laplacian = edge_index.get_sparse_normalized_laplacian(num_nodes=2)
    dense_laplacian = laplacian.to_dense()

    # L = I - D^{-1/2} A D^{-1/2}, so diagonal should be non-negative
    for i in range(2):
        assert dense_laplacian[i, i] >= 0


def test_get_sparse_normalized_laplacian_does_not_contain_nan_or_inf():
    edge_index = EdgeIndex(torch.tensor([[0, 1, 2], [1, 2, 0]]))
    laplacian = edge_index.get_sparse_normalized_laplacian(num_nodes=4)
    dense_laplacian = laplacian.to_dense()

    assert not torch.any(torch.isnan(dense_laplacian))
    assert not torch.any(torch.isinf(dense_laplacian))


def test_get_sparse_normalized_laplacian_when_single_edge():
    edge_index = EdgeIndex(torch.tensor([[0, 1], [1, 0]]))
    laplacian = edge_index.get_sparse_normalized_laplacian(num_nodes=2)
    dense_laplacian = laplacian.to_dense()

    # D^{-1/2} = diag(1, 1), A = [[0,1],[1,0]], so D^{-1/2} A D^{-1/2} = A
    # L = I - A = [[1,-1],[-1,1]]
    expected = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
    assert torch.allclose(dense_laplacian, expected, atol=1e-6)


def test_remove_duplicate_edges():
    edge_index = EdgeIndex(torch.tensor([[0, 0, 1], [1, 1, 2]]))
    edge_index.remove_duplicate_edges()

    edges = list(zip(edge_index.item[0].tolist(), edge_index.item[1].tolist()))

    assert len(edges) == 2
    # (0, 1) should appear only once
    assert edges.count((0, 1)) == 1


def test_remove_duplicate_edges_when_no_duplicates():
    edge_index = EdgeIndex(torch.tensor([[0, 1, 2], [1, 2, 3]]))
    original_edges = set(zip(edge_index.item[0].tolist(), edge_index.item[1].tolist()))

    edge_index.remove_duplicate_edges()

    new_edges = set(zip(edge_index.item[0].tolist(), edge_index.item[1].tolist()))
    assert original_edges == new_edges


@pytest.mark.parametrize(
    "edge_index_tensor, expected_num_edges_after_removal",
    [
        pytest.param(torch.tensor([[], []]), 0, id="empty_edge_index"),
        pytest.param(torch.tensor([[0, 1], [1, 2]]), 2, id="no_duplicates"),
        pytest.param(torch.tensor([[0, 1, 0], [1, 2, 1]]), 2, id="one_duplicate"),
        pytest.param(torch.tensor([[0, 0, 0], [1, 1, 1]]), 1, id="three_duplicates"),
        pytest.param(
            torch.tensor([[0, 2, 0, 0], [1, 2, 0, 0]]), 3, id="mixed_duplicates_non_duplicates"
        ),
    ],
)
def test_remove_duplicate_edges(edge_index_tensor, expected_num_edges_after_removal):
    edge_index = EdgeIndex(edge_index_tensor)
    edge_index.remove_duplicate_edges()

    assert edge_index.num_edges == expected_num_edges_after_removal


def test_remove_duplicate_edges_preserves_device():
    device = torch.device("cpu")

    edge_index = EdgeIndex(torch.tensor([[0, 0, 1], [1, 1, 2]], device=device))
    edge_index.remove_duplicate_edges()

    assert edge_index.item.device == device


def test_remove_duplicate_edges_preserves_dtype():
    edge_index = EdgeIndex(torch.tensor([[0, 0, 1], [1, 1, 2]], dtype=torch.long))
    edge_index.remove_duplicate_edges()

    assert edge_index.item.dtype == torch.long


def test_to_undirected_edge_index_single_directed_edge():
    """A single directed edge (0 -> 1) should produce bidirectional edges."""
    edge_index = EdgeIndex(torch.tensor([[0], [1]]))
    edge_index.to_undirected()
    edges = set(zip(edge_index.item[0].tolist(), edge_index.item[1].tolist()))

    # Should contain both (0, 1) and (1, 0)
    assert (0, 1) in edges
    assert (1, 0) in edges
    assert len(edges) == 2


def test_to_undirected_edge_index_already_undirected_does_not_create_duplicates():
    edge_index = EdgeIndex(torch.tensor([[0, 1], [1, 0]]))
    edge_index.to_undirected()
    edges = set(zip(edge_index.item[0].tolist(), edge_index.item[1].tolist()))

    # (0, 1) and (1, 0) should still be present, but no duplicates
    expected_edges = {(0, 1), (1, 0)}
    assert edges == expected_edges


def test_to_undirected_edge_index_removes_duplicate_edges():
    edge_index = EdgeIndex(torch.tensor([[0, 0, 1], [1, 1, 0]]))
    edge_index.to_undirected()
    edges = set(zip(edge_index.item[0].tolist(), edge_index.item[1].tolist()))

    expected_edges = {(0, 1), (1, 0)}  # Duplicates should be removed
    assert edges == expected_edges


def test_to_undirected_edge_index_triangle_directed():
    """
    A directed triangle should become a bidirectional triangle.

    Example:
        Directed cycle: 0 -> 1 -> 2 -> 0
        Bidirectional traingle: 0 <-> 1 <-> 2 <-> 0
    """
    edge_index = EdgeIndex(torch.tensor([[0, 1, 2], [1, 2, 0]]))
    edge_index.to_undirected()
    edges = set(zip(edge_index.item[0].tolist(), edge_index.item[1].tolist()))

    bidirectional_triangle = {(0, 1), (1, 0), (1, 2), (2, 1), (2, 0), (0, 2)}
    assert edges == bidirectional_triangle


def test_to_undirected_edge_index_empty_edge_index_returns_empty_tensor():
    edge_index = EdgeIndex(torch.tensor([[], []]))
    edge_index.to_undirected()

    assert edge_index.item.shape == (2, 0)


def test_to_undirected_edge_index_with_selfloops_adds_all_selfloops():
    edge_index = EdgeIndex(torch.tensor([[0, 1], [1, 2]]))
    edge_index.to_undirected(with_selfloops=True)
    edges = set(zip(edge_index.item[0].tolist(), edge_index.item[1].tolist()))

    # Should have self-loops for nodes 0, 1, 2 (inferred from max index)
    assert (0, 0) in edges
    assert (1, 1) in edges
    assert (2, 2) in edges


def test_to_undirected_edge_index_preserves_selfloops_in_input():
    # (1, 1) is a self-loop in the input, so it should still be present
    edge_index = EdgeIndex(torch.tensor([[0, 1, 1], [1, 0, 1]]))
    edge_index.to_undirected()
    edges = set(zip(edge_index.item[0].tolist(), edge_index.item[1].tolist()))

    assert (1, 1) in edges


def test_to_undirected_edge_index_with_selfloops_does_not_duplicate_selfloops():
    # (1, 1) is already a self-loop
    edge_index = EdgeIndex(torch.tensor([[0, 1], [1, 1]]))
    edge_index.to_undirected(with_selfloops=True)
    edges = list(zip(edge_index.item[0].tolist(), edge_index.item[1].tolist()))

    assert (0, 0) in edges
    assert (1, 1) in edges


@pytest.mark.parametrize(
    "with_selfloops",
    [
        pytest.param(True, id="with_selfloops"),
        pytest.param(False, id="without_selfloops"),
    ],
)
def test_to_undirected_edge_index_preserves_device(with_selfloops):
    edge_index = EdgeIndex(torch.tensor([[0], [1]], device="cpu"))
    edge_index.to_undirected(with_selfloops=with_selfloops)

    assert edge_index.item.device == edge_index.item.device


def test_to_undirected_edge_index_disconnected_components():
    # Two disconnected components: (0, 1) and (2, 3)
    edge_index = EdgeIndex(torch.tensor([[0, 2], [1, 3]]))
    edge_index.to_undirected()
    edges = set(zip(edge_index.item[0].tolist(), edge_index.item[1].tolist()))

    expected_edges = {(0, 1), (1, 0), (2, 3), (3, 2)}
    assert edges == expected_edges


@pytest.mark.parametrize(
    "edge_index, expected_num_undirected_edges",
    [
        pytest.param(
            torch.tensor([[0], [1]]),
            2,
            id="single_edge_becomes_two",
        ),
        pytest.param(
            torch.tensor([[0, 1], [1, 0]]),
            2,
            id="bidirectional_stays_two",
        ),
        pytest.param(
            torch.tensor([[0, 1, 2], [1, 2, 0]]),
            6,
            id="directed_triangle_becomes_six",
        ),
        pytest.param(
            torch.tensor([[0, 0], [1, 2]]),
            4,
            id="star_two_edges_becomes_four",
        ),
    ],
)
def test_to_undirected_edge_index_edge_count(edge_index, expected_num_undirected_edges):
    edge_index = EdgeIndex(edge_index)
    edge_index.to_undirected()

    assert edge_index.item.shape[1] == expected_num_undirected_edges


def test_to_undirected_edge_index_dtype_preserved():
    dtype = torch.long

    edge_index = EdgeIndex(torch.tensor([[0, 1], [1, 2]], dtype=dtype))
    edge_index.to_undirected()

    assert edge_index.item.dtype == dtype
