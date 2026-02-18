import torch

from torch import Tensor
from typing import Optional, List, Dict, Any, Literal, Set, TypeAlias
from hyperbench.utils import to_0based_ids

from .graph import Graph


Neighborhood: TypeAlias = Set[int]


class HIFHypergraph:
    """
    A hypergraph data structure that supports directed/undirected hyperedges
    with incidence-based representation.
    """

    def __init__(
        self,
        network_type: Optional[Literal["asc", "directed", "undirected"]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        incidences: Optional[List[Dict[str, Any]]] = None,
        nodes: Optional[List[Dict[str, Any]]] = None,
        edges: Optional[List[Dict[str, Any]]] = None,
    ):
        self.network_type = network_type
        self.metadata = metadata if metadata is not None else {}
        self.incidences = incidences if incidences is not None else []
        self.nodes = nodes if nodes is not None else []
        self.edges = edges if edges is not None else []

    @classmethod
    def empty(cls) -> "HIFHypergraph":
        return cls(
            network_type="undirected",
            nodes=[],
            edges=[],
            incidences=[],
            metadata=None,
        )

    @classmethod
    def from_hif(cls, data: Dict[str, Any]) -> "HIFHypergraph":
        """
        Create a Hypergraph from a HIF (Hypergraph Interchange Format).

        Args:
            data: Dictionary with keys: network-type, metadata, incidences, nodes, edges

        Returns:
            Hypergraph instance
        """
        network_type = data.get("network-type") or data.get("network_type")
        metadata = data.get("metadata", {})
        incidences = data.get("incidences", [])
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])

        return cls(
            network_type=network_type,
            metadata=metadata,
            incidences=incidences,
            nodes=nodes,
            edges=edges,
        )

    @property
    def num_nodes(self) -> int:
        """Return the number of nodes in the hypergraph."""
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        """Return the number of edges in the hypergraph."""
        return len(self.edges)


class Hypergraph:
    """
    A simple hypergraph data structure using edge list representation.
    """

    def __init__(self, hyperedges: List[List[int]]):
        self.hyperedges = hyperedges

    @property
    def num_nodes(self) -> int:
        """Return the number of nodes in the hypergraph."""
        nodes = set()
        for edge in self.hyperedges:
            nodes.update(edge)
        return len(nodes)

    @property
    def num_hyperedges(self) -> int:
        """Return the number of hyperedges in the hypergraph."""
        return len(self.hyperedges)

    def neighbors_of(self, node: int) -> Neighborhood:
        """
        Return the set of nodes that share at least one hyperedge with node.

        A node u is a neighbor of v if there exists a hyperedge e such that
        both u and v are in e. The node itself is excluded from the result.

        Args:
            node: The node ID to find neighbors for.

        Returns:
            A set of neighbor node IDs (excluding the node itself).
        """
        neighbors: Neighborhood = set()
        for hyperedge in self.hyperedges:
            if node in hyperedge:
                neighbors.update(hyperedge)

        neighbors.discard(node)
        return neighbors

    def neighbors_of_all(self) -> Dict[int, Neighborhood]:
        """
        Build a mapping from every node to its neighbors.

        This precomputes ``neighbors_of`` for all nodes at once, which is
        more efficient when scoring many candidate hyperedges.

        Returns:
            A dictionary mapping each node ID to its set of neighbors.
        """
        nodes: Set[int] = set()
        for hyperedge in self.hyperedges:
            nodes.update(hyperedge)

        node_to_neighbors: Dict[int, Neighborhood] = {}
        for node in nodes:
            node_to_neighbors[node] = self.neighbors_of(node)

        return node_to_neighbors

    @classmethod
    def from_hyperedge_index(cls, hyperedge_index: Tensor) -> "Hypergraph":
        """
        Create a Hypergraph from a hyperedge index representation.

        Args:
            hyperedge_index: Tensor of shape (2, |E|) representing hyperedges, where each column is (node, hyperedge).

        Returns:
            Hypergraph instance
        """
        if hyperedge_index.size(1) < 1:
            return cls(hyperedges=[])

        unique_hyperedge_ids = hyperedge_index[1].unique()
        hyperedges = [
            hyperedge_index[0, hyperedge_index[1] == hyperedge_id].tolist()
            for hyperedge_id in unique_hyperedge_ids
        ]

        return cls(hyperedges=hyperedges)


class HyperedgeIndex:
    """
    A wrapper for hyperedge index representation.
    Hyperedge index is a tensor of shape (2, |E|) that encodes the relationships between nodes and hyperedges.
    Each column in the tensor represents an incidence between a node and a hyperedge, with the first row containing node indices
    and the second row containing corresponding hyperedge indices.

    Examples:
        >>> hyperedge_index = [[0, 1, 2, 0],
        ...                    [0, 0, 0, 1]]

        This represents two hyperedges:
            - Hyperedge 0 connects nodes 0, 1, and 2.
            - Hyperedge 1 connects node 0.

        The number of nodes in this hypergraph is 3 (nodes 0, 1, and 2).
        The number of hyperedges is 2 (hyperedges 0 and 1).
    """

    def __init__(self, hyperedge_index: Tensor):
        self.__hyperedge_index = hyperedge_index

    @property
    def item(self) -> Tensor:
        """Return the hyperedge index tensor."""
        return self.__hyperedge_index

    @property
    def num_hyperedges(self) -> int:
        """Return the number of hyperedges in the hypergraph."""
        if self.__hyperedge_index.size(1) < 1:
            return 0

        hyperedges = self.__hyperedge_index[1]
        return len(hyperedges.unique())

    @property
    def num_nodes(self) -> int:
        """Return the number of nodes in the hypergraph."""
        if self.__hyperedge_index.size(1) < 1:
            return 0

        nodes = self.__hyperedge_index[0]
        return len(nodes.unique())

    def nodes_in(self, hyperedge_id: int) -> List[int]:
        """Return the list of node IDs that belong to the given hyperedge."""
        return self.__hyperedge_index[0, self.__hyperedge_index[1] == hyperedge_id].tolist()

    def reduce_to_edge_index_on_random_direction(
        self,
        x: Tensor,
        with_mediators: bool = False,
        remove_selfloops: bool = True,
    ) -> Tensor:
        """
        Construct a graph from a hypergraph with methods proposed in `HyperGCN: A New Method of Training Graph Convolutional Networks on Hypergraphs <https://arxiv.org/pdf/1809.02589.pdf>`_ paper.
        Reference implementation: `source <https://deephypergraph.readthedocs.io/en/latest/_modules/dhg/structure/graphs/graph.html#Graph.from_hypergraph_hypergcn>`_.

        Args:
            x: Node feature matrix. Size ``(|V|, C)``.
            with_mediators: Whether to use mediator to transform the hyperedges to edges in the graph. Defaults to ``False``.
            remove_selfloops: Whether to remove self-loops. Defaults to ``True``.

        Returns:
            The edge index. Size ``(2, |E'|)``.

        Raises:
            ValueError: If any hyperedge contains fewer than 2 nodes.
        """
        device = x.device

        hypergraph = Hypergraph.from_hyperedge_index(self.__hyperedge_index)
        hypergraph_edges: List[List[int]] = hypergraph.hyperedges
        graph_edges: List[List[int]] = []

        # Random direction (feature_dim, 1) for projecting nodes in each hyperedge
        # Geometrically, we are choosing a random line through the origin in ℝᵈ, where ᵈ = feature_dim
        random_direction = torch.rand((x.shape[1], 1), device=device)

        for edge in hypergraph_edges:
            num_nodes_in_edge = len(edge)
            if num_nodes_in_edge < 2:
                raise ValueError("The number of vertices in an hyperedge must be >= 2.")

            # projections (num_nodes_in_edge,) contains a scalar value for each node in the hyperedge,
            # indicating its projection on the random vector 'random_direction'.
            # Key idea: If two points are very far apart in ℝᵈ, there is a high probability
            # that a random projection will still separate them
            projections = torch.matmul(x[edge], random_direction).squeeze()

            # The indices of the nodes that the farthest apart in the direction of 'random_direction'
            node_max_proj_idx = torch.argmax(projections)
            node_min_proj_idx = torch.argmin(projections)

            if not with_mediators:  # Just connect the two farthest nodes
                graph_edges.append([edge[node_min_proj_idx], edge[node_max_proj_idx]])
                continue

            for node_idx in range(num_nodes_in_edge):
                if node_idx != node_max_proj_idx and node_idx != node_min_proj_idx:
                    graph_edges.append([edge[node_min_proj_idx], edge[node_idx]])
                    graph_edges.append([edge[node_max_proj_idx], edge[node_idx]])

        graph = Graph(edges=graph_edges)
        if remove_selfloops:
            graph.remove_selfloops()

        return graph.to_edge_index()

    def to_0based(
        self,
        node_ids_to_rebase: Tensor,
        hyperedge_ids_to_rebase: Tensor,
    ) -> "HyperedgeIndex":
        """
        Convert hyperedge index to the 0-based format by rebasing node IDs to the range ``[0, num_nodes-1]`` and hyperedge IDs ``[0, num_hyperedges-1]``.

        Args:
            node_ids_to_rebase: Tensor of shape ``(num_nodes,)`` containing the original node IDs that need to be rebased to 0-based format.
            hyperedge_ids_to_rebase: Tensor of shape ``(num_hyperedges,)`` containing the original hyperedge IDs that need to be rebased to 0-based format.

        Returns:
            A new :class:`HyperedgeIndex` instance with the hyperedge index converted to 0-based format.
        """
        node_ids = self.__hyperedge_index[0]
        hyperedge_ids = self.__hyperedge_index[1]

        # Example: negative_hyperedge_index after sorting: [[0, 0, 1, 2, 3, 4],
        #                                                    [3, 4, 4, 3, 4, 3]]
        #          -> negative_hyperedge_index after remapping: [[0, 0, 1, 2, 3, 4],
        #                                                        [3, 4, 4, 3, 4, 3]]
        self.__hyperedge_index[0] = to_0based_ids(node_ids, node_ids_to_rebase)

        # Example: negative_hyperedge_index after remapping nodes: [[0, 0, 1, 2, 3, 4],
        #                                                           [3, 4, 4, 3, 4, 3]]
        #          negative_hyperedge_ids = [3, 4]
        #          -> negative_hyperedge_index after remapping hyperedges: [[0, 0, 1, 2, 3, 4],
        #                                                                   [0, 0, 1, 0, 1, 0]]
        self.__hyperedge_index[1] = to_0based_ids(hyperedge_ids, hyperedge_ids_to_rebase)

        return self
