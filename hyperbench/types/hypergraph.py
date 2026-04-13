import torch

from torch import Tensor
from typing import Optional, List, Dict, Any, Literal, Set, TypeAlias

from hyperbench.utils import to_0based_ids

from .graph import EdgeIndex, Graph


Neighborhood: TypeAlias = Set[int]


class HIFHypergraph:
    """
    A hypergraph data structure that supports directed/undirected hyperedges
    with incidence-based representation.

    Args:
        network_type: The type of hypergraph, which can be "asc" (or "directed") for directed hyperedges, or "undirected" for undirected hyperedges.
        metadata: Optional dictionary of metadata about the hypergraph.
        incidences: A list of incidences, where each incidence is a dictionary with keys "node" and "edge" representing the relationship between a node and a hyperedge.
        nodes: A list of node dictionaries, where each dictionary contains information about a node (e.g., id, features).
        edges: A list of edge dictionaries, where each dictionary contains information about a hyperedge (e.g., id, features).
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

    def stats(self) -> Dict[str, Any]:
        """
        Compute statistics for the HIFhypergraph.
        The fields returned in the dictionary include:
        - ``num_nodes``: The number of nodes in the hypergraph.
        - ``num_hyperedges``: The number of hyperedges in the hypergraph.
        - ``avg_degree_node_raw``: The average degree of nodes, calculated as the mean number of hyperedges each node belongs to.
        - ``avg_degree_node``: The floored node average degree.
        - ``avg_degree_hyperedge_raw``: The average size of hyperedges, calculated as the mean number of nodes each hyperedge contains.
        - ``avg_degree_hyperedge``: The floored hyperedge average size.
        - ``node_degree_max``: The maximum degree of any node in the hypergraph.
        - ``hyperedge_degree_max``: The maximum size of any hyperedge in the hypergraph.
        - ``node_degree_median``: The median degree of nodes in the hypergraph.
        - ``hyperedge_degree_median``: The median size of hyperedges in the hypergraph.
        - ``distribution_node_degree``: A list where the value at index ``i`` represents the count of nodes with degree ``i``.
        - ``distribution_hyperedge_size``: A list where the value at index ``i`` represents the count of hyperedges with size ``i``.
        - ``distribution_node_degree_hist``: A dictionary where the keys are node degrees and the values are the count of nodes with that degree.
        - ``distribution_hyperedge_size_hist``: A dictionary where the keys are hyperedge sizes and the values are the count of hyperedges with that size.

        Returns:
            A dictionary containing various statistics about the hypergraph.
        """

        node_degree: Dict[Any, int] = {}
        hyperedge_size: Dict[Any, int] = {}

        for incidence in self.incidences:
            node_id = incidence.get("node")
            edge_id = incidence.get("edge")
            node_degree[node_id] = node_degree.get(node_id, 0) + 1
            hyperedge_size[edge_id] = hyperedge_size.get(edge_id, 0) + 1

        num_nodes = len(self.nodes)
        num_hyperedges = len(self.edges)
        total_incidences = len(self.incidences)

        distribution_node_degree: List[int] = sorted(node_degree.values())
        distribution_hyperedge_size: List[int] = sorted(hyperedge_size.values())

        avg_degree_node_raw = total_incidences / num_nodes if num_nodes else 0
        avg_degree_node = int(avg_degree_node_raw)
        avg_degree_hyperedge_raw = total_incidences / num_hyperedges if num_hyperedges else 0
        avg_degree_hyperedge = int(avg_degree_hyperedge_raw)

        node_degree_max = max(distribution_node_degree) if distribution_node_degree else 0
        hyperedge_degree_max = (
            max(distribution_hyperedge_size) if distribution_hyperedge_size else 0
        )

        n_n = len(distribution_node_degree)
        node_degree_median = (
            (
                distribution_node_degree[n_n // 2]
                if n_n % 2
                else (distribution_node_degree[n_n // 2 - 1] + distribution_node_degree[n_n // 2])
                / 2
            )
            if n_n
            else 0
        )

        n_e = len(distribution_hyperedge_size)
        hyperedge_degree_median = (
            (
                distribution_hyperedge_size[n_e // 2]
                if n_e % 2
                else (
                    distribution_hyperedge_size[n_e // 2 - 1]
                    + distribution_hyperedge_size[n_e // 2]
                )
                / 2
            )
            if n_e
            else 0
        )

        distribution_node_degree_hist: Dict[int, int] = {}
        for d in distribution_node_degree:
            distribution_node_degree_hist[d] = distribution_node_degree_hist.get(d, 0) + 1

        distribution_hyperedge_size_hist: Dict[int, int] = {}
        for s in distribution_hyperedge_size:
            distribution_hyperedge_size_hist[s] = distribution_hyperedge_size_hist.get(s, 0) + 1

        return {
            "num_nodes": num_nodes,
            "num_hyperedges": num_hyperedges,
            "avg_degree_node_raw": avg_degree_node_raw,
            "avg_degree_node": avg_degree_node,
            "avg_degree_hyperedge_raw": avg_degree_hyperedge_raw,
            "avg_degree_hyperedge": avg_degree_hyperedge,
            "node_degree_max": node_degree_max,
            "hyperedge_degree_max": hyperedge_degree_max,
            "node_degree_median": node_degree_median,
            "hyperedge_degree_median": hyperedge_degree_median,
            "distribution_node_degree": distribution_node_degree,
            "distribution_hyperedge_size": distribution_hyperedge_size,
            "distribution_node_degree_hist": distribution_node_degree_hist,
            "distribution_hyperedge_size_hist": distribution_hyperedge_size_hist,
        }


class Hypergraph:
    """
    A simple hypergraph data structure using edge list representation.

    Args:
        hyperedges: A list of hyperedges, where each hyperedge is represented as a list of node IDs.
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

    def stats(self) -> Dict[str, Any]:
        """Return basic statistics about the hypergraph."""
        node_degree: Dict[int, int] = {}
        distribution_hyperedge_size: List[int] = []
        total_incidences = 0

        for hyperedge in self.hyperedges:
            size = len(hyperedge)
            distribution_hyperedge_size.append(size)
            total_incidences += size
            for node in hyperedge:
                node_degree[node] = node_degree.get(node, 0) + 1

        num_nodes = len(node_degree)
        num_hyperedges = len(self.hyperedges)
        distribution_node_degree: List[int] = sorted(node_degree.values())

        avg_degree_hyperedge = total_incidences / num_hyperedges if num_hyperedges else 0
        total_incidences_nodes = sum(distribution_node_degree)
        avg_degree_node = total_incidences_nodes / num_nodes if num_nodes else 0

        hyperedge_degree_max = (
            max(distribution_hyperedge_size) if distribution_hyperedge_size else 0
        )
        node_degree_max = max(distribution_node_degree) if distribution_node_degree else 0

        sorted_hyperedge_sizes = sorted(distribution_hyperedge_size)
        n_e = len(sorted_hyperedge_sizes)
        hyperedge_degree_median = (
            (
                sorted_hyperedge_sizes[n_e // 2]
                if n_e % 2
                else (sorted_hyperedge_sizes[n_e // 2 - 1] + sorted_hyperedge_sizes[n_e // 2]) / 2
            )
            if n_e
            else 0
        )

        n_n = len(distribution_node_degree)
        node_degree_median = (
            (
                distribution_node_degree[n_n // 2]
                if n_n % 2
                else (distribution_node_degree[n_n // 2 - 1] + distribution_node_degree[n_n // 2])
                / 2
            )
            if n_n
            else 0
        )

        distribution_hyperedge_size_hist: Dict[int, int] = {}
        for s in distribution_hyperedge_size:
            distribution_hyperedge_size_hist[s] = distribution_hyperedge_size_hist.get(s, 0) + 1

        distribution_node_degree_hist: Dict[int, int] = {}
        for d in distribution_node_degree:
            distribution_node_degree_hist[d] = distribution_node_degree_hist.get(d, 0) + 1

        return {
            "num_nodes": num_nodes,
            "num_hyperedges": num_hyperedges,
            "avg_degree_node": avg_degree_node,
            "avg_degree_hyperedge": avg_degree_hyperedge,
            "node_degree_max": node_degree_max,
            "hyperedge_degree_max": hyperedge_degree_max,
            "node_degree_median": node_degree_median,
            "hyperedge_degree_median": hyperedge_degree_median,
            "distribution_node_degree": distribution_node_degree,
            "distribution_hyperedge_size": distribution_hyperedge_size,
            "distribution_node_degree_hist": distribution_node_degree_hist,
            "distribution_hyperedge_size_hist": distribution_hyperedge_size_hist,
        }

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

    Args:
        hyperedge_index: A tensor of shape ``(2, |E|)`` representing hyperedges, where each column is (node, hyperedge).
    """

    def __init__(self, hyperedge_index: Tensor):
        self.__hyperedge_index = hyperedge_index

    @property
    def all_node_ids(self) -> Tensor:
        """Return the tensor of all node IDs in the hyperedge index."""
        return self.__hyperedge_index[0]

    @property
    def all_hyperedge_ids(self) -> Tensor:
        """Return the tensor of all hyperedge IDs in the hyperedge index."""
        return self.__hyperedge_index[1]

    @property
    def item(self) -> Tensor:
        """Return the hyperedge index tensor."""
        return self.__hyperedge_index

    @property
    def node_ids(self) -> Tensor:
        """Return the sorted unique node IDs from the hyperedge index."""
        return self.__hyperedge_index[0].unique(sorted=True)

    @property
    def hyperedge_ids(self) -> Tensor:
        """Return the sorted unique hyperedge IDs from the hyperedge index."""
        return self.__hyperedge_index[1].unique(sorted=True)

    @property
    def num_hyperedges(self) -> int:
        """Return the number of hyperedges in the hypergraph."""
        if self.num_incidences < 1:
            return 0

        hyperedges = self.__hyperedge_index[1]
        return len(hyperedges.unique())

    @property
    def num_nodes(self) -> int:
        """Return the number of nodes in the hypergraph."""
        if self.num_incidences < 1:
            return 0

        nodes = self.__hyperedge_index[0]
        return len(nodes.unique())

    @property
    def num_incidences(self) -> int:
        """Return the number of incidences in the hypergraph, which is the number of columns in the hyperedge index."""
        return self.__hyperedge_index.size(1)

    def nodes_in(self, hyperedge_id: int) -> List[int]:
        """Return the list of node IDs that belong to the given hyperedge."""
        return self.__hyperedge_index[0, self.__hyperedge_index[1] == hyperedge_id].tolist()

    def num_nodes_if_isolated_exist(self, num_nodes: int) -> int:
        """
        Return the number of nodes in the hypergraph, accounting for isolated nodes that may not appear in the hyperedge index.

        Args:
            num_nodes: The total number of nodes in the hypergraph, including isolated nodes.

        Returns:
            The number of nodes in the hypergraph, which is the maximum of the number of unique nodes in the hyperedge index and the provided ``num_nodes``.
        """
        return max(self.num_nodes, num_nodes)

    def reduce_to_edge_index_on_clique_expansion(self) -> Tensor:
        """
        Construct a graph from a hypergraph via clique expansion using ``H @ H^T``, where ``H`` is the incidence matrix of the hypergraph.
        In clique expansion, each hyperedge is replaced by a clique connecting all its member nodes.

        For each hyperedge, all pairs of member nodes become edges in the resulting graph.
        This is computed efficiently using the incidence matrix: ``A = H @ H^T``, where ``H`` is
        the sparse incidence matrix of shape ``[num_nodes, num_hyperedges]`` and ``A`` is the adjacency matrix of the clique-expanded graph.

        Returns:
            The edge index of the clique-expanded graph. Size ``(2, |E'|)``.
        """
        # Build sparse incidence matrix of shape [num_nodes, num_hyperedges]
        values = torch.ones(
            size=(self.num_incidences,),
            dtype=torch.float,
            device=self.__hyperedge_index.device,
        )
        incidence_matrix = torch.sparse_coo_tensor(
            indices=torch.stack([self.all_node_ids, self.all_hyperedge_ids], dim=0),
            values=values,
            size=(self.num_nodes, self.num_hyperedges),
        ).coalesce()

        # A = H @ H^T gives adjacency with self-loops on diagonal
        # Example: For hyperedge_index = [[0, 1, 2, 0],
        #                                 [0, 0, 0, 1]]
        #                         hyperedges 0  1
        #          -> incidence_matrix H = [[1, 1], node 0
        #                                   [1, 0], node 1
        #                                   [1, 0]] node 2
        #               nodes 0  1  2
        #          -> H^T = [[1, 1, 1], hyperedge 0
        #                    [1, 0, 0]] hyperedge 1
        #                       nodes 0  1  2
        #          -> A = H @ H^T = [[2, 1, 1], node 0
        #                            [1, 1, 1], node 1
        #                            [1, 1, 1]] node 2
        #                                         nodes 0  1  2
        #          -> A (after removing self-loops) = [[0, 1, 1], node 0
        #                                              [1, 0, 1], node 1
        #                                              [1, 1, 0]] node 2
        adj_matrix = torch.sparse.mm(incidence_matrix, incidence_matrix.t()).coalesce()

        # Extract edge_index, make undirected, and deduplicate
        return EdgeIndex(adj_matrix.indices()).to_undirected().item

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

        return graph.to_edge_index().to(device)

    def remove_duplicate_edges(self) -> "HyperedgeIndex":
        """Remove duplicate edges from the hyperedge index. Keeps the tensor contiguous in memory."""
        # Example: hyperedge_index = [[0, 1, 2, 2, 0, 3, 2],
        #                             [3, 4, 4, 3, 4, 3, 3]], shape (2, 7)
        #          -> after torch.unique(..., dim=1):
        #             hyperedge_index = [[0, 1, 2, 2, 0, 3],
        #                                [3, 4, 4, 3, 4, 3]], shape (2, |E'| = 6)
        # Note: we need to call contiguous() after torch.unique() to ensure
        # the resulting tensor is contiguous in memory, which is important for efficient indexing
        # and further operations (e.g., searchsorted)
        self.__hyperedge_index = torch.unique(self.__hyperedge_index, dim=1).contiguous()
        return self

    def remove_hyperedges_with_fewer_than_k_nodes(self, k: int) -> "HyperedgeIndex":
        """
        Remove hyperedges that contain fewer than k nodes.

        Example:
            >>> hyperedge_index = [[0, 1, 2, 3, 5, 4],
            ...                    [0, 0, 1, 1, 2, 1]], shape (2, |E| = 6)

            >>> k = 3
            >>> unique_hyperedge_ids: [0, 1, 2]
            ... # inverse -> idx_to_hyperedge_id, counts -> num_nodes_per_hyperedge
            ... inverse           = [0, 0, 1, 1, 2, 1]  # (index into unique_hyperedge_ids per column)
            ... counts            = [2, 3, 1]
            >>> # counts[inverse] is equivalent to:
            ... # for i, inv in enumerate(inverse): keep_mask[i] = counts[inv]
            >>> counts[inverse]   = [2, 2, 3, 3, 1, 3]
            >>> keep_mask         = [F, F, T, T, F, T]

            >>> # after filtering hyperedges with fewer than k=3 nodes:
            >>> hyperedge_index = [[2, 3, 4],
            ...                    [1, 1, 1]], shape (2, |E'| = 3)

        Args:
            k: The minimum number of nodes a hyperedge must contain to be kept.

        Returns:
            A new :class:`HyperedgeIndex` instance with hyperedges containing fewer than k nodes.
        """
        _, idx_to_hyperedge_id, num_nodes_per_hyperedge = torch.unique(
            self.all_hyperedge_ids,
            return_inverse=True,
            return_counts=True,
        )
        keep_mask = num_nodes_per_hyperedge[idx_to_hyperedge_id] >= k
        self.__hyperedge_index = self.__hyperedge_index[:, keep_mask]
        return self

    def to_0based(
        self,
        node_ids_to_rebase: Optional[Tensor] = None,
        hyperedge_ids_to_rebase: Optional[Tensor] = None,
    ) -> "HyperedgeIndex":
        """
        Convert hyperedge index to the 0-based format by rebasing node IDs to the range ``[0, num_nodes-1]`` and hyperedge IDs ``[0, num_hyperedges-1]``.

        Args:
            node_ids_to_rebase: Tensor of shape ``(num_nodes,)`` containing the original node IDs that need to be rebased to 0-based format.
                If ``None``, all node IDs in the hyperedge index will be rebased to 0-based format based on their unique sorted order.
            hyperedge_ids_to_rebase: Tensor of shape ``(num_hyperedges,)`` containing the original hyperedge IDs that need to be rebased to 0-based format.
                If ``None``, all hyperedge IDs in the hyperedge index will be rebased to 0-based format based on their unique sorted order.

        Returns:
            A new :class:`HyperedgeIndex` instance with the hyperedge index converted to 0-based format.
        """
        # Example: hyperedge_index after sorting: [[0, 0, 1, 2, 3, 4],
        #                                          [3, 4, 4, 3, 4, 3]]
        #          node_ids_to_rebase = [0, 1, 2, 3, 4]
        #          -> hyperedge_index after remapping: [[0, 0, 1, 2, 3, 4],
        #                                               [3, 4, 4, 3, 4, 3]]
        self.__hyperedge_index[0] = to_0based_ids(self.all_node_ids, node_ids_to_rebase)

        # Example: hyperedge_index after remapping nodes: [[0, 0, 1, 2, 3, 4],
        #                                                  [3, 4, 4, 3, 4, 3]]
        #          hyperedge_ids_to_rebase = [3, 4]
        #          -> hyperedge_index after remapping hyperedges: [[0, 0, 1, 2, 3, 4],
        #                                                          [0, 0, 1, 0, 1, 0]]
        self.__hyperedge_index[1] = to_0based_ids(self.all_hyperedge_ids, hyperedge_ids_to_rebase)

        return self
