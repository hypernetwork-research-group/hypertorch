import torch

from torch import Tensor
from typing import Optional, List, Dict, Any, Literal, Set, Tuple, TypeAlias
from hyperbench.utils import sparse_dropout, to_0based_ids

from hyperbench.types.graph import EdgeIndex, Graph


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
        hyperedges: A list of edge dictionaries, where each dictionary contains information about a hyperedge (e.g., id, features).
    """

    def __init__(
        self,
        network_type: Optional[Literal["asc", "directed", "undirected"]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        incidences: Optional[List[Dict[str, Any]]] = None,
        nodes: Optional[List[Dict[str, Any]]] = None,
        hyperedges: Optional[List[Dict[str, Any]]] = None,
    ):
        self.network_type = network_type
        self.metadata = metadata if metadata is not None else {}
        self.incidences = incidences if incidences is not None else []
        self.nodes = nodes if nodes is not None else []
        self.hyperedges = hyperedges if hyperedges is not None else []

    @classmethod
    def empty(cls) -> "HIFHypergraph":
        return cls(
            network_type="undirected",
            nodes=[],
            hyperedges=[],
            incidences=[],
            metadata=None,
        )

    @classmethod
    def from_hif(cls, data: Dict[str, Any]) -> "HIFHypergraph":
        """
        Create a Hypergraph from a HIF (Hypergraph Interchange Format).

        Args:
            data: Dictionary with keys: network-type, metadata, incidences, nodes, hyperedges

        Returns:
            Hypergraph instance
        """
        network_type = data.get("network-type") or data.get("network_type")
        metadata = data.get("metadata", {})
        incidences = data.get("incidences", [])
        nodes = data.get("nodes", [])
        hyperedges = data.get("edges", [])

        return cls(
            network_type=network_type,
            metadata=metadata,
            incidences=incidences,
            nodes=nodes,
            hyperedges=hyperedges,
        )

    @property
    def num_nodes(self) -> int:
        """Return the number of nodes in the hypergraph."""
        return len(self.nodes)

    @property
    def num_hyperedges(self) -> int:
        """Return the number of hyperedges in the hypergraph."""
        return len(self.hyperedges)

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
        num_hyperedges = len(self.hyperedges)
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

    @staticmethod
    def smoothing_with_matrix(
        x: Tensor,
        matrix: Tensor,
        drop_rate: float = 0.0,
    ) -> Tensor:
        """
        Return the feature matrix smoothed with a smoothing matrix.
        Computes ``M @ X`` where ``M`` is the smoothing matrix and ``X`` is the node feature matrix.

        Args:
            x: Node feature matrix. Size ``(num_nodes, C)``.
            matrix: The smoothing matrix. Size ``(num_nodes, num_nodes)``.
            drop_rate: Randomly dropout the connections in the smoothing matrix with probability ``drop_rate``. Defaults to ``0.0``.

        Returns:
            The smoothed feature matrix. Size ``(num_nodes, C)``.
        """
        if drop_rate > 0.0:
            matrix = sparse_dropout(matrix, drop_rate)
        return matrix.matmul(x)


class HyperedgeIndex:
    """
    A wrapper for hyperedge index representation.
    Hyperedge index is a tensor of shape ``(2, num_incidences)`` that encodes the relationships between nodes and hyperedges.
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
        hyperedge_index: A tensor of shape ``(2, num_incidences)`` representing hyperedges, where each column is (node, hyperedge).
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

    def get_sparse_incidence_matrix(
        self,
        num_nodes: Optional[int] = None,
        num_hyperedges: Optional[int] = None,
    ) -> Tensor:
        """
        Compute the sparse incidence matrix H of shape ``(num_nodes, num_hyperedges)``.
        Each entry ``H[v, e] = 1`` if node ``v`` belongs to hyperedge ``e``, and 0 otherwise.

        Args:
            num_nodes: Total number of nodes. If ``None``, inferred from hyperedge index.
            num_hyperedges: Total number of hyperedges. If ``None``, inferred from hyperedge index.

        Returns:
            The sparse incidence matrix H of shape ``(num_nodes, num_hyperedges)``.
        """
        device = self.__hyperedge_index.device
        num_nodes = num_nodes if num_nodes is not None else self.num_nodes
        num_hyperedges = num_hyperedges if num_hyperedges is not None else self.num_hyperedges

        incidence_values = torch.ones(self.num_incidences, dtype=torch.float, device=device)
        incidence_indices = torch.stack([self.all_node_ids, self.all_hyperedge_ids], dim=0)
        incidence_matrix = torch.sparse_coo_tensor(
            indices=incidence_indices,
            values=incidence_values,
            size=(num_nodes, num_hyperedges),
        )
        return incidence_matrix.coalesce()

    def get_sparse_normalized_node_degree_matrix(
        self,
        incidence_matrix: Tensor,
        power: float,
        num_nodes: Optional[int] = None,
    ) -> Tensor:
        """
        Compute a sparse diagonal node degree matrix from row-sums of the incidence matrix.

        Args:
            incidence_matrix: The sparse incidence matrix H of shape ``(num_nodes, num_hyperedges)``.
            power: Exponent applied to node degrees before placing them on the diagonal.
            num_nodes: Total number of nodes. If ``None``, inferred from hyperedge index.

        Returns:
            The sparse diagonal matrix of shape ``(num_nodes, num_nodes)``.
        """
        device = self.__hyperedge_index.device
        num_nodes = num_nodes if num_nodes is not None else self.num_nodes

        degrees = torch.sparse.sum(incidence_matrix, dim=1).to_dense()
        normalized_degrees = degrees.pow(power)
        normalized_degrees[normalized_degrees == float("inf")] = 0

        diagonal_indices = torch.arange(num_nodes, device=device).unsqueeze(0).repeat(2, 1)
        degree_matrix = torch.sparse_coo_tensor(
            indices=diagonal_indices,
            values=normalized_degrees,
            size=(num_nodes, num_nodes),
        )
        return degree_matrix.coalesce()

    def get_sparse_rownormalized_node_degree_matrix(
        self,
        incidence_matrix: Tensor,
        num_nodes: Optional[int] = None,
    ) -> Tensor:
        """
        Compute the sparse normalized node degree matrix D_n^-1.
        The node degree ``d_n[i]`` is the number of hyperedges containing node ``i``
        (i.e., the row-sum of the incidence matrix H).

        Args:
            incidence_matrix: The sparse incidence matrix H of shape ``(num_nodes, num_hyperedges)``.
            num_nodes: Total number of nodes. If ``None``, inferred from hyperedge index.

        Returns:
            The sparse diagonal matrix D_n^-1 of shape ``(num_nodes, num_nodes)``.
        """
        # Example: hyperedge_index = [[0, 1, 2, 0],
        #                             [0, 0, 0, 1]]
        #                         hyperedges 0  1
        #          -> incidence_matrix H = [[1, 1], node 0
        #                                   [1, 0], node 1
        #                                   [1, 0]] node 2
        #                                          nodes 0  1  2
        #          -> row-sum gives node degrees: d_n = [2, 1, 1]
        #          -> D_n^{-1} has diagonal [1/2, 1, 1]
        return self.get_sparse_normalized_node_degree_matrix(
            incidence_matrix=incidence_matrix,
            power=-1,
            num_nodes=num_nodes,
        )

    def get_sparse_symnormalized_node_degree_matrix(
        self,
        incidence_matrix: Tensor,
        num_nodes: Optional[int] = None,
    ) -> Tensor:
        """
        Compute the sparse normalized node degree matrix D_n^-1/2.
        The node degree ``d_n[i]`` is the number of hyperedges containing node ``i``
        (i.e., the row-sum of the incidence matrix H).

        Args:
            incidence_matrix: The sparse incidence matrix H of shape ``(num_nodes, num_hyperedges)``.
            num_nodes: Total number of nodes. If ``None``, inferred from hyperedge index.

        Returns:
            The sparse diagonal matrix D_n^-1/2 of shape ``(num_nodes, num_nodes)``.
        """
        # Example: hyperedge_index = [[0, 1, 2, 0],
        #                             [0, 0, 0, 1]]
        #                         hyperedges 0  1
        #          -> incidence_matrix H = [[1, 1], node 0
        #                                   [1, 0], node 1
        #                                   [1, 0]] node 2
        #                                          nodes 0  1  2
        #          -> row-sum gives node degrees: d_n = [2, 1, 1]
        #          -> D_n^{-1/2} has diagonal [1/sqrt(2), 1, 1]
        return self.get_sparse_normalized_node_degree_matrix(
            incidence_matrix=incidence_matrix,
            power=-0.5,
            num_nodes=num_nodes,
        )

    def get_sparse_normalized_hyperedge_degree_matrix(
        self,
        incidence_matrix: Tensor,
        num_hyperedges: Optional[int] = None,
    ) -> Tensor:
        """
        Compute the sparse normalized hyperedge degree matrix D_e^-1.

        The hyperedge degree ``d_e[j]`` is the number of nodes in hyperedge ``j``
        (i.e., the column-sum of the incidence matrix H).

        Args:
            incidence_matrix: The sparse incidence matrix H of shape ``(num_nodes, num_hyperedges)``.
            num_hyperedges: Total number of hyperedges. If ``None``, inferred from hyperedge index.

        Returns:
            The sparse diagonal matrix D_e^-1 of shape ``(num_hyperedges, num_hyperedges)``.
        """
        device = self.__hyperedge_index.device
        num_hyperedges = num_hyperedges if num_hyperedges is not None else self.num_hyperedges

        # Example: hyperedge_index = [[0, 1, 2, 0],
        #                             [0, 0, 0, 1]]
        #                         hyperedges 0  1
        #          -> incidence_matrix H = [[1, 1], node 0
        #                                   [1, 0], node 1
        #                                   [1, 0]] node 2
        #          -> column-sum gives hyperedge degrees: d_e = [3, 1], shape (num_hyperedges,)
        degrees = torch.sparse.sum(incidence_matrix, dim=0).to_dense()

        # Example: d_e = [3, 1]
        #          -> degree_inv = [1/3, 1]
        degree_inv = degrees.pow(-1)
        degree_inv[degree_inv == float("inf")] = 0

        # Construct the sparse diagonal matrix D_e^{-1}
        # Example: degree_inv = [1/3, 1] as the diagonal values,
        #          diagonal_indices = [[0, 0],
        #                              [1, 1]]
        #               hyperedges 0  1
        #          -> D_e^{-1} = [[1/3, 0], hyperedge 0
        #                         [0, 1]]   hyperedge 1
        diagonal_indices = torch.arange(num_hyperedges, device=device).unsqueeze(0).repeat(2, 1)
        degree_matrix = torch.sparse_coo_tensor(
            indices=diagonal_indices,
            values=degree_inv,
            size=(num_hyperedges, num_hyperedges),
        )
        return degree_matrix.coalesce()

    def get_sparse_hgnn_smoothing_matrix(
        self,
        num_nodes: Optional[int] = None,
        num_hyperedges: Optional[int] = None,
    ) -> Tensor:
        """
        Compute the sparse HGNN Laplacian matrix for hypergraph spectral convolution.

        Implements: ``L_HGNN = D_n^{-1/2} H D_e^{-1} H^T D_n^{-1/2}``

        where:
            - H is the incidence matrix of shape ``(num_nodes, num_hyperedges)``
            - D_n^-1/2 is the normalized node degree matrix
            - D_e^-1 is the inverse hyperedge degree matrix (with W = I)

        Args:
            num_nodes: Total number of nodes. If ``None``, inferred from hyperedge index.
            num_hyperedges: Total number of hyperedges. If ``None``, inferred from hyperedge index.

        Returns:
            The sparse HGNN Laplacian matrix of shape ``(num_nodes, num_nodes)``.
        """
        num_nodes = num_nodes if num_nodes is not None else self.num_nodes
        num_hyperedges = num_hyperedges if num_hyperedges is not None else self.num_hyperedges

        incidence_matrix = self.get_sparse_incidence_matrix(num_nodes, num_hyperedges)
        node_degree_matrix = self.get_sparse_symnormalized_node_degree_matrix(
            incidence_matrix,
            num_nodes,
        )
        hyperedge_degree_matrix = self.get_sparse_normalized_hyperedge_degree_matrix(
            incidence_matrix, num_hyperedges
        )

        normalized_laplacian_matrix = torch.sparse.mm(
            node_degree_matrix,
            torch.sparse.mm(
                incidence_matrix,
                torch.sparse.mm(
                    hyperedge_degree_matrix,
                    torch.sparse.mm(incidence_matrix.t(), node_degree_matrix),
                ),
            ),
        )
        return normalized_laplacian_matrix.coalesce()

    def get_sparse_hgnnp_smoothing_matrix(
        self,
        num_nodes: Optional[int] = None,
        num_hyperedges: Optional[int] = None,
    ) -> Tensor:
        """
        Compute the sparse HGNN+ smoothing matrix for hypergraph mean aggregation.

        Implements: ``M_HGNN+ = D_v^{-1} H D_e^{-1} H^T``

        This matrix is row-stochastic for non-isolated nodes and corresponds to
        the two-stage mean aggregation used by HGNN+:
            1. ``D_e^{-1} H^T X``: mean over nodes in each hyperedge.
            2. ``D_v^{-1} H (...)``: mean over hyperedges incident to each node.

        Args:
            num_nodes: Total number of nodes. If ``None``, inferred from hyperedge index.
            num_hyperedges: Total number of hyperedges. If ``None``, inferred from hyperedge index.

        Returns:
            The sparse HGNN+ smoothing matrix of shape ``(num_nodes, num_nodes)``.
        """
        num_nodes = num_nodes if num_nodes is not None else self.num_nodes
        num_hyperedges = num_hyperedges if num_hyperedges is not None else self.num_hyperedges

        incidence_matrix = self.get_sparse_incidence_matrix(num_nodes, num_hyperedges)
        node_degree_matrix = self.get_sparse_rownormalized_node_degree_matrix(
            incidence_matrix,
            num_nodes,
        )
        hyperedge_degree_matrix = self.get_sparse_normalized_hyperedge_degree_matrix(
            incidence_matrix,
            num_hyperedges,
        )

        smoothing_matrix = torch.sparse.mm(
            node_degree_matrix,
            torch.sparse.mm(
                incidence_matrix,
                torch.sparse.mm(hyperedge_degree_matrix, incidence_matrix.t()),
            ),
        )
        return smoothing_matrix.coalesce()

    def reduce(self, strategy: Literal["clique_expansion"], **kwargs) -> Tensor:
        """
        Reduce the hypergraph to a graph represented by edge index using the specified strategy.

        Args:
            strategy: The reduction strategy to use. Defaults to ``clique_expansion``.
            **kwargs: Additional keyword arguments for specific strategies.

        Returns:
            The edge index of the reduced graph. Size ``(2, num_edges)``.
        """
        match strategy:
            case _:
                return self.reduce_to_edge_index_on_clique_expansion()

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
        incidence_matrix = self.get_sparse_incidence_matrix()

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
        return_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Construct a graph from a hypergraph with methods proposed in `HyperGCN: A New Method of Training Graph Convolutional Networks on Hypergraphs <https://arxiv.org/pdf/1809.02589.pdf>`_ paper.
        Reference implementation: `source <https://deephypergraph.readthedocs.io/en/latest/_modules/dhg/structure/graphs/graph.html#Graph.from_hypergraph_hypergcn>`_.

        Args:
            x: Node feature matrix. Size ``(num_nodes, C)``.
            with_mediators: Whether to use mediator to transform the hyperedges to edges in the graph. Defaults to ``False``.
            remove_selfloops: Whether to remove self-loops. Defaults to ``True``.
            return_weights: Whether to return the DHG-style reduced-edge weights alongside the edge index. Defaults to ``False``.

        Returns:
            A tuple ``(edge_index, edge_weights)`` where:
            - ``edge_index`` has size ``(2, |num_edges|)``.
            - ``edge_weights`` has size ``(|num_edges|,)`` when ``return_weights=True``, otherwise ``None``.

        Raises:
            ValueError: If any hyperedge contains fewer than 2 nodes.
        """
        device = x.device

        hypergraph = Hypergraph.from_hyperedge_index(self.__hyperedge_index)
        hypergraph_edges: List[List[int]] = hypergraph.hyperedges
        graph_edges: List[List[int]] = []
        graph_edge_weights: List[float] = []

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
                graph_edge_weights.append(1.0 / num_nodes_in_edge)
                continue

            edge_weight = 1.0 / (2 * num_nodes_in_edge - 3)
            for node_idx in range(num_nodes_in_edge):
                if node_idx != node_max_proj_idx and node_idx != node_min_proj_idx:
                    graph_edges.append([edge[node_min_proj_idx], edge[node_idx]])
                    graph_edges.append([edge[node_max_proj_idx], edge[node_idx]])
                    graph_edge_weights.extend([edge_weight, edge_weight])

        graph = Graph(
            edges=graph_edges,
            edge_weights=graph_edge_weights if return_weights else None,
        )
        if remove_selfloops:
            graph.remove_selfloops()

        return (
            graph.to_edge_index().to(device),
            graph.edge_weights_tensor.to(device) if return_weights else None,
        )

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
