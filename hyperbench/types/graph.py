import torch

from torch import Tensor
from typing import List, Optional
from hyperbench import utils


class Graph:
    """
    A simple graph data structure using edge list representation.

    Args:
        edges: A list of edges, where each edge is represented as a list of two integers (source_node, destination_node).
    """

    def __init__(self, edges: List[List[int]]):
        self.edges = edges

    @property
    def num_nodes(self) -> int:
        """Return the number of nodes in the graph."""
        nodes = set()
        for edge in self.edges:
            nodes.update(edge)
        return len(nodes)

    @property
    def num_edges(self) -> int:
        """Return the number of edges in the graph."""
        return len(self.edges)

    def remove_selfloops(self) -> "Graph":
        """
        Remove self-loops from the graph.

        Returns:
            List of edges without self-loops.
        """
        if self.num_edges == 0:
            return self

        edges_tensor = torch.tensor(self.edges, dtype=torch.long)

        # Example: edges = [[0, 1],
        #                   [1, 1],
        #                   [2, 3]] shape (|E|, 2)
        #          -> no_selfloop_mask = [True, False, True]
        #          -> edges without self-loops = [[0, 1],
        #                                         [2, 3]]
        no_selfloop_mask = edges_tensor[:, 0] != edges_tensor[:, 1]
        self.edges = edges_tensor[no_selfloop_mask].tolist()
        return self

    def to_edge_index(self) -> Tensor:
        """
        Convert the graph to edge index representation.

        Returns:
            edge_index: Tensor of shape (2, |E|) representing edges.
        """
        if self.num_edges == 0:
            return torch.empty((2, 0), dtype=torch.long)

        # Example: edges = [[0, 1],
        #                   [1, 2],
        #                   [2, 3]] shape (|E|, 2)
        #          ->  edge_index = [[0, 1, 2],
        #                            [1, 2, 3]] shape (2, |E|)
        edge_index = torch.tensor(self.edges, dtype=torch.long).t()
        return edge_index

    @staticmethod
    def smoothing_with_laplacian_matrix(
        x: Tensor,
        laplacian_matrix: Tensor,
        drop_rate: float = 0.0,
    ) -> Tensor:
        r"""
        Return the feature matrix smoothed with a Laplacian matrix.

        Args:
            x: Node feature matrix. Size ``(|V|, C)``.
            laplacian_matrix: The Laplacian matrix. Size ``(|V|, |V|)``.
            drop_rate: Randomly dropout the connections in the Laplacian with probability ``drop_rate``. Defaults to ``0.0``.

        Returns:
            The smoothed feature matrix. Size ``(|V|, C)``.
        """
        if drop_rate > 0.0:
            laplacian_matrix = utils.sparse_dropout(laplacian_matrix, drop_rate)
        return laplacian_matrix.matmul(x)


class EdgeIndex:
    """
    A wrapper for edge index representation of a graph.
    Edge index is a tensor of shape ``(2, num_edges)`` where the first row contains source node indices and the second row contains destination node indices for each edge.

    Examples:
        >>> edge_index = [[0, 1, 2],
        ...               [1, 0, 3]]

        This represents a graph with edges (0, 1), (1, 0), and (2, 3).
        The number of nodes in this graph is 4 (nodes 0, 1, 2, and 3) and the number of edges is 3.

    Args:
        edge_index: A tensor of shape ``(2, num_edges)`` representing the edges in the graph.
    """

    def __init__(self, edge_index: Tensor):
        self.__edge_index = edge_index

    @property
    def item(self) -> Tensor:
        """Return the edge index tensor."""
        return self.__edge_index

    @property
    def max_node_id(self) -> int:
        """Return the maximum node ID in the edge index."""
        if self.__edge_index.size(1) < 1:
            return -1
        return int(self.__edge_index.max())

    @property
    def num_edges(self) -> int:
        """Return the number of edges in the graph."""
        if self.__edge_index.size(1) < 1:
            return 0
        # Number of edges is the number of columns in edge_index, which is dim=1,
        # as each column represents an edge (source, destination)
        return self.__edge_index.size(1)

    @property
    def num_nodes(self) -> int:
        """Return the number of nodes in the graph."""
        if self.__edge_index.size(1) < 1:
            return 0
        unique_nodes = torch.unique(self.__edge_index)
        return len(unique_nodes)

    def add_selfloops(
        self,
        num_nodes: Optional[int] = None,
        with_duplicate_removal: bool = True,
    ) -> "EdgeIndex":
        """
        Add self-loops to each node in the edge index.

        Examples:
            >>> edge_index = [[0, 1, 2],
            ...               [1, 0, 3]]
            >>> edge_index_with_selfloops = [[0, 1, 2, 0, 1, 2, 3],
            ...                              [1, 0, 3, 0, 1, 2, 3]]

            When ``num_nodes`` is higher than the number of nodes in ``edge_index``,
            self-loops are added for all nodes from ``0`` to ``num_nodes - 1``,
            including nodes not present in the original edges:

            >>> edge_index = [[0, 1, 2],
            ...               [1, 0, 3]]
            >>> num_nodes = 6
            >>> edge_index_with_selfloops = [[0, 1, 2, 0, 1, 2, 3, 4, 5],
            ...                              [1, 0, 3, 0, 1, 2, 3, 4, 5]]

        Args:
            num_nodes: Total number of nodes. When provided, self-loops are added for nodes ``0`` to ``num_nodes - 1``. When ``None``, defaults to ``self.num_nodes``.
                This parameter is important when ``edge_index`` does not contain all nodes (e.g., some nodes are isolated and have no edges or have been removed),
                as it ensures that the resulting Laplacian matrix has the correct size and includes all nodes. For instance, for self-loops.
            with_duplicate_removal: Whether to remove duplicate edges after adding self-loops. Defaults to ``True``.

        Returns:
            This :class:`EdgeIndex` instance with self-loops added.

        Raises:
            ValueError: If the input edge index has no edges (i.e., ``shape (2, 0)``).
        """
        if self.__edge_index.size(1) < 1:
            raise ValueError("Edge index must have at least one edge to add self-loops.")

        device = self.__edge_index.device
        src, dest = self.__edge_index[0], self.__edge_index[1]

        # Add self-loops: A_hat = A + I (works as we assume node indices are in [0, num_nodes-1])
        # Example: edge_index = [[0, 1, 2],
        #                        [1, 0, 3]], num_nodes = None
        #          -> num_selfloop_nodes = 4 (self.num_nodes, as num_nodes is None)
        #          -> selfloop_indices = [0, 1, 2, 3]
        #          -> src = [0, 1, 2, 0, 1, 2, 3]
        #          -> dest = [1, 0, 3, 0, 1, 2, 3]
        #          -> edge_index_with_selfloops = [[0, 1, 2, 0, 1, 2, 3],
        #                                          [1, 0, 3, 0, 1, 2, 3]]
        #
        # Example with num_nodes=6 (higher than 4 nodes in edge_index):
        #          -> num_selfloop_nodes = 6
        #          -> selfloop_indices = [0, 1, 2, 3, 4, 5]
        #          -> src = [0, 1, 2, 0, 1, 2, 3, 4, 5]
        #          -> dest = [1, 0, 3, 0, 1, 2, 3, 4, 5]
        #          -> edge_index_with_selfloops = [[0, 1, 2, 0, 1, 2, 3, 4, 5],
        #                                          [1, 0, 3, 0, 1, 2, 3, 4, 5]]
        num_selfloop_nodes = self.num_nodes if num_nodes is None else num_nodes
        selfloop_indices = torch.arange(num_selfloop_nodes, device=device)
        src = torch.cat([src, selfloop_indices])
        dest = torch.cat([dest, selfloop_indices])
        edge_index_with_selfloops = torch.stack([src, dest], dim=0)

        self.__edge_index = edge_index_with_selfloops
        if with_duplicate_removal:
            self.remove_duplicate_edges()

        return self

    def get_sparse_adjacency_matrix(self, num_nodes: Optional[int] = None) -> Tensor:
        """
        Compute the sparse adjacency matrix from a graph edge index.
        To get the normalized adjacency matrix, add self-loops to the edge_index.

        Examples:
            >>> edge_index = [[0, 1, 2],
            ...               [1, 0, 3]]
            >>> num_nodes = 4
            >>> adj_values = [1, 1, 1]
            >>> adj_indices = [[0, 1, 2],
            ...                [1, 0, 3]]
            >>>                0  1  2  3
            ... adj_matrix = [[0, 1, 0, 0], 0
            ...               [1, 0, 0, 0], 1
            ...               [0, 0, 0, 1], 2
            ...               [0, 0, 1, 0]] 3

        Args:
            num_nodes: The number of nodes in the graph.
                If ``None``, it will be inferred from ``self.num_nodes``.
                Note that the node indices in ``edge_index`` are assumed to be in the range [0, num_nodes-1].

        Returns:
            The sparse adjacency matrix of shape ``(num_nodes, num_nodes)``.
        """
        src, dest = self.__edge_index
        num_nodes = self.num_nodes if num_nodes is None else num_nodes

        # Example: edge_index = [[0, 1, 2, 3],
        #                       [1, 0, 3, 2]]
        #         -> adj_values = [1, 1, 1, 1]
        #         -> adj_indices = [[0, 1, 2, 3],
        #                           [1, 0, 3, 2]]
        #                  0  1  2  3
        #         -> A = [[0, 1, 0, 0], 1
        #                 [1, 0, 0, 0], 0
        #                 [0, 0, 0, 1], 3
        #                 [0, 0, 1, 0]] 2
        adj_values = torch.ones(src.size(0), device=self.__edge_index.device)
        adj_indices = torch.stack([src, dest], dim=0)
        adj_matrix = torch.sparse_coo_tensor(adj_indices, adj_values, size=(num_nodes, num_nodes))
        return adj_matrix.coalesce()

    def get_sparse_identity_matrix(self, num_nodes: Optional[int] = None) -> Tensor:
        """
        Compute the sparse identity matrix I of shape (num_nodes, num_nodes).

        Examples:
            >>> num_nodes = 3
            >>> identity_indices = [[0, 1, 2],
            ...                     [0, 1, 2]]
            >>> values = [1, 1, 1]
            >>> I = [[1, 0, 0],
            ...      [0, 1, 0],
            ...      [0, 0, 1]]

        Args:
            num_nodes: The number of nodes in the graph.
                If ``None``, it will be inferred from ``self.num_nodes``.

        Returns:
            The sparse identity matrix I of shape ``(num_nodes, num_nodes)``.
        """
        device = self.__edge_index.device
        num_nodes = self.num_nodes if num_nodes is None else num_nodes

        # Example: num_nodes = 3
        #          -> identity_indices = [[0, 1, 2],
        #                                 [0, 1, 2]]
        #             we use repeat(2, 1) as I is a matrix NxN, so we need indices for both rows and columns
        #          -> values = [1, 1, 1]
        #                   0  1  2
        #          -> I = [[1, 0, 0], 0
        #                  [0, 1, 0], 1
        #                  [0, 0, 1]] 2
        identity_indices = torch.arange(num_nodes, device=device).unsqueeze(0).repeat(2, 1)
        identity_matrix = torch.sparse_coo_tensor(
            indices=identity_indices,
            values=torch.ones(num_nodes, device=device),
            size=(num_nodes, num_nodes),
        )
        return identity_matrix.coalesce()

    def get_sparse_normalized_degree_matrix(
        self,
        num_nodes: Optional[int] = None,
    ) -> Tensor:
        """
        Compute the sparse normalized degree matrix D^-1/2 from a graph edge index.

        Args:
            num_nodes: The number of nodes in the graph.
                If ``None``, it will be inferred from ``self.num_nodes``.
                Note that the node indices in ``edge_index`` are assumed to be in the range [0, num_nodes-1].

        Returns:
            The sparse normalized degree matrix D^-1/2 of shape ``(num_nodes, num_nodes)``.
        """
        device = self.__edge_index.device
        src, _ = self.__edge_index

        num_nodes = self.num_nodes if num_nodes is None else num_nodes

        # Compute degree for each node, initially degree matrix D has all zeros
        degrees: Tensor = torch.zeros(num_nodes, device=device)

        # Example: src = [0, 1, 2, 1], degrees = [0, 0, 0, 0]
        #          -> degrees[0] += 1 = degrees = [1,0,0,0]
        #          -> degrees[1] += 1 = degrees = [1,1,0,0]
        #          -> degrees[2] += 1 = degrees = [1,1,1,0]
        #          -> degrees[1] += 1 = degrees = [1,2,1,0]
        #          -> final degrees = [1,2,1,0]
        degree_initial_values = torch.ones(
            src.size(0), device=device
        )  # Each edge contributes 1 to the degree of the source node
        degrees.scatter_add_(dim=0, index=src, src=degree_initial_values)

        # Compute D^-1/2 == D^-0.5
        degree_inv_sqrt: Tensor = degrees.pow(-0.5)
        # Handle isolated nodes where degree is 0, which lead to inf values in degree_inv_sqrt
        degree_inv_sqrt[degree_inv_sqrt == float("inf")] = 0

        # Convert degree vector to a diagonal sparse normalized matrix D
        # Example: degree_inv_sqrt = [1, 0.707, 1, 0]
        #          -> diagonal_indices = [[0, 1, 2, 3],
        #                                 [0, 1, 2, 3]]
        #                   0  1      2  3
        #          -> D = [[1, 0,     0, 0], 0
        #                  [0, 0.707, 0, 0], 1
        #                  [0, 0,     1, 0], 2
        #                  [0, 0,     0, 0]] 3
        diagonal_indices = torch.arange(num_nodes, device=device).unsqueeze(0).repeat(2, 1)
        degree_matrix = torch.sparse_coo_tensor(
            indices=diagonal_indices,
            values=degree_inv_sqrt,
            size=(num_nodes, num_nodes),
        )
        return degree_matrix.coalesce()

    def get_sparse_normalized_laplacian(
        self,
        num_nodes: Optional[int] = None,
    ) -> Tensor:
        """
        Compute the sparse symmetric normalized Laplacian matrix: L = I - D^{-1/2} A D^{-1/2}.

        Unlike ``get_sparse_normalized_gcn_laplacian``, this method does not add self-loops
        and computes the standard Laplacian (not the GCN propagation matrix).

        Args:
            num_nodes: The number of nodes in the graph. If ``None``,
                it will be inferred from ``self.num_nodes``.

        Returns:
            The sparse symmetric normalized Laplacian matrix of shape ``(num_nodes, num_nodes)``.
        """
        self.to_undirected(with_selfloops=False)

        num_nodes = self.num_nodes if num_nodes is None else num_nodes

        degree_matrix = self.get_sparse_normalized_degree_matrix(num_nodes)
        adj_matrix = self.get_sparse_adjacency_matrix(num_nodes)

        # D^{-1/2} A D^{-1/2}
        normalized_adj_matrix = torch.sparse.mm(
            degree_matrix,
            torch.sparse.mm(adj_matrix, degree_matrix),
        )

        # L = I - D^{-1/2} A D^{-1/2}
        identity_matrix = self.get_sparse_identity_matrix(num_nodes)
        return (identity_matrix - normalized_adj_matrix).coalesce()

    def get_sparse_normalized_gcn_laplacian(
        self,
        num_nodes: Optional[int] = None,
    ) -> Tensor:
        """
        Compute the sparse Laplacian matrix from a graph edge index.

        The GCN Laplacian is defined as: L_GCN = D_hat^-1/2 * A_hat * D_hat^-1/2,
        where A_hat = A + I (adjacency with self-loops) and D_hat is the degree matrix of A_hat.

        Args:
            num_nodes: The number of nodes in the graph. If ``None``,
                it will be inferred from ``self.num_nodes``.
                Note that the node indices in ``edge_index`` are assumed to be in the range [0, num_nodes-1].
                This parameter is important when ``edge_index`` does not contain all nodes (e.g., some nodes are isolated and have no edges or have been removed),
                as it ensures that the resulting Laplacian matrix has the correct size and includes all nodes. For instance, for self-loops.

        Returns:
            The sparse symmetrically normalized Laplacian matrix of shape ``(num_nodes, num_nodes)``.
        """
        self.to_undirected(with_selfloops=True, num_nodes=num_nodes)

        num_nodes = self.num_nodes if num_nodes is None else num_nodes

        degree_matrix = self.get_sparse_normalized_degree_matrix(num_nodes)
        adj_matrix = self.get_sparse_adjacency_matrix(num_nodes)

        # Compute normalized Laplacian matrix: L = D^-1/2 * A * D^-1/2
        normalized_laplacian_matrix = torch.sparse.mm(
            degree_matrix,
            torch.sparse.mm(adj_matrix, degree_matrix),
        )
        return normalized_laplacian_matrix.coalesce()

    def remove_selfloops(self) -> "EdgeIndex":
        """Remove self-loops from the edge index."""
        # Example: edge_index = [[0, 1, 2, 3],
        #                        [1, 1, 3, 2]], shape (2, |E| = 4)
        #          -> keep_mask = [True, False, True, True]
        #          -> edge_index = [[0, 2, 3],
        #                           [1, 3, 2]], shape (2, |E'| = 3)
        keep_mask = self.__edge_index[0] != self.__edge_index[1]
        self.__edge_index = self.__edge_index[:, keep_mask]
        return self

    def remove_duplicate_edges(self) -> "EdgeIndex":
        """Remove duplicate edges from the edge index. Keeps the tensor contiguous in memory."""
        # Example: edge_index = [[0, 1, 2, 2, 0, 3, 2],
        #                        [1, 0, 3, 2, 1, 2, 2]], shape (2, |E| = 7)
        #          -> after torch.unique(..., dim=1):
        #             edge_index = [[0, 1, 2, 2, 3],
        #                           [1, 0, 3, 2, 2]], shape (2, |E'| = 5)
        # Note: we call contiguous() to ensure that the resulting tensor is contiguous in memory,
        # which can improve performance for subsequent operations that require contiguous tensors
        self.__edge_index = torch.unique(self.__edge_index, dim=1).contiguous()
        return self

    def to_undirected(
        self,
        with_selfloops: bool = False,
        num_nodes: Optional[int] = None,
    ) -> "EdgeIndex":
        """
        Convert the edge index to an undirected edge index by adding reverse edges.

        Args:
            with_selfloops: Whether to add self-loops to each node. Defaults to ``False``.
            num_nodes: Total number of nodes. Propagated to ``add_selfloops`` when ``with_selfloops`` is ``True``.
                This parameter is useful when ``edge_index`` does not contain all nodes (e.g., some nodes are isolated and have no edges or have been removed),
                as it ensures that the resulting Laplacian matrix has the correct size and includes all nodes. For instance, for self-loops.

        Returns:
            This :class:`EdgeIndex` instance converted to undirected.
        """
        device = self.__edge_index.device

        src, dest = self.__edge_index[0], self.__edge_index[1]
        src, dest = torch.cat([src, dest]), torch.cat([dest, src])

        # Example: edge_index = [[0, 1, 2],
        #                        [1, 0, 3]]
        #          -> after torch.stack([...], dim=0):
        #             undirected_edge_index = [[0, 1, 2, 1, 0, 3],
        #                                      [1, 0, 3, 0, 1, 2]]
        #          -> after torch.unique(..., dim=1):
        #             undirected_edge_index = [[0, 1, 2, 3],
        #                                      [1, 0, 3, 2]]
        undirected_edge_index: Tensor = torch.stack([src, dest], dim=0).to(device)
        self.__edge_index = undirected_edge_index

        if with_selfloops:
            # Don't remove duplicate edges when adding self-loops, as we need to remove them
            # even if with_selfloops is False, to ensure that the edge index is clean and doesn't contain duplicate edges.
            # In this way, we don't do the duplicate edge removal twice, which would be redundant and inefficient
            self.add_selfloops(num_nodes=num_nodes, with_duplicate_removal=False)

        self.remove_duplicate_edges()

        return self
