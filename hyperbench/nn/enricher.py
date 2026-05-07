import warnings
import random
import torch

from abc import ABC
from torch import Tensor, optim
from typing import Literal, Optional, TypeAlias
from hyperbench.types import EdgeIndex, HyperedgeIndex
from torch_geometric.nn import Node2Vec as PyGNode2Vec

EnrichmentMode: TypeAlias = Literal["concatenate", "replace"]


class Enricher(ABC):
    """
    Generic base class for enrichers.

    Args:
        cache_dir: Directory for saving/loading cached features. If ``None``, caching is disabled.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
    ):
        self.cache_dir = cache_dir

    def enrich(self, hyperedge_index: Tensor) -> Tensor:
        raise NotImplementedError("Subclasses must implement the enrich method.")


class NodeEnricher(Enricher, ABC):
    """
    Base class for node enrichers.
    """

    pass


class HyperedgeEnricher(Enricher, ABC):
    """
    Base class for hyperedge enrichers.
    """

    pass


class HyperedgeAttrsEnricher(HyperedgeEnricher):
    """
    Base class for enrichers that generate hyperedge attributes (features).

    Args:
        cache_dir: Directory for saving/loading cached features. If ``None``, caching is disabled.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
    ):
        super().__init__(cache_dir=cache_dir)

    def enrich(self, hyperedge_index: Tensor) -> Tensor:
        """
        Generate hyperedge attributes. By default, this method creates a simple feature of 1.0 for each hyperedge.

        Args:
            hyperedge_index: Hyperedge index tensor of shape ``(2, num_hyperedges)``.

        Returns:
            Tensor of shape ``(num_hyperedges, 1)`` containing the generated attribute for each hyperedge.
        """
        # Add a feature of 1.0 for each hyperedge, which can be used
        # as a baseline or for methods that require hyperedge features.
        num_hyperedges = HyperedgeIndex(hyperedge_index).num_hyperedges
        hyperedge_attrs = torch.ones(size=(num_hyperedges, 1), device=hyperedge_index.device)
        return hyperedge_attrs


class HyperedgeWeightsEnricher(HyperedgeEnricher):
    """
    Generates hyperedge weights based on the number of nodes in each hyperedge.

    Args:
        cache_dir: Directory for saving/loading cached features. If ``None``, caching is disabled.
        alpha: Scaling factor for the random component added to weights. Must be between 0.0 and 1.0.
        beta: If provided, the random component is alpha * beta. If None, no random component is added.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        alpha: float = 1.0,
        beta: Optional[float] = None,
    ):
        super().__init__(cache_dir=cache_dir)
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError("Alpha must be between 0.0 and 1.0.")

        self.alpha = alpha
        self.beta = beta

    def enrich(self, hyperedge_index: Tensor) -> Tensor:
        """
        Compute edge weights as the number of nodes in each hyperedge.

        Args:
            hyperedge_index: Hyperedge index tensor of shape ``(2, num_hyperedges)``.
            alpha: Scaling factor for the random component added to weights.
            beta: If provided, the random component is alpha * beta.

        Returns:
            Tensor of shape ``(num_hyperedges,)`` containing the weight of each hyperedge.
        """
        # Count the number of nodes in each hyperedge by counting occurrences of each hyperedge index.
        # Example: if hyperedge_index[1] = [0, 0, 1, 1, 1], then we have 2 nodes in hyperedge 0 and 3 nodes in hyperedge 1.
        num_hyperedges = int(hyperedge_index[1].max().item()) + 1
        weights = torch.bincount(hyperedge_index[1], minlength=num_hyperedges).float()

        random_alpha = random.uniform(0, self.alpha)
        if self.beta is not None:
            weights += random_alpha * self.beta
        return weights


class Node2VecEnricher(NodeEnricher):
    def __init__(
        self,
        num_features: int,
        walk_length: int = 20,
        context_size: int = 10,
        num_walks_per_node: int = 10,
        p: float = 1.0,
        q: float = 1.0,
        num_negative_samples: int = 1,
        num_nodes: int = 0,
        graph_reduction_strategy: Literal["clique_expansion"] = "clique_expansion",
        num_epochs: int = 5,
        learning_rate: float = 0.01,
        batch_size: int = 128,
        sparse: bool = True,
        cache_dir: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Enrich node features using Node2Vec embeddings computed from the clique expansion of the hypergraph.

        Args:
            num_features: Dimensionality of the node embeddings to generate.
            walk_length: Length of each random walk.
            context_size: Window size for the skip-gram model (number of neighbors in the walk considered as context).
                For example, if ``context_size=2`` and ``walk_length=5``, then for a random walk ``[v0, v1, v2, v3, v4]``,
                the context for ``v2`` would be ``[v0, v1, v3, v4]`` as we take neighbors within distance 2 in the walk.
                The pairs generated by skip-gram would be ``[(v2, v0), (v2, v1), (v2, v3), (v2, v4)]``.
                Rule of thumb: Graphs with strong local structure (5-10), Graphs with communities/long-range patterns (10-20).
                Defaults to ``10``.
            num_walks_per_node: Number of random walks to start at each node.
            p: Return hyperparameter for Node2Vec. Default is ``1.0`` (unbiased).
                This controls the probability of stepping back to the node visited in the previous step.
                Lower values of ``p`` make immediate backtracking more likely, which keeps walks closer to the
                local neighborhood. Higher values of ``p`` discourage returning to the previous node, so walks
                are less likely to bounce back and forth across the same edge.
            q: In-out hyperparameter for Node2Vec. Default is ``1.0`` (unbiased).
                This controls whether walks stay near the source node or explore further outward.
                Lower values of ``q`` bias the walk toward outward exploration, behaving more like DFS and
                emphasizing structural roles. Higher values of ``q`` bias the walk toward nearby nodes,
                behaving more like BFS and emphasizing community structure and homophily.
            num_negative_samples: Number of negative samples to use for training the skip-gram model.
                If set to ``X``, then for each positive pair ``(u, v)`` generated from the random walks, ``X`` negative pairs ``(u, v_neg)`` will be generated,
                where ``v_neg`` is a node sampled uniformly at random from all nodes in the graph.
                Defaults to ``1``, meaning one negative sample per positive pair.
            num_nodes: Total number of nodes in the graph. If not provided, it will be inferred from the hyperedge_index.
                This is only needed if the hyperedge_index does not include all nodes (e.g., some isolated nodes are missing).
            graph_reduction_strategy: Strategy for reducing the hyperedge graph. Defaults to ``clique_expansion``.
            num_epochs: Number of epochs used to optimize Node2Vec embeddings. Defaults to ``5``.
            learning_rate: Learning rate for embedding optimization. Defaults to ``0.01``.
            batch_size: Batch size used by the random-walk loader. Defaults to ``128``.
            sparse: Whether Node2Vec embeddings should use sparse gradients.
            cache_dir: Optional directory to cache computed embeddings. If ``None``, caching is disabled.
            verbose: Whether to print verbose output during training. Defaults to ``False``.
        """
        super().__init__(cache_dir=cache_dir)
        if walk_length < context_size:
            raise ValueError(
                f"Expected walk_length >= context_size, got "
                f"walk_length={walk_length}, context_size={context_size}."
            )

        self.embedding_dim = num_features
        self.walk_length = walk_length
        self.context_size = context_size
        self.num_walks_per_node = num_walks_per_node
        self.p = p
        self.q = q
        self.num_negative_samples = num_negative_samples
        self.num_nodes = num_nodes
        self.graph_reduction_strategy = graph_reduction_strategy
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.sparse = sparse
        self.verbose = verbose

    def enrich(self, hyperedge_index: Tensor) -> Tensor:
        """
        Compute Node2Vec embeddings from the clique expansion of the hypergraph.

        The hypergraph is converted to a regular graph via clique expansion, where each hyperedge of size k
        contributes a k x k block of edges between its member nodes.
        The resulting ``edge_index`` is then used to train a Node2Vec model using random walks and the skip-gram objective.

        Args:
            hyperedge_index: Hyperedge index tensor of shape ``(2, num_hyperedges)``.

        Returns:
            Tensor of shape ``(num_nodes, embedding_dim)`` containing the Node2Vec embeddings for each node.
        """
        device = hyperedge_index.device

        if self.verbose:
            print(f"Reducing hypergraph to graph via {self.graph_reduction_strategy}...")

        hyperedge_index_wrapper = HyperedgeIndex(hyperedge_index)
        num_nodes = hyperedge_index_wrapper.num_nodes_if_isolated_exist(self.num_nodes)
        if num_nodes == 0:
            warnings.warn(
                "Found no nodes. Returning empty node features.",
                category=UserWarning,
                stacklevel=2,
            )
            return torch.empty((0, self.embedding_dim), device=device)

        reduced_edge_index = hyperedge_index_wrapper.reduce(self.graph_reduction_strategy)
        edge_index_wrapper = EdgeIndex(reduced_edge_index).remove_selfloops()
        if edge_index_wrapper.num_edges == 0:
            warnings.warn(
                "Clique expansion produced no non-self-loop edges. Returning zero node features.",
                category=UserWarning,
                stacklevel=2,
            )
            return torch.zeros((num_nodes, self.embedding_dim), device=device)

        edge_index = edge_index_wrapper.item.to(device)
        model = PyGNode2Vec(
            edge_index=edge_index,
            embedding_dim=self.embedding_dim,
            walk_length=self.walk_length,
            context_size=self.context_size,
            walks_per_node=self.num_walks_per_node,
            p=self.p,
            q=self.q,
            num_negative_samples=self.num_negative_samples,
            num_nodes=num_nodes,
            sparse=self.sparse,
        ).to(device)

        data_loader = model.loader(batch_size=self.batch_size, shuffle=True)
        optimizer = (
            optim.SparseAdam(model.parameters(), lr=self.learning_rate)
            if self.sparse
            else optim.Adam(model.parameters(), lr=self.learning_rate)
        )

        if self.verbose:
            print(f"Training Node2Vec model for {self.num_epochs} epochs...")

        model.train()
        for epoch in range(self.num_epochs):
            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.num_epochs}")

            # Iterate over batches of positive and negative random walks
            for positive_random_walk, negative_random_walk in data_loader:
                positive_random_walk = positive_random_walk.to(device)
                negative_random_walk = negative_random_walk.to(device)

                optimizer.zero_grad()
                loss = model.loss(positive_random_walk, negative_random_walk)
                loss.backward()
                optimizer.step()

        if self.verbose:
            print("Training complete. Generating node embeddings...")

        model.eval()
        with torch.no_grad():
            x: Tensor = model()  # shape (num_nodes, num_features)

        # Detach node embeddings from computation graph and return them
        return x.detach().to(device)


class LaplacianPositionalEncodingEnricher(NodeEnricher):
    """
    Enrich node features with Laplacian Positional Encodings computed from the symmetric normalized Laplacian of the clique expansion of the hypergraph.

    Args:
        num_features: Number of positional encoding features to generate for each node.
        num_nodes: Total number of nodes in the graph. If not provided, it will be inferred from the hyperedge_index.
            This is only needed if the hyperedge_index does not include all nodes (e.g., some isolated nodes are missing).
            Another instance is when the setting is transductive and the hyperedge index contains some hyperedges
            that do not contain all the nodes in the node space.
        cache_dir: Optional directory to cache computed features. If ``None``, caching is disabled.
    """

    def __init__(
        self,
        num_features: int,
        num_nodes: int = 0,
        cache_dir: Optional[str] = None,
    ):
        super().__init__(cache_dir=cache_dir)
        self.num_features = num_features
        self.num_nodes = num_nodes

    def enrich(self, hyperedge_index: Tensor) -> Tensor:
        """
        Compute Laplacian Positional Encoding: the k smallest non-trivial eigenvectors
        of the symmetric normalized Laplacian L = I - D^{-1/2} A D^{-1/2}.

        The first eigenvector (constant, eigenvalue ~0) is skipped.
        The next num_features eigenvectors are returned as positional features.

        Args:
            hyperedge_index: Hyperedge index tensor of shape ``(2, num_hyperedges)``.

        Returns:
            Tensor of shape ``(num_nodes, num_features)``.
        """
        edge_index = HyperedgeIndex(hyperedge_index).reduce_to_edge_index_on_clique_expansion()
        edge_index_wrapper = EdgeIndex(edge_index)
        num_nodes = self.num_nodes if self.num_nodes > 0 else None
        laplacian_matrix = edge_index_wrapper.get_sparse_normalized_laplacian(num_nodes=num_nodes)
        laplacian_matrix_dense = (
            laplacian_matrix.to_dense()  # torch.linalg.eigh only works on dense tensors
        )

        # Compute eigenvalues and eigenvectors of the symmetric Laplacian.
        # torch.linalg.eigh returns them sorted in ascending order of eigenvalue.
        # The smallest eigenvalue is ~0 with a constant eigenvector (all entries equal),
        # which carries no positional information and will be skipped.
        # Example: eigenvalues ~ [0, 1, 2],
        #          eigenvectors ~ [[0.577, -0.707, 0.408],
        #                          [0.577,  0.000, -0.816],
        #                          [0.577,  0.707,  0.408]]
        # Column 0 (eigenvalue ~0) is the trivial constant vector, all entries ~0.577.
        # eigenvectors shape is ``(num_nodes, num_nodes)``, each column is an eigenvector.
        with torch.no_grad():
            _, eigenvectors = torch.linalg.eigh(laplacian_matrix_dense)

        # We skip the first (trivial) eigenvector, so at most num_nodes - 1 are usable.
        # Example: 3 nodes -> 2 available non-trivial eigenvectors
        num_nodes = int(eigenvectors.size(0))
        num_nontrivial_eigenvectors = num_nodes - 1

        # If we have enough eigenvectors, slice columns 1 through num_features (inclusive).
        # Each row will be the positional encoding for that node.
        # Example: num_features = 2, eigenvectors.shape = (3, 3)
        #          -> return columns 1 and 2
        #             shape (3, 2)  # (num_nodes, num_features)
        if num_nontrivial_eigenvectors >= self.num_features:
            return eigenvectors[:, 1 : self.num_features + 1]

        # If the graph has fewer usable eigenvectors than requested
        # (e.g., num_features = 5 but only 2 available), we create a zero-padded tensor and fill what we have.
        # Example: num_nontrivial_eigenvectors = 2, num_features = 5
        #          -> shape (3, 5)  # columns 0-1 filled, 2-4 are zeros.
        x = torch.zeros(size=(num_nodes, self.num_features), device=edge_index.device)
        x[:, :num_nontrivial_eigenvectors] = eigenvectors[:, 1:]
        return x
