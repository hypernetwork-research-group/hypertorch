import random
import torch
import warnings

from abc import ABC, abstractmethod
from torch import Tensor, optim
from typing import Literal, TypeAlias, cast
from torch_geometric.nn import Node2Vec as PyGNode2Vec
from hypertorch.types import (
    EdgeIndex,
    HyperedgeIndex,
    GraphReductionStrategy,
    GraphReductionStrategyEnum,
)
from hypertorch.models import VilLain
from hypertorch.utils import (
    validate_is_between,
    validate_is_finite,
    validate_is_finite_when_provided,
    validate_is_non_negative,
    validate_is_positive,
)


EnrichmentMode: TypeAlias = Literal["concatenate", "replace"]
"""Mode used to combine generated features with existing features."""


class _VilLainTrainer:
    """
    Shared training helper for VilLain-based enrichers.

    The helper owns the common configuration, node and hyperedge count resolution,
    model construction, and training loop used by node feature and hyperedge attribute enrichers.

    Attributes:
        embedding_dim: Dimensionality of the embeddings to generate.
        num_nodes: Total number of nodes, including isolated nodes missing from ``hyperedge_index``.
        num_hyperedges: Total number of hyperedges, including empty hyperedges missing
            from ``hyperedge_index``.
        labels_per_subspace: Number of virtual labels per VilLain subspace.
        training_steps: Propagation steps used for VilLain self-supervised loss.
        generation_steps: Propagation steps averaged for final embeddings.
        tau: Gumbel-Softmax temperature.
        eps: Numerical stability constant.
        num_epochs: Number of optimization epochs.
        learning_rate: Adam learning rate.
        weight_decay: Adam weight decay.
        verbose: Whether to print training progress.
    """

    def __init__(
        self,
        num_features: int,
        num_nodes: int = 0,
        num_hyperedges: int = 0,
        labels_per_subspace: int = 2,
        training_steps: int = 4,
        generation_steps: int = 100,
        tau: float = 1.0,
        eps: float = 1e-10,
        num_epochs: int = 5,
        learning_rate: float = 0.01,
        weight_decay: float = 0.0,
        verbose: bool = False,
    ):
        """
        Initialize the VilLain training helper.

        Args:
            num_features: Dimensionality of the embeddings to generate.
            num_nodes: Total number of nodes, including isolated nodes missing
                from ``hyperedge_index``.
            num_hyperedges: Total number of hyperedges, including empty hyperedges missing
                from ``hyperedge_index``.
            labels_per_subspace: Number of virtual labels per VilLain subspace.
            training_steps: Propagation steps used for VilLain self-supervised loss.
            generation_steps: Propagation steps averaged for final embeddings.
            tau: Gumbel-Softmax temperature.
            eps: Numerical stability constant.
            num_epochs: Number of optimization epochs.
            learning_rate: Adam learning rate.
            weight_decay: Adam weight decay.
            verbose: Whether to print training progress.
        """
        self.embedding_dim: int = num_features
        self.num_nodes: int = num_nodes
        self.num_hyperedges: int = num_hyperedges
        self.labels_per_subspace: int = labels_per_subspace
        self.training_steps: int = training_steps
        self.generation_steps: int = generation_steps
        self.tau: float = tau
        self.eps: float = eps
        self.num_epochs: int = num_epochs
        self.learning_rate: float = learning_rate
        self.weight_decay: float = weight_decay
        self.verbose: bool = verbose

        self.__validate()

    def _empty_features(self, hyperedge_index: Tensor) -> Tensor:
        """
        Return an empty feature matrix on the same device as ``hyperedge_index``.

        Args:
            hyperedge_index: Hyperedge index used only to select the output device.

        Returns:
            Empty tensor of shape ``(0, embedding_dim)``.
        """
        return torch.empty(
            size=(0, self.embedding_dim),
            dtype=torch.float,
            device=hyperedge_index.device,
        )

    def _num_hyperedges(self, hyperedge_index: Tensor) -> int:
        """
        Return the explicit hyperedge count or infer it from ``hyperedge_index``.

        Args:
            hyperedge_index: Hyperedge index tensor used to infer the hyperedge count when
                no explicit count was provided.

        Returns:
            Total number of hyperedges to preserve during VilLain propagation.
        """
        return (
            self.num_hyperedges
            if self.num_hyperedges > 0
            else HyperedgeIndex(hyperedge_index).num_hyperedges
        )

    def _num_nodes(self, hyperedge_index: Tensor) -> int:
        """
        Return the explicit node count or infer it from ``hyperedge_index``.

        Args:
            hyperedge_index: Hyperedge index tensor used to infer the node count when
                no explicit count was provided.

        Returns:
            Total number of nodes to preserve during VilLain training and embedding generation.
        """
        return HyperedgeIndex(hyperedge_index).num_nodes_if_isolated_exist(self.num_nodes)

    def _train(self, hyperedge_index: Tensor) -> VilLain:
        """
        Train a VilLain model on the provided hypergraph topology.

        Args:
            hyperedge_index: Hyperedge index tensor of shape ``(2, num_incidences)``.

        Returns:
            Trained VilLain model ready to generate node or hyperedge embeddings.
        """
        model = VilLain(
            num_nodes=self._num_nodes(hyperedge_index),
            embedding_dim=self.embedding_dim,
            labels_per_subspace=self.labels_per_subspace,
            training_steps=self.training_steps,
            generation_steps=self.generation_steps,
            tau=self.tau,
            eps=self.eps,
        ).to(hyperedge_index.device)

        optimizer = optim.Adam(
            params=model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        if self.verbose:
            print(f"Training VilLain model for {self.num_epochs} epochs...")

        model.train()
        for epoch in range(self.num_epochs):
            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.num_epochs}\r", end="")

            optimizer.zero_grad()
            loss, _ = model.loss(
                hyperedge_index=hyperedge_index,
                num_hyperedges=self._num_hyperedges(hyperedge_index),
            )
            loss.backward()
            optimizer.step()

        return model

    def __validate(self) -> None:
        """
        Validate VilLain training configuration.

        Raises:
            ValueError: If any configuration value is outside its supported range.
        """
        validate_is_positive("num_features", self.embedding_dim)
        validate_is_non_negative("num_nodes", self.num_nodes)
        validate_is_non_negative("num_hyperedges", self.num_hyperedges)

        if self.labels_per_subspace < 2:
            raise ValueError(
                f"'labels_per_subspace' must be at least 2, got {self.labels_per_subspace}."
            )

        validate_is_positive("training_steps", self.training_steps)
        validate_is_positive("generation_steps", self.generation_steps)
        validate_is_finite("tau", self.tau)
        validate_is_positive("tau", self.tau)
        validate_is_finite("eps", self.eps)
        validate_is_positive("eps", self.eps)
        validate_is_positive("num_epochs", self.num_epochs)
        validate_is_positive("learning_rate", self.learning_rate)
        validate_is_non_negative("weight_decay", self.weight_decay)
        validate_is_finite("learning_rate", self.learning_rate)
        validate_is_finite("weight_decay", self.weight_decay)


class Enricher(ABC):
    """
    Generic base class for enrichers.

    Attributes:
        cache_dir: Directory for saving/loading cached features. If ``None``, caching is disabled.
    """

    def __init__(
        self,
        cache_dir: str | None = None,
    ):
        """
        Initialize the enricher.

        Args:
            cache_dir: Directory for saving/loading cached features.
                If ``None``, caching is disabled.
        """
        self.cache_dir: str | None = cache_dir

    @abstractmethod
    def enrich(self, hyperedge_index: Tensor) -> Tensor:
        """
        Enrich data derived from a hyperedge index.

        Args:
            hyperedge_index: Hyperedge index tensor.

        Returns:
            features: Enriched tensor.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("Subclasses must implement the enrich method.")


class HyperedgeEnricher(Enricher, ABC):
    """
    Base class for hyperedge enrichers.
    """

    pass


HyperedgeAttrsEnricher: TypeAlias = HyperedgeEnricher
"""Type alias for enrichers that generate hyperedge attributes."""


HyperedgeWeightsEnricher: TypeAlias = HyperedgeEnricher
"""Type alias for enrichers that generate hyperedge weights."""


class NodeEnricher(Enricher, ABC):
    """
    Base class for node enrichers.
    """

    pass


class FillValueHyperedgeAttrsEnricher(HyperedgeAttrsEnricher):
    """
    Generates simple hyperedge attributes by filling them with a constant value.

    Attributes:
        fill_value: The constant value to fill the hyperedge attributes with. Defaults to ``1.0``.
    """

    def __init__(
        self,
        cache_dir: str | None = None,
        fill_value: float = 1.0,
    ):
        """
        Initialize the fill-value hyperedge attribute enricher.

        Args:
            cache_dir: Directory for saving/loading cached features.
                If ``None``, caching is disabled.
            fill_value: The constant value to fill the hyperedge attributes with.
        """
        super().__init__(cache_dir=cache_dir)
        self.fill_value: float = fill_value

    def enrich(self, hyperedge_index: Tensor) -> Tensor:
        """
        Generate hyperedge attributes.

        Args:
            hyperedge_index: Hyperedge index tensor of shape ``(2, num_hyperedges)``.

        Returns:
            hyperedge_attr: Tensor of shape ``(num_hyperedges, 1)`` containing
                the generated attribute for each hyperedge.
        """
        num_hyperedges = HyperedgeIndex(hyperedge_index).num_hyperedges
        hyperedge_attrs = torch.full(
            size=(num_hyperedges, 1),
            fill_value=self.fill_value,
            dtype=torch.float,
            device=hyperedge_index.device,
        )
        return hyperedge_attrs


class VilLainHyperedgeAttrsEnricher(_VilLainTrainer, HyperedgeAttrsEnricher):
    """
    Enrich hyperedge attributes with VilLain embeddings learned from hypergraph topology.
    """

    def __init__(
        self,
        num_features: int,
        num_nodes: int = 0,
        num_hyperedges: int = 0,
        labels_per_subspace: int = 2,
        training_steps: int = 4,
        generation_steps: int = 100,
        tau: float = 1.0,
        eps: float = 1e-10,
        num_epochs: int = 5,
        learning_rate: float = 0.01,
        weight_decay: float = 0.0,
        cache_dir: str | None = None,
        verbose: bool = False,
    ):
        """
        Initialize the VilLain hyperedge attribute enricher.

        Args:
            num_features: Dimensionality of the hyperedge embeddings to generate.
            num_nodes: Total number of nodes, including isolated nodes missing from
                ``hyperedge_index``.
            num_hyperedges: Total number of hyperedges, including empty hyperedges missing
                from ``hyperedge_index``.
            labels_per_subspace: Number of virtual labels per subspace.
            training_steps: Propagation steps used for VilLain self-supervised loss.
            generation_steps: Propagation steps averaged for final embeddings.
            tau: Gumbel-Softmax temperature.
            eps: Numerical stability constant.
            num_epochs: Number of epochs used to optimize VilLain embeddings.
            learning_rate: Learning rate for embedding optimization.
            weight_decay: Weight decay for the optimizer.
            cache_dir: Optional directory to cache computed features. If ``None``,
                caching is disabled.
            verbose: Whether to print verbose output during training.
        """
        HyperedgeAttrsEnricher.__init__(self, cache_dir=cache_dir)
        _VilLainTrainer.__init__(
            self,
            num_features=num_features,
            num_nodes=num_nodes,
            num_hyperedges=num_hyperedges,
            labels_per_subspace=labels_per_subspace,
            training_steps=training_steps,
            generation_steps=generation_steps,
            tau=tau,
            eps=eps,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            verbose=verbose,
        )

    def enrich(self, hyperedge_index: Tensor) -> Tensor:
        """
        Train VilLain on the hypergraph and return hyperedge embeddings.

        Args:
            hyperedge_index: Hyperedge index tensor of shape ``(2, num_hyperedges)``.

        Returns:
            hyperedge_embeddings: Tensor of shape ``(num_hyperedges, num_features)``
                containing VilLain hyperedge embeddings.
        """
        num_hyperedges = self._num_hyperedges(hyperedge_index)
        if num_hyperedges == 0:
            warnings.warn(
                "Found no hyperedges. Returning empty hyperedge attributes.",
                category=UserWarning,
                stacklevel=2,
            )
            return self._empty_features(hyperedge_index)

        model = self._train(hyperedge_index)
        model.eval()
        with torch.no_grad():
            hyperedge_attr = model.hyperedge_embeddings(
                hyperedge_index=hyperedge_index,
                num_hyperedges=num_hyperedges,
            )
        return hyperedge_attr.detach().to(hyperedge_index.device)


class ABHyperedgeWeightsEnricher(HyperedgeWeightsEnricher):
    """
    Generates hyperedge weights based on the number of nodes in each hyperedge.

    Attributes:
        alpha: Scaling factor for the random component added to weights.
            Must be between ``0.0`` and ``1.0``.
        beta: If provided, the random component is alpha * beta.
            If ``None``, no random component is added.
    """

    def __init__(
        self,
        cache_dir: str | None = None,
        alpha: float = 1.0,
        beta: float | None = None,
    ):
        """
        Initialize the hyperedge weight enricher.

        Args:
            cache_dir: Directory for saving/loading cached features.
                If ``None``, caching is disabled.
            alpha: Scaling factor for the random component added to weights.
            beta: If provided, the random component is ``alpha * beta``.

        Raises:
            ValueError: If ``alpha`` or ``beta`` is invalid.
        """
        super().__init__(cache_dir=cache_dir)

        validate_is_between("alpha", alpha, 0.0, 1.0)
        validate_is_finite_when_provided("beta", beta)

        self.alpha: float = alpha
        self.beta: float | None = beta

    def enrich(self, hyperedge_index: Tensor) -> Tensor:
        """
        Compute edge weights as the number of nodes in each hyperedge.

        Args:
            hyperedge_index: Hyperedge index tensor of shape ``(2, num_hyperedges)``.

        Returns:
            hyperedge_weight: Tensor of shape ``(num_hyperedges,)`` containing
                the weight of each hyperedge.
        """
        # Count the number of nodes in each hyperedge by counting occurrences of
        # each hyperedge index.
        # Example: if hyperedge_index[1] = [0, 0, 1, 1, 1], then we have 2 nodes
        # in hyperedge 0 and 3 nodes in hyperedge 1.
        num_hyperedges = int(hyperedge_index[1].max().item()) + 1
        weights = torch.bincount(hyperedge_index[1], minlength=num_hyperedges).float()

        random_alpha = random.uniform(0, self.alpha)
        if self.beta is not None:
            weights += random_alpha * self.beta
        return weights


class Node2VecEnricher(NodeEnricher):
    """
    Enrich node features using Node2Vec embeddings computed from the clique expansion of the
    hypergraph.

    Attributes:
        embedding_dim: Dimensionality of the node embeddings to generate.
        walk_length: Length of each random walk.
        context_size: Window size for the skip-gram model
            (number of neighbors in the walk considered as context).
            For example, if ``context_size=2`` and ``walk_length=5``, then for
            a random walk ``[v0, v1, v2, v3, v4]``,
            the context for ``v2`` would be ``[v0, v1, v3, v4]`` as we take neighbors within
            distance 2 in the walk.
            The pairs generated by skip-gram would be ``[(v2, v0), (v2, v1), (v2, v3), (v2, v4)]``.
            Rule of thumb: Graphs with strong local structure (5-10), Graphs with
            communities/long-range patterns (10-20).
            Defaults to ``10``.
        num_walks_per_node: Number of random walks to start at each node.
        p: Return hyperparameter for Node2Vec. Default is ``1.0`` (unbiased).
            This controls the probability of stepping back to the node visited in the previous step.
            Lower values of ``p`` make immediate backtracking more likely,
            which keeps walks closer to the local neighborhood. Higher values of ``p`` discourage
            returning to the previous node, so walks are less likely to bounce back
            and forth across the same edge.
        q: In-out hyperparameter for Node2Vec. Default is ``1.0`` (unbiased).
            This controls whether walks stay near the source node or explore further outward.
            Lower values of ``q`` bias the walk toward outward exploration, behaving more like DFS
            and emphasizing structural roles. Higher values of ``q`` bias the walk toward
            nearby nodes, behaving more like BFS and emphasizing community structure and homophily.
        num_negative_samples: Number of negative samples used for skip-gram training.
            If set to ``X``, then for each positive pair ``(u, v)`` generated from the random walks,
            ``X`` negative pairs ``(u, v_neg)`` will be generated,
            where ``v_neg`` is a node sampled uniformly at random from all nodes in the graph.
            Defaults to ``1``, meaning one negative sample per positive pair.
        num_nodes: Total number of nodes to preserve. If not provided, it will be inferred from
            ``hyperedge_index``. This is only needed if ``hyperedge_index`` does not include
            all nodes (e.g., some isolated nodes are missing).
        graph_reduction_strategy: Strategy for reducing the hyperedge graph.
            Defaults to ``clique_expansion``.
        num_epochs: Number of epochs used to optimize Node2Vec embeddings. Defaults to ``5``.
        learning_rate: Learning rate for embedding optimization. Defaults to ``0.01``.
        batch_size: Batch size used by the random-walk loader. Defaults to ``128``.
        sparse: Whether Node2Vec embeddings should use sparse gradients.
        verbose: Whether to print verbose output during training. Defaults to ``False``.
    """

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
        graph_reduction_strategy: GraphReductionStrategy = (
            GraphReductionStrategyEnum.CLIQUE_EXPANSION
        ),
        num_epochs: int = 5,
        learning_rate: float = 0.01,
        batch_size: int = 128,
        sparse: bool = True,
        cache_dir: str | None = None,
        verbose: bool = False,
    ):
        """
        Initialize the Node2Vec enricher.

        Args:
            num_features: Dimensionality of the node embeddings to generate.
            walk_length: Length of each random walk.
            context_size: Window size for the skip-gram model. For example, if
                ``context_size=2`` and ``walk_length=5``, then for a random walk
                ``[v0, v1, v2, v3, v4]``, the context for ``v2`` is
                ``[v0, v1, v3, v4]``.
            num_walks_per_node: Number of random walks to start at each node.
            p: Return hyperparameter for Node2Vec.
            q: In-out hyperparameter for Node2Vec.
            num_negative_samples: Number of negative samples to use for skip-gram training.
            num_nodes: Total number of nodes in the graph. If not provided, it is inferred from
                ``hyperedge_index``.
            graph_reduction_strategy: Strategy for reducing the hypergraph.
            num_epochs: Number of epochs used to optimize Node2Vec embeddings.
            learning_rate: Learning rate for embedding optimization.
            batch_size: Batch size used by the random-walk loader.
            sparse: Whether Node2Vec embeddings should use sparse gradients.
            cache_dir: Optional directory to cache computed embeddings.
                If ``None``, caching is disabled.
            verbose: Whether to print verbose output during training.
        """
        super().__init__(cache_dir=cache_dir)
        self.embedding_dim: int = num_features
        self.walk_length: int = walk_length
        self.context_size: int = context_size
        self.num_walks_per_node: int = num_walks_per_node
        self.p: float = p
        self.q: float = q
        self.num_negative_samples: int = num_negative_samples
        self.num_nodes: int = num_nodes
        self.graph_reduction_strategy: GraphReductionStrategy = graph_reduction_strategy
        self.num_epochs: int = num_epochs
        self.learning_rate: float = learning_rate
        self.batch_size: int = batch_size
        self.sparse: bool = sparse
        self.verbose: bool = verbose

        self.__validate()

    def enrich(self, hyperedge_index: Tensor) -> Tensor:
        """
        Compute Node2Vec embeddings from the clique expansion of the hypergraph.

        The hypergraph is converted to a regular graph via clique expansion, where each hyperedge
        of size k contributes a k x k block of edges between its member nodes.
        The resulting ``edge_index`` is then used to train a Node2Vec model using random walks
        and the skip-gram objective.

        Args:
            hyperedge_index: Hyperedge index tensor of shape ``(2, num_hyperedges)``.

        Returns:
            x: Tensor of shape ``(num_nodes, embedding_dim)`` containing the Node2Vec embeddings
                for each node.
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
            return torch.empty(size=(0, self.embedding_dim), dtype=torch.float, device=device)

        reduced_edge_index = hyperedge_index_wrapper.reduce(
            self.graph_reduction_strategy,
            num_nodes=num_nodes,
        )
        edge_index_wrapper = EdgeIndex(reduced_edge_index).remove_selfloops()
        if edge_index_wrapper.num_edges == 0:
            warnings.warn(
                """
                Clique expansion produced no non-self-loop edges. Returning zero node features.
                """,
                category=UserWarning,
                stacklevel=2,
            )
            return torch.zeros(
                size=(num_nodes, self.embedding_dim),
                dtype=torch.float,
                device=device,
            )

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
                print(f"Epoch {epoch + 1}/{self.num_epochs}\r", end="")

            # Iterate over batches of positive and negative random walks
            for positive_random_walk, negative_random_walk in data_loader:
                positive_random_walk_on_device = positive_random_walk.to(device)
                negative_random_walk_on_device = negative_random_walk.to(device)

                optimizer.zero_grad()
                loss = model.loss(positive_random_walk_on_device, negative_random_walk_on_device)
                loss.backward()
                optimizer.step()

        if self.verbose:
            print("Training complete. Generating node embeddings...")

        model.eval()
        with torch.no_grad():
            x: Tensor = model()  # shape (num_nodes, num_features)

        # Detach node embeddings from computation graph and return them
        return x.detach().to(device)

    def __validate(self) -> None:
        """
        Validate Node2Vec enrichment configuration.

        Raises:
            ValueError: If any configuration value is outside its supported range.
        """
        validate_is_positive("num_features", self.embedding_dim)
        validate_is_positive("walk_length", self.walk_length)
        validate_is_positive("context_size", self.context_size)
        if self.walk_length < self.context_size:
            raise ValueError(
                "Expected walk_length >= context_size, got "
                f"walk_length={self.walk_length}, context_size={self.context_size}."
            )

        validate_is_positive("num_walks_per_node", self.num_walks_per_node)
        validate_is_finite("p", self.p)
        validate_is_positive("p", self.p)
        validate_is_finite("q", self.q)
        validate_is_positive("q", self.q)
        validate_is_positive("num_negative_samples", self.num_negative_samples)
        validate_is_non_negative("num_nodes", self.num_nodes)
        validate_is_positive("num_epochs", self.num_epochs)
        validate_is_finite("learning_rate", self.learning_rate)
        validate_is_positive("learning_rate", self.learning_rate)
        validate_is_positive("batch_size", self.batch_size)


class LaplacianPositionalEncodingEnricher(NodeEnricher):
    """
    Enrich node features with Laplacian Positional Encodings computed from the symmetric normalized
    Laplacian of the clique expansion of the hypergraph.

    Attributes:
        num_features: Number of positional encoding features to generate for each node.
        num_nodes: Total number of nodes in the graph. If not provided, it will be inferred
            from ``hyperedge_index``. This is only needed if ``hyperedge_index`` does not include
            all nodes (e.g., some isolated nodes are missing). Another instance is when the setting
            is transductive and ``hyperedge_index`` contains some hyperedges that do not contain
            all the nodes in the node space.
    """

    def __init__(
        self,
        num_features: int,
        num_nodes: int = 0,
        cache_dir: str | None = None,
    ):
        """
        Initialize the Laplacian positional encoding enricher.

        Args:
            num_features: Number of positional encoding features to generate for each node.
            num_nodes: Total number of nodes in the graph.
                If not provided, it is inferred from ``hyperedge_index``.
            cache_dir: Optional directory to cache computed features.
                If ``None``, caching is disabled.

        Raises:
            ValueError: If ``num_features`` or ``num_nodes`` is invalid.
        """
        super().__init__(cache_dir=cache_dir)

        validate_is_positive("num_features", num_features)
        validate_is_non_negative("num_nodes", num_nodes)

        self.num_features: int = num_features
        self.num_nodes: int = num_nodes

    def enrich(self, hyperedge_index: Tensor) -> Tensor:
        """
        Compute Laplacian Positional Encoding: the k smallest non-trivial eigenvectors
        of the symmetric normalized Laplacian L = I - D^{-1/2} A D^{-1/2}.

        The first eigenvector (constant, eigenvalue ~0) is skipped.
        The next num_features eigenvectors are returned as positional features.

        Args:
            hyperedge_index: Hyperedge index tensor of shape ``(2, num_hyperedges)``.

        Returns:
            node_features: Tensor of shape ``(num_nodes, num_features)``.
        """
        num_nodes = self.num_nodes if self.num_nodes > 0 else None
        edge_index = HyperedgeIndex(hyperedge_index).reduce_to_edge_index_on_clique_expansion(
            num_nodes=num_nodes
        )
        edge_index_wrapper = EdgeIndex(edge_index)
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
            eigenvectors = cast(Tensor, eigenvectors)

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
        # (e.g., num_features = 5 but only 2 available), we create a zero-padded tensor
        # and fill what we have.
        # Example: num_nontrivial_eigenvectors = 2, num_features = 5
        #          -> shape (3, 5)  # columns 0-1 filled, 2-4 are zeros.
        x = torch.zeros(
            size=(num_nodes, self.num_features),
            dtype=eigenvectors.dtype,
            device=eigenvectors.device,
        )
        x[:, :num_nontrivial_eigenvectors] = eigenvectors[:, 1:]
        return x


class VilLainEnricher(_VilLainTrainer, NodeEnricher):
    """
    Enrich node features with VilLain embeddings learned from hypergraph topology.
    """

    def __init__(
        self,
        num_features: int,
        num_nodes: int = 0,
        num_hyperedges: int = 0,
        labels_per_subspace: int = 2,
        training_steps: int = 4,
        generation_steps: int = 100,
        tau: float = 1.0,
        eps: float = 1e-10,
        num_epochs: int = 5,
        learning_rate: float = 0.01,
        weight_decay: float = 0.0,
        cache_dir: str | None = None,
        verbose: bool = False,
    ):
        """
        Initialize the VilLain node feature enricher.

        Args:
            num_features: Dimensionality of the node embeddings to generate.
            num_nodes: Total number of nodes, including isolated nodes missing
                from ``hyperedge_index``.
            num_hyperedges: Total number of hyperedges, including empty hyperedges missing
                from ``hyperedge_index``.
            labels_per_subspace: Number of virtual labels per subspace.
            training_steps: Propagation steps used for VilLain self-supervised loss.
            generation_steps: Propagation steps averaged for final embeddings.
            tau: Gumbel-Softmax temperature.
            eps: Numerical stability constant.
            num_epochs: Number of epochs used to optimize VilLain embeddings.
            learning_rate: Learning rate for embedding optimization.
            weight_decay: Weight decay for the optimizer.
            cache_dir: Optional directory to cache computed features. If ``None``,
                caching is disabled.
            verbose: Whether to print verbose output during training.
        """
        NodeEnricher.__init__(self, cache_dir=cache_dir)
        _VilLainTrainer.__init__(
            self,
            num_features=num_features,
            num_nodes=num_nodes,
            num_hyperedges=num_hyperedges,
            labels_per_subspace=labels_per_subspace,
            training_steps=training_steps,
            generation_steps=generation_steps,
            tau=tau,
            eps=eps,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            verbose=verbose,
        )

    def enrich(self, hyperedge_index: Tensor) -> Tensor:
        """
        Train VilLain on the hypergraph and return node embeddings.

        Args:
            hyperedge_index: Hyperedge index tensor of shape ``(2, num_hyperedges)``.

        Returns:
            node_embeddings: Tensor of shape ``(num_nodes, num_features)`` containing
                VilLain node embeddings.
        """
        num_nodes = self._num_nodes(hyperedge_index)
        if num_nodes == 0:
            warnings.warn(
                "Found no nodes. Returning empty node features.",
                category=UserWarning,
                stacklevel=2,
            )
            return self._empty_features(hyperedge_index)

        model = self._train(hyperedge_index)
        model.eval()
        with torch.no_grad():
            x = model.node_embeddings(
                hyperedge_index=hyperedge_index,
                num_hyperedges=self._num_hyperedges(hyperedge_index),
            )
        return x.detach().to(hyperedge_index.device)
