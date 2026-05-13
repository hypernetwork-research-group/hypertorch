import torch

from torch import Tensor, optim
from hyperbench.types.hypergraph import HyperedgeIndex


class _VilLainTrainer:
    """
    Shared training helper for VilLain-based enrichers.

    The helper owns the common configuration, node and hyperedge count resolution,
    model construction, and training loop used by node feature and hyperedge attribute enrichers.

    Args:
        num_features: Dimensionality of the embeddings to generate.
        num_nodes: Total number of nodes, including isolated nodes missing from ``hyperedge_index``.
        num_hyperedges: Total number of hyperedges, including empty hyperedges missing from ``hyperedge_index``.
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
        self.embedding_dim = num_features
        self.num_nodes = num_nodes
        self.num_hyperedges = num_hyperedges
        self.labels_per_subspace = labels_per_subspace
        self.training_steps = training_steps
        self.generation_steps = generation_steps
        self.tau = tau
        self.eps = eps
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.verbose = verbose

    def _empty_features(self, hyperedge_index: Tensor) -> Tensor:
        """
        Return an empty feature matrix on the same device as ``hyperedge_index``.

        Args:
            hyperedge_index: Hyperedge index used only to select the output device.

        Returns:
            Empty tensor of shape ``(0, embedding_dim)``.
        """
        return torch.empty((0, self.embedding_dim), device=hyperedge_index.device)

    def _num_hyperedges(self, hyperedge_index: Tensor) -> int:
        """
        Return the explicit hyperedge count or infer it from ``hyperedge_index``.

        Args:
            hyperedge_index: Hyperedge index tensor used to infer the hyperedge count when no explicit count was provided.

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
            hyperedge_index: Hyperedge index tensor used to infer the node count when no explicit count was provided.

        Returns:
            Total number of nodes to preserve during VilLain training and embedding generation.
        """
        return HyperedgeIndex(hyperedge_index).num_nodes_if_isolated_exist(self.num_nodes)

    def _train(self, hyperedge_index: Tensor):
        """
        Train a VilLain model on the provided hypergraph topology.

        Args:
            hyperedge_index: Hyperedge index tensor of shape ``(2, num_incidences)``.

        Returns:
            Trained VilLain model ready to generate node or hyperedge embeddings.
        """
        # We need it here to avoid circular imports,
        # this is internal logic anyway and not part of the public API of this module.
        # This is changing when we refactor as per https://github.com/hypernetwork-research-group/hyperbench/issues/213
        from hyperbench.models.villain import VilLain

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
                print(f"Epoch {epoch + 1}/{self.num_epochs}")

            optimizer.zero_grad()
            loss, _ = model.loss(
                hyperedge_index=hyperedge_index,
                num_hyperedges=self._num_hyperedges(hyperedge_index),
            )
            loss.backward()
            optimizer.step()

        return model
