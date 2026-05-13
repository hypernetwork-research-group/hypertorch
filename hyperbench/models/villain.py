import math
import torch
import torch.nn.functional as F

from torch import Tensor, nn
from typing import Literal
from hyperbench.nn import HyperedgeAggregator, NodeAggregator, VilLainLoss, VilLainLossParts
from hyperbench.types import HyperedgeIndex


class VilLain(nn.Module):
    """
    VilLain learns node-specific virtual-label logits instead of consuming existing node features.
    The model is transductive: rows in ``node_embedding`` correspond to the fixed global node space used during training.
    - Proposed in `VilLain: Self-Supervised Learning on Homogeneous Hypergraphs without Features via Virtual Label Propagation <https://dl.acm.org/doi/pdf/10.1145/3589334.3645454>`_ paper (WWW 2024).
    - Reference implementation: `source <https://github.com/geon0325/VilLain/>`_.

    Each forward pass:
    1. Samples differentiable virtual-label assignments with Gumbel-Softmax.
    2. Propagates them over the incidence structure.
    3. Returns averaged propagated node embeddings.


    Args:
        num_nodes: Total number of trainable nodes.
        embedding_dim: Returned embedding dimension. Defaults to ``128``.
        labels_per_subspace: Number of virtual labels per subspace. Defaults to ``2``.
        training_steps: Propagation steps used for self-supervised loss. Defaults to ``4``.
        generation_steps: Propagation steps averaged for final embeddings. Defaults to ``100``.
        tau: Gumbel-Softmax temperature. Defaults to ``1.0``.
        eps: Numerical stability constant. Defaults to ``1e-10``.
    """

    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int = 128,
        labels_per_subspace: int = 2,
        training_steps: int = 4,
        generation_steps: int = 100,
        tau: float = 1.0,
        eps: float = 1e-10,
    ):
        super().__init__()
        self.__validate_args(
            num_nodes=num_nodes,
            embedding_dim=embedding_dim,
            labels_per_subspace=labels_per_subspace,
            training_steps=training_steps,
            generation_steps=generation_steps,
            tau=tau,
            eps=eps,
        )

        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.labels_per_subspace = labels_per_subspace
        self.training_steps = training_steps
        self.generation_steps = generation_steps
        self.tau = tau
        self.eps = eps

        self.num_subspaces = math.ceil(embedding_dim / labels_per_subspace)
        self.raw_embedding_dim = self.num_subspaces * labels_per_subspace
        self.node_embedding = nn.Parameter(torch.empty(size=(num_nodes, self.raw_embedding_dim)))

        self.loss_fn = VilLainLoss(
            num_subspaces=self.num_subspaces,
            labels_per_subspace=self.labels_per_subspace,
            eps=self.eps,
        )

        self.reset_parameters()

    def forward(
        self,
        hyperedge_index: Tensor,
        node_ids: Tensor | None = None,
        num_hyperedges: int | None = None,
    ) -> tuple[Tensor, VilLainLossParts]:
        """
        Compute the self-supervised VilLain objective.
        Use ``hyperedge_embeddings`` or ``node_embeddings`` to generate final embeddings for inference after training.

        Args:
            hyperedge_index: Incidence tensor of shape ``(2, num_incidences)``.
            node_ids: Optional global node ids matching local node ids the embedding table in the transductive setting.
                Use this when a batch has rebased local node ids but the learned logits live in the full transductive node table.
                This is needed as the model keeps an internal embedding table with a row for every node in the global node space.
            num_hyperedges: Optional explicit hyperedge count used during node-to-hyperedge pooling to preserve empty hyperedges.
                If not provided, the hyperedge count is inferred from ``hyperedge_index``.

        Returns:
            Node embeddings of shape ``(num_local_nodes, embedding_dim)``.
        """
        return self.loss(
            hyperedge_index=hyperedge_index,
            node_ids=node_ids,
            num_hyperedges=num_hyperedges,
        )

    def loss(
        self,
        hyperedge_index: Tensor,
        node_ids: Tensor | None = None,
        num_hyperedges: int | None = None,
    ) -> tuple[Tensor, VilLainLossParts]:
        """
        Compute the self-supervised VilLain objective.

        Args:
            hyperedge_index: Incidence tensor of shape ``(2, num_incidences)``.
            node_ids: Optional global node ids matching local node ids the embedding table in the transductive setting.
                Use this when a batch has rebased local node ids but the learned logits live in the full transductive node table.
                This is needed as the model keeps an internal embedding table with a row for every node in the global node space.
            num_hyperedges: Optional explicit hyperedge count used during node-to-hyperedge pooling to preserve empty hyperedges.
                If not provided, the hyperedge count is inferred from ``hyperedge_index``.

        Returns:
            A tuple ``(total_loss, loss_parts)`` where ``loss_parts`` contains ``local_loss`` and ``global_loss`` scalar tensors.
        """
        node_embeddings = self.__get_initial_virtual_node_features(node_ids=node_ids)
        actual_num_hyperedges = self.__num_hyperedges(hyperedge_index, num_hyperedges)

        local_loss = node_embeddings.new_zeros(size=())
        global_loss = node_embeddings.new_zeros(size=())
        for _ in range(self.training_steps):
            node_embeddings, hyperedge_embeddings = self.__message_passing(
                x=node_embeddings,
                hyperedge_index=hyperedge_index,
                num_hyperedges=actual_num_hyperedges,
            )
            local_loss = local_loss + self.loss_fn.local_loss(node_embeddings, hyperedge_embeddings)
            global_loss = global_loss + self.loss_fn.global_loss(
                node_embeddings, hyperedge_embeddings
            )

        return self.loss_fn.total_loss(local_loss, global_loss), {
            "local_loss": local_loss,
            "global_loss": global_loss,
        }

    def hyperedge_embeddings(
        self,
        hyperedge_index: Tensor,
        node_ids: Tensor | None = None,
        num_hyperedges: int | None = None,
    ) -> Tensor:
        """
        Generate hyperedge embeddings by averaging propagated hyperedge states.
        Every generation step computes hyperedge states from the current node states, then updates node states for the next step.

        Args:
            hyperedge_index: Incidence tensor of shape ``(2, num_incidences)``.
            node_ids: Optional global node ids matching local node ids the embedding table in the transductive setting.
                Use this when a batch has rebased local node ids but the learned logits live in the full transductive node table.
                This is needed as the model keeps an internal embedding table with a row for every node in the global node space.
            num_hyperedges: Optional explicit hyperedge count used during node-to-hyperedge pooling to preserve empty hyperedges.
                If not provided, the hyperedge count is inferred from ``hyperedge_index``.

        Returns:
            Hyperedge embeddings of shape ``(num_hyperedges, embedding_dim)``.
        """
        return self.__embeddings(
            hyperedge_index=hyperedge_index,
            node_ids=node_ids,
            num_hyperedges=num_hyperedges,
            mode="hyperedge",
        )

    def node_embeddings(
        self,
        hyperedge_index: Tensor,
        node_ids: Tensor | None = None,
        num_hyperedges: int | None = None,
    ) -> Tensor:
        """
        Generate node embeddings by averaging propagated node states.

        Args:
            hyperedge_index: Incidence tensor of shape ``(2, num_incidences)``.
            node_ids: Optional global node ids matching local node ids the embedding table in the transductive setting.
                Use this when a batch has rebased local node ids but the learned logits live in the full transductive node table.
                This is needed as the model keeps an internal embedding table with a row for every node in the global node space.
            num_hyperedges: Optional explicit hyperedge count used during node-to-hyperedge pooling to preserve empty hyperedges.
                If not provided, the hyperedge count is inferred from ``hyperedge_index``.

        Returns:
            Node embeddings of shape ``(num_local_nodes, embedding_dim)``.
        """
        return self.__embeddings(
            hyperedge_index=hyperedge_index,
            node_ids=node_ids,
            num_hyperedges=num_hyperedges,
            mode="node",
        )

    def reset_parameters(self) -> None:
        """Initialize trainable virtual-label logits near zero."""
        nn.init.normal_(self.node_embedding, mean=0.0, std=0.1)

    def __embeddings(
        self,
        hyperedge_index: Tensor,
        node_ids: Tensor | None,
        num_hyperedges: int | None,
        mode: Literal["node", "hyperedge"] = "node",
    ) -> Tensor:
        """
        Generate final node or hyperedge embeddings for inference.

        Args:
            hyperedge_index: Incidence tensor of shape ``(2, num_incidences)``.
            node_ids: Optional global node ids matching local node ids the embedding table in the transductive setting.
            num_hyperedges: Optional explicit hyperedge count to preserve empty hyperedges during propagation.
            mode: Selects whether to accumulate propagated node states or hyperedge states.

        Returns:
            Averaged embeddings truncated to ``embedding_dim``.
        """
        with torch.no_grad():
            x = self.__get_initial_virtual_node_features(node_ids=node_ids)
            actual_num_hyperedges = self.__num_hyperedges(hyperedge_index, num_hyperedges)

            final_embeddings_size = (
                (x.size(0), self.raw_embedding_dim)
                if mode == "node"
                else (actual_num_hyperedges, self.raw_embedding_dim)
            )
            final_embeddings = x.new_zeros(size=final_embeddings_size)
            for _ in range(self.generation_steps):
                x, hyperedge_embeddings = self.__message_passing(
                    x=x,
                    hyperedge_index=hyperedge_index,
                    num_hyperedges=actual_num_hyperedges,
                )

                # Suppose generation_steps = 100.
                # Average 100 propagated embeddings for each node/hyperedge to get more stable final embeddings.
                # Sum here and divide by generation_steps later to avoid storing all 100 embeddings in memory at once.
                final_embeddings = final_embeddings + (
                    x if mode == "node" else hyperedge_embeddings
                )
            final_embeddings = final_embeddings / self.generation_steps

            # Example: final_embeddings.shape = (num_nodes/num_hyperedges, 8) with raw_embedding_dim=8
            #          -> returned shape = (num_nodes/num_hyperedges, 4) with embedding_dim=4
            #             as it takes the first 4 channels of the raw embedding as the final embedding.
            return final_embeddings[:, : self.embedding_dim]

    def __get_initial_virtual_node_features(self, node_ids: Tensor | None = None) -> Tensor:
        """
        Convert trainable node logits into flattened virtual-label probabilities.

        Args:
            node_ids: Optional global node ids matching local node ids the embedding table in the transductive setting.
                If ``None``, all node rows are used.

        Returns:
            A tensor of shape ``(num_selected_nodes, raw_embedding_dim)``.
        """
        logits = self.node_embedding if node_ids is None else self.node_embedding[node_ids]

        # Split flat logits into independent virtual-label subspaces.
        # Example: with raw_embedding_dim=8, num_subspaces=4, labels_per_subspace=2:
        #          logits.shape = (num_nodes, 8)
        #          -> viewed_logits shape = (num_nodes, 4, 2)
        #          viewed_logits[0] = [[l00, l01],  # node 0, subspace 0
        #                              [l02, l03],  # node 0, subspace 1
        #                              [l04, l05],  # node 0, subspace 2
        #                              [l06, l07]]  # node 0, subspace 3
        viewed_logits = logits.view(-1, self.num_subspaces, self.labels_per_subspace)

        # Convert each subspace's logits into a differentiable virtual-label assignment.
        # Example: viewed_logits[0, 0] = [0.03, -0.02]
        #          -> probs[0, 0] might be [0.47, 0.53] with tau=1.0
        #          probs.shape remains (num_nodes, 4, 2).
        probs = F.gumbel_softmax(viewed_logits, tau=self.tau, dim=2, hard=False)

        # Flatten subspaces back into a standard node-by-channel node feature matrix.
        # The aggregators expect matrices shaped (num_nodes, num_channels==raw_embedding_dim),
        # so propagation happens on the flattened channel dimension.
        # Example: probs.shape = (num_nodes, 4, 2) -> shape = (num_nodes, 8)
        return probs.reshape(-1, self.raw_embedding_dim)

    def __message_passing(
        self,
        x: Tensor,
        hyperedge_index: Tensor,
        num_hyperedges: int,
    ) -> tuple[Tensor, Tensor]:
        """
        One round of message passing, where nodes send messages to hyperedges and then hyperedges send messages back to nodes.

        Args:
            x: Virtual node features of shape (num_nodes, raw_embedding_dim).
            hyperedge_index: Hyperedge index tensor of shape (2, num_edges).
            num_hyperedges: Total number of hyperedges.

        Returns:
            The updated node and hyperedge embeddings after one round of message passing.
        """
        hyperedge_embeddings = HyperedgeAggregator(
            hyperedge_index=hyperedge_index,
            node_embeddings=x,
            num_hyperedges=num_hyperedges,
        ).pool("mean")

        node_embeddings = NodeAggregator(
            hyperedge_index=hyperedge_index,
            hyperedge_embeddings=hyperedge_embeddings,
            num_nodes=x.size(0),
        ).pool("mean")

        return node_embeddings, hyperedge_embeddings

    def __num_hyperedges(
        self,
        hyperedge_index: Tensor,
        num_hyperedges: int | None,
    ) -> int:
        """
        Return the explicit hyperedge count or infer it from the ``hyperedge_index``, if not provided.
        Explicit counts are required when empty hyperedges must remain in the hypergraph.
        """
        if num_hyperedges is not None:
            return num_hyperedges
        return HyperedgeIndex(hyperedge_index).num_hyperedges

    def __validate_args(
        self,
        num_nodes: int,
        embedding_dim: int,
        labels_per_subspace: int,
        training_steps: int,
        generation_steps: int,
        tau: float,
        eps: float,
    ) -> None:
        if num_nodes < 1:
            raise ValueError("num_nodes must be positive.")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be positive.")
        if labels_per_subspace < 2:
            raise ValueError("labels_per_subspace must be at least 2.")
        if training_steps < 1:
            raise ValueError("training_steps must be positive.")
        if generation_steps < 1:
            raise ValueError("generation_steps must be positive.")
        if tau <= 0:
            raise ValueError("tau must be positive.")
        if eps <= 0:
            raise ValueError("eps must be positive.")
