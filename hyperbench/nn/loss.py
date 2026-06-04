import torch
import torch.nn.functional as F

from torch import Tensor, nn
from typing import TypedDict


class NHPRankingLoss(nn.Module):
    """
    Ranking loss that pushes positive hyperedges above sampled negatives.

    Examples:
        >>> logits = [2.0, 1.0, -1.0]
        >>> labels = [1.0, 1.0, 0.0]
        >>> loss = NHPRankingLoss()
        >>> loss(logits, labels)
        >>> loss.ndim
        ... 0
    """

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        """
        Compute the ranking loss.

        Args:
            logits: Logit scores for each candidate hyperedge, of shape ``(num_hyperedges,)``.
            labels: Binary labels indicating positive (1) and negative (0) hyperedges, of shape ``(num_hyperedges,)``.

        Returns:
            loss: Scalar loss value.
        """
        # Split logits by label as we need to compare positive scores against negative scores.
        # Example: logits = [2.0, 1.0, -1.0]
        #          labels = [1.0, 1.0, 0.0]
        #          -> positive_logits = [2.0, 1.0]
        #          -> negative_logits = [-1.0]
        positive_logits = logits[labels == 1]
        negative_logits = logits[labels == 0]

        positive_scores = torch.sigmoid(positive_logits)
        negative_scores = torch.sigmoid(negative_logits)
        if positive_scores.numel() == 0 or negative_scores.numel() == 0:
            raise ValueError("NHPRankingLoss requires both positive and negative hyperedges.")

        # Objective: enforce that each positive score is higher than the average negative score.
        # For each positive score pos_i:
        #   margin_i = mean(negative_scores) - pos_i
        # We interpret margin_i as follows:
        # - If pos_i > mean(negatives), then margin_i < 0    -> desirable
        # - If pos_i <= mean(negatives), then margin_i >= 0  -> violation
        margins = negative_scores.mean(dtype=torch.float) - positive_scores

        # Then softplus(margin_i):
        # - Is ~0 when margin_i is strongly negative (good ranking).
        # - Grows smoothly when margin_i > 0 (penalizing violations).
        # Final loss is the average over all positive samples.
        return F.softplus(margins).mean(dtype=torch.float)


class VilLainLoss:
    """
    VilLain self-supervised loss formulas.

    This class is intentionally stateless with respect to propagation.
    The VilLain model owns message passing and accumulation over steps
    and this class owns the per-step formulas for local and global loss,

    Args:
        num_subspaces: Number of virtual-label subspaces in each embedding.
        labels_per_subspace: Number of virtual labels in each subspace.
        eps: Numerical stability constant used in logarithms and cosine similarity.
    """

    def __init__(
        self,
        num_subspaces: int,
        labels_per_subspace: int,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.num_subspaces = num_subspaces
        self.labels_per_subspace = labels_per_subspace
        self.eps = eps

    def local_loss(self, node_embeddings: Tensor, hyperedge_embeddings: Tensor) -> Tensor:
        """
        Compute the local entropy loss for one propagation step.

        Local loss is minimized to encourage propagated node and hyperedge distributions
        to become confident within each virtual-label subspace.

        Args:
            node_embeddings: Propagated node states of shape ``(num_nodes, num_subspaces * labels_per_subspace)``.
            hyperedge_embeddings: Propagated hyperedge states with the same channel dimension as ``node_embeddings``.

        Returns:
            loss: Scalar tensor containing node plus hyperedge entropy losses.
        """
        return self.entropy_loss(node_embeddings) + self.entropy_loss(hyperedge_embeddings)

    def global_loss(self, node_embeddings: Tensor, hyperedge_embeddings: Tensor) -> Tensor:
        """
        Compute global anti-collapse losses for one propagation step.

        Global loss combines negative global entropy, which encourages balanced label usage
        with a distinctiveness term that separates label columns inside each subspace.

        Args:
            node_embeddings: Propagated node states of shape ``(num_nodes, num_subspaces * labels_per_subspace)``.
            hyperedge_embeddings: Propagated hyperedge states with the same channel dimension as ``node_embeddings``.

        Returns:
            loss: Scalar tensor containing node plus hyperedge global losses.
        """
        return (
            self.balance_loss(node_embeddings)
            + self.distinctiveness_loss(node_embeddings)
            + self.balance_loss(hyperedge_embeddings)
            + self.distinctiveness_loss(hyperedge_embeddings)
        )

    def total_loss(self, local_loss: Tensor, global_loss: Tensor) -> Tensor:
        """
        Combine accumulated local and global VilLain losses.

        Args:
            local_loss: Accumulated local entropy loss.
            global_loss: Accumulated balance plus distinctiveness loss.

        Returns:
            loss: Scalar tensor to minimize.
        """
        return local_loss + global_loss

    def entropy_loss(self, x: Tensor) -> Tensor:
        """
        Compute mean entropy within each virtual-label subspace.

        Args:
            x: Flattened virtual-label probabilities of shape ``(num_items, num_subspaces * labels_per_subspace)``.

        Returns:
            loss: Scalar entropy loss.
        """
        if x.size(0) == 0:
            return x.sum(dtype=torch.float) * 0.0

        # Example: x.shape = (num_nodes, 8)
        #          -> probs.shape = (num_nodes, 4, 2)
        #          probs[0, 0] = [0.12, 0.88] is node/hyperedge item 0's
        #          virtual-label distribution in subspace 0.
        probs = x.view(-1, self.num_subspaces, self.labels_per_subspace)

        # With this, we induce structurally close nodes (or hyperedges) to be assigned to the same label.
        # Example: probs.shape = (num_nodes, 4, 2)
        #          -> entropy.shape = (num_nodes, 4), one entropy per item and subspace
        entropy = -(probs * torch.log(probs + self.eps)).sum(dim=2, dtype=torch.float)
        return entropy.mean(dtype=torch.float)

    def balance_loss(self, x: Tensor) -> Tensor:
        """
        Compute negative entropy of global virtual-label usage.

        This term is minimized, so the negative sign makes optimization maximize entropy of average label usage
        and reduces collapse to one virtual label.

        Args:
            x: Flattened virtual-label probabilities of shape ``(num_items, num_subspaces * labels_per_subspace)``.

        Returns:
            loss: Scalar balance loss.
        """
        if x.size(0) == 0:
            return x.sum(dtype=torch.float) * 0.0

        # Example: with raw_embedding_dim=8, num_subspaces=4, labels_per_subspace=2:
        #          x.shape = (num_nodes, 8)
        #          -> probs.shape = (num_nodes, 4, 2)
        #          -> mean_probs.shape = (4, 2)
        #          mean_probs[0] = average usage of the two labels in subspace 0
        #          across all num_nodes nodes/hyperedges in this tensor.
        probs = x.view(-1, self.num_subspaces, self.labels_per_subspace)
        mean_probs = probs.mean(dim=0, dtype=torch.float)

        # Negative entropy to maximize global label diversity and prevents collapse.
        # Example: mean_probs[0] = [0.50, 0.50] has higher entropy than mean_probs[0] = [0.99, 0.01].
        entropy = -(mean_probs * torch.log(mean_probs + self.eps)).sum(dim=1, dtype=torch.float)
        return -entropy.mean(dtype=torch.float)

    def distinctiveness_loss(self, x: Tensor) -> Tensor:
        """
        Penalize similar virtual-label columns inside each subspace.

        For every subspace, this compares all label columns across items with cosine similarity and applies a diagonal classification objective.
        The diagonal target encourages each label column to be most similar to itself and less similar to other labels.

        Args:
            x: Flattened virtual-label probabilities of shape ``(num_items, num_subspaces * labels_per_subspace)``.

        Returns:
            loss: Scalar distinctiveness loss.
        """
        if x.size(0) == 0:
            return x.sum(dtype=torch.float) * 0.0

        # Distinctiveness compares virtual-label columns inside each subspace across all items.
        # Example: with raw_embedding_dim=8, num_subspaces=4, labels_per_subspace=2:
        #          x.shape = (num_nodes, 8)
        #          -> probs.shape = (num_nodes, 4, 2)
        probs = x.view(-1, self.num_subspaces, self.labels_per_subspace)

        # Build all ordered pairs of virtual-label column ids inside a subspace.
        # Example with num_subspaces=4 and labels_per_subspace=2:
        #         idx_i = [0, 1, 0, 1], shape = (4,)
        #         idx_j = [0, 0, 1, 1], shape = (4,)
        #         pairs are (0,0), (1,0), (0,1), (1,1)
        idx_i = torch.arange(self.labels_per_subspace, dtype=torch.long, device=x.device).repeat(
            self.labels_per_subspace
        )
        idx_j = torch.arange(
            self.labels_per_subspace, dtype=torch.long, device=x.device
        ).repeat_interleave(self.labels_per_subspace)

        # Compare every virtual-label column against every other column.
        # Two different labels in the same subspace should not describe the same pattern of nodes/hyperedges.
        # Example: with num_subspaces=4:
        #          probs[:, :, idx_i] and probs[:, :, idx_j] both have shape (4, 4, 4),
        #          where the last dimension enumerates the four ordered label pairs above
        #          probs[:, :, idx_i] == [[[p00, p01, p00, p01],   # node/hyperedge 0's label probabilities for the four pairs
        #                                  [p10, p11, p10, p11],   # node/hyperedge 1's label probabilities for the four pairs
        #                                  [p20, p21, p20, p21],   # node/hyperedge 2's label probabilities for the four pairs
        #                                  [p30, p31, p30, p31]],  # node/hyperedge 3's label probabilities for the four pairs
        #                                 ...]
        #          probs[:, :, idx_j] == [[[p00, p00, p01, p01],   # node/hyperedge 0's label probabilities for the four pairs
        #                                  [p10, p10, p11, p11],   # node/hyperedge 1's label probabilities for the four pairs
        #                                  [p20, p20, p21, p21],   # node/hyperedge 2's label probabilities for the four pairs
        #                                  [p30, p30, p31, p31]],  # node/hyperedge 3's label probabilities for the four pairs
        #                                 ...]
        #          F.cosine_similarity(..., dim=0) compares each pair across the 4 items, producing shape (4, 4)
        #          view(-1, 2, 2) restores one 2x2 similarity matrix per subspace, so shape becomes (4, 2, 2)
        similarity = F.cosine_similarity(
            probs[:, :, idx_i],
            probs[:, :, idx_j],
            dim=0,
            eps=self.eps,
        ).view(-1, self.labels_per_subspace, self.labels_per_subspace)

        # Turn each similarity row into a classification distribution and keep the diagonal self-match probabilities.
        # Example: similarity[subspace 0].shape = (2, 2)
        #          - row 0 scores how label 0 matches labels [0, 1]
        #          - row 1 scores how label 1 matches labels [0, 1]
        #          -> assignment_probs has rows summing to 1 via softmax(dim=2)
        #          -> diagonal_probs keeps P(label 0 matches 0) and P(label 1 matches 1).
        # Minimizing -log(diagonal_probs) encourages each label column to be:
        # - Most similar to itself
        # - Less similar to other label columns
        assignment_probs = torch.softmax(similarity, dim=2, dtype=torch.float)
        diagonal_probs = torch.diagonal(assignment_probs, dim1=1, dim2=2)
        return torch.mean(-torch.log(diagonal_probs + self.eps), dtype=torch.float)


class VilLainLossParts(TypedDict):
    """
    Named VilLain self-supervised loss parts returned by ``VilLain.loss``.

    Attributes:
        local_loss: Sum of node and hyperedge local entropy losses over all training propagation steps.
        global_loss: Sum of balance and distinctiveness losses over all training propagation steps.
    """

    local_loss: Tensor
    global_loss: Tensor
