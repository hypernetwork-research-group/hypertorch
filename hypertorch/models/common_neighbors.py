import torch

from torch import Tensor, nn
from typing import Literal
from hypertorch.nn import CommonNeighborsHyperedgeScorer, NeighborScorer
from hypertorch.types import Neighborhood


class CommonNeighbors(nn.Module):
    """
    Computes Common Neighbors scores for hyperedges.

    Attributes:
        scorer: An instance of a NeighborScorer that computes the scores for hyperedges.
    """

    def __init__(
        self,
        aggregation: Literal["mean", "min", "sum"],
        scorer: NeighborScorer | None = None,
    ) -> None:
        """
        Initialize the common-neighbors model.

        Args:
            aggregation: Method used by the default scorer to aggregate pairwise counts.
                Defaults to ``mean``.
            scorer: Optional custom neighbor scorer.
        """
        super().__init__()
        self.scorer: NeighborScorer = (
            scorer if scorer is not None else CommonNeighborsHyperedgeScorer(aggregation)
        )

    def forward(
        self,
        candidate_nodes: Tensor,
        hyperedge_index: Tensor | None = None,
        node_to_neighbors: dict[int, Neighborhood] | None = None,
    ) -> Tensor:
        """
        Compute CN scores for all hyperedges in the batch.

        Args:
            candidate_nodes: Tensor containing node IDs to score of shape ``(num_nodes,)``.
            hyperedge_index: Tensor containing the hyperedge indices. Defaults to ``None``.
            node_to_neighbors: Optional mapping from nodes to their neighborhoods.
                Defaults to ``None``.

        Returns:
            scores: A 1-D tensor of shape (num_hyperedges,) with CN scores.
        """
        scores = self.scorer.score_batch(
            candidate_nodes=candidate_nodes,
            hyperedge_index=hyperedge_index,
            node_to_neighbors=node_to_neighbors,
        )
        torch.log1p(scores, out=scores)
        return scores
