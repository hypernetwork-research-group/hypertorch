import torch

from torch import Tensor, nn
from typing import Literal
from hyperbench.nn import CommonNeighborsScorer, NeighborScorer
from hyperbench.types import Neighborhood


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
        self.scorer = scorer if scorer is not None else CommonNeighborsScorer(aggregation)

    def forward(
        self,
        hyperedge_index: Tensor,
        node_to_neighbors: dict[int, Neighborhood] | None = None,
    ) -> Tensor:
        """
        Compute CN scores for all hyperedges in the batch.

        Args:
            hyperedge_index: Tensor containing the hyperedge indices.
            node_to_neighbors: Optional mapping from nodes to their neighborhoods.
                Defaults to ``None``.

        Returns:
            scores: A 1-D tensor of shape (num_hyperedges,) with CN scores.
        """
        scores = self.scorer.score_batch(hyperedge_index, node_to_neighbors)
        torch.log1p(scores, out=scores)
        return scores
