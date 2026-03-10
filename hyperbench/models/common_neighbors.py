import torch

from torch import Tensor, nn
from typing import Dict, Literal, Optional
from hyperbench.nn import CommonNeighborsScorer, NeighborScorer
from hyperbench.types import Neighborhood


class CommonNeighbors(nn.Module):
    def __init__(
        self,
        aggregation: Literal["mean", "min", "sum"],
        scorer: Optional[NeighborScorer] = None,
    ) -> None:
        super().__init__()
        self.scorer = scorer if scorer is not None else CommonNeighborsScorer(aggregation)

    def forward(
        self,
        hyperedge_index: Tensor,
        node_to_neighbors: Optional[Dict[int, Neighborhood]] = None,
    ) -> Tensor:
        """
        Compute CN scores for all hyperedges in the batch.

        Args:
            hyperedge_index: Tensor containing the hyperedge indices.
            node_to_neighbors: Optional mapping from nodes to their neighborhoods.

        Returns:
            A 1-D tensor of shape (num_hyperedges,) with CN scores.
        """
        scores = self.scorer.score_batch(hyperedge_index, node_to_neighbors)
        torch.log1p(scores, out=scores)
        return scores
