import torch

from abc import ABC, abstractmethod
from torch import Tensor
from typing import Dict, List, Optional
from hyperbench.nn import Aggregation
from hyperbench.types import Neighborhood, Hypergraph, HyperedgeIndex


class NeighborScorer(ABC):
    @abstractmethod
    def score(
        self,
        candidate_nodes: List[int],
        candidate_to_neighbors: Dict[int, Neighborhood],
    ) -> float:
        raise NotImplementedError

    @abstractmethod
    def score_batch(
        self,
        hyperedge_index: Tensor,
        node_to_neighbors: Optional[Dict[int, Neighborhood]] = None,
    ) -> Tensor:
        raise NotImplementedError


class CommonNeighborsScorer(NeighborScorer):
    __DEFAULT_SCORE = 0.0

    def __init__(self, aggregation: Aggregation = Aggregation.MEAN) -> None:
        self.aggregation = aggregation

    def score(
        self,
        candidate_nodes: List[int],
        candidate_to_neighbors: Dict[int, Neighborhood],
    ) -> float:
        """
        Compute the CN score for a single candidate hyperedge.

        Args:
            candidate_nodes: List of node IDs forming the candidate hyperedge.
                If less than 2 nodes are provided, the function returns a default score of ``0.0``.
            candidate_to_neighbors: Mapping from node IDs to their set of neighbors.

        Returns:
            The aggregated common neighbors score.
        """
        if len(candidate_nodes) < 2:
            return self.__DEFAULT_SCORE

        pairwise_counts: List[int] = []
        candidates_tensor = torch.tensor(candidate_nodes)

        # Example: candidate_nodes = [1, 2, 3]
        #          -> compute common neighbors for pairs (1, 2), (1, 3), and (2, 3)
        for u, v in torch.combinations(candidates_tensor, 2):
            neighbors_u: Neighborhood = candidate_to_neighbors.get(u.item(), set())
            neighbors_v: Neighborhood = candidate_to_neighbors.get(v.item(), set())

            common_neighbors = neighbors_u & neighbors_v
            pairwise_counts.append(len(common_neighbors))

        return self.__to_score_by_aggregation(pairwise_counts)

    def score_batch(
        self,
        hyperedge_index: Tensor,
        node_to_neighbors: Optional[Dict[int, Neighborhood]] = None,
    ) -> Tensor:
        """
        Score all hyperedges in a hyperedge index tensor.

        Args:
            hyperedge_index: Tensor of shape ``(2, |E|)``.
            node_to_neighbors: Optional precomputed node to neighborhood mapping. If None, it will be computed from ``hyperedge_index``.

        Returns:
            A 1-D tensor of shape ``(num_hyperedges,)`` with the CN score for each hyperedge.
        """
        if node_to_neighbors is None:
            node_to_neighbors = Hypergraph.from_hyperedge_index(hyperedge_index).neighbors_of_all()

        scores: List[float] = []
        hyperedge_index_wrapper = HyperedgeIndex(hyperedge_index)
        for hyperedge_id in range(hyperedge_index_wrapper.num_hyperedges):
            node_ids = hyperedge_index_wrapper.nodes_in(hyperedge_id)
            hyperedge_score = self.score(node_ids, node_to_neighbors)
            scores.append(hyperedge_score)

        return torch.tensor(scores, dtype=torch.float32, device=hyperedge_index.device)

    def __to_score_by_aggregation(self, pairwise_counts: List[int]) -> float:
        if not pairwise_counts:
            return self.__DEFAULT_SCORE

        match self.aggregation:
            case Aggregation.MEAN:
                return sum(pairwise_counts) / len(pairwise_counts)
            case Aggregation.MIN:
                return float(min(pairwise_counts))
            case Aggregation.SUM:
                return float(sum(pairwise_counts))

        # Fallback, should never reach here due to validation in __init__()
        return self.__DEFAULT_SCORE
