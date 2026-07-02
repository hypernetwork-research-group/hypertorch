import torch

from abc import ABC, abstractmethod
from torch import Tensor
from typing import Literal
from hypertorch.types import Neighborhood, Hypergraph, HyperedgeIndex


class NeighborScorer(ABC):
    """
    Abstract base class for neighbor scorers.
    """

    @abstractmethod
    def score(
        self,
        candidate_nodes: list[int],
        candidate_to_neighbors: dict[int, Neighborhood],
    ) -> float:
        """
        Score a single candidate hyperedge.

        Args:
            candidate_nodes: Node IDs in the candidate hyperedge.
            candidate_to_neighbors: Mapping from node IDs to their neighborhoods.

        Returns:
            score: Candidate score.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError

    @abstractmethod
    def score_batch(
        self,
        candidate_nodes: Tensor,
        hyperedge_index: Tensor | None = None,
        node_to_neighbors: dict[int, Neighborhood] | None = None,
    ) -> Tensor:
        """
        Score a batch of hyperedges or nodes.

        Args:
            candidate_nodes: Tensor containing node IDs to score of shape ``(num_nodes,)``.
            hyperedge_index: Optional tensor containing the hyperedge index. Defaults to ``None``.
            node_to_neighbors: Optional precomputed node-neighborhood mapping. Defaults to ``None``.

        Returns:
            scores: Score tensor with per-hyperedge or per-node scores.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError


class CommonNeighborsHyperedgeScorer(NeighborScorer):
    """
    Scorer for computing the Common Neighbors (CN) score for hyperedges.

    Attributes:
        aggregation: Method to aggregate node embeddings per hyperedge. Can be one of
            ``"mean"``, ``"min"``, or ``"sum"``.
    """

    __DEFAULT_SCORE = 0.0

    def __init__(self, aggregation: Literal["mean", "min", "sum"]) -> None:
        """
        Initialize the common-neighbors scorer.

        Args:
            aggregation: Method used to aggregate pairwise common-neighbor counts.
        """
        self.aggregation: Literal["mean", "min", "sum"] = aggregation

    def score(
        self,
        candidate_nodes: list[int],
        candidate_to_neighbors: dict[int, Neighborhood],
    ) -> float:
        """
        Compute the CN score for a single candidate hyperedge.

        Args:
            candidate_nodes: List of node IDs forming the candidate hyperedge.
                If less than 2 nodes are provided, the function returns a default score of ``0.0``.
            candidate_to_neighbors: Mapping from node IDs to their set of neighbors.

        Returns:
            score: The aggregated common neighbors score.
        """
        if len(candidate_nodes) < 2:
            return self.__DEFAULT_SCORE

        pairwise_counts: list[int] = []
        candidates_tensor = torch.tensor(candidate_nodes, dtype=torch.long)

        # Example: candidate_nodes = [1, 2, 3]
        #          -> compute common neighbors for pairs (1, 2), (1, 3), and (2, 3)
        for u, v in torch.combinations(candidates_tensor, 2):
            neighbors_u: Neighborhood = candidate_to_neighbors.get(u.item(), set())
            neighbors_v: Neighborhood = candidate_to_neighbors.get(v.item(), set())

            common_neighbors = neighbors_u & neighbors_v
            pairwise_counts.append(len(common_neighbors))

        return _to_score_by_aggregation(
            pairwise_counts=pairwise_counts,
            aggregation=self.aggregation,
            default_score=self.__DEFAULT_SCORE,
        )

    def score_batch(
        self,
        candidate_nodes: Tensor,
        hyperedge_index: Tensor | None = None,
        node_to_neighbors: dict[int, Neighborhood] | None = None,
    ) -> Tensor:
        """
        Score a batch of hyperedges.

        Args:
            candidate_nodes: Tensor containing gloal node IDs so that nodes in the batch
                can be scored by mapping them to the ID space of the training hypergraph.
                The tensor shape is ``(num_nodes,)``.
            hyperedge_index: Tensor containing the hyperedge index. Defaults to ``None``.
            node_to_neighbors: Optional precomputed node to neighborhood mapping.
                Defaults to ``None``.
            node_to_neighbors: Optional precomputed node to neighborhood mapping.
                If ``None``, it will be computed from ``data``.
                Defaults to ``None``.

        Returns:
            scores: A 1-D tensor of shape ``(num_hyperedges,)`` with the CN score
                or each hyperedge.
        """
        if hyperedge_index is None:
            raise ValueError("'hyperedge_index' must be provided.")

        hyperedge_index_wrapper = HyperedgeIndex(hyperedge_index).to_global(candidate_nodes)

        if node_to_neighbors is None:
            node_to_neighbors = Hypergraph.from_hyperedge_index(
                hyperedge_index=hyperedge_index_wrapper.item,
            ).neighbors_of_all()

        scores: list[float] = []
        for hyperedge_id in range(hyperedge_index_wrapper.num_hyperedges):
            node_ids = hyperedge_index_wrapper.nodes_in(hyperedge_id)
            hyperedge_score = self.score(node_ids, node_to_neighbors)
            scores.append(hyperedge_score)

        return torch.tensor(scores, dtype=torch.float32, device=hyperedge_index.device)


class CommonNeighborsNodeScorer(NeighborScorer):
    """
    Scorer for computing the Common Neighbors (CN) score for nodes.

    Attributes:
        num_classes: Number of node classes to score.
        class_to_node_ids: Mapping from class IDs to labeled reference node IDs.
        aggregation: Method used to aggregate pairwise node-reference scores per class.
        exclude_self_reference: Whether a node should ignore itself when it also
            appears among labeled reference nodes.
    """

    __DEFAULT_SCORE = 0.0

    def __init__(
        self,
        num_classes: int,
        class_to_node_ids: dict[int, list[int]] | None = None,
        aggregation: Literal["mean", "min", "sum"] = "sum",
        exclude_self_reference: bool = True,
    ) -> None:
        """
        Initialize the common-neighbors node scorer.

        Args:
            num_classes: Number of node classes to score.
            class_to_node_ids: Mapping from class IDs to labeled reference node IDs.
                Defaults to an empty mapping for each class.
            aggregation: Method used to aggregate pairwise node-reference scores
                per class. Defaults to ``"sum"``.
            exclude_self_reference: Whether a node should ignore itself when it also
                appears among labeled reference nodes. Defaults to ``True``.
        """
        self.num_classes: int = num_classes
        self.class_to_node_ids: dict[int, list[int]] = (
            class_to_node_ids
            if class_to_node_ids is not None
            else {class_id: [] for class_id in range(num_classes)}
        )
        self.aggregation: Literal["mean", "min", "sum"] = aggregation
        self.exclude_self_reference: bool = exclude_self_reference

    def score(
        self,
        candidate_nodes: list[int],
        candidate_to_neighbors: dict[int, Neighborhood],
    ) -> float:
        """
        Score one node against reference nodes from a single class.

        The first item in ``candidate_nodes`` is the node being classified.
        Remaining items are labeled reference nodes for one class.

        Args:
            candidate_nodes: Node IDs containing one target node followed by
                reference nodes for one class.
            candidate_to_neighbors: Mapping from node IDs to their neighborhoods.

        Returns:
            score: Aggregated node-class score.
        """
        if len(candidate_nodes) < 2:
            return self.__DEFAULT_SCORE

        node_id = candidate_nodes[0]
        reference_node_ids = candidate_nodes[1:]
        return self.__score_node_class(
            node_id=node_id,
            reference_node_ids=reference_node_ids,
            node_to_neighbors=candidate_to_neighbors,
        )

    def score_batch(
        self,
        candidate_nodes: Tensor,
        hyperedge_index: Tensor | None = None,
        node_to_neighbors: dict[int, Neighborhood] | None = None,
    ) -> Tensor:
        """
        Score nodes against labeled reference nodes grouped by class.
        A reference node is a training node labeled with a class ID, and the
        score of a node for each class is computed by aggregating the
        pairwise common-neighbor counts between the node being
        scored and each reference node for a class.

        Args:
            candidate_nodes: Tensor containing node IDs to score of shape ``(num_nodes,)``.
            hyperedge_index: Tensor of shape ``(2, num_hyperedges)`` containing the hyperedge index.
                Defaults to ``None``. If provided, it raises. a ValueError since the hyperedge index
                is not used in this scorer.
            node_to_neighbors: Mapping from node IDs to their training-world neighborhoods.

        Returns:
            scores: Raw node-class scores of shape ``(num_nodes, num_classes)``.

        Raises:
            ValueError: If ``hyperedge_index`` is provided, since it is not used in this scorer.
        """
        if hyperedge_index is not None:
            raise ValueError(
                "'hyperedge_index' must not be provided, as it is not used in this scorer."
            )

        node_to_neighbors = node_to_neighbors if node_to_neighbors is not None else {}

        # Convert the 1-D node tensor into Python IDs so each output row follows
        # the same node order as the input.
        # Example: data = tensor([3, 0, 2])
        #          -> data.tolist() = [3, 0, 2]
        #          -> output rows score node 3, then node 0, then node 2.
        node_ids = candidate_nodes.tolist()

        # Materialize the nested Python score matrix as a floating tensor.
        #                   class 0    1
        # Example: raw scores = [[0.0, 2.0],  node 0
        #                        [1.0, 0.0]]  node 1
        #          -> tensor shape = (2, 2), one row per node and one column per class.
        return torch.tensor(
            [
                [
                    # Score one node against all reference nodes for one class
                    # Example: node_id = 0, class_id = 1
                    #          -> reference_node_ids = [1, 2]
                    #          -> score = CN(node 0, reference nodes [1, 2])
                    #          -> output row = [CN(node 0, class 0), CN(node 0, class 1)]
                    self.__score_node_class(
                        node_id=int(node_id),
                        reference_node_ids=self.class_to_node_ids.get(class_id, []),
                        node_to_neighbors=node_to_neighbors,
                    )
                    for class_id in range(self.num_classes)
                ]
                for node_id in node_ids  # Score each node in the input tensor
            ],
            dtype=torch.float32,
            device=candidate_nodes.device,
        )

    def __score_node_class(
        self,
        node_id: int,
        reference_node_ids: list[int],
        node_to_neighbors: dict[int, Neighborhood],
    ) -> float:
        """
        Score one node against labeled reference nodes for a single class.
        The score is computed by aggregating the pairwise common-neighbor
        counts between the node being scored and each reference node.

        Args:
            node_id: Node ID to classify.
            reference_node_ids: Labeled reference node IDs for one class.
            node_to_neighbors: Mapping from node IDs to their neighborhoods.

        Returns:
            score: Aggregated node-class score.
        """
        pairwise_counts: list[int] = [
            self.__score_pair(node_id, reference_node_id, node_to_neighbors)
            for reference_node_id in reference_node_ids
            # If no self-reference is allowed, skip the reference node
            # if it is the same as the node being scored.
            if not self.exclude_self_reference or reference_node_id != node_id
        ]
        return _to_score_by_aggregation(
            pairwise_counts=pairwise_counts,
            aggregation=self.aggregation,
            default_score=self.__DEFAULT_SCORE,
        )

    def __score_pair(
        self,
        node_id: int,
        reference_node_id: int,
        node_to_neighbors: dict[int, Neighborhood],
    ) -> int:
        """
        Compute the pairwise common-neighbor count for two nodes.
        The count is the number of nodes that are neighbors of
        both the node being scored and the reference node.

        Args:
            node_id: Node ID to classify.
            reference_node_id: Labeled reference node ID for one class.
            node_to_neighbors: Mapping from node IDs to their neighborhoods.

        Returns:
            score: Pairwise common-neighbor count.
        """
        node_neighbors = node_to_neighbors.get(node_id, set())
        reference_neighbors = node_to_neighbors.get(reference_node_id, set())
        score = len(node_neighbors & reference_neighbors)
        return score


def _to_score_by_aggregation(
    pairwise_counts: list[int],
    aggregation: str,
    default_score: float = 0.0,
) -> float:
    """
    Aggregate pairwise common-neighbor counts into a score.

    Args:
        pairwise_counts: Pairwise common-neighbor counts.
        aggregation: Method used to aggregate pairwise counts.
        default_score: Default score to return if no pairwise counts are available.
            Defaults to ``0.0``.

    Returns:
        score: Aggregated common-neighbors score.
    """
    score = default_score
    if len(pairwise_counts) < 1:
        return score

    match aggregation:
        case "mean":
            score = sum(pairwise_counts) / len(pairwise_counts)
        case "min":
            score = float(min(pairwise_counts))
        case "sum":
            score = float(sum(pairwise_counts))

    return score
