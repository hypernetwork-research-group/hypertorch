import torch

from abc import ABC, abstractmethod
from enum import Enum
from torch import Tensor
from typing import List
from hyperbench.types import HData


class SamplingStrategy(Enum):
    NODE = "node"
    HYPEREDGE = "hyperedge"


class BaseSampler(ABC):
    @abstractmethod
    def sample(self, index: int | List[int], hdata: HData) -> HData:
        """
        Sample a sub-hypergraph and return HData with global IDs.

        Args:
            index: An integer or list of integers specifying which items to sample.
            hdata: The original HData to sample from.

        Returns:
            A new HData instance containing only the sampled items and their associated data.
        """
        raise NotImplementedError("Subclasses must implement the sample method.")

    @abstractmethod
    def len(self, hdata: HData) -> int:
        """
        Return the number of sampleable items (nodes or hyperedges).

        Args:
            hdata: The HData to query for the number of sampleable items.
        """
        raise NotImplementedError("Subclasses must implement the len method.")

    def _normalize_index(self, index: int | List[int], size: int) -> List[int]:
        """
        Convert index to list, deduplicate, validate length.

        Args:
            index: An integer or a list of integers representing IDs to sample.
            size: The total number of sampleable items (e.g., nodes or hyperedges) for validation.

        Returns:
            List of IDs to sample.

        Raises:
            ValueError: If the provided index is invalid (e.g., empty list or list length exceeds number of sampleable items).
        """
        if isinstance(index, list):
            if len(index) < 1:
                raise ValueError("Index list cannot be empty.")
            if len(index) > size:
                raise ValueError(
                    f"Index list length ({len(index)}) cannot exceed the number of sampleable items ({size})."
                )
            return list(set(index))
        return [index]

    def _sample_hyperedge_index(
        self,
        hyperedge_index: Tensor,
        sampled_hyperedge_ids: Tensor,
    ) -> Tensor:
        """
        Sample the hyperedge index to keep only incidences belonging to the specified sampled hyperedge IDs.

        Args:
            hyperedge_index: The original hyperedge index tensor of shape ``[2, num_incidences]``.
            sampled_hyperedge_ids: A tensor containing the IDs of hyperedges to sample.

        Returns:
            A new hyperedge index tensor containing only the incidences of the sampled hyperedges.
        """
        hyperedge_ids = hyperedge_index[1]

        # Find incidences where the hyperedge is in our sampled hyperedges
        # Example: hyperedge_ids = [0, 0, 0, 1, 2, 2], sampled_hyperedge_ids = [0, 2]
        #          -> sampled_hyperedges_mask = [True, True, True, False, True, True]
        sampled_hyperedges_mask = torch.isin(hyperedge_ids, sampled_hyperedge_ids)

        # Keep all incidences belonging to the sampled hyperedges
        # Example: hyperedge_index = [[0, 0, 1, 2, 3, 4],
        #                             [0, 0, 0, 1, 2, 2]],
        #          sampled_hyperedges_mask = [True, True, True, False, True, True]
        #          -> sampled_hyperedge_index = [[0, 0, 1, 3, 4],
        #                                        [0, 0, 0, 2, 2]]
        sampled_hyperedge_index = hyperedge_index[:, sampled_hyperedges_mask]
        return sampled_hyperedge_index

    def _validate_bounds(self, ids: List[int], size: int, label: str) -> None:
        """
        Check all IDs are in [0, self.len).

        Args:
            ids: List of IDs to validate.
            size: The total number of sampleable items (e.g., nodes or hyperedges).
            label: A string label for error messages (e.g., "Node ID" or "Hyperedge ID").

        Raises:
            IndexError: If any ID is out of bounds.
        """
        for id in ids:
            if id < 0 or id >= size:
                raise IndexError(f"{label} {id} is out of bounds (0, {size - 1}).")


class HyperedgeSampler(BaseSampler):
    def sample(self, index: int | List[int], hdata: HData) -> HData:
        """
        Sample hyperedges by their IDs and return the sub-hypergraph containing only those hyperedges and their incident nodes.

        Examples:
        >>> hyperedge_index = [[0, 0, 1, 2, 3, 4],
        ...                    [0, 0, 0, 1, 2, 2]]
        >>> hdata = HData.from_hyperedge_index(hyperedge_index)
        >>> strategy = HyperedgeSampler()
        >>> sampled_hdata = strategy.sample([0, 2], hdata)
        >>> sampled_hdata.hyperedge_index
        >>> tensor([[0, 0, 1, 3, 4],
        ...         [0, 0, 0, 2, 2]])

        Args:
            index: An integer or a list of integers representing hyperedge IDs to sample.
            hdata: The original HData to sample from.

        Returns:
            An HData instance containing only the sampled hyperedges and their incident nodes.

        Raises:
            ValueError: If the provided index is invalid (e.g., empty list or list length exceeds number of hyperedges).
            IndexError: If any hyperedge ID is out of bounds.
        """
        ids = self._normalize_index(index, self.len(hdata))
        self._validate_bounds(ids, self.len(hdata), "Hyperedge ID")

        hyperedge_index = hdata.hyperedge_index

        sampled_hyperedge_ids = torch.tensor(ids, device=hyperedge_index.device)

        # Example: sampled_hyperedge_ids = [0, 2],
        #          hyperedge_index = [[0, 0, 1, 2, 3, 4],
        #                             [0, 0, 0, 1, 2, 2]],
        #          -> sampled_hyperedges_mask = [True, True, True, False, True, True]
        #          -> sampled_hyperedge_index = [[0, 0, 1, 3, 4],
        #                                        [0, 0, 0, 2, 2]]
        sampled_hyperedge_index = self._sample_hyperedge_index(
            hyperedge_index, sampled_hyperedge_ids
        )

        return HData.from_hyperedge_index(sampled_hyperedge_index)

    def len(self, hdata: HData) -> int:
        """
        Return the number of hyperedges in the given HData.

        Args:
            hdata: The HData to query for the number of hyperedges.

        Returns:
            The number of hyperedges in the HData.
        """
        return hdata.num_hyperedges


class NodeSampler(BaseSampler):
    def sample(self, index: int | List[int], hdata: HData) -> HData:
        """
        Sample nodes by their IDs and return the sub-hypergraph containing only those nodes and their incident hyperedges.

        Examples:
        >>> hyperedge_index = [[0, 0, 1, 2, 3, 4],
        ...                    [0, 0, 0, 1, 2, 2]]
        >>> hdata = HData.from_hyperedge_index(hyperedge_index)
        >>> strategy = NodeSampler()
        >>> sampled_hdata = strategy.sample([0, 3], hdata)
        >>> sampled_hdata.hyperedge_index
        >>> tensor([[0, 0, 1, 3, 4],
        ...         [0, 0, 0, 2, 2]])

        Args:
            index: An integer or a list of integers representing node IDs to sample.
            hdata: The original HData to sample from.

        Returns:
            An HData instance containing only the sampled nodes and their incident hyperedges.

        Raises:
            ValueError: If the provided index is invalid (e.g., empty list or list length exceeds number of nodes).
            IndexError: If any node ID is out of bounds.
        """
        ids = self._normalize_index(index, self.len(hdata))
        self._validate_bounds(ids, self.len(hdata), "Node ID")

        hyperedge_index = hdata.hyperedge_index
        node_ids = hyperedge_index[0]
        hyperedge_ids = hyperedge_index[1]

        sampled_node_ids = torch.tensor(ids, device=node_ids.device)

        # Find incidences where the node is in our sampled nodes
        # Example: node_ids = [0, 0, 1, 2, 3, 4], sampled_node_ids = [0, 3]
        #          -> sampled_nodes_mask = [True, True, False, False, True, False]
        sampled_nodes_mask = torch.isin(node_ids, sampled_node_ids)

        # Get unique hyperedges that have at least one sampled node
        # Example: hyperedge_ids = [0, 0, 0, 1, 2, 2], sampled_nodes_mask = [True, True, False, False, True, False]
        #          -> sampled_hyperedge_ids = [0, 2] as they connect to sampled nodes
        sampled_hyperedge_ids = hyperedge_ids[sampled_nodes_mask].unique()

        # Example: sampled_hyperedge_ids = [0, 2],
        #          hyperedge_index = [[0, 0, 1, 2, 3, 4],
        #                             [0, 0, 0, 1, 2, 2]],
        #          -> sampled_hyperedges_mask = [True, True, True, False, True, True]
        #          -> sampled_hyperedge_index = [[0, 0, 1, 3, 4],
        #                                        [0, 0, 0, 2, 2]]
        sampled_hyperedge_index = self._sample_hyperedge_index(
            hyperedge_index, sampled_hyperedge_ids
        )

        return HData.from_hyperedge_index(sampled_hyperedge_index)

    def len(self, hdata: HData) -> int:
        """
        Return the number of nodes in the given HData.

        Args:
            hdata: The HData to query for the number of nodes.

        Returns:
            The number of nodes in the HData.
        """
        return hdata.num_nodes


def create_sampler_from_strategy(strategy: SamplingStrategy) -> BaseSampler:
    """
    Factory function to create a sampler instance based on the provided sampling strategy type.

    Args:
        strategy: An instance of SamplingStrategy enum indicating which sampling strategy to use.

    Returns:
        An instance of a subclass of BaseSampler corresponding to the specified strategy. If strategy is not recognized, defaults to ``HyperedgeSampler``.
    """
    match strategy:
        case SamplingStrategy.NODE:
            return NodeSampler()
        case _:
            return HyperedgeSampler()
