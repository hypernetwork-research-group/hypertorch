import torch

from abc import ABC, abstractmethod
from typing import cast
from torch import Tensor
from hyperbench.types import HData
from hyperbench.utils import (
    create_seeded_torch_generator,
    validate_is_non_empty,
    validate_ratios,
)


class Splitter(ABC):
    """
    Abstract base class for splitters.
    """

    @abstractmethod
    def split(self, to_split: Tensor, ratios: list[float]) -> tuple[list[Tensor], list[float]]:
        """
        Split the input tensor into multiple tensors according to the provided ratios.

        The output tensors are not guaranteed to be non-empty, so downstream validation may
        be necessary.

        Args:
            to_split: The tensor to split. For example, a list of hyperedge IDs.
            ratios: The ratios for the splits. For example, ``[0.7, 0.1, 0.2]`` for
                a 70/10/20 split.

        Returns:
            (Values per split, ratios after splitting): A tuple of (list of split tensors, list
                of actual ratios achieved by the splits).

        """
        pass


class HyperedgeIDSplitter(Splitter):
    """
    Initialize a splitter for hyperedge-ID based dataset partitioning.

    Args:
        hdata: Hypergraph data whose hyperedges and node coverage drive the split logic.

    """

    def __init__(self, hdata: HData) -> None:
        self.hdata = hdata

    def ensure_split_covers_all_nodes(
        self,
        hyperedge_ids_by_split: list[Tensor],
        split_idx: int = 0,
    ) -> tuple[list[Tensor], list[float]]:
        """
        Rebalance a split until its hyperedges cover every node in the hypergraph.

        Hyperedges are moved from the other splits into the target split, always
        choosing the donor hyperedge that covers the largest number of currently missing nodes.

        Args:
            hyperedge_ids_by_split: Hyperedge IDs assigned to each split.
            split_idx: Index of the split that must cover the full node space.

        Returns:
            hyperedge_ids_by_split: The updated hyperedge IDs for each split.
            ratios: The final ratios of hyperedges in each split after rebalancing.

        Raises:
            ValueError: If one or more nodes do not appear in any hyperedge of the
                source hypergraph.

        """
        validate_is_non_empty("hyperedge_ids_by_split", hyperedge_ids_by_split)
        if split_idx < 0 or split_idx >= len(hyperedge_ids_by_split):
            raise ValueError(
                f"split_idx must reference an existing split, got {split_idx} "
                f"for {len(hyperedge_ids_by_split)} splits."
            )

        required_node_ids = torch.arange(self.hdata.num_nodes, device=self.hdata.device)
        available_node_ids = self.hdata.hyperedge_index[0].unique()
        missing_from_hypergraph_mask = torch.logical_not(
            torch.isin(required_node_ids, available_node_ids)
        )
        if bool(missing_from_hypergraph_mask.any()):
            missing_node_ids = required_node_ids[missing_from_hypergraph_mask].tolist()
            raise ValueError(
                "Cannot create a transductive first split covering all nodes because "
                f"these node ids do not appear in any hyperedge: {missing_node_ids}."
            )

        missing_node_ids = self.__missing_node_ids(
            hyperedge_ids=hyperedge_ids_by_split[split_idx],
            required_node_ids=required_node_ids,
        )
        while missing_node_ids.numel() > 0:
            donor_split_idx, hyperedge_id = self.__pick_covering_hyperedge(
                hyperedge_ids_by_split=hyperedge_ids_by_split,
                missing_node_ids=missing_node_ids,
                split_to_cover_idx=split_idx,
            )
            hyperedge_ids_by_split[split_idx] = torch.cat(
                [hyperedge_ids_by_split[split_idx], hyperedge_id.view(1)]
            )
            donor_hyperedge_ids = hyperedge_ids_by_split[donor_split_idx]
            hyperedge_ids_by_split[donor_split_idx] = donor_hyperedge_ids[
                donor_hyperedge_ids != hyperedge_id
            ]
            missing_node_ids = self.__missing_node_ids(
                hyperedge_ids=hyperedge_ids_by_split[split_idx],
                required_node_ids=required_node_ids,
            )

        return hyperedge_ids_by_split, self.get_split_ratios(hyperedge_ids_by_split)

    def validate_splits_have_hyperedges(self, hyperedge_ids_by_split: list[Tensor]) -> None:
        """
        Validate that every split retains at least one hyperedge.

        Args:
            hyperedge_ids_by_split: Hyperedge IDs assigned to each split.

        Raises:
            ValueError: If any split is empty after splitting or rebalancing.

        """
        empty_split_indices = [
            split_idx
            for split_idx, split_hyperedge_ids in enumerate(hyperedge_ids_by_split)
            if split_hyperedge_ids.numel() == 0
        ]
        if empty_split_indices:
            final_ratios = self.get_split_ratios(hyperedge_ids_by_split)
            raise ValueError(
                f"Cannot create dataset splits because splits {empty_split_indices} "
                f"contain no hyperedges. Final ratios: {final_ratios}."
            )

    def get_hyperedge_ids_permutation(self, shuffle: bool | None, seed: int | None) -> Tensor:
        """
        Return hyperedge IDs in deterministic or shuffled order.

        Args:
            shuffle: Whether to randomly permute the hyperedge IDs.
            seed: Optional random seed used when ``shuffle`` is truthy.

        Returns:
            hyperedge_ids_permutation: Ordered or shuffled hyperedge IDs on the HData device.

        """
        device = self.hdata.device
        num_hyperedges = self.hdata.num_hyperedges

        # Shuffle hyperedge IDs if shuffle is requested, otherwise keep original order
        # for deterministic splits
        if shuffle:
            generator = create_seeded_torch_generator(device=device, seed=seed)

            random_hyperedge_ids_permutation = torch.randperm(
                n=num_hyperedges,
                generator=generator,
                device=device,
            )
            return random_hyperedge_ids_permutation

        ranged_hyperedge_ids_permutation = torch.arange(num_hyperedges, device=device)
        return ranged_hyperedge_ids_permutation

    def get_split_ratios(self, hyperedge_ids_by_split: list[Tensor]) -> list[float]:
        """
        Compute realized split ratios from hyperedge counts.

        Args:
            hyperedge_ids_by_split: Hyperedge IDs assigned to each split.

        Returns:
            ratios: Ratios derived from the number of hyperedges in each split.

        """
        num_hyperedges_by_split = [
            int(split_hyperedge_ids.numel()) for split_hyperedge_ids in hyperedge_ids_by_split
        ]
        num_hyperedges = sum(num_hyperedges_by_split)
        if num_hyperedges == 0:
            return [0.0 for _ in hyperedge_ids_by_split]

        return [
            round(split_num_hyperedges / num_hyperedges, 4)
            for split_num_hyperedges in num_hyperedges_by_split
        ]

    def split(self, to_split: Tensor, ratios: list[float]) -> tuple[list[Tensor], list[float]]:
        """
        Split hyperedge IDs by cumulative ratio boundaries.

        Early splits use cumulative floor boundaries to avoid over-consuming hyperedges.
        The final split receives any remaining hyperedges caused by rounding.

        Args:
            to_split: Hyperedge IDs to partition.
            ratios: Requested split ratios.

        Returns:
            hyperedge_ids_by_split: The updated hyperedge IDs for each split.
            ratios: The final ratios of hyperedges in each split after rebalancing.

        """
        validate_ratios(ratios)

        # Cumulative floor boundaries keep early splits from over-consuming hyperedges.
        # The last split absorbs any rounding remainder.
        num_hyperedges = int(to_split.size(0))

        start = 0
        cumulative_ratio = 0.0
        hyperedge_ids_by_split = []
        for split_idx, ratio in enumerate(ratios):
            cumulative_ratio += ratio
            end = (
                num_hyperedges
                if split_idx == len(ratios) - 1
                else int(cumulative_ratio * num_hyperedges)
            )
            hyperedge_ids_by_split.append(to_split[start:end])
            start = end

        return hyperedge_ids_by_split, self.get_split_ratios(hyperedge_ids_by_split)

    def __missing_node_ids(self, hyperedge_ids: Tensor, required_node_ids: Tensor) -> Tensor:
        """
        Return the node IDs not covered by the given hyperedges.

        Args:
            hyperedge_ids: Hyperedge IDs whose node coverage should be inspected.
            required_node_ids: Node IDs that must be covered.

        Returns:
            missing_node_ids: Required node IDs that are still uncovered.

        """
        covered_node_ids = self.__nodes_covered_by_hyperedges(hyperedge_ids)
        covered_node_ids_mask = torch.isin(required_node_ids, covered_node_ids)
        not_covered_node_ids_mask = torch.logical_not(covered_node_ids_mask)
        return required_node_ids[not_covered_node_ids_mask]

    def __nodes_covered_by_hyperedges(self, hyperedge_ids: Tensor) -> Tensor:
        """
        Collect unique node IDs incident to the provided hyperedges.

        Args:
            hyperedge_ids: Hyperedge IDs to inspect.

        Returns:
            nodes_covered_by_hyperedge: Unique node IDs covered by the input hyperedges.

        """
        all_hyperedge_ids = self.hdata.hyperedge_index[1]
        nodes_in_input_hyperedges_mask = torch.isin(all_hyperedge_ids, hyperedge_ids)
        all_node_ids = self.hdata.hyperedge_index[0]
        return all_node_ids[nodes_in_input_hyperedges_mask].unique()

    def __pick_covering_hyperedge(
        self,
        hyperedge_ids_by_split: list[Tensor],
        missing_node_ids: Tensor,
        split_to_cover_idx: int,
    ) -> tuple[int, Tensor]:
        """
        Choose the donor hyperedge that covers the most currently missing nodes.

        Args:
            hyperedge_ids_by_split: Hyperedge IDs assigned to each split.
            missing_node_ids: Node IDs still missing from the target split.
            split_to_cover_idx: Index of the split being rebalanced.

        Returns:
            split_idx: The index of the donor split containing the selected hyperedge.
            hyperedge_id: The ID of the selected hyperedge.

        """
        best_gain = 0
        best_split_idx: int | None = None
        best_hyperedge_id: Tensor | None = None
        for split_idx, split_hyperedge_ids in enumerate(hyperedge_ids_by_split):
            if split_idx == split_to_cover_idx:
                continue

            for hyperedge_id in split_hyperedge_ids:
                covered_node_ids = self.__nodes_covered_by_hyperedges(hyperedge_id.view(1))
                missing_nodes_in_covered_nodes_mask = torch.isin(covered_node_ids, missing_node_ids)
                gain = int(missing_nodes_in_covered_nodes_mask.sum().item())
                if gain > best_gain:
                    best_split_idx = split_idx
                    best_hyperedge_id = hyperedge_id
                    best_gain = gain

        # Split construction partitions every available hyperedge, so
        # a valid HData has a covering donor whenever the first split still misses a node
        return cast(int, best_split_idx), cast(Tensor, best_hyperedge_id)
