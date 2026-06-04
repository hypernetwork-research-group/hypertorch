from __future__ import annotations

import torch

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar, cast
from torch import Tensor
from hyperbench.utils import (
    NodeSpaceSetting,
    create_seeded_torch_generator,
    is_transductive_setting,
    to_0based_ids,
    validate_is_between,
    validate_is_non_empty,
    validate_node_space_setting,
    validate_ratios,
)
from hyperbench.types.hypergraph import HyperedgeIndex

if TYPE_CHECKING:
    from hyperbench.types.hdata import HData
    from hyperbench.data.dataset import Dataset


_ToSplitType = TypeVar("_ToSplitType")
_SplitResultType = TypeVar("_SplitResultType")


class Splitter(ABC, Generic[_ToSplitType, _SplitResultType]):
    """
    Abstract base class for splitting objects into parts.
    """

    @abstractmethod
    def split(self, to_split: _ToSplitType, **kwargs) -> _SplitResultType:
        """
        Split the input object and return the split result.

        Args:
            to_split: The object to split.
            **kwargs: Additional keyword arguments that may be required by specific splitter implementations.

        Returns:
            The result of splitting the input object.
        """
        pass


class DefaultDatasetSplitter(Splitter["Dataset", tuple[list["Dataset"], list[float]]]):
    """
    Split a dataset by hyperedges and materialize dataset partitions.

    Args:
        ratios: List of floats summing to ``1.0``.
        node_space_setting: Whether to preserve full or local node spaces.
        cover_all_nodes_in_train_split: Whether transductive splits should move
            hyperedges into the first split until all nodes are incident to at
            least one selected training hyperedge.
            train_split_idx: The index of the split to treat as the train split. Defaults to ``0``,
                so the first split is the train split that gets the full node space in the
                transductive setting and is optionally rebalanced to cover all nodes.
                This is used only when ``node_space_setting=="transductive"`` and ``cover_all_nodes_in_train_split==True``,
                to determine which split should be rebalanced to cover all nodes.
                For the 'inductive' setting, splits are always returned based on the provided ratios.
        shuffle: Whether to shuffle hyperedges before splitting.
        seed: Optional random seed for reproducibility.
    """

    def __init__(
        self,
        node_space_setting: NodeSpaceSetting = "transductive",
        shuffle: bool | None = False,
        seed: int | None = None,
    ) -> None:
        self.node_space_setting = node_space_setting
        self.shuffle = shuffle
        self.seed = seed

        validate_node_space_setting(self.node_space_setting)

    def split(self, to_split: Dataset, **kwargs) -> tuple[list[Dataset], list[float]]:
        """
        Split a dataset and return materialized split datasets plus final ratios.

        Args:
            to_split: The `Dataset` to split.
            ratios: Desired split ratios, used for initial split construction and
                as a reference during rebalancing. Expected as a keyword argument.

        Returns:
            datasets_and_ratios: Split datasets and final hyperedge-count ratios.

        Raises:
            ValueError: If ratios do not sum to ``1.0``, a final split has zero
                hyperedges, or a requested transductive train-cover split cannot
                cover the full node space.
        """
        ratios: list[float] = kwargs.get("ratios", [])
        validate_ratios(ratios)

        cover_all_nodes_in_train_split: bool = kwargs.get("cover_all_nodes_in_train_split", False)

        train_split_idx: int = kwargs.get("train_split_idx", 0)
        self.__validate_train_split_idx(train_split_idx, ratios)

        hdata = to_split.hdata
        hyperedge_splitter = HyperedgeIDSplitter(
            hyperedge_index=hdata.hyperedge_index,
            num_nodes=hdata.num_nodes,
            num_hyperedges=hdata.num_hyperedges,
        )
        hyperedge_ids_permutation = hyperedge_splitter.get_hyperedge_ids_permutation(
            shuffle=self.shuffle,
            seed=self.seed,
        )
        hyperedge_ids_by_split, final_ratios = hyperedge_splitter.split(
            to_split=hyperedge_ids_permutation,
            ratios=ratios,
        )
        if is_transductive_setting(self.node_space_setting) and cover_all_nodes_in_train_split:
            hyperedge_ids_by_split, final_ratios = hyperedge_splitter.ensure_split_covers_all_nodes(
                hyperedge_ids_by_split=hyperedge_ids_by_split,
                split_idx=train_split_idx,
            )
        hyperedge_splitter.validate_splits_have_hyperedges(hyperedge_ids_by_split)

        split_datasets: list[Dataset] = []
        for split_num, split_hyperedge_ids in enumerate(hyperedge_ids_by_split):
            split_node_space_setting: NodeSpaceSetting = (
                "transductive"
                if is_transductive_setting(self.node_space_setting) and split_num == train_split_idx
                else "inductive"
            )
            split_hdata = DefaultHDataSplitter(node_space_setting=split_node_space_setting).split(
                to_split=hdata,
                split_hyperedge_ids=split_hyperedge_ids,
            )
            split_hdata = split_hdata.to(device=hdata.device)

            split_dataset = to_split.__class__(
                hdata=split_hdata,
                sampling_strategy=to_split.sampling_strategy,
            )
            split_datasets.append(split_dataset)

        return split_datasets, final_ratios

    def __validate_train_split_idx(self, train_split_idx: int, ratios: list[float]) -> None:
        if self.node_space_setting != "transductive" and train_split_idx != 0:
            raise ValueError(
                f"'train_split_idx' is only relevant when 'node_space_setting' is 'transductive', "
                f"got 'node_space_setting={self.node_space_setting}' and 'train_split_idx={train_split_idx}'."
                "For the 'inductive' setting, splits are returned based on the provided ratios."
            )
        validate_is_between("train_split_idx", train_split_idx, 0, len(ratios) - 1)


class DefaultHDataSplitter(Splitter["HData", "HData"]):
    """
    Materialize an `HData` split from explicit hyperedge IDs.

    Args:
        node_space_setting: Whether to preserve the full node space in the split.
    """

    def __init__(
        self,
        node_space_setting: NodeSpaceSetting = "transductive",
    ) -> None:
        self.node_space_setting = node_space_setting
        validate_node_space_setting(self.node_space_setting)

    def split(self, to_split: HData, **kwargs) -> HData:
        """
        Build an `HData` for a single split from the given hyperedge IDs.

        Args:
            to_split: The original `HData` containing the full hypergraph.
            split_hyperedge_ids: The hyperedge IDs that should be included in the split, expected as a keyword argument.

        Returns:
            hdata: The splitted instance with remapped node and hyperedge IDs.
        """
        split_hyperedge_ids = kwargs.get("split_hyperedge_ids", [])
        validate_is_non_empty("split_hyperedge_ids", split_hyperedge_ids)

        keep_mask = torch.isin(to_split.hyperedge_index[1], split_hyperedge_ids)
        split_hyperedge_index = to_split.hyperedge_index[:, keep_mask]
        split_unique_hyperedge_ids = split_hyperedge_index[1].unique()

        split_y = to_split.y[split_unique_hyperedge_ids]

        split_hyperedge_attr = (
            to_split.hyperedge_attr[split_unique_hyperedge_ids]
            if to_split.hyperedge_attr is not None
            else None
        )
        split_hyperedge_weights = (
            to_split.hyperedge_weights[split_unique_hyperedge_ids]
            if to_split.hyperedge_weights is not None
            else None
        )

        if is_transductive_setting(self.node_space_setting):
            split_hyperedge_index[1] = to_0based_ids(
                original_ids=split_hyperedge_index[1],
                ids_to_rebase=split_unique_hyperedge_ids,
            )
            return to_split.__class__(
                x=to_split.x.clone(),
                hyperedge_index=split_hyperedge_index,
                hyperedge_weights=split_hyperedge_weights,
                hyperedge_attr=split_hyperedge_attr,
                num_nodes=to_split.num_nodes,
                num_hyperedges=len(split_unique_hyperedge_ids),
                global_node_ids=to_split.global_node_ids.clone()
                if to_split.global_node_ids is not None
                else None,
                y=split_y,
            )

        split_unique_node_ids = split_hyperedge_index[0].unique()
        split_hyperedge_index = (
            HyperedgeIndex(split_hyperedge_index)
            .to_0based(
                node_ids_to_rebase=split_unique_node_ids,
                hyperedge_ids_to_rebase=split_unique_hyperedge_ids,
            )
            .item
        )

        return to_split.__class__(
            x=to_split.x[split_unique_node_ids],
            hyperedge_index=split_hyperedge_index.clone(),
            hyperedge_weights=split_hyperedge_weights,
            hyperedge_attr=split_hyperedge_attr,
            num_nodes=len(split_unique_node_ids),
            num_hyperedges=len(split_unique_hyperedge_ids),
            global_node_ids=to_split.global_node_ids[split_unique_node_ids],
            y=split_y,
        )


class HyperedgeIDSplitter(Splitter["Tensor", tuple[list["Tensor"], list[float]]]):
    """
    Initialize a splitter for hyperedge-ID based dataset partitioning.

    Args:
        hyperedge_index: Hypergraph incidence index whose node coverage drives the split logic.
        num_nodes: Number of nodes in the source hypergraph.
        num_hyperedges: Number of hyperedges in the source hypergraph.
    """

    def __init__(
        self,
        hyperedge_index: Tensor,
        num_nodes: int,
        num_hyperedges: int,
    ) -> None:
        self.hyperedge_index = hyperedge_index
        self.num_nodes = num_nodes
        self.num_hyperedges = num_hyperedges
        self.device = hyperedge_index.device

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
            ValueError: If one or more nodes do not appear in any hyperedge of the source hypergraph.
        """
        validate_is_non_empty("hyperedge_ids_by_split", hyperedge_ids_by_split)
        validate_is_between("split_idx", split_idx, 0, len(hyperedge_ids_by_split) - 1)

        required_node_ids = torch.arange(self.num_nodes, device=self.device)
        available_node_ids = self.hyperedge_index[0].unique()
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
        if len(empty_split_indices) > 0:
            final_ratios = self.get_split_ratios(hyperedge_ids_by_split)
            raise ValueError(
                f"Splitting produced splits {empty_split_indices} "
                f"with no hyperedges. Final ratios: {final_ratios}."
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
        # Shuffle hyperedge IDs if shuffle is requested, otherwise keep original order for deterministic splits
        if shuffle:
            generator = create_seeded_torch_generator(device=self.device, seed=seed)
            random_hyperedge_ids_permutation = torch.randperm(
                n=self.num_hyperedges,
                generator=generator,
                device=self.device,
            )
            return random_hyperedge_ids_permutation

        ranged_hyperedge_ids_permutation = torch.arange(self.num_hyperedges, device=self.device)
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

    def split(self, to_split: Tensor, **kwargs) -> tuple[list[Tensor], list[float]]:
        """
        Split hyperedge IDs by cumulative ratio boundaries.

        Early splits use cumulative floor boundaries to avoid over-consuming hyperedges.
        The final split receives any remaining hyperedges caused by rounding.

        Args:
            to_split: Hyperedge IDs to partition.
            ratios: Desired split ratios, used for initial split construction and
                as a reference during rebalancing. Expected as a keyword argument.

        Returns:
            hyperedge_ids_by_split: The updated hyperedge IDs for each split.
            ratios: The final ratios of hyperedges in each split after rebalancing.
        """
        ratios: list[float] = kwargs.get("ratios", [])
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
        all_hyperedge_ids = self.hyperedge_index[1]
        nodes_in_input_hyperedges_mask = torch.isin(all_hyperedge_ids, hyperedge_ids)
        all_node_ids = self.hyperedge_index[0]
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
