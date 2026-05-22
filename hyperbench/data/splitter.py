import torch

from abc import ABC, abstractmethod
from typing import cast
from torch import Tensor
from hyperbench.types import HData


class Splitter(ABC):
    @abstractmethod
    def split(self, to_split: Tensor, ratios: list[float]) -> tuple[list[Tensor], list[float]]:
        """
        Split the input tensor into multiple tensors according to the provided ratios.
        The output tensors are not guaranteed to be non-empty, so downstream validation may be necessary.

        Args:
            to_split: The tensor to split. For example, a list of hyperedge IDs.
            ratios: The ratios for the splits. For example, ``[0.7, 0.1, 0.2]`` for a 70/10/20 split.

        Returns:
            (Values per split, ratios after splitting): A tuple of (list of split tensors, list of actual ratios achieved by the splits).
        """
        pass


class HyperedgeIDSplitter(Splitter):
    def __init__(self, hdata: HData) -> None:
        self.hdata = hdata

    def ensure_split_covers_all_nodes(
        self,
        hyperedge_ids_by_split: list[Tensor],
        split_idx: int = 0,
    ) -> tuple[list[Tensor], list[float]]:
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
        device = self.hdata.device
        num_hyperedges = self.hdata.num_hyperedges

        # Shuffle hyperedge IDs if shuffle is requested, otherwise keep original order for deterministic splits
        if shuffle:
            generator = torch.Generator(device=device)
            if seed is not None:
                generator.manual_seed(seed)

            random_hyperedge_ids_permutation = torch.randperm(
                n=num_hyperedges,
                generator=generator,
                device=device,
            )
            return random_hyperedge_ids_permutation

        ranged_hyperedge_ids_permutation = torch.arange(num_hyperedges, device=device)
        return ranged_hyperedge_ids_permutation

    def get_split_ratios(self, hyperedge_ids_by_split: list[Tensor]) -> list[float]:
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
        covered_node_ids = self.__nodes_covered_by_hyperedges(hyperedge_ids)
        covered_node_ids_mask = torch.isin(required_node_ids, covered_node_ids)
        not_covered_node_ids_mask = torch.logical_not(covered_node_ids_mask)
        return required_node_ids[not_covered_node_ids_mask]

    def __nodes_covered_by_hyperedges(self, hyperedge_ids: Tensor) -> Tensor:
        all_hyperedge_ids = self.hdata.hyperedge_index[1]
        nodes_in_input_hyperedges_mask = torch.isin(all_hyperedge_ids, hyperedge_ids)
        all_node_ids = self.hdata.hyperedge_index[0]
        return all_node_ids[nodes_in_input_hyperedges_mask].unique()

    def __pick_covering_hyperedge(
        self,
        hyperedge_ids_by_split: list[Tensor],
        missing_node_ids: Tensor,
    ) -> tuple[int, Tensor]:
        best_gain = 0
        best_split_idx: int | None = None
        best_hyperedge_id: Tensor | None = None

        for split_idx, split_hyperedge_ids in enumerate(
            hyperedge_ids_by_split[1:],
            start=1,
        ):
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
