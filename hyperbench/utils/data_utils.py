import torch

from torch import Tensor
from typing import Optional


def empty_nodefeatures() -> Tensor:
    return torch.empty((0, 0))


def empty_hyperedgeindex() -> Tensor:
    return torch.empty((2, 0))


def empty_edgeattr(num_edges: int) -> Tensor:
    return torch.empty((num_edges, 0))


def to_non_empty_edgeattr(edge_attr: Tensor | None) -> Tensor:
    num_edges = edge_attr.size(0) if edge_attr is not None else 0
    return empty_edgeattr(num_edges) if edge_attr is None else edge_attr


def to_0based_ids(original_ids: Tensor, ids_to_rebase: Optional[Tensor] = None) -> Tensor:
    """
    Remap IDs to contiguous 0-based indices.

    If ``ids_to_rebase`` is provided, only IDs present in it are kept and remapped.
    If ``ids_to_rebase`` is not provided, all unique IDs in ``original_ids`` are remapped.

    Examples:
        >>> to_0based_ids(torch.tensor([1, 3, 3, 7]), torch.tensor([3, 7]))
        ... -> tensor([0, 0, 1])  # 1 is excluded, 3 -> 0, 7 -> 1

        >>> to_0based_ids(torch.tensor([5, 3, 5, 8]))
        ... -> tensor([1, 0, 1, 2])  # 3 -> 0, 5 -> 1, 8 -> 2

    Args:
        original_ids: Tensor of original IDs.
        ids_to_rebase: Optional tensor of IDs to keep and remap. If None, all unique IDs are used.

    Returns:
        Tensor of 0-based IDs.
    """
    if ids_to_rebase is None:
        sorted_unique_original_ids = original_ids.unique(sorted=True)
        return torch.searchsorted(sorted_unique_original_ids, original_ids)

    keep_mask = torch.isin(original_ids, ids_to_rebase)
    ids_to_keep = original_ids[keep_mask]
    sorted_unique_ids_to_rebase = ids_to_rebase.unique(sorted=True)
    return torch.searchsorted(sorted_unique_ids_to_rebase, ids_to_keep)
