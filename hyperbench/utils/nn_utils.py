import torch

from enum import Enum
from typing import TypeAlias
from torch.nn import Module
from torch import Tensor
from torch_geometric.utils import scatter


INPUT_LAYER = 0


ActivationFn: TypeAlias = type[Module]
NormalizationFn: TypeAlias = type[Module]


class Stage(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


def is_layer(layer_idx: int, desired_layer: int) -> bool:
    return layer_idx == desired_layer


def is_input_layer(layer_idx: int) -> bool:
    return is_layer(layer_idx, INPUT_LAYER)


def maxmin_scatter(
    src: Tensor,
    index: Tensor,
    dim: int,
    dim_size: int | None = None,
) -> Tensor:
    """
    Performs a scatter reduction that computes the channel-wise range (max - min) for each index group.

    Args:
        src: The source tensor containing the values to scatter.
        index: The indices of elements to scatter.
        dim: The axis along which to index.
        dim_size: The size of the output tensor along the scatter dimension.
            If not provided, it will be inferred from the maximum index value.

    Returns:
        values: A tensor containing the max-min values for each index group.
    """
    max_embeddings = scatter(src=src, index=index, dim=dim, dim_size=dim_size, reduce="max")
    min_embeddings = scatter(src=src, index=index, dim=dim, dim_size=dim_size, reduce="min")
    return max_embeddings - min_embeddings


def validate_floating_tensor_dtype(name: str, tensor: Tensor) -> None:
    if not tensor.is_floating_point():
        raise ValueError(f"{name!r} must have a floating-point dtype, got {tensor.dtype}.")


def validate_long_tensor_dtype(name: str, tensor: Tensor) -> None:
    if tensor.dtype != torch.long:
        raise ValueError(f"{name!r} must have dtype torch.long, got {tensor.dtype}.")
