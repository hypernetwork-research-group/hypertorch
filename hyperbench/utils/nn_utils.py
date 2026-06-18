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
    """
    Training stage labels.
    """

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


def is_layer(layer_idx: int, desired_layer: int) -> bool:
    """
    Check whether a layer index matches a desired index.

    Args:
        layer_idx: Layer index to inspect.
        desired_layer: Target layer index.

    Returns:
        result: ``True`` when the indices match.
    """
    return layer_idx == desired_layer


def is_input_layer(layer_idx: int) -> bool:
    """
    Check whether a layer index points to the input layer.

    Args:
        layer_idx: Layer index to inspect.

    Returns:
        result: ``True`` when ``layer_idx`` is the input layer index.
    """
    return is_layer(layer_idx, INPUT_LAYER)


def maxmin_scatter(
    src: Tensor,
    index: Tensor,
    dim: int,
    dim_size: int | None = None,
) -> Tensor:
    """
    Performs a scatter reduction that computes the channel-wise range (max - min) for each
    index group.

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
    """
    Validate that a tensor has a floating-point dtype.

    Args:
        name: Name of the validated tensor.
        tensor: Tensor to validate.

    Raises:
        ValueError: If the tensor does not have a floating-point dtype.
    """
    if not tensor.is_floating_point():
        raise ValueError(f"{name!r} must have a floating-point dtype, got {tensor.dtype}.")


def validate_long_tensor_dtype(name: str, tensor: Tensor) -> None:
    """
    Validate that a tensor has dtype ``torch.long``.

    Args:
        name: Name of the validated tensor.
        tensor: Tensor to validate.

    Raises:
        ValueError: If the tensor does not have dtype ``torch.long``.
    """
    if tensor.dtype != torch.long:
        raise ValueError(f"{name!r} must have dtype torch.long, got {tensor.dtype}.")
