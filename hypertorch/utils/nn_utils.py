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
            Defaults to ``None``.

    Returns:
        values: A tensor containing the max-min values for each index group.
    """
    max_embeddings = scatter(src=src, index=index, dim=dim, dim_size=dim_size, reduce="max")
    min_embeddings = scatter(src=src, index=index, dim=dim, dim_size=dim_size, reduce="min")
    return max_embeddings - min_embeddings


def node_labels_from_node_degrees(
    node_incidences: Tensor,
    num_nodes: int,
    num_classes: int = 3,
) -> Tensor:
    """
    Create node labels by binning nodes according to hypergraph degree.

    The input is the node row of a hyperedge index, where each occurrence
    represents one node-hyperedge incidence. The function counts those
    occurrences to get one degree per node, then splits the degree distribution
    into ``num_classes`` quantile bins.

    Examples:
        >>> node_incidences = torch.tensor([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
        >>> labels = node_labels_from_node_degrees(node_incidences, num_nodes=5, num_classes=3)
        >>> labels
        ... tensor([0, 0, 1, 2, 2])

        Here, ``node_degrees = [0, 1, 2, 3, 4]``:
        node ``0`` is isolated, node ``1`` appears once, node ``2`` appears twice,
        node ``3`` appears three times, and node ``4`` appears four times. With
        three classes, the labels represent low, medium, and high degree bins.

    Args:
        node_incidences: A 1D tensor containing one node ID per hypergraph incidence,
            typically ``hdata.hyperedge_index[0]``.
        num_nodes: Total number of nodes, including isolated nodes that may not
            appear in ``node_incidences``.
        num_classes: Number of degree classes to produce.

    Returns:
        labels: A long tensor of shape ``[num_nodes]`` containing one class label
            per node.
    """
    # Count one occurrence per node-hyperedge incidence.
    # Example: node_incidences = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4], num_nodes = 5
    #                       nodes 0  1  2  3  4  # node 0 is isolated
    #          -> node_degrees = [0, 1, 2, 3, 4]
    node_degrees = torch.bincount(input=node_incidences, minlength=num_nodes).float()

    # Build quantile cut points between classes.
    # Example: num_classes = 3
    #          -> quantiles = [1/3, 2/3]
    #          -> thresholds ~= [1.333, 2.667]
    #          -> thresholds split nodes into low, medium, and high degree bins
    thresholds = torch.quantile(
        input=node_degrees,
        q=torch.tensor(
            [i / num_classes for i in range(1, num_classes)],
            device=node_degrees.device,
        ),
    )

    # Convert each degree into the index of its quantile bin.
    # Example: node_degrees = [0, 1, 2, 3, 4], thresholds ~= [1.333, 2.667]
    #          -> labels = [0, 0, 1, 2, 2]
    labels = torch.bucketize(input=node_degrees, boundaries=thresholds).long()
    return labels


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
