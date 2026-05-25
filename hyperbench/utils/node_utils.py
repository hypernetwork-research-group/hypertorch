from torch import Tensor
from typing import Literal, TypeAlias
from collections.abc import Sequence


NodeSpaceFiller: TypeAlias = float | int | Sequence[float] | Tensor
NodeSpaceSetting: TypeAlias = Literal["inductive", "transductive"]


def assign_hyperedge_label_to_nodes(
    hyperedge_index: Tensor,
    y: Tensor,
    num_hyperedges: int,
) -> dict[frozenset[int], float]:
    labels_by_nodes: dict[frozenset[int], float] = {}
    for hyperedge_id in range(num_hyperedges):
        mask = hyperedge_index[1] == hyperedge_id
        nodes = frozenset(hyperedge_index[0][mask].tolist())
        labels_by_nodes[nodes] = y[hyperedge_id].item()
    return labels_by_nodes


def is_inductive_setting(node_space_setting: NodeSpaceSetting | None) -> bool:
    return node_space_setting == "inductive"


def is_transductive_setting(node_space_setting: NodeSpaceSetting | None) -> bool:
    return node_space_setting == "transductive"


def validate_node_space_setting(node_space_setting: NodeSpaceSetting) -> None:
    """
    Validate that the node space setting is one of the supported values.

    Args:
        node_space_setting: The node space setting to validate, which should be either "inductive" or "transductive".

    Raises:
        ValueError: If the node space setting is not one of the supported values.
    """
    if is_transductive_setting(node_space_setting) or is_inductive_setting(node_space_setting):
        return

    raise ValueError(
        f"node_space_setting must be one of 'transductive' or 'inductive', got {node_space_setting!r}."
    )
