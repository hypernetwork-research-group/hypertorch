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
    labels_by_nodes = {}
    for hyperedge_id in range(num_hyperedges):
        mask = hyperedge_index[1] == hyperedge_id
        nodes = frozenset(hyperedge_index[0][mask].tolist())
        labels_by_nodes[nodes] = y[hyperedge_id]
    return labels_by_nodes


def is_inductive_setting(node_space_setting: NodeSpaceSetting | None) -> bool:
    return node_space_setting == "inductive"


def is_transductive_setting(node_space_setting: NodeSpaceSetting | None) -> bool:
    return node_space_setting == "transductive"
