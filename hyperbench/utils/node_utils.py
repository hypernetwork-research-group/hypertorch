from torch import Tensor
from typing import Literal, Optional, Sequence, TypeAlias


NodeSpaceAssignment: TypeAlias = Literal["first", "all"]
NodeSpaceFiller: TypeAlias = float | int | Sequence[float] | Tensor
NodeSpaceSetting: TypeAlias = Literal["inductive", "transductive"]


def is_assigned_to_all(node_space_assignment: Optional[NodeSpaceAssignment]) -> bool:
    return node_space_assignment == "all"


def is_assigned_to_first(node_space_assignment: Optional[NodeSpaceAssignment]) -> bool:
    return node_space_assignment == "first"


def is_inductive_setting(node_space_setting: Optional[NodeSpaceSetting]) -> bool:
    return node_space_setting == "inductive"


def is_transductive_setting(node_space_setting: Optional[NodeSpaceSetting]) -> bool:
    return node_space_setting == "transductive"


def is_transductive_split(
    node_space_setting: Optional[NodeSpaceSetting],
    assign_node_space_to: Optional[NodeSpaceAssignment],
    split_num: int,
) -> bool:
    if not is_transductive_setting(node_space_setting):
        return False
    if is_assigned_to_all(assign_node_space_to):
        return True
    return is_assigned_to_first(assign_node_space_to) and split_num == 0
