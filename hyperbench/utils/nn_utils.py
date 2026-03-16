from enum import Enum
from typing import Literal, Type, TypeAlias
from torch.nn import Module


INPUT_LAYER = 0


ActivationFn: TypeAlias = Type[Module]
NormalizationFn: TypeAlias = Type[Module]


# We can't use StrEnum as we support python 3.10,
# which doesn't have it. So we use Literal instead.
class Aggregation:
    MAX: Literal["max"] = "max"
    MEAN: Literal["mean"] = "mean"
    MIN: Literal["min"] = "min"
    SUM: Literal["sum"] = "sum"


class Stage(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


def is_layer(layer_idx: int, desired_layer: int) -> bool:
    return layer_idx == desired_layer


def is_input_layer(layer_idx: int) -> bool:
    return is_layer(layer_idx, INPUT_LAYER)
