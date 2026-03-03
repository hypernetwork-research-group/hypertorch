from enum import Enum
from typing import Callable, Dict, Type, TypeAlias
from torch import Tensor
from torch.nn import Module


INPUT_LAYER = 0


ActivationFn: TypeAlias = Type[Module]
MetricFn: TypeAlias = Callable[[Tensor, Tensor], Tensor]
Metrics: TypeAlias = Dict[str, MetricFn]


class Aggregation(Enum):
    MEAN = "mean"
    MIN = "min"
    SUM = "sum"


class Stage(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


def is_layer(layer_idx: int, desired_layer: int) -> bool:
    return layer_idx == desired_layer


def is_input_layer(layer_idx: int) -> bool:
    return is_layer(layer_idx, INPUT_LAYER)
