from torch import nn, Tensor
from typing import Dict, Optional
from hyperbench.utils import (
    ActivationFn,
    NormalizationFn,
    is_input_layer,
    is_layer,
)


class MLP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: Optional[int] = None,
        num_layers: int = 1,
        activation_fn: Optional[ActivationFn] = None,
        activation_fn_kwargs: Optional[Dict] = None,
        normalization_fn: Optional[NormalizationFn] = None,
        normalization_fn_kwargs: Optional[Dict] = None,
        bias: bool = True,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.__validate_num_layers(num_layers, hidden_channels)

        hidden_channels = hidden_channels if hidden_channels is not None else 0
        activation_fn = activation_fn if activation_fn is not None else nn.ReLU
        activation_fn_kwargs = activation_fn_kwargs if activation_fn_kwargs is not None else {}
        normalization_fn_kwargs = (
            normalization_fn_kwargs if normalization_fn_kwargs is not None else {}
        )

        layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            is_output_layer = is_layer(layer_idx, num_layers - 1)

            linear_layer = nn.Linear(
                in_features=in_channels if is_input_layer(layer_idx) else hidden_channels,
                out_features=out_channels if is_output_layer else hidden_channels,
                bias=bias,
            )
            layers.append(linear_layer)

            if not is_output_layer:
                if normalization_fn is not None:
                    layers.append(normalization_fn(hidden_channels, **normalization_fn_kwargs))

                layers.append(activation_fn(**activation_fn_kwargs))

                if drop_rate > 0.0:
                    layers.append(nn.Dropout(drop_rate))

        self.layers = nn.Sequential(*layers)

    def forward(self, x) -> Tensor:
        return self.layers(x)

    def __validate_num_layers(self, num_layers: int, hidden_channels: Optional[int]) -> None:
        if num_layers < 1:
            raise ValueError("At least one layer is required for MLP.")
        if num_layers > 1 and hidden_channels is None:
            raise ValueError("hidden_channels must be specified for MLP with more than 1 layer.")


class SLP(MLP):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=1,
        )
