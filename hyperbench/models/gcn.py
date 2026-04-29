from torch import Tensor, nn
from typing import Dict, Optional, TypedDict
from typing_extensions import NotRequired
from torch_geometric.nn import GCNConv
from hyperbench.utils import ActivationFn, is_layer


class GCNConfig(TypedDict):
    """
    Configuration for the GCN model.

    Args:
        in_channels: Dimension of the input node embeddings to the GCN layers.
        out_channels: Dimension of the output node embeddings from the GCN layers.
        hidden_channels: Dimension of the hidden node embeddings in the GCN layers.
        num_layers: Number of GCN layers. Must be at least 1. Defaults to ``2``.
        drop_rate: Dropout rate applied after each GCN layer (except the last one). Defaults to ``0.0`` (no dropout).
        activation_fn: Activation function to use after each hidden layer. Defaults to ``nn.ReLU``.
        activation_fn_kwargs: Keyword arguments for the activation function. Defaults to empty dict.
        bias: Whether to include a bias term in the GCN layers. Defaults to ``True``.
        improved: Whether to use the improved version of GCNConv. Defaults to ``False``.
        add_self_loops: Whether to add self-loops to the input graph. Defaults to ``True``.
        normalize: Whether to symmetrically normalize the adjacency matrix in GCNConv. Defaults to ``True``.
        cached: Whether to cache the normalized adjacency matrix in GCNConv.
            Only applicable if the graph structure does not change between epochs. Defaults to ``False``.
    """

    in_channels: int
    out_channels: int
    hidden_channels: NotRequired[int]
    num_layers: NotRequired[int]
    drop_rate: NotRequired[float]
    bias: NotRequired[bool]
    activation_fn: NotRequired[ActivationFn]
    activation_fn_kwargs: NotRequired[Dict]
    improved: NotRequired[bool]
    add_self_loops: NotRequired[bool]
    normalize: NotRequired[bool]
    cached: NotRequired[bool]


class GCN(nn.Module):
    """
    A reusable multi-layer GCN stack built from ``torch_geometric.nn.GCNConv``.

    Args:
        in_channels: Dimension of the input node embeddings to the GCN layers.
        out_channels: Dimension of the output node embeddings from the GCN layers.
        hidden_channels: Dimension of the hidden node embeddings in the GCN layers.
            Defaults to ``in_channels``.
        num_layers: Number of GCN layers. Must be at least 1. Defaults to ``2``.
        drop_rate: Dropout rate applied after each GCN layer except the last one.
        bias: Whether to include a bias term in the GCN layers.
        activation_fn: Activation function to use after each hidden layer. Defaults to ``nn.ReLU``.
        activation_fn_kwargs: Keyword arguments for the activation function. Defaults to empty dict.
        improved: Whether to use the improved version of ``GCNConv``.
        add_self_loops: Whether to add self-loops to the input graph.
        normalize: Whether to symmetrically normalize the adjacency matrix in ``GCNConv``.
        cached: Whether to cache the normalized adjacency matrix in ``GCNConv``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: Optional[int] = None,
        num_layers: int = 2,
        drop_rate: float = 0.0,
        bias: bool = True,
        activation_fn: Optional[ActivationFn] = None,
        activation_fn_kwargs: Optional[Dict] = None,
        improved: bool = False,
        add_self_loops: bool = True,
        normalize: bool = True,
        cached: bool = False,
    ):
        super().__init__()
        activation_fn = activation_fn if activation_fn is not None else nn.ReLU
        activation_fn_kwargs = activation_fn_kwargs if activation_fn_kwargs is not None else {}

        self.dropout = nn.Dropout(drop_rate)
        self.activation = activation_fn(**activation_fn_kwargs)
        self.layers = self.__build_layers(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            bias=bias,
            improved=improved,
            add_self_loops=add_self_loops,
            normalize=normalize,
            cached=cached,
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        num_layers = len(self.layers)
        for idx, layer in enumerate(self.layers):
            x = layer(x, edge_index)

            is_not_last_layer = not is_layer(idx, num_layers - 1)
            if is_not_last_layer:
                x = self.activation(x)
                x = self.dropout(x)

        return x

    def __build_layers(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: Optional[int],
        num_layers: int,
        bias: bool,
        improved: bool,
        add_self_loops: bool,
        normalize: bool,
        cached: bool,
    ) -> nn.ModuleList:
        if num_layers < 1:
            raise ValueError(f"Expected num_layers >= 1 for GCN, got {num_layers}.")

        hidden_channels = hidden_channels if hidden_channels is not None else 0
        if num_layers > 1 and hidden_channels <= 0:
            raise ValueError(
                f"Expected positive hidden_channels for GCN with multiple layers, got {hidden_channels}."
            )

        common_kwargs: Dict[str, bool] = {
            "bias": bias,
            "improved": improved,
            "add_self_loops": add_self_loops,
            "normalize": normalize,
            "cached": cached,
        }

        if num_layers == 1:
            return nn.ModuleList([GCNConv(in_channels, out_channels, **common_kwargs)])

        layers = [GCNConv(in_channels, hidden_channels, **common_kwargs)]
        for _ in range(num_layers - 2):
            layers.append(GCNConv(hidden_channels, hidden_channels, **common_kwargs))
        layers.append(GCNConv(hidden_channels, out_channels, **common_kwargs))

        return nn.ModuleList(layers)
