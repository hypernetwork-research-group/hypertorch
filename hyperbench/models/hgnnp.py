from torch import Tensor, nn
from hyperbench.nn import HGNNPConv


class HGNNP(nn.Module):
    """
    HGNN+ performs hypergraph convolution with two-stage mean aggregation using the
    incidence structure directly: nodes -> hyperedges -> nodes.
    - Proposed in `HGNN+: General Hypergraph Neural Networks <https://ieeexplore.ieee.org/document/9795251>`_ paper (IEEE T-PAMI 2022).
    - Reference implementation: `source <https://deephypergraph.readthedocs.io/en/latest/_modules/dhg/models/hypergraphs/hgnnp.html#HGNNP>`_.

    Args:
        in_channels: The number of input channels.
        hidden_channels: The number of hidden channels.
        num_classes: The number of output channels.
        bias: If set to ``False``, the layer will not learn the bias parameter. Defaults to ``True``.
        use_batch_normalization: If set to ``True``, layers will use batch normalization. Defaults to ``False``.
        drop_rate: Dropout ratio. Defaults to ``0.5``.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        bias: bool = True,
        use_batch_normalization: bool = False,
        drop_rate: float = 0.5,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                HGNNPConv(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    bias=bias,
                    use_batch_normalization=use_batch_normalization,
                    drop_rate=drop_rate,
                ),
                HGNNPConv(
                    in_channels=hidden_channels,
                    out_channels=num_classes,
                    bias=bias,
                    use_batch_normalization=use_batch_normalization,
                    is_last=True,
                ),
            ]
        )

    def forward(self, x: Tensor, hyperedge_index: Tensor) -> Tensor:
        """
        Apply two stacked ``HGNNPConv`` layers to produce node embeddings.

        Args:
            x: Input node feature matrix. Size ``(num_nodes, in_channels)``.
            hyperedge_index: Hyperedge incidence in COO format. Size ``(2, num_incidences)``,
                where row 0 contains node IDs and row 1 contains hyperedge IDs.

        Returns:
            The output node feature matrix. Size ``(num_nodes, num_classes)``.
        """
        for layer in self.layers:
            x = layer(x, hyperedge_index)
        return x
