from torch import Tensor, nn
from hyperbench.nn import HGNNConv


class HGNN(nn.Module):
    """
    HGNN performs spectral convolution directly on the hypergraph structure using the
    node-hyperedge incidence matrix, without any reduction to a pairwise graph.
    Unlike HyperGCN (which approximates each hyperedge by selecting representative pairwise
    edges via random projection), HGNN preserves all higher-order relationships by passing
    messages through the full incidence structure: nodes -> hyperedges -> nodes.

    References:
        - Proposed in [Hypergraph Neural Networks](https://arxiv.org/pdf/1809.09401) (AAAI 2019).
        - Reference implementation: [Code](https://deephypergraph.readthedocs.io/en/latest/_modules/dhg/models/hypergraphs/hgnn.html#HGNN).

    Args:
        in_channels: The number of input channels.
        hidden_channels: The number of hidden channels.
        num_classes: The number of output channels.
        bias: If set to ``False``, the layer will not learn the bias parameter.
            Defaults to ``True``.
        use_batch_normalization: If set to ``True``, layers will use batch normalization.
            Defaults to ``False``.
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
                HGNNConv(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    bias=bias,
                    use_batch_normalization=use_batch_normalization,
                    drop_rate=drop_rate,
                ),
                HGNNConv(
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
        Apply two stacked ``HGNNConv`` layers to produce node embeddings.

        The first layer applies ReLU + dropout and maps ``in_channels -> hidden_channels``.
        The second layer is the output layer (no activation/dropout) and maps
        ``hidden_channels -> num_classes``.

        Args:
            x: Input node feature matrix. Size ``(num_nodes, in_channels)``.
            hyperedge_index: Hyperedge incidence in COO format. Size ``(2, num_incidences)``,
                where row 0 contains node IDs and row 1 contains hyperedge IDs.

        Returns:
            x: The output node feature matrix. Size ``(num_nodes, num_classes)``.

        """
        for layer in self.layers:
            x = layer(x, hyperedge_index)
        return x
