from torch import Tensor, nn

from hyperbench.nn import HNHNConv


class HNHN(nn.Module):
    """
    HNHN performs incidence-based hypergraph convolution with explicit hyperedge
    embeddings between the node -> hyperedge -> node propagation steps.
    - Proposed in `HNHN: Hypergraph Networks with Hyperedge Neurons <https://arxiv.org/abs/2006.12278>`_ paper.
    - Reference implementation: `source <https://deephypergraph.readthedocs.io/en/latest/_modules/dhg/models/hypergraphs/hnhn.html#HNHN>`_.

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
                HNHNConv(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    bias=bias,
                    use_batch_normalization=use_batch_normalization,
                    drop_rate=drop_rate,
                ),
                HNHNConv(
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
        Apply two stacked ``HNHNConv`` layers to produce node embeddings.

        Args:
            x: Input node feature matrix of size ``(num_nodes, in_channels)``.
            hyperedge_index: Hyperedge incidence in COO format of size ``(2, num_incidences)``.

        Returns:
            The output node feature matrix of size ``(num_nodes, num_classes)``.
        """
        for layer in self.layers:
            x = layer(x, hyperedge_index)
        return x
