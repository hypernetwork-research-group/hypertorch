from typing import Optional
from torch import Tensor, nn
from hyperbench.nn import HGNNConv
from hyperbench.types import HyperedgeIndex


class HGNN(nn.Module):
    """
    HGNN performs spectral convolution directly on the hypergraph structure using the
    node-hyperedge incidence matrix, without any reduction to a pairwise graph.
    Unlike HyperGCN (which approximates each hyperedge by selecting representative pairwise
    edges via random projection), HGNN preserves all higher-order relationships by passing
    messages through the full incidence structure: nodes -> hyperedges -> nodes.
    - Proposed in `Hypergraph Neural Networks <https://arxiv.org/pdf/1809.09401>`_ paper (AAAI 2019).
    - Reference implementation: `source <https://deephypergraph.readthedocs.io/en/latest/_modules/dhg/models/hypergraphs/hgnn.html#HGNN>`_.

    Args:
        in_channels: The number of input channels.
        hidden_channels: The number of hidden channels.
        num_classes: The number of output channels.
        bias: If set to ``False``, the layer will not learn the bias parameter. Defaults to ``True``.
        use_batch_normalization: If set to ``True``, layers will use batch normalization. Defaults to ``False``.
        drop_rate: Dropout ratio. Defaults to ``0.5``.
        fast: If set to ``True``, the HGNN Laplacian will be computed once and cached.
            Defaults to ``False`` as the original paper does not mention caching.
            Setting it to ``True`` can speed up training when the hypergraph structure is static.
            For example, negatives added only before training (not per batch/epoch) will not change the hypergraph topology, so caching is safe.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        bias: bool = True,
        use_batch_normalization: bool = False,
        drop_rate: float = 0.5,
        fast: bool = False,
    ):
        super().__init__()
        self.fast = fast
        self.cached_hgnn_laplacian_matrix: Optional[Tensor] = None

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
        r"""
        Apply two stacked ``HGNNConv`` layers to produce node embeddings.

        The first layer applies ReLU + dropout and maps ``in_channels -> hidden_channels``.
        The second layer is the output layer (no activation/dropout) and maps
        ``hidden_channels -> num_classes``.

        When ``fast=True`` (default), the HGNN Laplacian ``D_n^{-1/2} H D_e^{-1} H^T D_n^{-1/2}``
        is computed once from ``hyperedge_index`` and cached. The cache is invalidated only when
        ``num_nodes`` changes (e.g., due to negative sampling adding nodes across batches).
        This is safe because the HGNN Laplacian depends solely on the hypergraph topology,
        unlike HyperGCN's Laplacian which depends on node features via random projection.

        Args:
            x: Input node feature matrix. Size ``(num_nodes, in_channels)``.
            hyperedge_index: Hyperedge incidence in COO format. Size ``(2, num_incidences)``,
                where row 0 contains node IDs and row 1 contains hyperedge IDs.

        Returns:
            The output node feature matrix. Size ``(num_nodes, num_classes)``.
        """
        if not self.fast:
            for layer in self.layers:
                x = layer(x, hyperedge_index)
            return x

        should_not_use_cached = (
            self.cached_hgnn_laplacian_matrix is None
            or self.cached_hgnn_laplacian_matrix.size(0) != x.size(0)
        )

        if should_not_use_cached:
            self.cached_hgnn_laplacian_matrix = HyperedgeIndex(
                hyperedge_index
            ).get_sparse_hgnn_laplacian(num_nodes=x.size(0))

        for layer in self.layers:
            x = layer(x, hyperedge_index, hgnn_laplacian_matrix=self.cached_hgnn_laplacian_matrix)
        return x
