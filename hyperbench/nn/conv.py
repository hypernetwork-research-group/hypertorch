from typing import Optional
from torch import Tensor, nn
from hyperbench.types import EdgeIndex, Graph, HyperedgeIndex


class HyperGCNConv(nn.Module):
    """
    The HyperGCNConv layer proposed in `HyperGCN: A New Method of Training Graph Convolutional Networks on Hypergraphs <https://dl.acm.org/doi/10.5555/3454287.3454422>`_ paper (NeurIPS 2019).
    Reference implementation: `source <https://deephypergraph.readthedocs.io/en/latest/_modules/dhg/nn/convs/hypergraphs/hypergcn_conv.html#HyperGCNConv>`_.

    Args:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        bias: If set to ``False``, the layer will not learn the bias parameter. Defaults to ``True``.
        use_batch_normalization: If set to ``True``, the layer will use batch normalization. Defaults to ``False``.
        drop_rate: If set to a positive number, the layer will use dropout. Defaults to ``0.5``.
        use_mediator: Whether to use mediator to transform the hyperedges to edges in the graph. Defaults to ``False``.
        is_last: If set to ``True``, the layer will not apply the final activation and dropout functions. Defaults to ``False``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        use_batch_normalization: bool = False,
        drop_rate: float = 0.5,
        use_mediator: bool = False,
        is_last: bool = False,
    ):
        super().__init__()
        self.is_last = is_last
        self.use_mediator = use_mediator
        self.batch_norm_1d = nn.BatchNorm1d(out_channels) if use_batch_normalization else None
        self.activation_fn = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(drop_rate)

        # θ is the learnable weight matrix (as in the HyperGCN paper)
        # # it projects node features from in_channels to out_channels
        # and learns how to mix feature channels
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(
        self,
        x: Tensor,
        hyperedge_index: Tensor,
        gcn_laplacian_matrix: Optional[Tensor] = None,
    ) -> Tensor:
        r"""
        The forward function.

        Args:
            x: Input node feature matrix. Size ``(N, in_channels)``.
            hyperedge_index: Hyperedge indices representing the hypergraph structure. Size ``(2, E)``.
            gcn_laplacian_matrix: Optional precomputed normalized GCN Laplacian matrix. Size ``(N, N)``. Defaults to ``None``.
                If provided, it will be used directly for smoothing, so we can skip computing it from edge_index.

        Returns:
            The output node feature matrix. Size ``(N, out_channels)``.
        """
        x = self.theta(x)

        if gcn_laplacian_matrix is not None:
            x = Graph.smoothing_with_gcn_laplacian_matrix(x, gcn_laplacian_matrix)
        else:
            edge_index = HyperedgeIndex(hyperedge_index).reduce_to_edge_index_on_random_direction(
                x,
                with_mediators=self.use_mediator,
            )

            normalized_gcn_laplacian_matrix = EdgeIndex(
                edge_index
            ).get_sparse_normalized_gcn_laplacian()

            x = Graph.smoothing_with_gcn_laplacian_matrix(x, normalized_gcn_laplacian_matrix)

        if not self.is_last:
            x = self.activation_fn(x)
            if self.batch_norm_1d is not None:
                x = self.batch_norm_1d(x)
            x = self.dropout(x)

        return x
