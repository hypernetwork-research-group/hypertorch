from typing import Optional
from torch import Tensor, nn
from hyperbench.nn import HyperGCNConv
from hyperbench.types import EdgeIndex, HyperedgeIndex


class HyperGCN(nn.Module):
    """
    HyperGCN approximates each hyperedge of the hypergraph by a set of pairwise edges connecting the vertices of the hyperedge
    and treats the learning problem as a graph learning problem on the approximation.
    - Proposed in `HyperGCN: A New Method of Training Graph Convolutional Networks on Hypergraphs <https://dl.acm.org/doi/10.5555/3454287.3454422>`_ paper (NeurIPS 2019).
    - Reference implementation: `source <https://deephypergraph.readthedocs.io/en/latest/_modules/dhg/models/hypergraphs/hypergcn.html#HyperGCN>`_.

    Args:
        in_channels: The number of input channels.
        hidden_channels: The number of hidden channels.
        num_classes: The number of classes of the classification task as HyperGCB is a node classification model.
        bias: If set to ``False``, the layer will not learn the bias parameter. Defaults to ``True``.
        use_batch_normalization: If set to ``True``, layers will use batch normalization. Defaults to ``False``.
        drop_rate: Dropout ratio. Defaults to ``0.5``.
        use_mediator: Whether to use mediator to transform the hyperedges to edges in the graph. Defaults to ``False``.
        fast: If set to ``True``, the transformed graph structure will be computed once from the input hypergraph and vertex features, and cached for future use. Defaults to ``True``.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        bias: bool = True,
        use_batch_normalization: bool = False,
        drop_rate: float = 0.5,
        use_mediator: bool = False,
        fast: bool = True,
    ):
        super().__init__()
        self.fast = fast
        self.use_mediator = use_mediator
        self.cached_gcn_laplacian_matrix: Optional[Tensor] = None

        self.layers = nn.ModuleList(
            [
                HyperGCNConv(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    bias=bias,
                    use_batch_normalization=use_batch_normalization,
                    drop_rate=drop_rate,
                    use_mediator=use_mediator,
                ),
                HyperGCNConv(
                    in_channels=hidden_channels,
                    out_channels=num_classes,
                    bias=bias,
                    use_batch_normalization=use_batch_normalization,
                    use_mediator=use_mediator,
                    is_last=True,
                ),
            ]
        )

    def forward(self, x: Tensor, hyperedge_index: Tensor) -> Tensor:
        r"""
        The forward function.

        Args:
            x: Input node feature matrix. Size ``(num_nodes, in_channels)``.
            hyperedge_index: The hyperedge indices of the hypergraph. Size ``(2, num_hyperedges)``.

        Returns:
            The output node feature matrix. Size ``(num_nodes, num_classes)``.
        """
        if not self.fast:
            for layer in self.layers:
                x = layer(x, hyperedge_index)
            return x

        # If the GCN Laplacian is cached, we need to check if the node feature size has changed
        # with cached_gcn_laplacian_matrix.size(0) != x.size(0), this can happen, for example, due to:
        # adding new negative samples or having validation/test sets with different node features
        should_not_use_cached_gcn_laplacian_matrix = (
            self.cached_gcn_laplacian_matrix is None  # Not cached yet
            or self.cached_gcn_laplacian_matrix.size(0)
            != x.size(0)  # Node feature size has changed
        )

        if should_not_use_cached_gcn_laplacian_matrix:
            edge_index = HyperedgeIndex(hyperedge_index).reduce_to_edge_index_on_random_direction(
                x,
                with_mediators=self.use_mediator,
            )

            self.cached_gcn_laplacian_matrix = EdgeIndex(
                edge_index
            ).get_sparse_normalized_gcn_laplacian(num_nodes=x.size(0))

        for layer in self.layers:
            x = layer(x, hyperedge_index, gcn_laplacian_matrix=self.cached_gcn_laplacian_matrix)
        return x
