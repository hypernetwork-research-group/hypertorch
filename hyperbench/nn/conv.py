from typing import Literal, Optional
from torch import Tensor, nn
from hyperbench.nn.aggregator import HyperedgeAggregator, NodeAggregator
from hyperbench.types import EdgeIndex, Graph, HyperedgeIndex, Hypergraph


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

        # θ is the learnable weight matrix (as in the HyperGCN paper),
        # it projects node features from in_channels to out_channels and learns how to mix feature channels
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(
        self,
        x: Tensor,
        hyperedge_index: Tensor,
        gcn_laplacian_matrix: Optional[Tensor] = None,
    ) -> Tensor:
        """
        The forward function.

        Args:
            x: Input node feature matrix. Size ``(num_nodes, in_channels)``.
            hyperedge_index: Hyperedge indices representing the hypergraph structure. Size ``(2, num_hyperedges)``.
            gcn_laplacian_matrix: Optional precomputed normalized GCN Laplacian matrix. Size ``(num_nodes, num_nodes)``. Defaults to ``None``.
                If provided, it will be used directly for smoothing, so we can skip computing it from edge_index.

        Returns:
            The output node feature matrix. Size ``(num_nodes, out_channels)``.
        """
        x = self.theta(x)

        if gcn_laplacian_matrix is not None:
            x = Graph.smoothing_with_laplacian_matrix(x, gcn_laplacian_matrix)
        else:
            edge_index, edge_weights = HyperedgeIndex(
                hyperedge_index
            ).reduce_to_edge_index_on_random_direction(
                x=x,
                with_mediators=self.use_mediator,
                return_weights=True,
            )

            normalized_gcn_laplacian_matrix = EdgeIndex(
                edge_index=edge_index,
                edge_weights=edge_weights,
            ).get_sparse_normalized_gcn_laplacian(num_nodes=x.size(0))

            x = Graph.smoothing_with_laplacian_matrix(x, normalized_gcn_laplacian_matrix)

        if not self.is_last:
            x = self.activation_fn(x)
            if self.batch_norm_1d is not None:
                x = self.batch_norm_1d(x)
            x = self.dropout(x)

        return x


class HGNNConv(nn.Module):
    """
    The HGNNConv layer proposed in `Hypergraph Neural Networks <https://arxiv.org/pdf/1809.09401>`_ paper (AAAI 2019).
    Reference implementation: `source <https://deephypergraph.readthedocs.io/en/latest/_modules/dhg/nn/convs/hypergraphs/hgnn_conv.html#HGNNConv>`_.

    Each layer performs: ``X' = sigma(L_HGNN X Theta)`` where ``L_HGNN = D_n^{-1/2} H D_e^{-1} H^T D_n^{-1/2}``
    is the hypergraph Laplacian computed from the incidence matrix H. This smooths node features through
    the hypergraph structure (nodes -> hyperedges -> nodes) without reducing to a pairwise graph.

    Unlike ``HyperGCNConv``, which uses a GCN Laplacian on a graph reduced from the hypergraph,
    ``HGNNConv`` operates entirely in hypergraph space and preserves all higher-order relationships.

    Args:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        bias: If set to ``False``, the layer will not learn the bias parameter. Defaults to ``True``.
        use_batch_normalization: If set to ``True``, the layer will use batch normalization. Defaults to ``False``.
        drop_rate: If set to a positive number, the layer will use dropout. Defaults to ``0.5``.
        is_last: If set to ``True``, the layer will not apply the final activation and dropout functions. Defaults to ``False``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        use_batch_normalization: bool = False,
        drop_rate: float = 0.5,
        is_last: bool = False,
    ):
        super().__init__()
        self.is_last = is_last
        self.batch_norm_1d = nn.BatchNorm1d(out_channels) if use_batch_normalization else None
        self.activation_fn = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x: Tensor, hyperedge_index: Tensor) -> Tensor:
        """
        Apply one HGNN convolution layer: project features, smooth via hypergraph Laplacian,
        then apply activation, batch norm, and dropout (unless this is the last layer).

        The full per-layer formula is:
            ``X' = sigma( D_n^{-1/2} H D_e^{-1} H^T D_n^{-1/2} (X Theta) )``

        where the Laplacian ``L = D_n^{-1/2} H D_e^{-1} H^T D_n^{-1/2}`` is computed from
        the hyperedge_index and can be passed in precomputed as ``hgnn_laplacian_matrix``
        for efficiency when the hypergraph structure does not change across forward passes.

        Args:
            x: Input node feature matrix of size ``(num_nodes, in_channels)``.
            hyperedge_index: Hyperedge incidence in COO format of size ``(2, num_incidences)``.

        Returns:
            The output node feature matrix of size ``(num_nodes, out_channels)``.
        """
        x = self.theta(x)

        smoothing_matrix = HyperedgeIndex(hyperedge_index).get_sparse_hgnn_smoothing_matrix(
            num_nodes=x.size(0),
        )
        x = Hypergraph.smoothing_with_matrix(x, smoothing_matrix)

        if not self.is_last:
            x = self.activation_fn(x)
            if self.batch_norm_1d is not None:
                x = self.batch_norm_1d(x)
            x = self.dropout(x)

        return x


class HGNNPConv(nn.Module):
    """
    The HGNNPConv layer proposed in `HGNN+: General Hypergraph Neural Networks <https://ieeexplore.ieee.org/document/9795251>`_ paper (IEEE T-PAMI 2022).
    Reference implementation: `source <https://deephypergraph.readthedocs.io/en/latest/_modules/dhg/nn/convs/hypergraphs/hgnnp_conv.html#HGNNPConv>`_.

    Each layer performs: ``X' = sigma(M_HGNN+ X Theta)`` where
    ``M_HGNN+ = D_v^{-1} H D_e^{-1} H^T`` is the HGNN+ smoothing matrix.

    Unlike ``HGNNConv``, which uses symmetric ``D_v^{-1/2}`` normalization for a
    spectral Laplacian, ``HGNNPConv`` uses plain inverse degrees and performs
    two-stage mean aggregation: nodes -> hyperedges -> nodes.

    Args:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        bias: If set to ``False``, the layer will not learn the bias parameter. Defaults to ``True``.
        use_batch_normalization: If set to ``True``, the layer will use batch normalization. Defaults to ``False``.
        drop_rate: If set to a positive number, the layer will use dropout. Defaults to ``0.5``.
        is_last: If set to ``True``, the layer will not apply the final activation and dropout functions. Defaults to ``False``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        use_batch_normalization: bool = False,
        drop_rate: float = 0.5,
        is_last: bool = False,
    ):
        super().__init__()
        self.is_last = is_last
        self.batch_norm_1d = nn.BatchNorm1d(out_channels) if use_batch_normalization else None
        self.activation_fn = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x: Tensor, hyperedge_index: Tensor) -> Tensor:
        """
        Apply one HGNN+ convolution layer using row-stochastic hypergraph smoothing.

        The full per-layer formula is:
            ``X' = sigma( D_v^{-1} H D_e^{-1} H^T (X Theta) )``

        Args:
            x: Input node feature matrix of size ``(num_nodes, in_channels)``.
            hyperedge_index: Hyperedge incidence in COO format of size ``(2, num_incidences)``.

        Returns:
            The output node feature matrix of size ``(num_nodes, out_channels)``.
        """
        x = self.theta(x)

        smoothing_matrix = HyperedgeIndex(hyperedge_index).get_sparse_hgnnp_smoothing_matrix(
            num_nodes=x.size(0),
        )
        x = Hypergraph.smoothing_with_matrix(x, smoothing_matrix)

        if not self.is_last:
            x = self.activation_fn(x)
            if self.batch_norm_1d is not None:
                x = self.batch_norm_1d(x)
            x = self.dropout(x)

        return x


class HNHNConv(nn.Module):
    """
    The HNHNConv layer proposed in `HNHN: Hypergraph Networks with Hyperedge Neurons <https://arxiv.org/abs/2006.12278>`_ paper.
    Reference implementation: `source <https://deephypergraph.readthedocs.io/en/latest/_modules/dhg/nn/convs/hypergraphs/hnhn_conv.html#HNHNConv>`_.

    Args:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        bias: If set to ``False``, the layer will not learn the bias parameter. Defaults to ``True``.
        use_batch_normalization: If set to ``True``, the layer will use batch normalization. Defaults to ``False``.
        drop_rate: If set to a positive number, the layer will use dropout. Defaults to ``0.5``.
        is_last: If set to ``True``, the layer will not apply the final activation and dropout functions. Defaults to ``False``.
    """

    __AGGREGATION: Literal["mean"] = "mean"

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        use_batch_normalization: bool = False,
        drop_rate: float = 0.5,
        is_last: bool = False,
    ):
        super().__init__()
        self.is_last = is_last
        self.batch_norm_1d = nn.BatchNorm1d(out_channels) if use_batch_normalization else None
        self.activation_fn = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(drop_rate)
        self.theta_v2e = nn.Linear(in_channels, out_channels, bias=bias)
        self.theta_e2v = nn.Linear(out_channels, out_channels, bias=bias)

    def forward(self, x: Tensor, hyperedge_index: Tensor) -> Tensor:
        """
        Apply one HNHN convolution layer using two learned projections around
        node-to-hyperedge and hyperedge-to-node mean aggregation.

        Args:
            x: Input node feature matrix of size ``(num_nodes, in_channels)``.
            hyperedge_index: Hyperedge incidence in COO format of size ``(2, num_incidences)``.

        Returns:
            The output node feature matrix of size ``(num_nodes, out_channels)``.
        """
        x = self.theta_v2e(x)

        hyperedge_embeddings = HyperedgeAggregator(hyperedge_index, x).pool(self.__AGGREGATION)
        hyperedge_embeddings = self.activation_fn(hyperedge_embeddings)
        hyperedge_embeddings = self.theta_e2v(hyperedge_embeddings)

        x = NodeAggregator(
            hyperedge_index=hyperedge_index,
            hyperedge_embeddings=hyperedge_embeddings,
            num_nodes=x.size(0),
        ).pool(self.__AGGREGATION)

        if not self.is_last:
            x = self.activation_fn(x)
            if self.batch_norm_1d is not None:
                x = self.batch_norm_1d(x)
            x = self.dropout(x)

        return x
