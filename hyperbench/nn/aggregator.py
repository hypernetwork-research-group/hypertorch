from torch import Tensor
from typing import Literal, Optional
from torch_geometric.utils import scatter

from hyperbench.types import HyperedgeIndex


class HyperedgeAggregator:
    """
    Pool node embeddings into hyperedge embeddings using the incidence structure.

    Each node-hyperedge incidence selects one node embedding row, then reduces
    those rows per hyperedge with the requested scatter aggregation.

    Args:
        hyperedge_index: Hyperedge incidence in COO format of size ``(2, num_incidences)``.
        node_embeddings: Node embedding matrix of size ``(num_nodes, num_channels)``.
    """

    def __init__(
        self,
        hyperedge_index: Tensor,
        node_embeddings: Tensor,
    ):
        self.hyperedge_index_wrapper = HyperedgeIndex(hyperedge_index)
        self.node_embeddings = node_embeddings

    def pool(self, aggregation: Literal["max", "min", "mean", "mul", "sum"]) -> Tensor:
        """
        Aggregate node embeddings for each hyperedge.

        Examples:
            >>> hyperedge_index = [[0, 1, 2, 2, 3],
            ...                    [0, 0, 0, 1, 1]]
            >>> node_embeddings = [[1, 10], [2, 20], [3, 30], [4, 40]]
            >>> HyperedgeAggregator(hyperedge_index, node_embeddings).pool("mean")
            ... [[2, 20], [3.5, 35]]
            >>> HyperedgeAggregator(hyperedge_index, node_embeddings).pool("sum")
            ... [[6, 60], [7, 70]]
            >>> HyperedgeAggregator(hyperedge_index, node_embeddings).pool("max")
            ... [[3, 30], [4, 40]]

        Args:
            aggregation: Reduction applied across the nodes belonging to each hyperedge.

        Returns:
            A hyperedge embedding matrix of shape ``(num_hyperedges, num_channels)``.
        """
        # Gather the embeddings for each incidence.
        # A node appearing in multiple hyperedges is repeated, once per incidence.
        # Example: node_embeddings = [[1, 10],  # node 0
        #                             [2, 20],  # node 1
        #                             [3, 30],  # node 2
        #                             [4, 40]]  # node 3
        #          -> all_node_ids = [0, 1, 2, 2, 3]
        #          -> incidence_node_embeddings = [[1, 10],  # node 0 for hyperedge 0
        #                                          [2, 20],  # node 1 for hyperedge 0
        #                                          [3, 30],  # node 2 for hyperedge 0
        #                                          [3, 30],  # node 2 for hyperedge 1
        #                                          [4, 40]]  # node 3 for hyperedge 1
        #             shape: (num_incidences, num_channels)
        incidence_node_embeddings = self.node_embeddings[self.hyperedge_index_wrapper.all_node_ids]

        # Scatter-aggregate node embeddings into hyperedge embeddings.
        # Example: with aggregation="sum":
        #          [[1+2+3, 10+20+30],  # hyperedge 0 contains node 0, 1, 2
        #          [3+4, 30+40]]        # hyperedge 1 contains node 2, 3
        #          shape: (num_hyperedges, num_channels)
        #          with aggregation="max":
        #          [[max(1, 2, 3), max(10, 20, 30)],  # hyperedge 0 contains node 0, 1, 2
        #           [max(3, 4), max(30, 40)]]         # hyperedge 1 contains node 2, 3
        #          shape: (num_hyperedges, num_channels)
        return scatter(
            src=incidence_node_embeddings,
            index=self.hyperedge_index_wrapper.all_hyperedge_ids,
            dim=0,  # scatter along the hyperedge dimension
            dim_size=self.hyperedge_index_wrapper.num_hyperedges,
            reduce=aggregation,
        )


class NodeAggregator:
    """
    Pool hyperedge embeddings into node embeddings using the incidence structure.

    Each node-hyperedge incidence selects one hyperedge embedding row, then
    reduces those rows per node with the requested scatter aggregation.

    Args:
        hyperedge_index: Hyperedge incidence in COO format of size ``(2, num_incidences)``.
        hyperedge_embeddings: Hyperedge embedding matrix of size ``(num_hyperedges, num_channels)``.
        num_nodes: Optional explicit node count. When provided, the pooled output preserves isolated nodes that do not appear in ``hyperedge_index``.
    """

    def __init__(
        self,
        hyperedge_index: Tensor,
        hyperedge_embeddings: Tensor,
        num_nodes: Optional[int] = None,
    ):
        self.hyperedge_index_wrapper = HyperedgeIndex(hyperedge_index)
        self.hyperedge_embeddings = hyperedge_embeddings
        self.num_nodes = num_nodes

    def pool(self, aggregation: Literal["max", "min", "mean", "mul", "sum"]) -> Tensor:
        """
        Aggregate hyperedge embeddings for each node.

        Examples:
            >>> hyperedge_index = [[0, 1, 1, 2],
            ...                    [0, 0, 1, 1]]
            >>> hyperedge_embeddings = [[10, 100], [20, 200]]
            >>> NodeAggregator(hyperedge_index, hyperedge_embeddings).pool("mean")
            ... [[10, 100], [15, 150], [20, 200]]
            >>> NodeAggregator(hyperedge_index, hyperedge_embeddings).pool("sum")
            ... [[10, 100], [30, 300], [20, 200]]
            >>> NodeAggregator(hyperedge_index, hyperedge_embeddings).pool("max")
            ... [[10, 100], [20, 200], [20, 200]]

        Args:
            aggregation: Reduction applied across the hyperedges incident to each node.

        Returns:
            A node embedding matrix of shape ``(num_nodes, num_channels)``.
        """
        # Gather the embeddings for each incidence.
        # A hyperedge appearing in multiple node incidences is repeated, once per incidence.
        # Example: hyperedge_embeddings = [[10, 100],  # hyperedge 0
        #                                  [20, 200]]  # hyperedge 1
        #          -> all_hyperedge_ids = [0, 0, 1, 1]
        #          -> incidence_hyperedge_embeddings = [[10, 100],   # hyperedge 0 for node 0
        #                                               [10, 100],   # hyperedge 0 for node 1
        #                                               [20, 200],   # hyperedge 1 for node 1
        #                                               [20, 200]]   # hyperedge 1 for node 2
        #             shape: (num_incidences, num_channels)
        incidence_hyperedge_embeddings = self.hyperedge_embeddings[
            self.hyperedge_index_wrapper.all_hyperedge_ids
        ]
        num_nodes = (
            self.num_nodes if self.num_nodes is not None else self.hyperedge_index_wrapper.num_nodes
        )

        # Scatter-aggregate hyperedge embeddings into node embeddings.
        # Example: with aggregation="sum":
        #          [[10, 100],         # node 0 belongs to hyperedge 0
        #           [10+20, 100+200],  # node 1 belongs to hyperedge 0 and 1
        #           [20, 200]]         # node 2 belongs to hyperedge 1
        #          shape: (num_nodes, num_channels)
        #          with aggregation="max":
        #          [[10, 100],                     # node 0 belongs to hyperedge 0
        #           [max(10, 20), max(100, 200)],  # node 1 belongs to hyperedge 0 and 1
        #           [20, 200]]                     # node 2 belongs to hyperedge 1
        return scatter(
            src=incidence_hyperedge_embeddings,
            index=self.hyperedge_index_wrapper.all_node_ids,
            dim=0,  # scatter along the node dimension
            dim_size=num_nodes,
            reduce=aggregation,
        )
