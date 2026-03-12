from torch import Tensor
from typing import Literal
from torch_geometric.utils import scatter

from hyperbench.types import HyperedgeIndex


class HyperedgeAggregator:
    def __init__(
        self,
        hyperedge_index: Tensor,
        node_embeddings: Tensor,
    ):
        self.hyperedge_index_wrapper = HyperedgeIndex(hyperedge_index)
        self.node_embeddings = node_embeddings

    def pool(self, aggregation: Literal["max", "min", "mean", "mul", "sum"]) -> Tensor:
        # Gather the embeddings for each incidence.
        # A node appearing in multiple hyperedges is repeated, once per incidence.
        # Example: all_node_ids = [0, 1, 2, 2, 3] (node 2 appears twice, once per hyperedge)
        #          -> incidence_node_embeddings = [[e00, e01],   # node 0
        #                                          [e10, e11],   # node 1
        #                                          [e20, e21],   # node 2 (for hyperedge 0)
        #                                          [e20, e21],   # node 2 (for hyperedge 1)
        #                                          [e30, e31]]   # node 3
        #          shape: (num_incidences, out_channels)
        incidence_node_embeddings = self.node_embeddings[self.hyperedge_index_wrapper.all_node_ids]

        # Scatter-aggregate node embeddings into hyperedge embeddings.
        # Example: with aggregation="sum":
        #          [[e00+e10+e20, e01+e11+e21],  # hyperedge 0 contains node 0, 1, 2
        #           [e20+e30, e21+e31]]          # hyperedge 1 contains node 2, 3
        #          shape: (num_hyperedges, out_channels)
        #          with aggregation="max":
        #          [[max(e00, e10, e20), max(e01, e11, e21)],  # hyperedge 0 contains node 0, 1, 2
        #           [max(e20, e30), max(e21, e31)]]            # hyperedge 1 contains node 2, 3
        return scatter(
            src=incidence_node_embeddings,
            index=self.hyperedge_index_wrapper.all_hyperedge_ids,
            dim=0,  # scatter along the hyperedge dimension
            dim_size=self.hyperedge_index_wrapper.num_hyperedges,
            reduce=aggregation,
        )
