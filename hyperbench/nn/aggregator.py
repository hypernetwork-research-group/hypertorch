import torch

from torch import Tensor
from typing import Literal, Tuple
from hyperbench.types import HyperedgeIndex


class HyperedgeAggregator:
    def __init__(
        self,
        hyperedge_index: Tensor,
        node_embeddings: Tensor,
    ):
        self.HYPEREDGE_AGGREGATION_DIM = 0

        self.hyperedge_index_wrapper = HyperedgeIndex(hyperedge_index)

        self.node_embeddings = node_embeddings
        self.out_channels = node_embeddings.size(1)

        self.device = hyperedge_index.device

    def max_pooling(self) -> Tensor:
        # Example: incidence_node_embeddings =
        #          [[e00, e01],   # node 0
        #           [e10, e11],   # node 1
        #           [e20, e21],   # node 2 (for hyperedge 0)
        #           [e20, e21],   # node 2 (for hyperedge 1)
        #           [e30, e31]]   # node 3
        #          shape: (num_incidences, out_channels)
        #          scatter_index =
        #          [[0, 0],   # node 0 belongs to hyperedge 0, so all its channels go to the corresponding channels of hyperedge 0
        #           [0, 0],   # node 1 belongs to hyperedge 0
        #           [0, 0],   # node 2 belongs to hyperedge 0
        #           [1, 1],   # node 2 belongs to hyperedge 1
        #           [1, 1]]   # node 3 belongs to hyperedge 1
        #          shape: (num_incidences, out_channels)
        #          hyperedge_embeddings =
        #          [[0, 0],   # hyperedge 0
        #           [0, 0]]   # hyperedge 1
        #          shape: (num_hyperedges, out_channels)
        hyperedge_embeddings, scatter_index, incidence_node_embeddings = self.__prepare()

        # Example: hyperedge_embeddings after `scatter_amax_`:
        #          [[max(e00, e10, e20), max(e01, e11, e21)],  # hyperedge 0 contains node 0, 1, 2
        #           [max(e20, e30), max(e21, e31)]]            # hyperedge 1 contains node 2, 3
        #          shape: (num_hyperedges, out_channels)
        hyperedge_embeddings.scatter_reduce_(
            reduce="amax",
            dim=self.HYPEREDGE_AGGREGATION_DIM,
            index=scatter_index,
            src=incidence_node_embeddings,
            include_self=False,  # To avoid the initial zeros in hyperedge_embeddings to be considered in the max reduction
        )
        return hyperedge_embeddings

    def min_pooling(self) -> Tensor:
        # Example: incidence_node_embeddings =
        #          [[e00, e01],   # node 0
        #           [e10, e11],   # node 1
        #           [e20, e21],   # node 2 (for hyperedge 0)
        #           [e20, e21],   # node 2 (for hyperedge 1)
        #           [e30, e31]]   # node 3
        #          shape: (num_incidences, out_channels)
        #          scatter_index =
        #          [[0, 0],   # node 0 belongs to hyperedge 0, so all its channels go to the corresponding channels of hyperedge 0
        #           [0, 0],   # node 1 belongs to hyperedge 0
        #           [0, 0],   # node 2 belongs to hyperedge 0
        #           [1, 1],   # node 2 belongs to hyperedge 1
        #           [1, 1]]   # node 3 belongs to hyperedge 1
        #          shape: (num_incidences, out_channels)
        #          hyperedge_embeddings =
        #          [[0, 0],   # hyperedge 0
        #           [0, 0]]   # hyperedge 1
        #          shape: (num_hyperedges, out_channels)
        hyperedge_embeddings, scatter_index, incidence_node_embeddings = self.__prepare()

        # Example: hyperedge_embeddings after `scatter_amin_`:
        #          [[min(e00, e10, e20), min(e01, e11, e21)],  # hyperedge 0 contains node 0, 1, 2
        #           [min(e20, e30), min(e21, e31)]]            # hyperedge 1 contains node 2, 3
        #          shape: (num_hyperedges, out_channels)
        hyperedge_embeddings.scatter_reduce_(
            reduce="amin",
            dim=self.HYPEREDGE_AGGREGATION_DIM,
            index=scatter_index,
            src=incidence_node_embeddings,
            include_self=False,  # To avoid the initial zeros in hyperedge_embeddings to be considered in the min reduction
        )
        return hyperedge_embeddings

    def mean_pooling(self) -> Tensor:
        # Example: hyperedge_embeddings after `sum_pooling`:
        #          [[e00+e10+e20, e01+e11+e21],  # hyperedge 0 contains node 0, 1, 2
        #           [e20+e30, e21+e31]]          # hyperedge 1 contains node 2, 3
        #          shape: (num_hyperedges, out_channels)
        hyperedge_embeddings = self.sum_pooling()

        # For mean pooling, we divide each sum by the number of nodes in that hyperedge.
        # counts[hi] = number of nodes in hyperedge hi.
        # We clamp it to min 1 to avoid division by zero as each hyperedge should have at least one node.
        # Example:
        # - hyperedge 0: (e00+e10+e20) / 3
        # - hyperedge 1: (e20+e30) / 2
        counts = (
            torch.bincount(
                input=self.hyperedge_index_wrapper.all_hyperedge_ids,
                minlength=self.hyperedge_index_wrapper.num_hyperedges,
            )
            .unsqueeze(1)
            .clamp(min=1)
        )

        return hyperedge_embeddings / counts

    def sum_pooling(self) -> Tensor:
        # Example: incidence_node_embeddings =
        #          [[e00, e01],   # node 0
        #           [e10, e11],   # node 1
        #           [e20, e21],   # node 2 (for hyperedge 0)
        #           [e20, e21],   # node 2 (for hyperedge 1)
        #           [e30, e31]]   # node 3
        #          shape: (num_incidences, out_channels)
        #          scatter_index =
        #          [[0, 0],   # node 0 belongs to hyperedge 0, so all its channels go to the corresponding channels of hyperedge 0
        #           [0, 0],   # node 1 belongs to hyperedge 0
        #           [0, 0],   # node 2 belongs to hyperedge 0
        #           [1, 1],   # node 2 belongs to hyperedge 1
        #           [1, 1]]   # node 3 belongs to hyperedge 1
        #          shape: (num_incidences, out_channels)
        #          hyperedge_embeddings =
        #          [[0, 0],   # hyperedge 0
        #           [0, 0]]   # hyperedge 1
        #          shape: (num_hyperedges, out_channels)
        hyperedge_embeddings, scatter_index, incidence_node_embeddings = self.__prepare()

        # Example: hyperedge_embeddings after `scatter_add_`:
        #          [[e00+e10+e20, e01+e11+e21],  # hyperedge 0 contains node 0, 1, 2
        #           [e20+e30, e21+e31]]          # hyperedge 1 contains node 2, 3
        #          shape: (num_hyperedges, out_channels)
        hyperedge_embeddings.scatter_add_(
            dim=self.HYPEREDGE_AGGREGATION_DIM,
            index=scatter_index,
            src=incidence_node_embeddings,
        )
        return hyperedge_embeddings

    def pool(self, aggregation: Literal["max", "min", "mean", "sum"]) -> Tensor:
        return getattr(self, f"{aggregation}_pooling")()

    def __prepare(self) -> Tuple[Tensor, Tensor, Tensor]:
        hyperedge_embeddings = torch.zeros(
            size=(self.hyperedge_index_wrapper.num_hyperedges, self.out_channels),
            device=self.device,
        )

        # Gather the embeddings for each incidence.
        # A node appearing in multiple hyperedges is repeated, once per incidence.
        # Example: node_ids = [0, 1, 2, 2, 3] (node 2 appears twice, once per hyperedge)
        #          -> [[e00, e01],   # node 0
        #              [e10, e11],   # node 1
        #              [e20, e21],   # node 2 (for hyperedge 0)
        #              [e20, e21],   # node 2 (for hyperedge 1)
        #              [e30, e31]]   # node 3
        #          shape: (num_incidences, out_channels)
        incidence_node_embeddings = self.node_embeddings[self.hyperedge_index_wrapper.all_node_ids]

        # Build the scatter index that tells `scatter_add_` which row of hyperedge_embeddings
        # each embedding should be added to. `unsqueeze(1)` adds a column dimension, then
        # `expand_as` repeats it across the embedding dimension so every feature of the same
        # node goes to the same hyperedge.
        # Example: hyperedge_ids = [0, 0, 0, 1, 1]
        #          -> unsqueeze(1): [[0], [0], [0], [1], [1]]
        #          -> expand_as:    [[0, 0], [0, 0], [0, 0], [1, 1], [1, 1]]
        #          shape: (num_incidences, out_channels)
        scatter_index = self.hyperedge_index_wrapper.all_hyperedge_ids.unsqueeze(1).expand_as(
            incidence_node_embeddings
        )

        return hyperedge_embeddings, scatter_index, incidence_node_embeddings
