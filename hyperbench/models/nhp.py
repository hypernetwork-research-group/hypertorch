import torch

from torch import Tensor, nn
from typing import Dict, Literal, Optional
from hyperbench.nn import HyperedgeAggregator
from hyperbench.utils import ActivationFn


class NHP(nn.Module):
    """
    Neural Hyperlink Predictor (NHP) for undirected hyperedge link prediction.
    - Proposed in `NHP: Neural Hypergraph Link Prediction <https://dl.acm.org/doi/10.1145/3340531.3411870>`_ paper (CIKM 2020).
    - Reference implementation: `source <https://github.com/cyixiao/NHP-reproduce/>`_.

    NHP scores each candidate hyperedge by building candidate-specific node embeddings.
    A node that appears in multiple candidate hyperedges can receive a different incidence embedding in each one,
    because its update depends on the other nodes in that candidate hyperedge.

    Examples:
        >>> x = [
        ...     [1., 0.],  # node 0
        ...     [0., 1.],  # node 1
        ...     [1., 1.],  # node 2
        ...     [1., 0.],  # node 3
        ... ]
        >>> hyperedge_index = [
        ...     [0, 1, 1, 2, 3],  # node IDs
        ...     [0, 0, 1, 1, 1],  # hyperedge IDs
        ... ]
        >>> # hyperedge 0 = {node 0, node 1}
        >>> # hyperedge 1 = {node 1, node 2, node 3}
        >>> model = NHP(in_channels=2, hidden_channels=8, aggregation="maxmin")
        >>> scores = model(x, hyperedge_index)
        >>> scores.shape
        ... torch.Size([2])

    Args:
        in_channels: Number of input features per node.
        hidden_channels: Number of hidden units in the node embeddings.
        activation_fn: Activation function to use after the linear transformations. Defaults to ``nn.ReLU``.
        activation_fn_kwargs: Keyword arguments for the activation function. Defaults to empty dict.
        aggregation: Method to aggregate the incidence embeddings into a hyperedge embedding. Must be either "maxmin" or "mean". Defaults to "maxmin".
        bias: Whether to include bias terms in the linear layers. Defaults to ``True``.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        activation_fn: Optional[ActivationFn] = None,
        activation_fn_kwargs: Optional[Dict] = None,
        aggregation: Literal["mean", "maxmin"] = "maxmin",
        bias: bool = True,
    ):
        super().__init__()

        activation_fn = activation_fn if activation_fn is not None else nn.ReLU
        activation_fn_kwargs = activation_fn_kwargs if activation_fn_kwargs is not None else {}

        self.aggregation = aggregation

        self.self_loop = nn.Linear(in_channels, hidden_channels, bias=bias)
        # GCN message passing is implemented through neighbor sum computation,
        # so one projection is enough for the hyperedge-aware term
        self.hyperedge_aware = nn.Linear(in_channels, hidden_channels, bias=bias)
        self.activation_fn = activation_fn(**activation_fn_kwargs)

        self.hyperedge_score = nn.Linear(hidden_channels, 1, bias=bias)

    def forward(self, x: Tensor, hyperedge_index: Tensor) -> Tensor:
        """
        Score each candidate hyperedge.

        Args:
            x: Node feature matrix of shape ``(num_nodes, in_channels)``.
            hyperedge_index: Incidence tensor of shape ``(2, num_incidences)``.

        Returns:
            Scores of shape ``(num_hyperedges,)``.
        """
        if hyperedge_index.numel() == 0:
            return x.new_empty((0,))

        # Example: hyperedge_index = [[0, 1, 1, 2, 3],  == node_ids
        #                             [0, 0, 1, 1, 1]]  == hyperedge_ids
        node_ids = hyperedge_index[0]
        hyperedge_ids = hyperedge_index[1]

        # Gather the node features for each incidence
        # Example: x = [[1, 0],  # node 0
        #               [0, 1],  # node 1
        #               [1, 1],  # node 2
        #               [1, 0]]  # node 3
        #          node_ids = [0, 1, 1, 2, 3]
        #          -> incidence_node_features = [[1, 0],  # node 0 in hyperedge 0
        #                                        [0, 1],  # node 1 in hyperedge 0
        #                                        [0, 1],  # node 1 in hyperedge 1
        #                                        [1, 1],  # node 2 in hyperedge 1
        #                                        [1, 0]]  # node 3 in hyperedge 1
        #             shape: (num_incidences, in_channels)
        incidence_node_features = x[node_ids]

        # Do one local message-passing step to sum original node features per hyperedge to get hyperedge features.
        # that are aware of all nodes in the candidate hyperedge.
        # Example: hyperedge 0 contains nodes (0, 1)    -> [1, 0] + [0, 1] = [1, 1]
        #          hyperedge 1 contains nodes (1, 2, 3) -> [0, 1] + [1, 1] + [1, 0] = [2, 2]
        #          -> hyperedge_features = [[1, 1],  # sum for hyperedge 0
        #                                   [2, 2]]  # sum for hyperedge 1
        #             shape: (num_hyperedges, in_channels)
        hyperedge_features = HyperedgeAggregator(
            hyperedge_index=hyperedge_index,
            node_embeddings=x,
        ).pool("sum")

        # Broadcast hyperedge features back to each of their incidences,
        # and remove the current node feature to give to each incidence
        # the features of its neighboring nodes in the candidate hyperedge.
        # Example: hyperedge_features = [[1, 1],  # sum for hyperedge 0
        #                                [2, 2]]  # sum for hyperedge 1
        #                               shape (num_hyperedges, in_channels),
        #          hyperedge_ids = [0, 0, 1, 1, 1],
        #          incidence_node_features = [[1, 0],  # node 0 in hyperedge 0
        #                                     [0, 1],  # node 1 in hyperedge 0
        #                                     [0, 1],  # node 1 in hyperedge 1
        #                                     [1, 1],  # node 2 in hyperedge 1
        #                                     [1, 0]]  # node 3 in hyperedge 1
        #                                    shape: (num_incidences, in_channels)
        #          -> hyperedge_features[hyperedge_ids] = [[1, 1],  # hyperedge 0 for node 0
        #                                                  [1, 1],  # hyperedge 0 for node 1
        #                                                  [2, 2],  # hyperedge 1 for node 1
        #                                                  [2, 2],  # hyperedge 1 for node 2
        #                                                  [2, 2]]  # hyperedge 1 for node 3
        #                                                 shape: (num_incidences, in_channels)
        #          -> neighbor_features_per_incidence = [[0, 1],  # node 0 sees node 1
        #                                                [1, 0],  # node 1 sees node 0
        #                                                [2, 1],  # node 1 sees node 2 and node 3
        #                                                [1, 1],  # node 2 sees node 1 and node 3
        #                                                [1, 2]]  # node 3 sees node 1 and node 2
        #                                               shape: (num_incidences, in_channels)
        neighbor_features_per_incidence = (
            hyperedge_features[hyperedge_ids] - incidence_node_features
        )

        # shape (num_incidences, hidden_channels)
        neighbor_aware_hyperedge_embeddings = self.hyperedge_aware(neighbor_features_per_incidence)
        # shape (num_incidences, hidden_channels)
        selfloop_embeddings = self.self_loop(incidence_node_features)

        # incidence_embeddings[0] = activation_fn(selfloop_embeddings[0] + neighbor_aware_hyperedge_embeddings[0])
        # is the embedding of the first incidence (i.e., node 0 in hyperedge 0)
        # after one local message-passing step inside that candidate hyperedge.
        incidence_embeddings = self.activation_fn(
            selfloop_embeddings + neighbor_aware_hyperedge_embeddings
        )  # shape (num_incidences, hidden_channels)

        # Treat each incidence embedding as a separately aggregatable set of features.
        # This is required because incidence embeddings are not global node embeddings:
        # node 1 may appear twice with two different embeddings as it participates in two different candidate hyperedges.
        # Example: incidence_ids = [0, 1, 2, 3, 4],
        #          hyperedge_ids = [0, 0, 1, 1, 1]
        #          -> incidence_hyperedge_index = [[0, 1, 2, 3, 4],
        #                                          [0, 0, 1, 1, 1]]
        num_incidences = incidence_embeddings.size(0)
        incidence_ids = torch.arange(num_incidences, device=hyperedge_index.device)
        incidence_hyperedge_index = torch.stack([incidence_ids, hyperedge_ids], dim=0)

        # Example: incidence_embeddings = [[1, 2],  # features 0, node 0 in hyperedge 0
        #                                  [3, 4],  # features 1, node 1 in hyperedge 0
        #                                  [5, 6],  # features 2, node 1 in hyperedge 1
        #                                  [7, 8],  # features 3, node 2 in hyperedge 1
        #                                  [9, 10]] # features 4, node 3 in hyperedge 1
        #          -> incidence_aggregator pools features (0, 1) for hyperedge 0 and features (2, 3, 4) for hyperedge 1
        #          if aggregation == "maxmin":
        #          -> hyperedge_embeddings = [[max(1, 3) - min(1, 3), max(2, 4) - min(2, 4)],                # hyperedge 0
        #                                     [max(5, 7, 9) - min(5, 7, 9), max(6, 8, 10) - min(6, 8, 10)]]  # hyperedge 1
        #                                    shape: (num_hyperedges, hidden_channels)
        #         if aggregation == "mean":
        #         -> hyperedge_embeddings = [[mean(1, 3), mean(2, 4)],         # hyperedge 0
        #                                    [mean(5, 7, 9), mean(6, 8, 10)]]  # hyperedge 1
        #                                   shape: (num_hyperedges, hidden_channels)
        incidence_aggregator = HyperedgeAggregator(
            hyperedge_index=incidence_hyperedge_index,
            node_embeddings=incidence_embeddings,
        )

        match self.aggregation:
            case "maxmin":
                max_embeddings = incidence_aggregator.pool("max")
                min_embeddings = incidence_aggregator.pool("min")
                hyperedge_embeddings = max_embeddings - min_embeddings
            case _:
                hyperedge_embeddings = incidence_aggregator.pool("mean")

        # Decode: linear projection to scalar score per hyperedge
        # shape: (num_hyperedges, 1) -> squeeze -> (num_hyperedges,)
        return self.hyperedge_score(hyperedge_embeddings).squeeze(-1)
