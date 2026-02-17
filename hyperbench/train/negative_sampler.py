import torch

from abc import ABC, abstractmethod
from enum import Enum
from torch import Tensor
from typing import List, Optional, Set, Tuple
from hyperbench.types import HData, HyperedgeIndex


class NegativeSamplingSchedule(Enum):
    """When to run negative sampling during training."""

    FIRST_EPOCH = "first_epoch"  # Only at epoch 0, cached for all subsequent epochs
    EVERY_N_EPOCHS = "every_n_epochs"  # Every N epochs (N provided separately)
    EVERY_EPOCH = "every_epoch"  # Negatives generated every epoch


class NegativeSampler(ABC):
    """
    Abstract base class for negative samplers.

    Args:
        return_0based_negatives (bool, optional):
            - If ``True``, the negative samples returned by the ``sample`` method will have 0-based node and hyperedge IDs.
            - If ``False``, the negative samples will retain the original global node and hyperedge IDs from the input data.
    """

    def __init__(self, return_0based_negatives: bool = False):
        super().__init__()
        self.return_0based_negatives: bool = return_0based_negatives

    @abstractmethod
    def sample(self, data: HData) -> HData:
        """
        Abstract method for negative sampling.

        Args:
            data: The input data object containing graph or hypergraph information.
            return_0based_negatives: If ``True``, return negative samples with 0-based node and hyperedge IDs.

        Returns:
            HData: The negative samples as a new :class:`HData` object.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def _new_negative_hyperedge_index(
        self,
        sampled_hyperedge_indexes: List[Tensor],
        negative_node_ids: Tensor,
        negative_hyperedge_ids: Tensor,
    ) -> Tensor:
        """
        Concatenate, sort, and remap the sampled hyperedge indexes for negative samples.

        Args:
            sampled_hyperedge_indexes (List[Tensor]): List of hyperedge index tensors for each negative sample.
            negative_node_ids (Tensor): Tensor of negative node IDs.
            negative_hyperedge_ids (Tensor): Tensor of negative hyperedge IDs.

        Returns:
            The concatenated, sorted, and remapped hyperedge index tensor.
            If ``self.return_0based_negatives`` is ``True``, the returned tensor will have 0-based node and hyperedge IDs.
            Otherwise, it will retain the original global node and hyperedge IDs from the input data.
        """
        negative_hyperedge_index = torch.cat(sampled_hyperedge_indexes, dim=1)
        if not self.return_0based_negatives:
            return negative_hyperedge_index

        negative_hyperedge_index_wrapper = HyperedgeIndex(negative_hyperedge_index).to_0based(
            node_ids_to_rebase=negative_node_ids,
            hyperedge_ids_to_rebase=negative_hyperedge_ids,
        )

        return negative_hyperedge_index_wrapper.item

    def _new_hyperedge_attr(
        self,
        sampled_hyperedge_attrs: List[Tensor],
        hyperedge_attr: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        """
        Concatenate the hyperedge attributes for the negative samples.

        Args:
            sampled_hyperedge_attrs (List[Tensor]): List of hyperedge attribute tensors for each negative sample.
            hyperedge_attr (Tensor, optional): The original hyperedge attributes from the input data.

        Returns:
            The concatenated hyperedge attribute tensor for the negative samples.
        """
        if hyperedge_attr is None or len(sampled_hyperedge_attrs) < 1:
            return None

        negative_hyperedge_attr = torch.stack(sampled_hyperedge_attrs, dim=0)
        return negative_hyperedge_attr

    def _new_x(self, x: Tensor, negative_node_ids: Tensor) -> Tuple[Tensor, int]:
        """
        Get the node feature matrix for the negative samples.

        Args:
            x (Tensor): The original node feature matrix from the input data.
            negative_node_ids (Tensor): Tensor of negative node IDs.

        Returns:
            The node feature matrix for the negative samples and the number of negative nodes.
        """
        return x[negative_node_ids], len(negative_node_ids)


class RandomNegativeSampler(NegativeSampler):
    """
    A random negative sampler. Negatives generated with ``return_0based_negatives = False`` aren't usable standalone
    as they have global node and hyperedge IDs. They must be concatenated with the original :class:`HData` object
    that is provided as input to the ``sample`` method, as it contains the global node and hyperedge IDs and features
    that can be indexed with the negative samples' IDs.

    Args:
        num_negative_samples (int): Number of negative hyperedges to generate.
        num_nodes_per_sample (int): Number of nodes per negative hyperedge.
        return_0based_negatives (bool, optional):
            - If ``True``, the negative samples returned by the ``sample`` method will have 0-based node and hyperedge IDs.
            - If ``False``, the negative samples will retain the original global node and hyperedge IDs from the input data.

    Raises:
        ValueError: If either argument is not positive.
    """

    def __init__(
        self,
        num_negative_samples: int,
        num_nodes_per_sample: int,
        return_0based_negatives: bool = False,
    ):
        if num_negative_samples <= 0:
            raise ValueError(f"num_negative_samples must be positive, got {num_negative_samples}.")
        if num_nodes_per_sample <= 0:
            raise ValueError(f"num_nodes_per_sample must be positive, got {num_nodes_per_sample}.")

        super().__init__(return_0based_negatives=return_0based_negatives)
        self.num_negative_samples = num_negative_samples
        self.num_nodes_per_sample = num_nodes_per_sample

    def sample(self, data: HData) -> HData:
        """
        Generate negative hyperedges by randomly sampling unique node IDs.
        Node IDs are sampled from the same node space as the input data, and the new negative hyperedge IDs
        start from the original number of hyperedges in the input data to avoid ID conflicts.
        The resulting negative samples are returned as a new :class:`HData` object with remapped 0-based node and hyperedge IDs, if ``self.return_0based_negatives == True``.
        Otherwise, the negative samples retain their original global node and hyperedge IDs from the input data.

        Example:
            With ``self.return_0based_negatives = True``:
            >>> num_negative_samples = 2, num_nodes_per_sample = 3
            >>> negative_hyperedge_index = [[0, 0, 1, 2, 3, 4],
                                            [0, 1, 1, 0, 1, 0]]
                The negative hyperedge 0 connects nodes 0, 2, 3.
                The second negative hyperedge 1 connects nodes 0, 1, 4.
            >>> negative_x = data.x[[0, 1, 2, 3, 4]]
            >>> negative_hyperedge_attr = random_attributes_for_2_negative_hyperedges if data.edge_attr is not None

            With ``self.return_0based_negatives = False``:
            >>> num_negative_samples = 2, num_nodes_per_sample = 3
            >>> negative_hyperedge_index = [[100, 120, 300, 450, 500, 501],
                                            [3, 3, 3, 4, 4, 4]]
            >>> negative_x = data.x  # since node IDs are not remapped, we can directly use the original feature matrix and index into it with the original node IDs of the negative samples
            >>> negative_hyperedge_attr = data.edge_attr.cat([data.edge_attr, random_attributes_for_2_negative_hyperedges]) if data.edge_attr is not None
        Args:
            data (HData): The input data object containing node and hyperedge information.

        Returns:
            A new :class:`HData` instance containing the negative samples.

        Raises:
            ValueError: If ``num_nodes_per_sample`` is greater than the number of available nodes.
        """
        if self.num_nodes_per_sample > data.num_nodes:
            raise ValueError(
                f"Asked to create samples with {self.num_nodes_per_sample} nodes, but only {data.num_nodes} nodes are available."
            )

        device = data.device

        negative_node_ids: Set[int] = set()
        sampled_hyperedge_indexes: List[Tensor] = []
        sampled_hyperedge_attrs: List[Tensor] = []

        new_hyperedge_id_offset = data.num_edges
        for new_hyperedge_id in range(self.num_negative_samples):
            # Sample with multinomial without replacement to ensure unique node ids
            # and assign each node id equal probability of being selected by setting all of them to 1
            # Example: num_nodes_per_sample=3, max_node_id=5
            #          -> possible output: [2, 0, 4]
            equal_probabilities = torch.ones(data.num_nodes, device=device)
            sampled_node_ids = torch.multinomial(
                input=equal_probabilities,
                num_samples=self.num_nodes_per_sample,
                replacement=False,
            )

            # Example: sampled_node_ids = [2, 0, 4], new_hyperedge_id=0, new_hyperedge_id_offset=3
            #          -> hyperedge_index = [[2, 0, 4],
            #                                [3, 3, 3]]  # this is sampled_hyperedge_id_tensor
            sampled_hyperedge_id_tensor = torch.full(
                (self.num_nodes_per_sample,),
                new_hyperedge_id + new_hyperedge_id_offset,
                device=device,
            )
            sampled_hyperedge_index = torch.stack(
                [sampled_node_ids, sampled_hyperedge_id_tensor], dim=0
            )
            sampled_hyperedge_indexes.append(sampled_hyperedge_index)

            # Example: nodes = [0, 1, 2],
            #          sampled_node_ids_0 = [0, 1], sampled_node_ids_1 = [1, 2],
            #          -> negative_node_ids = {0, 1, 2}
            negative_node_ids.update(sampled_node_ids.tolist())

            if data.edge_attr is not None:
                random_hyperedge_attr = torch.randn_like(data.edge_attr[0], device=device)
                sampled_hyperedge_attrs.append(random_hyperedge_attr)

        negative_node_ids_tensor = torch.tensor(list(negative_node_ids), device=device)
        new_x, num_negative_nodes = self._new_x(data.x, negative_node_ids_tensor)

        # Example: new_hyperedge_id_offset = 3 (if data.num_edges was 3)
        #          num_negative_samples = 2
        #          -> num_hyperedges_including_negatives = 5
        num_hyperedges_including_negatives = new_hyperedge_id_offset + self.num_negative_samples
        negative_hyperedge_ids = torch.arange(
            new_hyperedge_id_offset,
            num_hyperedges_including_negatives,
            device=device,
        )

        negative_hyperedge_index = self._new_negative_hyperedge_index(
            sampled_hyperedge_indexes,
            negative_node_ids_tensor,
            negative_hyperedge_ids,
        )

        return HData(
            x=new_x,
            edge_index=negative_hyperedge_index,
            edge_attr=self._new_hyperedge_attr(sampled_hyperedge_attrs, data.edge_attr),
            num_nodes=num_negative_nodes,
            num_edges=self.num_negative_samples,
        ).with_y_zeros()
