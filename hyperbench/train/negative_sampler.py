import torch

from abc import ABC, abstractmethod
from torch import Tensor
from typing import List, Optional, Set, Tuple
from hyperbench.types import HData, HyperedgeIndex


class NegativeSampler(ABC):
    """
    Abstract base class for negative samplers.

    Args:
        return_0based_negatives:
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

        Returns:
            The negative samples as a new :class:`HData` object.

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
            sampled_hyperedge_indexes: List of hyperedge index tensors for each negative sample.
            negative_node_ids: Tensor of negative node IDs.
            negative_hyperedge_ids: Tensor of negative hyperedge IDs.

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

    def _new_global_node_ids(
        self,
        global_node_ids: Optional[Tensor],
        negative_node_ids: Tensor,
    ) -> Optional[Tensor]:
        """
        Get the global node IDs for the negative samples.

        Args:
            global_node_ids: The original global node IDs from the input data.
            negative_node_ids: Tensor of negative node IDs.

        Returns:
            The global node IDs for the negative samples, or ``None`` if the input global node IDs are ``None``.
        """
        if global_node_ids is None:
            return None
        return global_node_ids[negative_node_ids]

    def _new_hyperedge_attr(
        self,
        sampled_hyperedge_attrs: List[Tensor],
        hyperedge_attr: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        """
        Concatenate the hyperedge attributes for the negative samples.

        Args:
            sampled_hyperedge_attrs: List of hyperedge attribute tensors for each negative sample.
            hyperedge_attr: The original hyperedge attributes from the input data.

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
            x: The original node feature matrix from the input data.
            negative_node_ids: Tensor of negative node IDs.

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
        num_negative_samples: Number of negative hyperedges to generate.
        num_nodes_per_sample: Number of nodes per negative hyperedge.
        return_0based_negatives:
            - If ``True``, the negative samples returned by the ``sample`` method will have 0-based node and hyperedge IDs.
            - If ``False``, the negative samples will retain the original global node and hyperedge IDs from the input data.

    Raises:
        ValueError: If either argument is not positive.
    """

    def __init__(
        self,
        num_negative_samples: int,
        num_nodes_per_sample: int,
        max_retries: int = 10,
        return_0based_negatives: bool = False,
    ):
        if num_negative_samples <= 0:
            raise ValueError(f"num_negative_samples must be positive, got {num_negative_samples}.")
        if num_nodes_per_sample <= 0:
            raise ValueError(f"num_nodes_per_sample must be positive, got {num_nodes_per_sample}.")
        if max_retries <= 0:
            raise ValueError(f"max_retries must be positive, got {max_retries}.")

        super().__init__(return_0based_negatives=return_0based_negatives)
        self.num_negative_samples = num_negative_samples
        self.num_nodes_per_sample = num_nodes_per_sample
        self.max_retries = max_retries

    def sample(self, data: HData) -> HData:
        """
        Generate negative hyperedges by randomly sampling unique node IDs.
        Node IDs are sampled from the same node space as the input data, and the new negative hyperedge IDs
        start from the original number of hyperedges in the input data to avoid ID conflicts.
        The resulting negative samples are returned as a new :class:`HData` object with remapped 0-based node and hyperedge IDs, if ``self.return_0based_negatives == True``.
        Otherwise, the negative samples retain their original global node and hyperedge IDs from the input data.

        Examples:
            With ``self.return_0based_negatives = True``:

            >>> num_negative_samples = 2
            >>> num_nodes_per_sample = 3
            >>> negative_hyperedge_index = [[0, 0, 1, 2, 3, 4],
            ...                             [0, 1, 1, 0, 1, 0]]

            The negative hyperedge 0 connects nodes 0, 2, 3.
            The second negative hyperedge 1 connects nodes 0, 1, 4.

            >>> negative_x = data.x[[0, 1, 2, 3, 4]]
            >>> negative_hyperedge_attr = random_attributes_for_2_negative_hyperedges

            With ``self.return_0based_negatives = False``:

            >>> num_negative_samples = 2
            >>> num_nodes_per_sample = 3
            >>> negative_hyperedge_index = [[100, 120, 300, 450, 500, 501],
            ...                             [3, 3, 3, 4, 4, 4]]

            Since node IDs are not remapped, the original feature matrix can be used directly.

            >>> negative_x = data.x

        Args:
            data: The input data object containing node and hyperedge information.

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

        # Build a set of existing positive hyperedges for dedup check
        # Each hyperedge is represented as a frozenset of node IDs
        positive_hyperedge_set: Set[frozenset] = set()
        he_index = data.hyperedge_index
        num_hyperedges = data.num_hyperedges
        for he_id in range(num_hyperedges):
            mask = he_index[1] == he_id
            nodes = frozenset(he_index[0, mask].tolist())
            positive_hyperedge_set.add(nodes)

        # Track sampled negatives to avoid duplicates
        sampled_negative_set: Set[frozenset] = set()

        new_hyperedge_id_offset = data.num_hyperedges
        generated_count = 0
        max_iterations = self.num_negative_samples * (1 + self.max_retries)

        for _ in range(max_iterations):
            if generated_count >= self.num_negative_samples:
                break

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

            # Check if this hyperedge is already a positive hyperedge
            sampled_set = frozenset(sampled_node_ids.tolist())
            if sampled_set in positive_hyperedge_set or sampled_set in sampled_negative_set:
                continue  # Skip: already exists as positive or duplicate negative

            sampled_negative_set.add(sampled_set)

            # Example: sampled_node_ids = [2, 0, 4], new_hyperedge_id=0, new_hyperedge_id_offset=3
            #          -> hyperedge_index = [[2, 0, 4],
            #                                [3, 3, 3]]  # this is sampled_hyperedge_id_tensor
            new_hyperedge_id = generated_count + new_hyperedge_id_offset
            sampled_hyperedge_id_tensor = torch.full(
                (self.num_nodes_per_sample,),
                new_hyperedge_id,
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

            if data.hyperedge_attr is not None:
                random_hyperedge_attr = torch.randn_like(data.hyperedge_attr[0], device=device)
                sampled_hyperedge_attrs.append(random_hyperedge_attr)

            generated_count += 1

        if generated_count < self.num_negative_samples:
            raise ValueError(
                f"Could only generate {generated_count} unique negative samples "
                f"out of {self.num_negative_samples} requested after "
                f"{max_iterations} attempts. "
                f"Try reducing num_negative_samples or num_nodes_per_sample, "
                f"or increasing max_retries (current: {self.max_retries})."
            )

        negative_node_ids_tensor = torch.tensor(sorted(negative_node_ids), device=device)
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
            hyperedge_index=negative_hyperedge_index,
            hyperedge_attr=self._new_hyperedge_attr(sampled_hyperedge_attrs, data.hyperedge_attr),
            num_nodes=num_negative_nodes,
            num_hyperedges=self.num_negative_samples,
            global_node_ids=self._new_global_node_ids(
                data.global_node_ids, negative_node_ids_tensor
            ),
        ).with_y_zeros()
