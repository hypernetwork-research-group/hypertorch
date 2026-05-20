import torch

from abc import ABC, abstractmethod
from math import comb
from torch import Tensor
from hyperbench.data.enricher import (
    HyperedgeAttrsEnricher,
    HyperedgeWeightsEnricher,
    NodeEnricher,
)
from hyperbench.types import HData, HyperedgeIndex
from hyperbench.utils import create_seeded_torch_generator


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
    def sample(self, hdata: HData, seed: int | None = None) -> HData:
        """
        Abstract method for negative sampling.

        Args:
            hdata: The input data object containing graph or hypergraph information.
            seed: Optional random seed for reproducible negative sampling.

        Returns:
            hdata: The negative samples as a new `HData` object.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def _new_negative_hyperedge_index(
        self,
        sampled_hyperedge_indexes: list[Tensor],
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
            hyperedge_index: The concatenated, sorted, and remapped hyperedge index tensor.
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
        global_node_ids: Tensor | None,
        negative_node_ids: Tensor,
    ) -> Tensor | None:
        """
        Get the global node IDs for the negative samples.

        Args:
            global_node_ids: The original global node IDs from the input data.
            negative_node_ids: Tensor of negative node IDs.

        Returns:
            global_node_ids: The global node IDs for the negative samples, or ``None`` if the input global node IDs are ``None``.
        """
        if global_node_ids is None:
            return None
        return global_node_ids[negative_node_ids]

    def _new_hyperedge_attr(
        self,
        sampled_hyperedge_attrs: list[Tensor],
        hyperedge_attr: Tensor | None = None,
    ) -> Tensor | None:
        """
        Concatenate the hyperedge attributes for the negative samples.

        Args:
            sampled_hyperedge_attrs: List of hyperedge attribute tensors for each negative sample.
            hyperedge_attr: The original hyperedge attributes from the input data.

        Returns:
            hyperedge_attr: The concatenated hyperedge attribute tensor for the negative samples.
        """
        if hyperedge_attr is None or len(sampled_hyperedge_attrs) < 1:
            return None

        negative_hyperedge_attr = torch.stack(sampled_hyperedge_attrs, dim=0)
        return negative_hyperedge_attr

    def _new_enriched_hyperedge_attr(
        self,
        hyperedge_attr_enricher: HyperedgeAttrsEnricher | None,
        negative_hyperedge_index: Tensor,
    ) -> Tensor | None:
        """
        Generate enriched hyperedge attributes for the negative samples.

        Args:
            hyperedge_attr_enricher: An optional `HyperedgeAttrsEnricher` to generate attributes for the new hyperedges.
            negative_hyperedge_index: The index tensor for the negative hyperedges.

        Returns:
            hyperedge_attr: The enriched hyperedge attribute tensor for the negative samples, or ``None`` if the enricher is not provided.
        """
        if hyperedge_attr_enricher is None:
            return None

        negative_hyperedge_index_0based = (
            HyperedgeIndex(negative_hyperedge_index.clone()).to_0based().item
        )
        return hyperedge_attr_enricher.enrich(negative_hyperedge_index_0based)

    def _new_enriched_hyperedge_weights(
        self,
        hyperedge_weights_enricher: HyperedgeWeightsEnricher | None,
        negative_hyperedge_index: Tensor,
    ) -> Tensor | None:
        """
        Generate enriched hyperedge weights for the negative samples.

        Args:
            hyperedge_weights_enricher: An optional `HyperedgeWeightsEnricher` to generate weights for the new hyperedges.
            negative_hyperedge_index: The index tensor for the negative hyperedges.

        Returns:
            hyperedge_weights: The enriched hyperedge weight tensor for the negative samples, or ``None`` if the enricher is not provided.
        """
        if hyperedge_weights_enricher is None:
            return None

        negative_hyperedge_index_0based = (
            HyperedgeIndex(negative_hyperedge_index.clone()).to_0based().item
        )
        return hyperedge_weights_enricher.enrich(negative_hyperedge_index_0based)

    def _new_x(self, x: Tensor, negative_node_ids: Tensor) -> tuple[Tensor, int]:
        """
        Get the node feature matrix for the negative samples.

        Args:
            x: The original node feature matrix from the input data.
            negative_node_ids: Tensor of negative node IDs.

        Returns:
            x_and_num_negative_nodes: The node feature matrix for the negative samples and the number of negative nodes.
        """
        return x[negative_node_ids], len(negative_node_ids)

    def _new_negative_hyperedge_ids(
        self,
        new_hyperedge_id_offset: int,
        num_negative_samples: int,
        device: torch.device,
    ) -> Tensor:
        """
        Build the hyperedge IDs assigned to sampled negative hyperedges.

        Args:
            new_hyperedge_id_offset: First negative hyperedge ID,
                usually the number of positive hyperedges in the input data.
            num_negative_samples: Number of negative hyperedge IDs to create.
            device: Device where the returned tensor should be allocated.

        Returns:
            hyperedge_ids: A tensor containing consecutive negative hyperedge IDs.
        """
        # Example: new_hyperedge_id_offset = 3 (if hdata.num_hyperedges was 3)
        #          num_negative_samples = 2
        #          -> negative_hyperedge_ids = [3, 4]
        num_hyperedges_including_negatives = new_hyperedge_id_offset + num_negative_samples
        return torch.arange(
            start=new_hyperedge_id_offset,
            end=num_hyperedges_including_negatives,
            device=device,
        )

    def _hyperedges_signatures(self, hyperedge_index: Tensor) -> set[tuple[int, ...]]:
        """
        Build order-independent node signatures for every hyperedge in a hyperedge index.

        Args:
            hyperedge_index: Tensor of shape ``[2, num_incidences]`` containing node
                and hyperedge IDs.

        Returns:
            signatures: A set of sorted node ID tuples, one tuple per hyperedge.
        """
        all_hyperedge_ids = hyperedge_index[1]
        unique_hyperedge_ids = all_hyperedge_ids.unique().tolist()

        signatures: set[tuple[int, ...]] = set()
        for hyperedge_id in unique_hyperedge_ids:
            node_ids_in_hyperedge_mask = all_hyperedge_ids == hyperedge_id
            node_ids_in_hyperedge = hyperedge_index[0][node_ids_in_hyperedge_mask]
            signatures.add(self._hyperedge_nodes_signature(node_ids_in_hyperedge.unique()))
        return signatures

    def _hyperedge_nodes_signature(
        self,
        node_ids: Tensor | list[int] | tuple[int, ...],
    ) -> tuple[int, ...]:
        """
        Convert node IDs into a sorted tuple that can be used as a hyperedge signature.

        Args:
            node_ids: A tensor or sequence containing node IDs for one hyperedge.

        Returns:
            signature: A sorted tuple of Python ``int`` values.
        """
        if isinstance(node_ids, Tensor):
            return tuple(sorted(int(node_id) for node_id in node_ids.tolist()))
        return tuple(sorted(int(node_id) for node_id in node_ids))


class SameNodeSpaceNegativeSampler(NegativeSampler, ABC):
    """
    Base class for negative samplers that sample only from existing nodes.

    Args:
        hyperedge_attr_enricher: An optional `HyperedgeAttrsEnricher` to generate attributes for the new hyperedges.
        hyperedge_weights_enricher: An optional `HyperedgeWeightsEnricher` to generate weights for the new hyperedges.
        return_0based_negatives:
            - If ``True``, the negative samples returned by the ``sample`` method will have 0-based node and hyperedge IDs.
            - If ``False``, the negative samples will retain the original global node and hyperedge IDs from the input data.
    """

    def __init__(
        self,
        hyperedge_attr_enricher: HyperedgeAttrsEnricher | None = None,
        hyperedge_weights_enricher: HyperedgeWeightsEnricher | None = None,
        return_0based_negatives: bool = False,
    ):
        super().__init__(return_0based_negatives=return_0based_negatives)
        self.hyperedge_attr_enricher = hyperedge_attr_enricher
        self.hyperedge_weights_enricher = hyperedge_weights_enricher


class GeneratedNodesNegativeSampler(NegativeSampler, ABC):
    """
    Base class for negative samplers that generate new nodes instead of sampling from existing ones.

    Args:
        node_feature_enricher: A `NodeEnricher` to generate features for the new nodes.
        hyperedge_attr_enricher: An optional `HyperedgeAttrsEnricher` to generate attributes for the new hyperedges.
        hyperedge_weights_enricher: An optional `HyperedgeWeightsEnricher` to generate weights for the new hyperedges.
        return_0based_negatives:
            - If ``True``, the negative samples returned by the ``sample`` method will have 0-based node and hyperedge IDs.
            - If ``False``, the negative samples will retain the original global node and hyperedge IDs from the input data.
    """

    def __init__(
        self,
        node_feature_enricher: NodeEnricher,
        hyperedge_attr_enricher: HyperedgeAttrsEnricher | None = None,
        hyperedge_weights_enricher: HyperedgeWeightsEnricher | None = None,
        return_0based_negatives: bool = False,
    ):
        super().__init__(return_0based_negatives=return_0based_negatives)
        self.node_feature_enricher = node_feature_enricher
        self.hyperedge_attr_enricher = hyperedge_attr_enricher
        self.hyperedge_weights_enricher = hyperedge_weights_enricher


class RandomNegativeSampler(SameNodeSpaceNegativeSampler):
    """
    A random negative sampler. Negatives generated with ``return_0based_negatives = False`` aren't usable standalone
    as they have global node and hyperedge IDs. They must be concatenated with the original `HData` object
    that is provided as input to the ``sample`` method, as it contains the global node and hyperedge IDs and features
    that can be indexed with the negative samples' IDs.

    Args:
        num_negative_samples: Number of negative hyperedges to generate.
        num_nodes_per_sample: Number of nodes per negative hyperedge.
        hyperedge_attr_enricher: An optional `HyperedgeAttrsEnricher` to generate attributes for the new hyperedges.
            If not provided, random attributes will be generated for the negative hyperedges if the input data has hyperedge attributes.
        hyperedge_weights_enricher: An optional `HyperedgeEnricher` to generate weights for the new hyperedges.
            If not provided, the negative hyperedges will not have weights.
        return_0based_negatives:
            - If ``True``, the negative samples returned by the ``sample`` method will have 0-based node and hyperedge IDs.
            - If ``False``, the negative samples will retain the original global node and hyperedge IDs from the input data.
        max_retry: Maximum number of rejected sampling attempts allowed per requested negative hyperedge before failing.
            If ``num_negative_samples`` is ``N``, the total maximum number of attempts will be ``N * max_retry``.

    Raises:
        ValueError: If any numeric argument is not positive.
    """

    def __init__(
        self,
        num_negative_samples: int,
        num_nodes_per_sample: int,
        hyperedge_attr_enricher: HyperedgeAttrsEnricher | None = None,
        hyperedge_weights_enricher: HyperedgeWeightsEnricher | None = None,
        return_0based_negatives: bool = False,
        max_retry: int = 100,
    ):
        if num_negative_samples <= 0:
            raise ValueError(f"num_negative_samples must be positive, got {num_negative_samples}.")
        if num_nodes_per_sample <= 0:
            raise ValueError(f"num_nodes_per_sample must be positive, got {num_nodes_per_sample}.")
        if max_retry <= 0:
            raise ValueError(f"max_retry must be positive, got {max_retry}.")

        super().__init__(
            hyperedge_attr_enricher=hyperedge_attr_enricher,
            hyperedge_weights_enricher=hyperedge_weights_enricher,
            return_0based_negatives=return_0based_negatives,
        )
        self.num_negative_samples = num_negative_samples
        self.num_nodes_per_sample = num_nodes_per_sample
        self.max_retry = max_retry

    def sample(self, hdata: HData, seed: int | None = None) -> HData:
        """
        Generate negative hyperedges by randomly sampling unique node IDs.
        Node IDs are sampled from the same node space as the input data, and the new negative hyperedge IDs
        start from the original number of hyperedges in the input data to avoid ID conflicts.
        The resulting negative samples are returned as a new `HData` object with remapped 0-based node and hyperedge IDs, if ``self.return_0based_negatives == True``.
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
            hdata: The input data object containing node and hyperedge information.
            seed: Optional random seed for reproducible negative sampling.

        Returns:
            hdata: A new `HData` instance containing the negative samples.

        Raises:
            ValueError: If ``num_nodes_per_sample`` is greater than the number of available nodes.
        """
        if self.num_nodes_per_sample > hdata.num_nodes:
            raise ValueError(
                f"Asked to create samples with {self.num_nodes_per_sample} nodes, but only {hdata.num_nodes} nodes are available."
            )

        device = hdata.device

        # Existing positive hyperedges of the requested size must not be returned as negatives
        # Example: hyperedge_index = [[0, 1, 2, 0, 3],
        #                             [0, 0, 0, 1, 1]],
        #          num_nodes_per_sample = 3
        #          -> positive_hyperedge_signatures = {(0, 1, 2), (0, 3)}
        #          so (0, 1, 2) and (0, 3) are rejected even though they are cliques
        positive_hyperedge_signatures = self._hyperedges_signatures(hdata.hyperedge_index)
        matching_size_positive_hyperedges_signatures = {
            signature
            for signature in positive_hyperedge_signatures
            if len(signature) == self.num_nodes_per_sample
        }
        self.__validate_enough_negative_hyperedges(
            hdata=hdata,
            positive_hyperedges_signatures=matching_size_positive_hyperedges_signatures,
        )

        (
            sampled_hyperedge_indexes,
            sampled_hyperedge_attrs,
            sampled_negative_node_ids,
            new_hyperedge_id_offset,
        ) = self.__sample_loop(
            hdata=hdata,
            positive_hyperedges_signatures=matching_size_positive_hyperedges_signatures,
            seed=seed,
        )

        negative_node_ids_tensor = torch.tensor(sorted(sampled_negative_node_ids), device=device)
        new_x, num_negative_nodes = self._new_x(hdata.x, negative_node_ids_tensor)

        negative_hyperedge_ids = self._new_negative_hyperedge_ids(
            new_hyperedge_id_offset=new_hyperedge_id_offset,
            num_negative_samples=self.num_negative_samples,
            device=device,
        )

        negative_hyperedge_index = self._new_negative_hyperedge_index(
            sampled_hyperedge_indexes,
            negative_node_ids_tensor,
            negative_hyperedge_ids,
        )

        negative_hyperedge_attr = self._new_enriched_hyperedge_attr(
            hyperedge_attr_enricher=self.hyperedge_attr_enricher,
            negative_hyperedge_index=negative_hyperedge_index,
        )
        # Default to the random attributes if no enricher is provided and the input data has hyperedge attributes
        if negative_hyperedge_attr is None:
            negative_hyperedge_attr = self._new_hyperedge_attr(
                sampled_hyperedge_attrs=sampled_hyperedge_attrs, hyperedge_attr=hdata.hyperedge_attr
            )

        return HData(
            x=new_x,
            hyperedge_index=negative_hyperedge_index,
            hyperedge_weights=self._new_enriched_hyperedge_weights(
                hyperedge_weights_enricher=self.hyperedge_weights_enricher,
                negative_hyperedge_index=negative_hyperedge_index,
            ),
            hyperedge_attr=negative_hyperedge_attr,
            num_nodes=num_negative_nodes,
            num_hyperedges=self.num_negative_samples,
            global_node_ids=self._new_global_node_ids(
                global_node_ids=hdata.global_node_ids, negative_node_ids=negative_node_ids_tensor
            ),
        ).with_y_zeros()

    def __sample_loop(
        self,
        hdata: HData,
        positive_hyperedges_signatures: set[tuple[int, ...]],
        seed: int | None = None,
    ) -> tuple[list[Tensor], list[Tensor], set[int], int]:
        """
        Sample unique negative hyperedges until the requested count or retry limit is reached.

        Args:
            hdata: The input hypergraph data used as the node and hyperedge ID source.
            positive_hyperedges_signatures: Existing positive hyperedge signatures that must not be sampled as negatives.
            seed: Optional random seed for reproducible sampling.

        Returns:
            samples: A tuple containing sampled hyperedge index tensors, sampled hyperedge attribute
            tensors, sampled node IDs, and the first negative hyperedge ID.

        Raises:
            ValueError: If the sampler cannot produce the requested number of unique negative
                hyperedges within the number of maximum allowed attempts.
        """
        device = hdata.device
        generator = create_seeded_torch_generator(device=device, seed=seed)

        sampled_negative_node_ids: set[int] = set()
        sampled_negative_hyperedge_signatures: set[tuple[int, ...]] = set()
        sampled_hyperedge_indexes: list[Tensor] = []
        sampled_hyperedge_attrs: list[Tensor] = []

        new_hyperedge_id_offset = hdata.num_hyperedges
        # max_retry is per requested negative, so scale the retries with the requested count.
        max_attempts = self.num_negative_samples * self.max_retry
        attempts = 0
        while (
            len(sampled_hyperedge_indexes) < self.num_negative_samples and attempts < max_attempts
        ):
            attempts += 1

            # Sample with multinomial without replacement to ensure unique node ids
            # and assign each node id equal probability of being selected by setting all of them to 1
            # Example: num_nodes_per_sample=3, max_node_id=5
            #          -> possible output: [2, 0, 4]
            equal_probabilities = torch.ones(hdata.num_nodes, device=device)
            sampled_node_ids = torch.multinomial(
                input=equal_probabilities,
                num_samples=self.num_nodes_per_sample,
                replacement=False,
                generator=generator,
            )

            sampled_nodes_signature = self._hyperedge_nodes_signature(sampled_node_ids)
            if (
                sampled_nodes_signature in positive_hyperedges_signatures
                or sampled_nodes_signature in sampled_negative_hyperedge_signatures
            ):
                # Reject this sample as it already exists as a positive
                # or as a previously sampled negative hyperedge
                continue
            sampled_negative_hyperedge_signatures.add(sampled_nodes_signature)

            # Example: sampled_node_ids = [2, 0, 4], new_hyperedge_id=0, new_hyperedge_id_offset=3
            #          -> hyperedge_index = [[2, 0, 4],
            #                                [3, 3, 3]]  # this is sampled_hyperedge_id_tensor
            new_hyperedge_id = len(sampled_hyperedge_indexes)
            sampled_hyperedge_id_tensor = torch.full(
                size=(self.num_nodes_per_sample,),
                fill_value=new_hyperedge_id + new_hyperedge_id_offset,
                device=device,
            )
            sampled_hyperedge_index = torch.stack(
                [sampled_node_ids, sampled_hyperedge_id_tensor], dim=0
            )
            sampled_hyperedge_indexes.append(sampled_hyperedge_index)

            # Example: nodes = [0, 1, 2],
            #          sampled_node_ids_0 = [0, 1], sampled_node_ids_1 = [1, 2],
            #          -> sampled_negative_node_ids = {0, 1, 2}
            sampled_negative_node_ids.update(sampled_node_ids.tolist())

            if hdata.hyperedge_attr is not None:
                random_hyperedge_attr = torch.randn(
                    size=hdata.hyperedge_attr[0].shape,
                    dtype=hdata.hyperedge_attr.dtype,
                    generator=generator,
                    device=device,
                )
                sampled_hyperedge_attrs.append(random_hyperedge_attr)

        if len(sampled_hyperedge_indexes) < self.num_negative_samples:
            raise ValueError(
                "Unable to sample "
                f"{self.num_negative_samples} unique negative hyperedges after "
                f"{max_attempts} attempts. Increase max_retry or request fewer samples."
            )

        return (
            sampled_hyperedge_indexes,
            sampled_hyperedge_attrs,
            sampled_negative_node_ids,
            new_hyperedge_id_offset,
        )

    def __validate_enough_negative_hyperedges(
        self,
        hdata: HData,
        positive_hyperedges_signatures: set[tuple[int, ...]],
    ) -> None:
        """
        Validate that enough unique negative hyperedges exist for the requested sample count.

        Args:
            hdata: The input hypergraph data that defines the number of available nodes.
            positive_hyperedges_signatures: Positive hyperedge signatures with the same size
                as the requested negative hyperedges.

        Raises:
            ValueError: If the requested number of negatives exceeds the number of possible
                unique non-positive hyperedges.
        """
        num_possible_hyperedges_by_size = comb(hdata.num_nodes, self.num_nodes_per_sample)
        num_positive_hyperedges = len(positive_hyperedges_signatures)
        num_possible_negative_hyperedges = num_possible_hyperedges_by_size - num_positive_hyperedges

        if self.num_negative_samples > num_possible_negative_hyperedges:
            raise ValueError(
                "Asked to create "
                f"{self.num_negative_samples} unique negative samples with "
                f"{self.num_nodes_per_sample} nodes each, but only "
                f"{num_possible_negative_hyperedges} are available."
            )


class CliqueNegativeSampler(SameNodeSpaceNegativeSampler):
    """
    Sample negative hyperedges that are cliques in the underlying graph.

    The underlying graph is obtained through clique expansion: two nodes are adjacent when
    they co-occur in at least one positive hyperedge. A sampled negative hyperedge contains
    ``num_nodes_per_sample`` nodes where every pair is adjacent, and the node set must not
    already exist as a positive hyperedge.

    Args:
        num_negative_samples: Number of negative hyperedges to generate.
        num_nodes_per_sample: Number of nodes per negative hyperedge. Must be at least 2.
        hyperedge_attr_enricher: Optional enricher to generate attributes for sampled negatives.
        hyperedge_weights_enricher: Optional enricher to generate weights for sampled negatives.
        return_0based_negatives: If ``True``, returned negative node and hyperedge IDs are rebased to 0-based IDs.
        max_candidates: Optional upper bound for full-size clique candidates enumerated during search
            If ``None``, it means no explicit cap. The limit counts every full-size clique candidate
            encountered before positive-hyperedge filtering, so positive hyperedges still consume the budget
            because they still require search work. This is a safety guard for dense graphs where clique enumeration
            can grow quickly. For example, ``max_candidates=10_000`` means the sampler stops if finding candidates
            requires enumerating more than 10,000 cliques of size ``num_nodes_per_sample``.
            It does not control how many negatives are returned, as that is controlled by ``num_negative_samples``.

    Raises:
        ValueError: If numeric arguments are invalid.
    """

    def __init__(
        self,
        num_negative_samples: int,
        num_nodes_per_sample: int,
        hyperedge_attr_enricher: HyperedgeAttrsEnricher | None = None,
        hyperedge_weights_enricher: HyperedgeWeightsEnricher | None = None,
        return_0based_negatives: bool = False,
        max_candidates: int | None = None,
    ):
        if num_negative_samples <= 0:
            raise ValueError(f"num_negative_samples must be positive, got {num_negative_samples}.")
        if num_nodes_per_sample < 2:
            raise ValueError(
                f"num_nodes_per_sample must be at least 2 for clique negative sampling, got {num_nodes_per_sample}."
            )
        if max_candidates is not None and max_candidates <= 0:
            raise ValueError(
                f"max_candidates must be positive when provided, got {max_candidates}."
            )

        super().__init__(
            hyperedge_attr_enricher=hyperedge_attr_enricher,
            hyperedge_weights_enricher=hyperedge_weights_enricher,
            return_0based_negatives=return_0based_negatives,
        )
        self.num_negative_samples = num_negative_samples
        self.num_nodes_per_sample = num_nodes_per_sample
        self.max_candidates = max_candidates

    def sample(self, hdata: HData, seed: int | None = None) -> HData:
        """
        Generate clique-based negative hyperedges from the input hypergraph.

        Args:
            hdata: Input hypergraph data.
            seed: Optional random seed for reproducible candidate selection.

        Returns:
            hdata: A new `HData` instance containing only sampled negative hyperedges.

        Raises:
            ValueError: If too few nodes or valid clique negatives are available.
        """
        if self.num_nodes_per_sample > hdata.num_nodes:
            raise ValueError(
                f"Asked to create samples with {self.num_nodes_per_sample} nodes, but only {hdata.num_nodes} nodes are available."
            )
        device = hdata.device

        # Example: hyperedge_index = [[0, 1, 2, 0, 3],
        #                             [0, 0, 0, 1, 1]],
        #          num_nodes_per_sample = 3
        #          -> positive_hyperedge_signatures = {(0, 1, 2)}
        #          so (0, 1, 2) is rejected even though it is a clique.
        positive_hyperedge_signatures = {
            signature
            for signature in self._hyperedges_signatures(hdata.hyperedge_index)
            if len(signature) == self.num_nodes_per_sample
        }

        # First build the clique-expanded graph, then enumerate node sets that are cliques
        # in that graph: these are the negatives with pairwise cohesion.
        # Example: hyperedge_index = [[0, 1, 2, 0, 3],
        #                             [0, 0, 0, 1, 1]],
        #          -> adjacency_list = [0: {1, 2, 3}, 1: {0, 2}, 2: {0, 1}, 3: {0}]
        #          -> clique candidates of size 3 = [(0, 1, 2)]
        #          -> valid_clique_candidates = [] because (0, 1, 2) is already positive
        #             if num_nodes_per_sample == 2 instead:
        #             clique candidates would be [(0, 1), (0, 2), (0, 3), (1, 2)]
        adjacency_list = HyperedgeIndex(hdata.hyperedge_index).get_clique_expansion_adjacency_list(
            num_nodes=hdata.num_nodes
        )
        valid_clique_candidates = self.__find_valid_clique_candidates(
            adjacency_list=adjacency_list,
            positive_hyperedge_signatures=positive_hyperedge_signatures,
        )

        (
            sampled_hyperedge_indexes,
            sampled_hyperedge_attrs,
            sampled_negative_node_ids,
            new_hyperedge_id_offset,
        ) = self.__sample_loop(
            hdata=hdata,
            clique_candidates=valid_clique_candidates,
            seed=seed,
        )

        negative_node_ids_tensor = torch.tensor(sorted(sampled_negative_node_ids), device=device)
        new_x, num_negative_nodes = self._new_x(hdata.x, negative_node_ids_tensor)

        negative_hyperedge_ids = self._new_negative_hyperedge_ids(
            new_hyperedge_id_offset=new_hyperedge_id_offset,
            num_negative_samples=self.num_negative_samples,
            device=device,
        )

        negative_hyperedge_index = self._new_negative_hyperedge_index(
            sampled_hyperedge_indexes,
            negative_node_ids_tensor,
            negative_hyperedge_ids,
        )

        negative_hyperedge_attr = self._new_enriched_hyperedge_attr(
            hyperedge_attr_enricher=self.hyperedge_attr_enricher,
            negative_hyperedge_index=negative_hyperedge_index,
        )
        if negative_hyperedge_attr is None:
            negative_hyperedge_attr = self._new_hyperedge_attr(
                sampled_hyperedge_attrs=sampled_hyperedge_attrs,
                hyperedge_attr=hdata.hyperedge_attr,
            )

        return HData(
            x=new_x,
            hyperedge_index=negative_hyperedge_index,
            hyperedge_weights=self._new_enriched_hyperedge_weights(
                hyperedge_weights_enricher=self.hyperedge_weights_enricher,
                negative_hyperedge_index=negative_hyperedge_index,
            ),
            hyperedge_attr=negative_hyperedge_attr,
            num_nodes=num_negative_nodes,
            num_hyperedges=self.num_negative_samples,
            global_node_ids=self._new_global_node_ids(
                global_node_ids=hdata.global_node_ids,
                negative_node_ids=negative_node_ids_tensor,
            ),
        ).with_y_zeros()

    def __expand_clique_candidates(
        self,
        prefix: tuple[int, ...],
        candidates: list[int],
        adjacency_list: list[set[int]],
        positive_hyperedge_signatures: set[tuple[int, ...]],
        valid_candidates: list[tuple[int, ...]],
        enumerated_candidates: int,
    ) -> int:
        """
        Recursively enumerate clique candidates from a clique-expanded adjacency list.

        Args:
            prefix: Current partial clique, represented as sorted node IDs.
            candidates: Node IDs that may extend ``prefix`` while preserving clique structure.
            adjacency_list: Clique-expanded graph adjacency list.
            positive_hyperedge_signatures: Positive hyperedge node signatures that must not be returned as negatives.
            valid_candidates: Output list mutated in place with valid negative clique candidates.
            enumerated_candidates: Number of full-size clique candidates visited so far.

        Returns:
            visited: Updated number of full-size clique candidates visited.

        Raises:
            ValueError: If ``max_candidates`` is set and clique enumeration exceeds it.
        """
        if len(prefix) == self.num_nodes_per_sample:  # Found a full-size clique candidate
            if self.max_candidates is not None and enumerated_candidates >= self.max_candidates:
                raise ValueError(
                    f"Clique negative candidate enumeration exceeded max_candidates={self.max_candidates}."
                )
            enumerated_candidates += 1

            signature = self._hyperedge_nodes_signature(prefix)
            if signature not in positive_hyperedge_signatures:
                valid_candidates.append(signature)

            return enumerated_candidates

        for node_idx, node_id in enumerate(candidates):
            # Keep only future candidates adjacent to the new node. Since previous
            # expansion steps already intersected with earlier prefix nodes, every
            # recursive prefix remains a clique
            # Example: prefix = (0,), candidates = [1, 2, 3],
            #          node_idx = 0, node_id = 1, adjacency_list[1] = {0, 2, 3},
            #          -> next_candidates = [2, 3]
            #             We don't pick 0 because it's already in the prefix
            next_candidates = [
                candidate_node_id
                for candidate_node_id in candidates[node_idx + 1 :]
                if candidate_node_id in adjacency_list[node_id]
            ]

            enumerated_candidates = self.__expand_clique_candidates(
                prefix=(*prefix, node_id),
                candidates=next_candidates,
                adjacency_list=adjacency_list,
                positive_hyperedge_signatures=positive_hyperedge_signatures,
                valid_candidates=valid_candidates,
                enumerated_candidates=enumerated_candidates,
            )

        return enumerated_candidates

    def __find_valid_clique_candidates(
        self,
        adjacency_list: list[set[int]],
        positive_hyperedge_signatures: set[tuple[int, ...]],
    ) -> list[tuple[int, ...]]:
        """
        Find valid clique negative candidates in the clique-expanded graph.

        Args:
            adjacency_list: Clique-expanded graph adjacency list.
            positive_hyperedge_signatures: Positive hyperedge node signatures with the requested sample size.

        Returns:
            candidates: Clique node signatures that are not positive hyperedges.

        Raises:
            ValueError: If fewer valid clique negatives exist than requested, or if
                ``max_candidates`` is exceeded during enumeration.
        """
        valid_clique_candidates: list[tuple[int, ...]] = []

        num_nodes = len(adjacency_list)
        # Initial candidates are all nodes, the recursive expansion
        # will narrow down to cliques of the right size
        all_nodes_as_candidates = list(range(num_nodes))
        self.__expand_clique_candidates(
            prefix=(),
            candidates=all_nodes_as_candidates,
            adjacency_list=adjacency_list,
            positive_hyperedge_signatures=positive_hyperedge_signatures,
            valid_candidates=valid_clique_candidates,
            enumerated_candidates=0,
        )

        if len(valid_clique_candidates) < self.num_negative_samples:
            raise ValueError(
                "Asked to create "
                f"{self.num_negative_samples} clique negative samples with "
                f"{self.num_nodes_per_sample} nodes each, but only "
                f"{len(valid_clique_candidates)} are available."
            )

        return valid_clique_candidates

    def __sample_loop(
        self,
        hdata: HData,
        clique_candidates: list[tuple[int, ...]],
        seed: int | None = None,
    ) -> tuple[list[Tensor], list[Tensor], set[int], int]:
        """
        Sample from valid clique candidates and build negative hyperedge tensors.

        Args:
            hdata: Input hypergraph data used for feature, attribute, and ID context.
            clique_candidates: Valid clique negative candidates to sample from.
            seed: Optional seed for reproducible candidate shuffling and random attributes.

        Returns:
            samples: A tuple containing sampled hyperedge index tensors, sampled hyperedge
            attribute tensors, sampled node IDs, and the first negative hyperedge ID.
        """
        device = hdata.device
        generator = create_seeded_torch_generator(device=device, seed=seed)

        # Shuffle the clique candidates with the optional generator
        # Example: clique_candidates = [(0, 1, 3),   # index 0
        #                               (0, 2, 3),   # index 1
        #                               (1, 2, 3)],  # index 2
        #          -> shuffled_clique_candidate_indexes = [2, 0, 1]
        #          -> sampled_clique_candidate_indexes = [2, 0] if num_negative_samples=2 as we only need 2 samples
        #          -> sampled_clique_candidates = [(1, 2, 3),  # index 2 in clique_candidates
        #                                          (0, 1, 3)]  # index 0 in clique_candidates
        num_valid_clique_candidates = len(clique_candidates)
        shuffled_clique_candidate_indexes = torch.randperm(
            n=num_valid_clique_candidates,
            generator=generator,
            device=device,
        )
        sampled_clique_candidate_indexes = shuffled_clique_candidate_indexes[
            : self.num_negative_samples
        ]
        sampled_clique_candidates = [
            clique_candidates[int(clique_candidate_idx)]
            for clique_candidate_idx in sampled_clique_candidate_indexes.tolist()
        ]

        sampled_negative_node_ids: set[int] = set()
        sampled_hyperedge_indexes: list[Tensor] = []
        sampled_hyperedge_attrs: list[Tensor] = []
        new_hyperedge_id_offset = hdata.num_hyperedges

        for new_hyperedge_id, sampled_clique_candidate in enumerate(sampled_clique_candidates):
            # Example: new_hyperedge_id_offset = 5, new_hyperedge_id = 0,
            #          sampled_candidate = (0, 2, 3)
            #          -> sampled_node_ids = [0, 2, 3]
            #          -> sampled_hyperedge_id_tensor = [5, 5, 5]
            #          -> sampled_hyperedge_index = [[0, 2, 3],
            #                                        [5, 5, 5]]
            sampled_node_ids = torch.tensor(
                sampled_clique_candidate, dtype=torch.long, device=device
            )
            sampled_hyperedge_id_tensor = torch.full(
                size=(self.num_nodes_per_sample,),
                fill_value=new_hyperedge_id + new_hyperedge_id_offset,
                device=device,
            )
            sampled_hyperedge_indexes.append(
                torch.stack([sampled_node_ids, sampled_hyperedge_id_tensor], dim=0)
            )
            sampled_negative_node_ids.update(sampled_clique_candidate)

            if hdata.hyperedge_attr is not None:
                random_hyperedge_attr = torch.randn(
                    size=hdata.hyperedge_attr[0].shape,
                    dtype=hdata.hyperedge_attr.dtype,
                    generator=generator,
                    device=device,
                )
                sampled_hyperedge_attrs.append(random_hyperedge_attr)

        return (
            sampled_hyperedge_indexes,
            sampled_hyperedge_attrs,
            sampled_negative_node_ids,
            new_hyperedge_id_offset,
        )
