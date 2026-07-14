import lightning as L
import torch

from torch import Generator, Tensor
from collections.abc import Callable
from typing import Any, TypeAlias
from torch.utils.data import DataLoader as TorchDataLoader
from hypertorch.data import Dataset
from hypertorch.types import HData, HyperedgeIndex


DataModule: TypeAlias = L.LightningDataModule


class DataLoader(TorchDataLoader):
    """
    DataLoader combines a dataset and a sampler, and provides an iterable
    over the given dataset. It extends ``torch.utils.data.DataLoader``.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool | None = False,
        sample_full_hypergraph: bool = False,
        drop_last: bool = False,
        num_workers: int = 0,
        persistent_workers: bool = False,
        collate_fn: Callable[[list[HData]], HData] | None = None,
        generator: Generator | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the data loader.

        Args:
            dataset: Dataset that provides sampled `HData` objects.
            batch_size: Number of samples per batch. Ignored when
                ``sample_full_hypergraph`` is ``True`` because the full dataset
                is loaded as one batch.
            shuffle: Whether to reshuffle sample indices at every epoch.
            sample_full_hypergraph: Whether each collated batch should ignore the sampled
                mini-batch subgraphs and return a clone of the dataset's full `HData`.
                If ``False``, batches contain only the sampled local subgraph.
                For dense transductive splits, this preserves the full graph as model
                context while `target_node_mask` or `target_hyperedge_mask` identifies the
                supervised nodes or hyperedges for the split.
                Defaults to ``False``.
            drop_last: Whether to drop the final incomplete batch when the dataset size
                is not divisible by the batch size. If ``True``, drop the last incomplete batch.
                If ``False``, the last batch will be kept and will be smaller.
                Defaults to ``False``.
            num_workers: Optional number of subprocesses to use for data loading.
                For instance, ``0`` means loading happens in the main process. Defaults to ``0``.
            persistent_workers: Whether worker processes should stay alive after a dataset has
                been consumed once. If ``True``, the data loader will not shut down
                the worker processes after a dataset has been consumed once. This allows to
                maintain the workers `Dataset` instances alive. Defaults to ``False``.
            collate_fn: Optional custom collate function. When ``None``, uses
                the default `collate` method.
            generator: Optional random generator used by the underlying Torch data loader.
            kwargs:
                sampler: Defines the strategy to draw samples from the dataset.
                    Can be any ``Iterable`` with ``__len__`` implemented.
                    Mutually exclusive with ``shuffle``. Defaults to ``None``.
                batch_sampler: Defines the strategy to draw batches of indices. Mutually exclusive
                    with ``batch_size``, ``shuffle``, ``sampler``, and ``drop_last``.
                    Defaults to ``None``.
                pin_memory: Whether to copy tensors into pinned memory before returning them.
                    If ``True``, the data loader will copy tensors into device/CUDA pinned memory
                    before returning them. If your data elements are a custom type, or `collate_fn`
                    returns a batch that is a custom type, look at ``torch.utils.data.DataLoader``
                    for examples. Defaults to ``False``.
                timeout: Timeout, in seconds, for collecting a batch from worker processes.
                    Should always be non-negative and defaults to ``0``.
                worker_init_fn: If not ``None``, this will be called on each worker subprocess
                    with the worker id (an int in ``[0, num_workers - 1]``) as input, after seeding
                    and before data loading. Defaults to ``None``.
                multiprocessing_context: Multiprocessing context or context name used to create
                    worker processes.
                prefetch_factor: Number of batches loaded in advance by each worker.
                    For example, ``2`` means there will be a total of ``2 * num_workers`` batches
                    prefetched across all workers. The default value depends on the set value
                    for num_workers. If ``num_workers=0``, defaults to ``None``.
                    If ``num_workers > 0``, defaults to ``2``.
                in_order: If ``True``, batches are returned in first-in, first-out order when
                    ``num_workers > 0``. Defaults to ``True``.
        """
        self.__sample_full_hypergraph: bool = sample_full_hypergraph

        super().__init__(
            dataset=dataset,
            batch_size=len(dataset) if self.__sample_full_hypergraph else batch_size,
            shuffle=shuffle,
            collate_fn=self.collate if collate_fn is None else collate_fn,
            generator=generator,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            drop_last=drop_last,
            **kwargs,
        )

        self.__cached_dataset_hdata: HData = dataset.hdata

    @classmethod
    def from_datasets(
        cls,
        train_dataset: Dataset | None = None,
        val_dataset: Dataset | None = None,
        test_dataset: Dataset | None = None,
        **kwargs: Any,
    ) -> DataModule:
        """
        Create a Lightning data module from datasets that share data loader parameters.

        Each provided dataset is wrapped in a separate `DataLoader`. The resulting data
        module can be passed directly to Lightning or to `MultiModelTrainer`.

        Examples:
            ```python
            data_module = DataLoader.from_datasets(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                batch_size=128,
                shuffle=False,
                num_workers=4,
                persistent_workers=True,
            )
            ```

        Args:
            train_dataset: Optional dataset used by the data module's training data loader.
            val_dataset: Optional dataset used by the data module's validation data loader.
            test_dataset: Optional dataset used by the data module's test data loader.
            kwargs: Parameters passed to every created `DataLoader`.

        Returns:
            data_module: A Lightning data module containing the created data loaders.
        """
        return DataLoaderDataModule(
            train_dataloader=cls(train_dataset, **kwargs) if train_dataset is not None else None,
            val_dataloader=cls(val_dataset, **kwargs) if val_dataset is not None else None,
            test_dataloader=cls(test_dataset, **kwargs) if test_dataset is not None else None,
        )

    def collate(self, batch: list[HData]) -> HData:
        """
        Collates a list of `HData` objects into a single batched `HData` object.

        This function combines multiple separate samples into a single batched representation
        suitable for mini-batch training.

        Handles:
            - Concatenating node features from all samples.
            - Concatenating and offsetting hyperedges from all samples.
            - Concatenating hyperedge attributes from all samples, if present.
            - Concatenating hyperedge weights from all samples, if present.

        Examples:
            Given ``batch = [HData_0, HData_1]``:

            For node features:

            >>> HData_0.x.shape  # (3, 64) — 3 nodes with 64 features
            >>> HData_1.x.shape  # (2, 64) — 2 nodes with 64 features
            >>> x.shape  # (5, 64) — all 5 nodes concatenated

            For hyperedge index:

            - ``HData_0`` (3 nodes, 2 hyperedges):

            >>> hyperedge_index = [[0, 1, 1, 2],  # Nodes 0, 1, 1, 2
            ...                    [0, 0, 1, 1]]  # HE 0 contains {0,1}, HE 1 contains {1,2}

            - ``HData_1`` (2 nodes, 1 hyperedge):

            >>> hyperedge_index = [[0, 1],  # Nodes 0, 1
            ...                    [0, 0]]  # Hyperedge 0 contains {0,1}

            Batched result:

            >>> hyperedge_index = [[0, 1, 1, 2, 3, 4],  # Node indices: original then offset by 3
            ...                    [0, 0, 1, 1, 2, 2]]  # Hyperedge IDs: original then offset by 2

        Args:
            batch: List of `HData` objects to collate.

        Returns:
            hdata: A single `HData` object containing the collated data.
        """
        if self.__sample_full_hypergraph:
            return self.__collate_full_hypergraph(batch)

        collated_hyperedge_index = torch.cat([data.hyperedge_index for data in batch], dim=1)
        hyperedge_index_wrapper = HyperedgeIndex(collated_hyperedge_index).remove_duplicate_edges()

        hyperedge_ids = hyperedge_index_wrapper.hyperedge_ids
        node_ids = hyperedge_index_wrapper.node_ids

        collated_x = self.__cached_dataset_hdata.x[node_ids]
        collated_global_node_ids = self.__cached_dataset_hdata.global_node_ids[node_ids]

        (
            collated_y,
            collated_target_node_mask,
            collated_target_hyperedge_mask,
        ) = self.__collate_y_and_target_masks_for_task(
            batch=batch,
            hyperedge_index_wrapper=hyperedge_index_wrapper,
        )

        collated_hyperedge_attr = (
            self.__cached_dataset_hdata.hyperedge_attr[hyperedge_ids]
            if self.__cached_dataset_hdata.hyperedge_attr is not None
            else None
        )

        collated_hyperedge_weights = (
            self.__cached_dataset_hdata.hyperedge_weights[hyperedge_ids]
            if self.__cached_dataset_hdata.hyperedge_weights is not None
            else None
        )

        collated_hyperedge_index = hyperedge_index_wrapper.to_0based().item

        collated_hdata = HData(
            x=collated_x,
            hyperedge_index=collated_hyperedge_index,
            hyperedge_weights=collated_hyperedge_weights,
            hyperedge_attr=collated_hyperedge_attr,
            num_nodes=hyperedge_index_wrapper.num_nodes,
            num_hyperedges=hyperedge_index_wrapper.num_hyperedges,
            global_node_ids=collated_global_node_ids,
            target_node_mask=collated_target_node_mask,
            target_hyperedge_mask=collated_target_hyperedge_mask,
            y=collated_y,
            task=self.__cached_dataset_hdata.task,
        )

        return collated_hdata.to(batch[0].device)

    def __collate_full_hypergraph(self, batch: list[HData]) -> HData:
        """
        Return the full hypergraph with target masks and labels rebuilt from the sampled batch.

        The sampled HData instances identify the current training targets, while the cached
        dataset HData provides the full hypergraph context.

        Args:
            batch: List of HData objects to collate.

        Returns:
            hdata: A single HData object containing the full hypergraph
                with target masks and labels updated from the sampled batch.
        """
        hyperedge_index_wrapper = HyperedgeIndex(self.__cached_dataset_hdata.hyperedge_index)

        (
            collated_y,
            collated_target_node_mask,
            collated_target_hyperedge_mask,
        ) = self.__collate_y_and_target_masks_for_task(
            batch=batch,
            hyperedge_index_wrapper=hyperedge_index_wrapper,
        )
        collated_hdata = self.__cached_dataset_hdata.clone()
        return (
            collated_hdata.with_y(collated_y)
            .with_target_node_mask(collated_target_node_mask)
            .with_target_hyperedge_mask(collated_target_hyperedge_mask)
            .to(batch[0].device)
        )

    def __collate_y_and_target_masks_for_task(
        self,
        batch: list[HData],
        hyperedge_index_wrapper: HyperedgeIndex,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        """
        Collates the labels (y) and target masks for a batch of
        HData instances based on the task type.

        Args:
            batch: List of HData instances containing the data to collate.
            hyperedge_index_wrapper: A HyperedgeIndex wrapping the collated hyperedge index.

        Returns:
            collated_y: A tensor containing the collated labels for the batch.
            collated_target_node_mask: A tensor containing the collated target node mask
                for the batch, or ``None`` if not applicable.
            collated_target_hyperedge_mask: A tensor containing the collated target hyperedge
                mask for the batch, or ``None`` if not applicable.
        """
        node_ids = hyperedge_index_wrapper.node_ids
        hyperedge_ids = hyperedge_index_wrapper.hyperedge_ids

        if self.__cached_dataset_hdata.is_node_related_task:
            collated_y = self.__cached_dataset_hdata.y[node_ids]
            collated_target_hyperedge_mask = None

            target_node_ids_list = [
                HyperedgeIndex(batch_hdata.hyperedge_index).node_ids[batch_hdata.target_node_mask]
                for batch_hdata in batch
            ]
            target_node_ids = torch.cat(target_node_ids_list, dim=0)
            collated_target_node_mask = torch.isin(node_ids, target_node_ids)
        elif self.__cached_dataset_hdata.is_hyperedge_related_task:
            collated_y = self.__cached_dataset_hdata.y[hyperedge_ids]
            collated_target_node_mask = None

            target_hyperedge_ids_list = [
                HyperedgeIndex(hdata.hyperedge_index).hyperedge_ids[hdata.target_hyperedge_mask]
                for hdata in batch
            ]
            target_hyperedge_ids = torch.cat(target_hyperedge_ids_list, dim=0)
            collated_target_hyperedge_mask = torch.isin(hyperedge_ids, target_hyperedge_ids)
        else:
            raise ValueError(
                f"Unsupported task category for task={self.__cached_dataset_hdata.task!r}."
            )

        return collated_y, collated_target_node_mask, collated_target_hyperedge_mask


class DataLoaderDataModule(DataModule):
    def __init__(
        self,
        train_dataloader: DataLoader | None,
        val_dataloader: DataLoader | None,
        test_dataloader: DataLoader | None,
    ) -> None:
        super().__init__()
        self.__train_dataloader = train_dataloader
        self.__val_dataloader = val_dataloader
        self.__test_dataloader = test_dataloader

    def train_dataloader(self) -> DataLoader | None:
        return self.__train_dataloader

    def val_dataloader(self) -> DataLoader | None:
        return self.__val_dataloader

    def test_dataloader(self) -> DataLoader | None:
        return self.__test_dataloader
