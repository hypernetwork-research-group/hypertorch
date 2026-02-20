import torch

from typing import List, Optional
from torch.utils.data import DataLoader as TorchDataLoader
from hyperbench.data import Dataset
from hyperbench.types import HData, HyperedgeIndex


class DataLoader(TorchDataLoader):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: Optional[bool] = False,
        **kwargs,
    ) -> None:
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate,
            **kwargs,
        )

        self.__cached_dataset_hdata = dataset.hdata

    def collate(self, batch: List[HData]) -> HData:
        """
        Collates a list of :class:`HData objects into a single batched :class:`HData object.

        This function combines multiple separate samples into a single batched representation suitable for mini-batch training.
        It handles:
        - Concatenating node features from all samples.
        - Concatenating and offsetting hyperedges from all samples.
        - Concatenating hyperedge attributes from all samples, if present.

        Examples:
            Given ``batch = [HData_0, HData_1]``:

            For node features:

            >>> HData_0.x.shape  # (3, 64) — 3 nodes with 64 features
            >>> HData_1.x.shape  # (2, 64) — 2 nodes with 64 features
            >>> x.shape  # (5, 64) — all 5 nodes concatenated

            For hyperedge index:

            - ``HData_0`` (3 nodes, 2 hyperedges):

            >>> hyperedge_index = [[0, 1, 1, 2],  # Nodes 0, 1, 1, 2
            ...                    [0, 0, 1, 1]]  # Hyperedge 0 contains {0,1}, Hyperedge 1 contains {1,2}

            - ``HData_1`` (2 nodes, 1 hyperedge):

            >>> hyperedge_index = [[0, 1],  # Nodes 0, 1
            ...                    [0, 0]]  # Hyperedge 0 contains {0,1}

            Batched result:

            >>> hyperedge_index = [[0, 1, 1, 2, 3, 4],  # Node indices: original then offset by 3
            ...                    [0, 0, 1, 1, 2, 2]]  # Hyperedge IDs: original then offset by 2

        Args:
            batch: List of :class:`HData objects to collate.

        Returns:
            A single :class:`HData` object containing the collated data.
        """
        collated_hyperedge_index = torch.cat([data.hyperedge_index for data in batch], dim=1)
        hyperedge_index_wrapper = HyperedgeIndex(collated_hyperedge_index).remove_duplicate_edges()

        hyperedge_ids = hyperedge_index_wrapper.hyperedge_ids
        collated_x = self.__cached_dataset_hdata.x[hyperedge_index_wrapper.node_ids]
        collated_y = self.__cached_dataset_hdata.y[hyperedge_ids]

        collated_hyeredge_attr = None
        if self.__cached_dataset_hdata.hyperedge_attr is not None:
            collated_hyeredge_attr = self.__cached_dataset_hdata.hyperedge_attr[hyperedge_ids]

        collated_hyperedge_index = hyperedge_index_wrapper.to_0based().item

        collated_hdata = HData(
            x=collated_x,
            hyperedge_index=collated_hyperedge_index,
            hyperedge_attr=collated_hyeredge_attr,
            num_nodes=hyperedge_index_wrapper.num_nodes,
            num_hyperedges=hyperedge_index_wrapper.num_hyperedges,
            y=collated_y,
        )

        return collated_hdata.to(batch[0].device)
