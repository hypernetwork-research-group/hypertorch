import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAccuracy,
    BinaryAveragePrecision,
    BinaryPrecision,
    BinaryRecall,
)
from hyperbench.data import (
    Dataset,
    Splitter,
    AlgebraDataset,
    SamplingStrategy,
)
from hyperbench.types import HData


class CustomSplitter(Splitter["Dataset", list["Dataset"]]):
    def __init__(self):
        super().__init__()

    def split(self, to_split: Dataset, **kwargs) -> list[Dataset]:
        hdata = to_split.hdata

        perm = hdata.hyperedge_index[0].argsort()
        num_hes = len(perm)
        if num_hes == 0:
            return []

        mid = num_hes // 2
        first_ids = perm[:mid]
        second_ids = perm[mid:]

        split_ids_list = [first_ids, second_ids]
        split_second_ids_first_half = split_ids_list[1][: len(split_ids_list[1]) // 2]
        split_second_ids_second_half = split_ids_list[1][len(split_ids_list[1]) // 2 :]
        split_ids_list = [
            split_ids_list[0],
            split_second_ids_first_half,
            split_second_ids_second_half,
        ]
        split_datasets: list[Dataset] = []
        for _, split_ids in enumerate(split_ids_list):
            keep_mask = torch.isin(hdata.hyperedge_index[1], split_ids)
            split_hyperedge_index = hdata.hyperedge_index[:, keep_mask]
            unique_hyperedge_indices = torch.unique(split_hyperedge_index[1])
            x = hdata.x
            split_hdata = HData(
                x=x,
                hyperedge_index=split_hyperedge_index,
                y=unique_hyperedge_indices.float(),
                num_nodes=hdata.num_nodes,
                num_hyperedges=len(unique_hyperedge_indices),
            )
            split_hdata = split_hdata.to(device=hdata.device)

            split_dataset = to_split.__class__(
                hdata=split_hdata, sampling_strategy=to_split.sampling_strategy
            )
            split_datasets.append(split_dataset)

        return split_datasets


if __name__ == "__main__":
    verbose = True
    num_workers = 8
    num_features = 32
    metrics = MetricCollection(
        {
            "auc": BinaryAUROC(),
            "accuracy": BinaryAccuracy(),
            "avg_precision": BinaryAveragePrecision(),
            "precision": BinaryPrecision(),
            "recall": BinaryRecall(),
        }
    )

    print("Loading and preparing dataset...")

    dataset = AlgebraDataset(sampling_strategy=SamplingStrategy.HYPEREDGE)
    if verbose:
        print(f"Dataset:\n {dataset.hdata}\n")

    custom_splitter = CustomSplitter()

    # Split dataset into train, val and test using ratios generated inside CustomSplitter.
    split_datasets = dataset.split(
        shuffle=True,
        seed=42,
        node_space_setting="transductive",
        cover_all_nodes_in_train_split=False,
        splitter=custom_splitter,  # pass the custom splitter to the split function
    )
    first_50, second_25, third_25 = split_datasets

    if verbose:
        print(f"Original dataset:\n {dataset.hdata}\n")
        print(f"First 50 dataset:\n {first_50.hdata}\n")
        print(f"Second 25 dataset:\n {second_25.hdata}\n")
        print(f"Third 25 dataset:\n {third_25.hdata}\n")

    print("Complete!")
