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
    HyperedgeIDSplitter,
    DefaultHDataSplitter,
)
from hyperbench.utils import (
    NodeSpaceSetting,
    is_transductive_setting,
)


class CustomSplitter(Splitter["Dataset", list["Dataset"]]):
    def __init__(self):
        super().__init__()

    def split(self, to_split: Dataset, **kwargs) -> list[Dataset]:
        hdata = to_split.hdata
        hyperedge_splitter = HyperedgeIDSplitter(
            hyperedge_index=hdata.hyperedge_index,
            num_nodes=hdata.num_nodes,
            num_hyperedges=hdata.num_hyperedges,
        )
        print("Using CustomSplitter to split the dataset...")
        perm = hyperedge_splitter.get_hyperedge_ids_permutation(
            shuffle=False,
            seed=None,
        )

        num_hes = int(perm.size(0))
        if num_hes == 0:
            return []

        mid = num_hes // 2
        first_ids = perm[:mid]
        second_ids = perm[mid:]

        split_ids_list = [first_ids, second_ids]

        split_datasets: list[Dataset] = []
        for idx, split_ids in enumerate(split_ids_list):
            node_space_setting: NodeSpaceSetting = (
                "transductive"
                if idx == 0
                and is_transductive_setting(kwargs.get("node_space_setting", "transductive"))
                else "inductive"
            )
            split_hdata = DefaultHDataSplitter(node_space_setting=node_space_setting).split(
                to_split=hdata, split_hyperedge_ids=split_ids
            )
            split_hdata = split_hdata.to(device=hdata.device)

            split_dataset = to_split.__class__(
                hdata=split_hdata, sampling_strategy=to_split.sampling_strategy
            )
            split_datasets.append(split_dataset)

        return split_datasets


if __name__ == "__main__":
    verbose = False
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
    first_half, second_half = split_datasets

    if verbose:
        print(f"Original dataset:\n {dataset.hdata}\n")
        print(f"First half dataset:\n {first_half.hdata}\n")
        print(f"Second half dataset:\n {second_half.hdata}\n")

    print("Complete!")
