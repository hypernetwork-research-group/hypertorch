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
    DefaultDatasetSplitter,
    AlgebraDataset,
    SamplingStrategy,
    HyperedgeIDSplitter,
    DefaultHDataSplitter,
)
from hyperbench.utils import (
    NodeSpaceSetting,
    is_transductive_setting,
)
import random


class CustomSplitter(DefaultDatasetSplitter):
    def __init__(self, seed: int = 42):
        super().__init__(node_space_setting="transductive")
        self.seed = seed

    def split(self, to_split: Dataset, **kwargs) -> tuple[list[Dataset], list[float]]:
        train_split_idx = 0
        cover_all_nodes_in_train_split: bool = kwargs.get("cover_all_nodes_in_train_split", False)

        random_ratios_train = random.Random(43).random()
        random_ratios_val = random.Random(42).random()
        random_ratios_test = 1 - random_ratios_train - random_ratios_val
        random_ratios = [random_ratios_train, random_ratios_val, random_ratios_test]
        hdata = to_split.hdata
        hyperedge_splitter = HyperedgeIDSplitter(
            hyperedge_index=hdata.hyperedge_index,
            num_nodes=hdata.num_nodes,
            num_hyperedges=hdata.num_hyperedges,
        )
        hyperedge_ids_permutation = hyperedge_splitter.get_hyperedge_ids_permutation(
            shuffle=self.shuffle,
            seed=self.seed,
        )
        hyperedge_ids_by_split, final_ratios = hyperedge_splitter.split(
            to_split=hyperedge_ids_permutation,
            ratios=random_ratios,
        )
        if is_transductive_setting(self.node_space_setting) and cover_all_nodes_in_train_split:
            hyperedge_ids_by_split, final_ratios = hyperedge_splitter.ensure_split_covers_all_nodes(
                hyperedge_ids_by_split=hyperedge_ids_by_split,
                split_idx=train_split_idx,
            )
        hyperedge_splitter.validate_splits_have_hyperedges(hyperedge_ids_by_split)

        split_datasets: list[Dataset] = []
        for split_num, split_hyperedge_ids in enumerate(hyperedge_ids_by_split):
            split_node_space_setting: NodeSpaceSetting = (
                "transductive"
                if is_transductive_setting(self.node_space_setting) and split_num == train_split_idx
                else "inductive"
            )
            split_hdata = DefaultHDataSplitter(node_space_setting=split_node_space_setting).split(
                to_split=hdata,
                split_hyperedge_ids=split_hyperedge_ids,
            )
            split_hdata = split_hdata.to(device=hdata.device)

            split_dataset = to_split.__class__(
                hdata=split_hdata,
                sampling_strategy=to_split.sampling_strategy,
            )
            split_datasets.append(split_dataset)

        return split_datasets, final_ratios


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

    custom_splitter = CustomSplitter(seed=123)

    # Split dataset into train, val and test using ratios generated inside CustomSplitter.
    split_datasets, split_ratios = dataset.split(
        shuffle=True,
        seed=42,
        node_space_setting="transductive",
        cover_all_nodes_in_train_split=False,
        splitter=custom_splitter,  # pass the custom splitter to the split function
    )
    train_dataset, val_dataset, test_dataset = split_datasets

    print(f"Custom split ratios: {split_ratios}")

    if verbose:
        print(f"Train dataset:\n {train_dataset.hdata}\n")
        print(f"Val dataset:\n {val_dataset.hdata}\n")
        print(f"Test dataset:\n {test_dataset.hdata}\n")

    print("Complete!")
