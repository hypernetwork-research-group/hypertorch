import torch
from hyperbench.data import (
    Dataset,
    Splitter,
    SamplingStrategy,
)
from hyperbench.types import HData


class CustomSplitter(Splitter["Dataset", list["Dataset"]]):
    def __init__(self):
        super().__init__()

    def split(self, to_split: Dataset, **kwargs) -> list[Dataset]:
        split_ratios = [0.5, 0.25, 0.25]  # Custom split ratios for train, val, test
        num_hyperedges_in_train = int(split_ratios[0] * to_split.hdata.num_hyperedges)
        num_hyperedges_in_val = int(split_ratios[1] * to_split.hdata.num_hyperedges)
        num_hyperedges_in_test = int(split_ratios[2] * to_split.hdata.num_hyperedges)

        end_train = start_val = num_hyperedges_in_train
        end_val = start_test = num_hyperedges_in_train + num_hyperedges_in_val
        end_test = num_hyperedges_in_train + num_hyperedges_in_val + num_hyperedges_in_test

        train_he = to_split.hdata.hyperedge_index[:, :end_train]
        val_he = to_split.hdata.hyperedge_index[:, start_val:end_val]
        test_he = to_split.hdata.hyperedge_index[:, start_test:end_test]

        train_hdata = HData.from_hyperedge_index(train_he)
        val_hdata = HData.from_hyperedge_index(val_he)
        test_hdata = HData.from_hyperedge_index(test_he)

        split_datasets = [
            Dataset.from_hdata(train_hdata, sampling_strategy=to_split.sampling_strategy),
            Dataset.from_hdata(val_hdata, sampling_strategy=to_split.sampling_strategy),
            Dataset.from_hdata(test_hdata, sampling_strategy=to_split.sampling_strategy),
        ]

        return split_datasets


if __name__ == "__main__":
    verbose = True

    x = torch.arange(7, dtype=torch.float).unsqueeze(1)
    hyperedge_index = torch.tensor(
        [
            [0, 1, 2, 0, 3, 1, 3, 2, 3, 3, 4, 4, 5, 6],
            [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5],
        ],
        dtype=torch.long,
    )
    hdata = HData(
        x=x,
        hyperedge_index=hyperedge_index,
        num_nodes=7,
        num_hyperedges=6,
    )
    dataset = Dataset.from_hdata(hdata, sampling_strategy=SamplingStrategy.HYPEREDGE)

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
        print(f"First 50% split:\n {first_50.hdata}\n")
        print(f"Second 25% split:\n {second_25.hdata}\n")
        print(f"Third 25% split:\n {third_25.hdata}\n")

    print("Complete!")
