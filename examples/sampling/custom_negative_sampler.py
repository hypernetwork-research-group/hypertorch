import torch

from hyperbench.data import (
    Dataset,
    NegativeSampler,
    SamplingStrategy,
)
from hyperbench.types import HData


def describe_negative_dataset(name: str, dataset: Dataset) -> None:
    hdata = dataset.hdata
    negative_mask = hdata.y == 0
    num_negative_hyperedges = negative_mask.sum(dtype=torch.int).item()

    print(f"{name}:")
    print(f"- Hyperedges after sampling: {hdata.num_hyperedges}")
    print(f"- Negative hyperedges added: {num_negative_hyperedges}")
    print(f"- Labels: {hdata.y.tolist()}")
    print()


class CustomNegativeSampler(NegativeSampler):
    def __init__(self):
        self.negative_tensor = torch.tensor(
            [
                [0, 1, 2, 0, 3, 1, 3, 2, 3, 3, 4],
                [5, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9],
            ],
            dtype=torch.long,
        )

    def sample(self, hdata: HData, seed: int | None = None) -> HData:
        return HData(
            x=hdata.x,
            hyperedge_index=self.negative_tensor,
            y=torch.zeros(5, dtype=torch.float),
            num_nodes=hdata.num_nodes,
            num_hyperedges=5,
        )


if __name__ == "__main__":
    x = torch.arange(5, dtype=torch.float).unsqueeze(1)
    hyperedge_index = torch.tensor(
        [
            [0, 1, 2, 0, 3, 1, 3, 2, 3, 3, 4],
            [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
        ],
        dtype=torch.long,
    )
    hdata = HData(
        x=x,
        hyperedge_index=hyperedge_index,
        num_nodes=5,
        num_hyperedges=5,
    )
    dataset = Dataset.from_hdata(hdata, sampling_strategy=SamplingStrategy.HYPEREDGE)

    print("Add negative samples to the dataset using CustomNegativeSampler...\n")

    dataset_with_custom_negatives = dataset.add_negative_samples(
        negative_sampler=CustomNegativeSampler(),
        seed=42,
    )

    print("Dataset after adding custom negative samples:")
    describe_negative_dataset("Custom Negative Sampler", dataset_with_custom_negatives)
