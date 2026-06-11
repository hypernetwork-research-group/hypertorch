import torch

from hyperbench.data import (
    CliqueNegativeSampler,
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
    def __init__(self, num_negative_samples: int, num_nodes_per_sample: int):
        self.num_negative_samples = num_negative_samples
        self.num_nodes_per_sample = num_nodes_per_sample

    def sample(self, hdata: HData, seed: int | None = None) -> HData:
        # pick one sample of the negative hyperedges (the last one) and duplicate it num_negative_samples times
        negative_hyperedge_index = (
            hdata.hyperedge_index[:, -1].unsqueeze(1).repeat(1, self.num_negative_samples)
        )
        negative_labels = torch.zeros(
            self.num_negative_samples, dtype=torch.long
        )  # label for negative samples is 0

        new_hyperedge_index = torch.cat([hdata.hyperedge_index, negative_hyperedge_index], dim=1)
        new_labels = torch.cat(
            [torch.ones(hdata.num_hyperedges, dtype=torch.long), negative_labels], dim=0
        )
        return HData(
            x=hdata.x,
            hyperedge_index=new_hyperedge_index,
            y=new_labels,
            num_nodes=hdata.num_nodes,
            num_hyperedges=hdata.num_hyperedges + self.num_negative_samples,
        )


if __name__ == "__main__":
    hdata = HData(
        x=torch.arange(5, dtype=torch.float).unsqueeze(1),
        hyperedge_index=torch.tensor(
            [
                [0, 1, 2, 0, 3, 1, 3, 2, 3, 3, 4],
                [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
            ],
            dtype=torch.long,
        ),
        num_nodes=5,
        num_hyperedges=5,
    )
    dataset = Dataset.from_hdata(hdata, sampling_strategy=SamplingStrategy.HYPEREDGE)

    print("Add negative samples to the dataset using CustomNegativeSampler...\n")

    custom_negative_sampler = CustomNegativeSampler(
        num_negative_samples=4,
        num_nodes_per_sample=4,
    )
    dataset_with_custom_negatives = dataset.add_negative_samples(
        custom_negative_sampler,
        seed=42,
    )

    print("Add negative samples to the dataset using CliqueNegativeSampler...\n")

    clique_negative_sampler = CliqueNegativeSampler(
        num_negative_samples=3,
        num_nodes_per_sample=3,
    )
    dataset_with_clique_negatives = dataset.add_negative_samples(
        clique_negative_sampler,
        seed=42,
    )

    print(f"Original dataset has {dataset.hdata.num_hyperedges} positive hyperedges\n")
    describe_negative_dataset("CliqueNegativeSampler", dataset_with_clique_negatives)
