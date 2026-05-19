import torch

from hyperbench.data import Dataset, SamplingStrategy
from hyperbench.train import CliqueNegativeSampler, RandomNegativeSampler
from hyperbench.types import HData


def describe_negative_dataset(name: str, dataset: Dataset) -> None:
    hdata = dataset.hdata
    negative_mask = hdata.y == 0
    num_negative_hyperedges = int(negative_mask.sum().item())

    print(f"{name}:")
    print(f"- Hyperedges after sampling: {hdata.num_hyperedges}")
    print(f"- Negative hyperedges added: {num_negative_hyperedges}")
    print(f"- Labels: {hdata.y.tolist()}")
    print()


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

    print("Add negative samples to the dataset using RandomNegativeSampler...\n")

    random_negative_sampler = RandomNegativeSampler(
        num_negative_samples=4,
        num_nodes_per_sample=4,
    )
    dataset_with_random_negatives = dataset.add_negative_samples(
        random_negative_sampler,
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
    describe_negative_dataset("RandomNegativeSampler", dataset_with_random_negatives)
    describe_negative_dataset("CliqueNegativeSampler", dataset_with_clique_negatives)
