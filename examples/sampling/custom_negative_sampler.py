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
    def __init__(self, num_negative_samples: int, num_nodes_per_sample: int):
        self.num_negative_samples = num_negative_samples
        self.num_nodes_per_sample = num_nodes_per_sample

    def sample(self, hdata: HData, seed: int | None = None) -> HData:
        if seed is not None:
            torch.manual_seed(seed)

        node_ids = torch.randint(
            low=0,
            high=hdata.num_nodes,
            size=(self.num_nodes_per_sample, self.num_negative_samples),
            dtype=torch.long,
        )

        hyperedge_ids = torch.arange(
            hdata.num_hyperedges,
            hdata.num_hyperedges + self.num_negative_samples,
            dtype=torch.long,
        ).repeat_interleave(self.num_nodes_per_sample)

        negative_hyperedge_index = torch.stack(
            [node_ids.reshape(-1), hyperedge_ids],
            dim=0,
        )

        return HData(
            x=hdata.x,
            hyperedge_index=negative_hyperedge_index,
            y=torch.zeros(self.num_negative_samples, dtype=torch.float),
            num_nodes=hdata.num_nodes,
            num_hyperedges=self.num_negative_samples,
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
        num_negative_samples=2,
        num_nodes_per_sample=2,
    )
    dataset_with_custom_negatives = dataset.add_negative_samples(
        custom_negative_sampler,
        seed=42,
    )

    print("Dataset after adding custom negative samples:")
    describe_negative_dataset("Custom Negative Sampler", dataset_with_custom_negatives)
