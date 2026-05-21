import torch

from os import PathLike
from torch import Tensor
from typing import IO, Any
from collections.abc import Iterator
from unittest.mock import MagicMock
from contextlib import contextmanager
from hyperbench.types import HData
from hyperbench.data import NegativeSampler


MOCK_BASE_PATH = "hyperbench/tests/mock"


def new_mock_trainer() -> MagicMock:
    trainer = MagicMock()
    trainer.fit = MagicMock()
    trainer.test = MagicMock(return_value=[{"acc": 0.9}])
    trainer.strategy = MagicMock()
    trainer.strategy.root_device = "cpu"
    return trainer


def new_mock_pyg_node2vec() -> MagicMock:
    weight = torch.nn.Parameter(torch.tensor(1.0))

    model = MagicMock()
    model.to.return_value = model
    model.loader.return_value = [
        (
            torch.tensor([[0, 1]], dtype=torch.long),
            torch.tensor([[2, 3]], dtype=torch.long),
        )
    ]
    model.parameters.return_value = [weight]
    model.return_value = (weight * torch.ones((4, 2))).requires_grad_(True)

    def loss(positive_random_walk: Tensor, negative_random_walk: Tensor) -> Tensor:
        return weight * (positive_random_walk.float().sum() + negative_random_walk.float().sum())

    model.loss.side_effect = loss
    return model


def new_mock_villain() -> MagicMock:
    weight = torch.nn.Parameter(torch.tensor(1.0))

    model = MagicMock()
    model.to.return_value = model
    model.parameters.return_value = [weight]
    model.hyperedge_embeddings.return_value = torch.ones((2, 3), requires_grad=True)
    model.node_embeddings.return_value = torch.ones((4, 2), requires_grad=True)

    def loss(hyperedge_index: Tensor, num_hyperedges: int) -> tuple[Tensor, dict[str, Any]]:
        return weight * 2.0, {}

    model.loss.side_effect = loss
    return model


@contextmanager
def new_mock_named_temporary_file(
    path: str | PathLike[str],
    mode: str = "wb",
) -> Iterator[IO[Any]]:
    with open(path, mode) as file_handle:
        yield file_handle


def new_mock_negative_sampler():
    def sample(data: HData, seed: int | None = None) -> HData:
        negative_nodes = torch.tensor([0, 2], dtype=torch.long, device=data.device)
        negative_hyperedge_id = torch.full(
            negative_nodes.shape,
            data.num_hyperedges,
            dtype=torch.long,
            device=data.device,
        )
        return HData(
            x=data.x,
            hyperedge_index=torch.stack([negative_nodes, negative_hyperedge_id]),
            num_nodes=data.num_nodes,
            num_hyperedges=1,
            global_node_ids=data.global_node_ids,
            y=torch.zeros(1, dtype=torch.float, device=data.device),
        )

    sampler = MagicMock(spec=NegativeSampler)
    sampler.sample.side_effect = sample
    return sampler
