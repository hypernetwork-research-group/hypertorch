import torch

from os import PathLike
from typing import IO, Any
from collections.abc import Iterator
from unittest.mock import MagicMock
from contextlib import contextmanager
from hyperbench.train import NegativeSampler
from hyperbench.types import HData


MOCK_BASE_PATH = "hyperbench/tests/mock"


def new_mock_trainer() -> MagicMock:
    trainer = MagicMock()
    trainer.fit = MagicMock()
    trainer.test = MagicMock(return_value=[{"acc": 0.9}])
    trainer.strategy = MagicMock()
    trainer.strategy.root_device = "cpu"
    return trainer


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
