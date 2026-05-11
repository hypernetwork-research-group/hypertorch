from os import PathLike
from typing import IO, Any
from collections.abc import Iterator
from unittest.mock import MagicMock
from contextlib import contextmanager

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
