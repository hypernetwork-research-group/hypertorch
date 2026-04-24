from unittest.mock import MagicMock


MOCK_BASE_PATH = "hyperbench/tests/mock"


def new_mock_trainer() -> MagicMock:
    trainer = MagicMock()
    trainer.fit = MagicMock()
    trainer.test = MagicMock(return_value=[{"acc": 0.9}])
    trainer.strategy = MagicMock()
    trainer.strategy.root_device = "cpu"
    return trainer
