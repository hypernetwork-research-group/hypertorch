import pytest

from unittest.mock import MagicMock
from hyperbench.types import ModelConfig


@pytest.fixture
def mock_model():
    return MagicMock()


@pytest.fixture
def mock_trainer():
    return MagicMock()


def test_model_config_initialization_with_trainer(mock_model, mock_trainer):
    model_config = ModelConfig(name="model", model=mock_model, version="0", trainer=mock_trainer)

    assert model_config.name == "model"
    assert model_config.version == "0"
    assert model_config.model is mock_model
    assert model_config.is_trainable is True
    assert model_config.trainer is mock_trainer


def test_model_config_initialization_without_trainer(mock_model):
    mock_config = ModelConfig(name="test_model", model=mock_model)

    assert mock_config.name == "test_model"
    assert mock_config.version == "default"
    assert mock_config.model is mock_model
    assert mock_config.is_trainable is True
    assert mock_config.trainer is None


def test_full_model_name(mock_model):
    mock_config = ModelConfig(name="foo", model=mock_model, version="bar")

    assert mock_config.full_model_name() == "foo:bar"


def test_full_model_name_default_version(mock_model):
    mock_config = ModelConfig(name="foo", model=mock_model)

    assert mock_config.full_model_name() == "foo:default"


def test_model_config_for_nontrainable_models(mock_model):
    model_config = ModelConfig(name="model", model=mock_model, is_trainable=False)

    assert model_config.is_trainable is False
