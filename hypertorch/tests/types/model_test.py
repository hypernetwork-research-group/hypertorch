import pytest

from types import MappingProxyType
from unittest.mock import MagicMock
from hypertorch.types import ModelConfig


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
    assert model_config.test_trainer is None


def test_model_config_initialization_with_test_trainer(mock_model, mock_trainer):
    model_config = ModelConfig(
        name="model",
        model=mock_model,
        version="0",
        test_trainer=mock_trainer,
    )

    assert model_config.test_trainer is mock_trainer


def test_model_config_initialization_without_trainer(mock_model):
    mock_config = ModelConfig(name="test_model", model=mock_model)

    assert mock_config.name == "test_model"
    assert mock_config.version == "default"
    assert mock_config.model is mock_model
    assert mock_config.is_trainable is True
    assert mock_config.trainer is None
    assert mock_config.test_trainer is None


def test_full_model_name(mock_model):
    mock_config = ModelConfig(name="foo", model=mock_model, version="bar")

    assert mock_config.full_model_name() == "foo:bar"


def test_full_model_name_default_version(mock_model):
    mock_config = ModelConfig(name="foo", model=mock_model)

    assert mock_config.full_model_name() == "foo:default"


def test_model_config_for_nontrainable_models(mock_model):
    model_config = ModelConfig(name="model", model=mock_model, is_trainable=False)

    assert model_config.is_trainable is False


def test_model_config_initialization_with_trainer_overrides(mock_model, tmp_path):
    callback = MagicMock()
    checkpoint_callback_kwargs = {"filename": "model-{epoch}", "save_last": True}
    trainer_kwargs = MappingProxyType(
        {
            "devices": 1,
            "test_devices": 1,
            "max_epochs": 3,
            "log_every_n_steps": 5,
            "check_val_every_n_epoch": 2,
            "callbacks": [callback],
            "enable_checkpointing": False,
            "checkpoint_callback_kwargs": checkpoint_callback_kwargs,
            "default_root_dir": tmp_path,
        }
    )

    model_config = ModelConfig(
        name="model",
        model=mock_model,
        trainer_kwargs=trainer_kwargs,
    )

    assert model_config.trainer_kwargs is trainer_kwargs
    assert model_config.trainer_kwargs["devices"] == 1
    assert model_config.trainer_kwargs["test_devices"] == 1
    assert model_config.trainer_kwargs["max_epochs"] == 3
    assert model_config.trainer_kwargs["log_every_n_steps"] == 5
    assert model_config.trainer_kwargs["check_val_every_n_epoch"] == 2
    assert model_config.trainer_kwargs["callbacks"] == [callback]
    assert model_config.trainer_kwargs["enable_checkpointing"] is False
    assert model_config.trainer_kwargs["checkpoint_callback_kwargs"] is checkpoint_callback_kwargs
    assert model_config.trainer_kwargs["default_root_dir"] == tmp_path
