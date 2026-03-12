import pytest

from unittest.mock import MagicMock, patch
from hyperbench.train import MultiModelTrainer
from hyperbench.types import ModelConfig
from hyperbench.tests import new_mock_trainer


@pytest.fixture
def mock_model_configs():
    model_configs = []

    for i in range(2):
        model = MagicMock()
        model.name = f"model{i}"
        model.version = f"{i}"

        model_config = MagicMock(spec=ModelConfig)
        model_config.name = f"model{i}"
        model_config.version = f"{i}"
        model_config.model = model
        model_config.trainer = None
        model_config.is_trainable = True
        model_config.full_model_name = lambda self=model_config: f"{self.name}:{self.version}"

        model_configs.append(model_config)

    return model_configs


@patch("hyperbench.train.trainer.L.Trainer")
def test_trainer_initialization(_, mock_model_configs):
    multi_model_trainer = MultiModelTrainer(mock_model_configs)

    assert len(multi_model_trainer.model_configs) == len(mock_model_configs)
    for config in multi_model_trainer.model_configs:
        assert config.trainer is not None


@patch("hyperbench.train.trainer.L.Trainer")
def test_trainer_initialization_with_initialized_trainer(mock_trainer, mock_model_configs):
    mock_model_configs[0].trainer = mock_trainer

    multi_model_trainer = MultiModelTrainer(mock_model_configs)

    assert len(multi_model_trainer.model_configs) == len(mock_model_configs)
    for config in multi_model_trainer.model_configs:
        assert config.trainer is not None


@patch("hyperbench.train.trainer.L.Trainer")
def test_models_property_returns_models(_, mock_model_configs):
    multi_model_trainer = MultiModelTrainer(mock_model_configs)
    models = multi_model_trainer.models

    assert len(models) == len(mock_model_configs)


@patch("hyperbench.train.trainer.L.Trainer")
def test_models_property_returns_empty_when_no_models(_):
    multi_model_trainer = MultiModelTrainer([])
    models = multi_model_trainer.models

    assert len(models) == 0


@patch("hyperbench.train.trainer.L.Trainer")
def test_model_returns_model_when_correct_name_and_no_version(_, mock_model_configs):
    mock_model_configs[0].version = "default"
    mock_model_configs[0].model.version = "default"

    multi_model_trainer = MultiModelTrainer(mock_model_configs)
    found = multi_model_trainer.model(name="model0")

    assert found is not None
    assert found.name == "model0"
    assert found.version == "default"


@patch("hyperbench.train.trainer.L.Trainer")
def test_model_returns_None_when_incorrect_name_and_no_version(_, mock_model_configs):
    multi_model_trainer = MultiModelTrainer(mock_model_configs)
    found = multi_model_trainer.model(name="nonexistent")

    assert found is None


@patch("hyperbench.train.trainer.L.Trainer")
def test_model_returns_model_when_correct_name_and_version(_, mock_model_configs):
    multi_model_trainer = MultiModelTrainer(mock_model_configs)
    found = multi_model_trainer.model(name="model0", version="0")

    assert found is not None
    assert found.name == "model0"
    assert found.version == "0"


@patch("hyperbench.train.trainer.L.Trainer")
def test_model_returns_None_when_incorrect_name_and_version(_, mock_model_configs):
    multi_model_trainer = MultiModelTrainer(mock_model_configs)
    not_found = multi_model_trainer.model(name="nonexistent", version="100")

    assert not_found is None


@patch("hyperbench.train.trainer.L.Trainer")
def test_model_returns_None_when_incorrect_name_and_correct_version(_, mock_model_configs):
    multi_model_trainer = MultiModelTrainer(mock_model_configs)
    not_found = multi_model_trainer.model(name="nonexistent", version="0")

    assert not_found is None


@patch(
    "hyperbench.train.trainer.L.Trainer",
    side_effect=lambda *args, **kwargs: new_mock_trainer(),
)
def test_fit_all_calls_fit(_, mock_model_configs):
    multi_model_trainer = MultiModelTrainer(mock_model_configs)

    multi_model_trainer.fit_all(verbose=False)
    for config in mock_model_configs:
        config.trainer.fit.assert_called_once()


@patch("hyperbench.train.trainer.L.Trainer")
def test_fit_all_with_no_models(_):
    multi_model_trainer = MultiModelTrainer([])

    with pytest.raises(ValueError, match="No models to fit."):
        multi_model_trainer.fit_all(verbose=False)


@patch("hyperbench.train.trainer.L.Trainer", return_value=None)
def test_fit_all_raises_when_None_trainer(_, mock_model_configs):
    multi_model_trainer = MultiModelTrainer(mock_model_configs)

    with pytest.raises(
        ValueError,
        match=f"Trainer not defined for model {mock_model_configs[0].full_model_name()}.",
    ):
        multi_model_trainer.fit_all(verbose=False)


@patch(
    "hyperbench.train.trainer.L.Trainer",
    side_effect=lambda *args, **kwargs: new_mock_trainer(),
)
def test_fit_all_with_verbose_true_prints(_, mock_model_configs, capsys):
    multi_model_trainer = MultiModelTrainer(mock_model_configs)

    multi_model_trainer.fit_all(verbose=True)

    for config in mock_model_configs:
        config.trainer.fit.assert_called_once()

    captured_out = capsys.readouterr().out
    logs = [line for line in captured_out.splitlines() if "Fit model" in line]
    assert len(logs) == len(mock_model_configs)


@patch(
    "hyperbench.train.trainer.L.Trainer",
    side_effect=lambda *args, **kwargs: new_mock_trainer(),
)
def test_fit_all_skips_non_trainable_model(_, mock_model_configs):
    mock_model_configs[0].is_trainable = False
    multi_model_trainer = MultiModelTrainer(mock_model_configs)

    multi_model_trainer.fit_all(verbose=False)

    mock_model_configs[0].trainer.fit.assert_not_called()
    mock_model_configs[1].trainer.fit.assert_called_once()


@patch(
    "hyperbench.train.trainer.L.Trainer",
    side_effect=lambda *args, **kwargs: new_mock_trainer(),
)
def test_fit_all_skips_non_trainable_model_with_verbose_prints(_, mock_model_configs, capsys):
    mock_model_configs[0].is_trainable = False
    multi_model_trainer = MultiModelTrainer(mock_model_configs)

    multi_model_trainer.fit_all(verbose=True)

    captured_out = capsys.readouterr().out
    assert "Skipping training for model" in captured_out
    assert mock_model_configs[0].full_model_name() in captured_out


@patch(
    "hyperbench.train.trainer.L.Trainer",
    side_effect=lambda *args, **kwargs: new_mock_trainer(),
)
def test_test_all_calls_test_and_returns_results(_, mock_model_configs):
    multi_model_trainer = MultiModelTrainer(mock_model_configs)

    results = multi_model_trainer.test_all(verbose=False)

    assert all("acc" in v for v in results.values())

    for config in mock_model_configs:
        config.trainer.test.assert_called_once()


@patch("hyperbench.train.trainer.L.Trainer")
def test_test_all_with_no_models(_):
    multi_model_trainer = MultiModelTrainer([])

    with pytest.raises(ValueError, match="No models to test."):
        multi_model_trainer.test_all(verbose=False)


@patch("hyperbench.train.trainer.L.Trainer", return_value=None)
def test_test_all_raises_when_None_trainer(_, mock_model_configs):
    multi_model_trainer = MultiModelTrainer(mock_model_configs)

    with pytest.raises(
        ValueError,
        match=f"Trainer not defined for model {mock_model_configs[0].full_model_name()}.",
    ):
        multi_model_trainer.test_all(verbose=False)


@patch(
    "hyperbench.train.trainer.L.Trainer",
    side_effect=lambda *args, **kwargs: new_mock_trainer(),
)
def test_test_all_with_verbose_true_prints(_, mock_model_configs, capsys):
    multi_model_trainer = MultiModelTrainer(mock_model_configs)

    multi_model_trainer.test_all(verbose=True)

    for config in mock_model_configs:
        config.trainer.test.assert_called_once()

    captured_out = capsys.readouterr().out
    logs = [line for line in captured_out.splitlines() if "Test model" in line]
    assert len(logs) == len(mock_model_configs)
