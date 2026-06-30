import pytest
import re

from unittest.mock import MagicMock, patch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks import ModelCheckpoint
from hypertorch.train import MultiModelTrainer
from hypertorch.types import ModelConfig
from hypertorch.tests import new_mock_trainer


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
        model_config.test_trainer = None
        model_config.is_trainable = True
        model_config.full_model_name = lambda self=model_config: f"{self.name}:{self.version}"
        model_config.train_dataloader = None
        model_config.val_dataloader = None
        model_config.test_dataloader = None

        model_configs.append(model_config)

    return model_configs


@patch("hypertorch.train.trainer.L.Trainer")
@patch("hypertorch.train.trainer.CSVLogger")
@patch("hypertorch.train.trainer.MarkdownTableLogger")
@patch("hypertorch.train.trainer.LaTexTableLogger")
def test_trainer_initialization(
    mock_latex_logger_cls,
    mock_md_logger_cls,
    mock_csv_logger_cls,
    mock_trainer_cls,
    mock_model_configs,
):
    multi_model_trainer = MultiModelTrainer(mock_model_configs)

    assert len(multi_model_trainer.model_configs) == len(mock_model_configs)
    for config in multi_model_trainer.model_configs:
        assert config.trainer is not None


def test_trainer_initialization_rejects_invalid_tensorboard_port(mock_model_configs):
    with pytest.raises(ValueError, match="'tensorboard_port' must be non-negative"):
        MultiModelTrainer(mock_model_configs, tensorboard_port=-1)


@patch("hypertorch.train.trainer.L.Trainer")
@patch("hypertorch.train.trainer.CSVLogger")
@patch("hypertorch.train.trainer.MarkdownTableLogger")
@patch("hypertorch.train.trainer.LaTexTableLogger")
def test_trainer_initialization_with_initialized_trainer(
    mock_latex_logger_cls,
    mock_md_logger_cls,
    mock_csv_logger_cls,
    mock_trainer_cls,
    mock_model_configs,
):
    mock_model_configs[0].trainer = mock_trainer_cls

    multi_model_trainer = MultiModelTrainer(mock_model_configs)

    assert len(multi_model_trainer.model_configs) == len(mock_model_configs)
    for config in multi_model_trainer.model_configs:
        assert config.trainer is not None


@patch("hypertorch.train.trainer.L.Trainer")
@patch("hypertorch.train.trainer.CSVLogger")
@patch("hypertorch.train.trainer.MarkdownTableLogger")
@patch("hypertorch.train.trainer.LaTexTableLogger")
def test_trainer_initialization_with_no_models(
    mock_latex_logger_cls,
    mock_md_logger_cls,
    mock_csv_logger_cls,
    mock_trainer_cls,
):
    with pytest.raises(ValueError, match=re.escape("'model_configs' cannot be empty.")):
        MultiModelTrainer(model_configs=[])


@patch("hypertorch.train.trainer.L.Trainer")
@patch("hypertorch.train.trainer.CSVLogger")
@patch("hypertorch.train.trainer.MarkdownTableLogger")
@patch("hypertorch.train.trainer.LaTexTableLogger")
def test_trainer_initialization_skips_configs_with_initialized_trainers(
    mock_latex_logger_cls,
    mock_md_logger_cls,
    mock_csv_logger_cls,
    mock_trainer_cls,
    mock_model_configs,
):
    initialized_trainers = [new_mock_trainer() for _ in mock_model_configs]
    for model_config, trainer in zip(mock_model_configs, initialized_trainers, strict=True):
        model_config.trainer = trainer

    MultiModelTrainer(mock_model_configs)

    mock_trainer_cls.assert_not_called()
    mock_csv_logger_cls.assert_not_called()
    mock_md_logger_cls.assert_not_called()
    mock_latex_logger_cls.assert_not_called()
    assert [config.trainer for config in mock_model_configs] == initialized_trainers


@patch("hypertorch.train.trainer.L.Trainer")
@patch("hypertorch.train.trainer.CSVLogger")
@patch("hypertorch.train.trainer.MarkdownTableLogger")
@patch("hypertorch.train.trainer.LaTexTableLogger")
def test_context_manager_enter_returns_self(
    mock_latex_logger_cls,
    mock_md_logger_cls,
    mock_csv_logger_cls,
    mock_trainer_cls,
    mock_model_configs,
):
    multi_model_trainer = MultiModelTrainer(mock_model_configs)
    with multi_model_trainer as trainer:
        assert trainer is multi_model_trainer


@patch("hypertorch.train.trainer.L.Trainer")
@patch("hypertorch.train.trainer.CSVLogger")
@patch("hypertorch.train.trainer.MarkdownTableLogger")
@patch("hypertorch.train.trainer.LaTexTableLogger")
@patch("hypertorch.train.trainer.MultiModelTrainer.finalize")
def test_context_manager_exit_calls_finalize(
    mock_finalize,
    mock_latex_logger_cls,
    mock_md_logger_cls,
    mock_csv_logger_cls,
    mock_trainer_cls,
    mock_model_configs,
):
    multi_model_trainer = MultiModelTrainer(mock_model_configs)
    with multi_model_trainer:
        pass

    mock_finalize.assert_called_once()


@patch("hypertorch.train.trainer.L.Trainer")
@patch("hypertorch.train.trainer.CSVLogger")
@patch("hypertorch.train.trainer.MarkdownTableLogger")
@patch("hypertorch.train.trainer.LaTexTableLogger")
@patch("hypertorch.train.trainer.MultiModelTrainer.finalize", side_effect=Exception("error"))
def test_del_suppresses_exception_from_finalize(
    mock_finalize,
    mock_latex_logger_cls,
    mock_md_logger_cls,
    mock_csv_logger_cls,
    mock_trainer_cls,
    mock_model_configs,
):
    trainer1 = MultiModelTrainer(mock_model_configs)
    trainer2 = MultiModelTrainer(mock_model_configs)

    with pytest.warns(
        UserWarning,
        match=re.escape("Exception occurred during MultiModelTrainer cleanup. Error: error"),
    ):
        trainer1.__del__()

    with pytest.warns(
        UserWarning,
        match=re.escape("Exception occurred during MultiModelTrainer cleanup. Error: error"),
    ):
        del trainer2


@patch("hypertorch.train.trainer.L.Trainer")
@patch("hypertorch.train.trainer.CSVLogger")
@patch("hypertorch.train.trainer.MarkdownTableLogger")
@patch("hypertorch.train.trainer.LaTexTableLogger")
def test_models_property_returns_models(
    mock_latex_logger_cls,
    mock_md_logger_cls,
    mock_csv_logger_cls,
    mock_trainer_cls,
    mock_model_configs,
):
    multi_model_trainer = MultiModelTrainer(mock_model_configs)
    models = multi_model_trainer.models

    assert len(models) == len(mock_model_configs)


@patch("hypertorch.train.trainer.L.Trainer")
@patch("hypertorch.train.trainer.CSVLogger")
@patch("hypertorch.train.trainer.MarkdownTableLogger")
@patch("hypertorch.train.trainer.LaTexTableLogger")
def test_model_returns_model_when_correct_name_and_no_version(
    mock_latex_logger_cls,
    mock_md_logger_cls,
    mock_csv_logger_cls,
    mock_trainer_cls,
    mock_model_configs,
):
    mock_model_configs[0].version = "default"
    mock_model_configs[0].model.version = "default"

    multi_model_trainer = MultiModelTrainer(mock_model_configs)
    found = multi_model_trainer.model(name="model0")

    assert found is not None
    assert found.name == "model0"
    assert found.version == "default"


@patch("hypertorch.train.trainer.L.Trainer")
@patch("hypertorch.train.trainer.CSVLogger")
@patch("hypertorch.train.trainer.MarkdownTableLogger")
@patch("hypertorch.train.trainer.LaTexTableLogger")
def test_model_returns_none_when_incorrect_name_and_no_version(
    mock_latex_logger_cls,
    mock_md_logger_cls,
    mock_csv_logger_cls,
    mock_trainer_cls,
    mock_model_configs,
):
    multi_model_trainer = MultiModelTrainer(mock_model_configs)
    found = multi_model_trainer.model(name="nonexistent")

    assert found is None


@patch("hypertorch.train.trainer.L.Trainer")
@patch("hypertorch.train.trainer.CSVLogger")
@patch("hypertorch.train.trainer.MarkdownTableLogger")
@patch("hypertorch.train.trainer.LaTexTableLogger")
def test_model_returns_model_when_correct_name_and_version(
    mock_latex_logger_cls,
    mock_md_logger_cls,
    mock_csv_logger_cls,
    mock_trainer_cls,
    mock_model_configs,
):
    multi_model_trainer = MultiModelTrainer(mock_model_configs)
    found = multi_model_trainer.model(name="model0", version="0")

    assert found is not None
    assert found.name == "model0"
    assert found.version == "0"


@patch("hypertorch.train.trainer.L.Trainer")
@patch("hypertorch.train.trainer.CSVLogger")
@patch("hypertorch.train.trainer.MarkdownTableLogger")
@patch("hypertorch.train.trainer.LaTexTableLogger")
def test_model_returns_none_when_incorrect_name_and_version(
    mock_latex_logger_cls,
    mock_md_logger_cls,
    mock_csv_logger_cls,
    mock_trainer_cls,
    mock_model_configs,
):
    multi_model_trainer = MultiModelTrainer(mock_model_configs)
    not_found = multi_model_trainer.model(name="nonexistent", version="100")

    assert not_found is None


@patch("hypertorch.train.trainer.L.Trainer")
@patch("hypertorch.train.trainer.CSVLogger")
@patch("hypertorch.train.trainer.MarkdownTableLogger")
@patch("hypertorch.train.trainer.LaTexTableLogger")
def test_model_returns_none_when_incorrect_name_and_correct_version(
    mock_latex_logger_cls,
    mock_md_logger_cls,
    mock_csv_logger_cls,
    mock_trainer_cls,
    mock_model_configs,
):
    multi_model_trainer = MultiModelTrainer(mock_model_configs)
    not_found = multi_model_trainer.model(name="nonexistent", version="0")

    assert not_found is None


@patch(
    "hypertorch.train.trainer.L.Trainer",
    side_effect=lambda *args, **kwargs: new_mock_trainer(),
)
@patch("hypertorch.train.trainer.CSVLogger")
@patch("hypertorch.train.trainer.MarkdownTableLogger")
@patch("hypertorch.train.trainer.LaTexTableLogger")
def test_fit_all_calls_fit(
    mock_latex_logger_cls,
    mock_md_logger_cls,
    mock_csv_logger_cls,
    mock_trainer_cls,
    mock_model_configs,
):
    multi_model_trainer = MultiModelTrainer(mock_model_configs)

    multi_model_trainer.fit_all(verbose=False)
    for config in mock_model_configs:
        config.trainer.fit.assert_called_once()


@patch("hypertorch.train.trainer.L.Trainer", return_value=None)
@patch("hypertorch.train.trainer.CSVLogger")
@patch("hypertorch.train.trainer.MarkdownTableLogger")
@patch("hypertorch.train.trainer.LaTexTableLogger")
def test_fit_all_raises_when_none_trainer(
    mock_latex_logger_cls,
    mock_md_logger_cls,
    mock_csv_logger_cls,
    mock_trainer_cls,
    mock_model_configs,
):
    multi_model_trainer = MultiModelTrainer(mock_model_configs)

    with pytest.raises(
        ValueError,
        match=f"Trainer not defined for model {mock_model_configs[0].full_model_name()}.",
    ):
        multi_model_trainer.fit_all(verbose=False)


@patch(
    "hypertorch.train.trainer.L.Trainer",
    side_effect=lambda *args, **kwargs: new_mock_trainer(),
)
@patch("hypertorch.train.trainer.CSVLogger")
@patch("hypertorch.train.trainer.MarkdownTableLogger")
@patch("hypertorch.train.trainer.LaTexTableLogger")
def test_fit_all_with_verbose_true_prints(
    mock_latex_logger_cls,
    mock_md_logger_cls,
    mock_csv_logger_cls,
    mock_trainer_cls,
    mock_model_configs,
    capsys,
):
    multi_model_trainer = MultiModelTrainer(mock_model_configs)

    multi_model_trainer.fit_all(verbose=True)

    for config in mock_model_configs:
        config.trainer.fit.assert_called_once()

    captured_out = capsys.readouterr().out
    logs = [line for line in captured_out.splitlines() if "Fit model" in line]
    assert len(logs) == len(mock_model_configs)
    assert all("device: cpu" in line for line in logs)
    assert all("log_dir:" in line for line in logs)
    assert all("ckpt_path: None" in line for line in logs)


def test_fit_all_with_missing_strategy_prints_unknown_device(capsys, mock_model_configs):
    for config in mock_model_configs:
        config.trainer = new_mock_trainer()
        config.trainer.strategy = None

    multi_model_trainer = MultiModelTrainer(mock_model_configs)

    multi_model_trainer.fit_all(verbose=True)

    captured_out = capsys.readouterr().out
    logs = [line for line in captured_out.splitlines() if "Fit model" in line]
    assert len(logs) == len(mock_model_configs)
    assert all("device: unknown" in line for line in logs)


@patch(
    "hypertorch.train.trainer.L.Trainer",
    side_effect=lambda *args, **kwargs: new_mock_trainer(),
)
@patch("hypertorch.train.trainer.CSVLogger")
@patch("hypertorch.train.trainer.MarkdownTableLogger")
@patch("hypertorch.train.trainer.LaTexTableLogger")
def test_fit_all_skips_non_trainable_model(
    mock_latex_logger_cls,
    mock_md_logger_cls,
    mock_csv_logger_cls,
    mock_trainer_cls,
    mock_model_configs,
):
    mock_model_configs[0].is_trainable = False
    multi_model_trainer = MultiModelTrainer(mock_model_configs)

    multi_model_trainer.fit_all(verbose=False)

    mock_model_configs[0].trainer.fit.assert_not_called()
    mock_model_configs[1].trainer.fit.assert_called_once()


@patch(
    "hypertorch.train.trainer.L.Trainer",
    side_effect=lambda *args, **kwargs: new_mock_trainer(),
)
@patch("hypertorch.train.trainer.CSVLogger")
@patch("hypertorch.train.trainer.MarkdownTableLogger")
@patch("hypertorch.train.trainer.LaTexTableLogger")
def test_fit_all_skips_non_trainable_model_with_verbose_prints(
    mock_latex_logger_cls,
    mock_md_logger_cls,
    mock_csv_logger_cls,
    mock_trainer_cls,
    mock_model_configs,
    capsys,
):
    mock_model_configs[0].is_trainable = False
    multi_model_trainer = MultiModelTrainer(mock_model_configs)

    multi_model_trainer.fit_all(verbose=True)

    captured_out = capsys.readouterr().out
    assert "Skipping training for model" in captured_out
    assert mock_model_configs[0].full_model_name() in captured_out


@patch(
    "hypertorch.train.trainer.L.Trainer",
    side_effect=lambda *args, **kwargs: new_mock_trainer(),
)
@patch("hypertorch.train.trainer.CSVLogger")
@patch("hypertorch.train.trainer.MarkdownTableLogger")
@patch("hypertorch.train.trainer.LaTexTableLogger")
def test_test_all_calls_test_and_returns_results(
    mock_latex_logger_cls,
    mock_md_logger_cls,
    mock_csv_logger_cls,
    mock_trainer_cls,
    mock_model_configs,
):
    multi_model_trainer = MultiModelTrainer(mock_model_configs)

    results = multi_model_trainer.test_all(verbose=False)

    assert all("acc" in v for v in results.values())

    for config in mock_model_configs:
        config.trainer.test.assert_called_once()


def test_test_all_uses_model_config_test_trainer(mock_model_configs):
    for config in mock_model_configs:
        config.trainer = new_mock_trainer()
        config.test_trainer = new_mock_trainer()

    multi_model_trainer = MultiModelTrainer(mock_model_configs)

    results = multi_model_trainer.test_all(verbose=False)

    assert all("acc" in v for v in results.values())
    for config in mock_model_configs:
        config.trainer.test.assert_not_called()
        config.test_trainer.test.assert_called_once()


@patch(
    "hypertorch.train.trainer.L.Trainer",
    side_effect=lambda *args, **kwargs: new_mock_trainer(),
)
@patch("hypertorch.train.trainer.CSVLogger")
@patch("hypertorch.train.trainer.MarkdownTableLogger")
@patch("hypertorch.train.trainer.LaTexTableLogger")
def test_init_creates_test_trainers_with_test_devices(
    mock_latex_logger_cls,
    mock_md_logger_cls,
    mock_csv_logger_cls,
    mock_trainer_cls,
    mock_model_configs,
):
    MultiModelTrainer(mock_model_configs, devices="auto", test_devices=1)

    assert mock_trainer_cls.call_count == len(mock_model_configs) * 2
    train_calls = mock_trainer_cls.call_args_list[::2]
    test_calls = mock_trainer_cls.call_args_list[1::2]

    assert all(call.kwargs["devices"] == "auto" for call in train_calls)
    assert all(call.kwargs["devices"] == 1 for call in test_calls)
    assert all(config.test_trainer is not None for config in mock_model_configs)


@patch(
    "hypertorch.train.trainer.L.Trainer",
    side_effect=lambda *args, **kwargs: new_mock_trainer(),
)
@patch("hypertorch.train.trainer.CSVLogger")
@patch("hypertorch.train.trainer.MarkdownTableLogger")
@patch("hypertorch.train.trainer.LaTexTableLogger")
def test_test_all_uses_auto_created_test_trainer_when_test_devices_is_set(
    mock_latex_logger_cls,
    mock_md_logger_cls,
    mock_csv_logger_cls,
    mock_trainer_cls,
    mock_model_configs,
):
    multi_model_trainer = MultiModelTrainer(mock_model_configs, test_devices=1)

    multi_model_trainer.test_all(verbose=False)

    for config in mock_model_configs:
        config.trainer.test.assert_not_called()
        config.test_trainer.test.assert_called_once()


@patch(
    "hypertorch.train.trainer.L.Trainer",
    side_effect=lambda *args, **kwargs: new_mock_trainer(),
)
@patch("hypertorch.train.trainer.CSVLogger")
@patch("hypertorch.train.trainer.MarkdownTableLogger")
@patch("hypertorch.train.trainer.LaTexTableLogger")
def test_test_all_uses_auto_created_test_trainer_even_if_trainer_is_set(
    mock_latex_logger_cls,
    mock_md_logger_cls,
    mock_csv_logger_cls,
    mock_trainer_cls,
    mock_model_configs,
):
    mock_trainer = new_mock_trainer()
    for config in mock_model_configs:
        config.trainer = mock_trainer

    multi_model_trainer = MultiModelTrainer(mock_model_configs, test_devices=1)

    assert mock_trainer_cls.call_count == len(mock_model_configs)  # No call for fit trainers

    test_calls = mock_trainer_cls.call_args_list[::2]
    assert all(call.kwargs["devices"] == 1 for call in test_calls)
    assert all(
        config.test_trainer is not None and config.test_trainer is not config.trainer
        for config in multi_model_trainer.model_configs
    )
    assert all(config.trainer is mock_trainer for config in multi_model_trainer.model_configs)


def test_test_all_skips_single_process_test_trainer_on_nonzero_rank(mock_model_configs):
    for config in mock_model_configs:
        config.trainer = new_mock_trainer()
        config.trainer.is_global_zero = False
        config.trainer.global_rank = 1
        config.trainer.world_size = 2
        config.test_trainer = new_mock_trainer()
        config.test_trainer.world_size = 1
        config.test_trainer.is_global_zero = False
        config.test_trainer.global_rank = 1

    mock_model_configs[0].trainer.is_global_zero = True
    mock_model_configs[0].trainer.global_rank = 0  # only config 0 has global zero trainer
    mock_model_configs[0].test_trainer.is_global_zero = True
    mock_model_configs[0].test_trainer.global_rank = 0

    multi_model_trainer = MultiModelTrainer(mock_model_configs)

    results = multi_model_trainer.test_all(verbose=False)

    assert results["model1:1"] == {}
    for config in mock_model_configs[1:]:
        config.trainer.test.assert_not_called()
        config.test_trainer.test.assert_not_called()

    assert "acc" in results["model0:0"]
    mock_model_configs[0].test_trainer.test.assert_called_once()


def test_test_all_runs_single_process_test_trainer_on_global_zero_rank(mock_model_configs):
    for config in mock_model_configs:
        config.trainer = new_mock_trainer()
        config.trainer.is_global_zero = True
        config.trainer.global_rank = 0  # global zero rank in all trainers
        config.trainer.world_size = 2
        config.test_trainer = new_mock_trainer()
        config.test_trainer.world_size = 1

    multi_model_trainer = MultiModelTrainer(mock_model_configs)

    results = multi_model_trainer.test_all(verbose=False)

    assert all("acc" in result for result in results.values())
    for config in mock_model_configs:
        config.trainer.test.assert_not_called()
        config.test_trainer.test.assert_called_once()


def test_test_all_does_not_skip_shared_distributed_trainer_on_nonzero_rank(mock_model_configs):
    for config in mock_model_configs:
        config.trainer = new_mock_trainer()
        config.trainer.is_global_zero = False
        config.trainer.global_rank = 1
        config.trainer.world_size = 2  # world size > 1 means it's a distributed training

    multi_model_trainer = MultiModelTrainer(mock_model_configs)

    results = multi_model_trainer.test_all(verbose=False)

    assert all("acc" in result for result in results.values())
    for config in mock_model_configs:
        config.trainer.test.assert_called_once()


def test_test_all_does_not_skip_distributed_test_trainer_on_nonzero_rank(mock_model_configs):
    for config in mock_model_configs:
        config.trainer = new_mock_trainer()
        config.trainer.is_global_zero = False
        config.trainer.global_rank = 1
        config.trainer.world_size = 2
        config.test_trainer = new_mock_trainer()
        config.test_trainer.world_size = 2  # world size > 1, so it's a distributed test trainer

    multi_model_trainer = MultiModelTrainer(mock_model_configs)

    results = multi_model_trainer.test_all(verbose=False)

    assert all("acc" in result for result in results.values())
    for config in mock_model_configs:
        config.trainer.test.assert_not_called()
        config.test_trainer.test.assert_called_once()


@pytest.mark.parametrize(
    "is_global_zero, expected",
    [
        pytest.param(False, False, id="non-global-zero-trainer"),
        pytest.param(True, True, id="global-zero-trainer"),
    ],
)
def test_is_global_zero(
    is_global_zero,
    expected,
    mock_model_configs,
):
    for config in mock_model_configs:
        config.trainer = new_mock_trainer()
        config.trainer.is_global_zero = is_global_zero

    multi_model_trainer = MultiModelTrainer(mock_model_configs)

    assert multi_model_trainer.is_global_zero is expected


@pytest.mark.parametrize(
    "rank, expected",
    [
        pytest.param(0, True, id="fallback-rank-zero"),
        pytest.param(1, False, id="fallback-nonzero-rank"),
    ],
)
@patch("hypertorch.train.trainer.torch.distributed.get_rank")
@patch("hypertorch.train.trainer.torch.distributed.is_initialized", return_value=True)
@patch("hypertorch.train.trainer.torch.distributed.is_available", return_value=True)
def test_is_global_zero_falls_back_to_distributed_global_rank(
    mock_is_available,
    mock_is_initialized,
    mock_get_rank,
    rank,
    expected,
    mock_model_configs,
):
    for config in mock_model_configs:
        config.trainer = new_mock_trainer()

    multi_model_trainer = MultiModelTrainer(mock_model_configs)
    mock_model_configs[0].trainer = None
    mock_model_configs[0].test_trainer = None
    mock_get_rank.return_value = rank

    assert multi_model_trainer.is_global_zero is expected
    mock_is_available.assert_called_once_with()
    mock_is_initialized.assert_called_once_with()
    mock_get_rank.assert_called_once_with()


@pytest.mark.parametrize(
    "is_available, is_initialized, rank, expected",
    [
        pytest.param(False, False, 3, 0, id="distributed-unavailable"),
        pytest.param(True, False, 3, 0, id="distributed-uninitialized"),
        pytest.param(True, True, 3, 3, id="distributed-initialized"),
    ],
)
@patch("hypertorch.train.trainer.torch.distributed.get_rank")
@patch("hypertorch.train.trainer.torch.distributed.is_initialized")
@patch("hypertorch.train.trainer.torch.distributed.is_available")
def test_distributed_global_rank(
    mock_is_available,
    mock_is_initialized,
    mock_get_rank,
    is_available,
    is_initialized,
    rank,
    expected,
    mock_model_configs,
):
    for config in mock_model_configs:
        config.trainer = new_mock_trainer()

    mock_is_available.return_value = is_available
    mock_is_initialized.return_value = is_initialized
    mock_get_rank.return_value = rank

    multi_model_trainer = MultiModelTrainer(mock_model_configs)

    assert multi_model_trainer.distributed_global_rank == expected
    if is_available and is_initialized:
        mock_get_rank.assert_called_once_with()
    else:
        mock_get_rank.assert_not_called()


@patch("hypertorch.train.trainer.torch.distributed.get_rank", return_value=1)
@patch("hypertorch.train.trainer.torch.distributed.is_initialized", return_value=True)
@patch("hypertorch.train.trainer.torch.distributed.is_available", return_value=True)
@patch(
    "hypertorch.train.trainer.torch.distributed.barrier", side_effect=lambda *args, **kwargs: None
)
@patch(
    "hypertorch.train.trainer.torch.distributed.destroy_process_group",
    side_effect=lambda *args, **kwargs: None,
)
def test_test_all_skips_test_trainer_without_train_trainer_on_nonzero_rank(
    mock_destroy_process_group,
    mock_barrier,
    mock_is_available,
    mock_is_initialized,
    mock_get_rank,
    mock_model_configs,
):
    for config in mock_model_configs:
        config.trainer = new_mock_trainer()
        config.test_trainer = new_mock_trainer()
        config.test_trainer.world_size = 1

    multi_model_trainer = MultiModelTrainer(mock_model_configs)
    for config in mock_model_configs:
        config.trainer = None

    results = multi_model_trainer.test_all(verbose=False)

    assert all(result == {} for result in results.values())
    for config in mock_model_configs:
        config.test_trainer.test.assert_not_called()
    assert mock_is_available.call_count == len(mock_model_configs) * 2
    assert mock_is_initialized.call_count == len(mock_model_configs) * 2
    assert mock_get_rank.call_count == len(mock_model_configs)
    assert mock_barrier.call_count == len(mock_model_configs)
    assert mock_destroy_process_group.call_count == len(mock_model_configs)


@patch("hypertorch.train.trainer.L.Trainer", return_value=None)
@patch("hypertorch.train.trainer.CSVLogger")
@patch("hypertorch.train.trainer.MarkdownTableLogger")
@patch("hypertorch.train.trainer.LaTexTableLogger")
def test_test_all_raises_when_none_trainer(
    mock_latex_logger_cls,
    mock_md_logger_cls,
    mock_csv_logger_cls,
    mock_trainer_cls,
    mock_model_configs,
):
    multi_model_trainer = MultiModelTrainer(mock_model_configs)

    with pytest.raises(
        ValueError,
        match=f"Trainer not defined for model {mock_model_configs[0].full_model_name()}.",
    ):
        multi_model_trainer.test_all(verbose=False)


@patch(
    "hypertorch.train.trainer.L.Trainer",
    side_effect=lambda *args, **kwargs: new_mock_trainer(),
)
@patch("hypertorch.train.trainer.CSVLogger")
@patch("hypertorch.train.trainer.MarkdownTableLogger")
@patch("hypertorch.train.trainer.LaTexTableLogger")
def test_test_all_with_verbose_true_prints(
    mock_latex_logger_cls,
    mock_md_logger_cls,
    mock_csv_logger_cls,
    mock_trainer_cls,
    mock_model_configs,
    capsys,
):
    multi_model_trainer = MultiModelTrainer(mock_model_configs)

    multi_model_trainer.test_all(verbose=True)

    for config in mock_model_configs:
        config.trainer.test.assert_called_once()

    captured_out = capsys.readouterr().out
    logs = [line for line in captured_out.splitlines() if "Test model" in line]
    assert len(logs) == len(mock_model_configs)
    assert all("device: cpu" in line for line in logs)
    assert all("log_dir:" in line for line in logs)
    assert all("ckpt_path: None" in line for line in logs)


def test_test_all_with_missing_root_device_prints_unknown_device(capsys, mock_model_configs):
    for config in mock_model_configs:
        config.trainer = new_mock_trainer()
        config.trainer.strategy.root_device = None

    multi_model_trainer = MultiModelTrainer(mock_model_configs)

    multi_model_trainer.test_all(verbose=True)

    captured_out = capsys.readouterr().out
    logs = [line for line in captured_out.splitlines() if "Test model" in line]
    assert len(logs) == len(mock_model_configs)
    assert all("device: unknown" in line for line in logs)


@patch("hypertorch.train.trainer.L.Trainer")
@patch("hypertorch.train.trainer.CSVLogger")
@patch("hypertorch.train.trainer.MarkdownTableLogger")
@patch("hypertorch.train.trainer.LaTexTableLogger")
def test_init_auto_increments_experiment_name(
    mock_latex_logger_cls,
    mock_md_logger_cls,
    mock_csv_logger_cls,
    mock_trainer_cls,
    mock_model_configs,
    tmp_path,
):
    (tmp_path / "experiment_0").mkdir()
    (tmp_path / "experiment_1").mkdir()

    MultiModelTrainer(
        mock_model_configs,
        default_root_dir=str(tmp_path),
    )

    for call in mock_csv_logger_cls.call_args_list:
        assert "experiment_2" in str(call.kwargs["save_dir"])

    for call in mock_md_logger_cls.call_args_list:
        assert "experiment_2" in str(call.kwargs["save_dir"])

    for call in mock_latex_logger_cls.call_args_list:
        assert "experiment_2" in str(call.kwargs["save_dir"])


@patch("hypertorch.train.trainer.L.Trainer")
@patch("hypertorch.train.trainer.CSVLogger")
@patch("hypertorch.train.trainer.MarkdownTableLogger")
@patch("hypertorch.train.trainer.LaTexTableLogger")
def test_init_defaults_to_experiment_0_when_default_root_dir_does_not_exist(
    mock_latex_logger_cls,
    mock_md_logger_cls,
    mock_csv_logger_cls,
    mock_trainer_cls,
    mock_model_configs,
    tmp_path,
):
    MultiModelTrainer(
        mock_model_configs,
        default_root_dir=str(tmp_path / "nonexistent_dir"),
    )

    for call in mock_csv_logger_cls.call_args_list:
        assert "experiment_0" in str(call.kwargs["save_dir"])

    for call in mock_md_logger_cls.call_args_list:
        assert "experiment_0" in str(call.kwargs["save_dir"])

    for call in mock_latex_logger_cls.call_args_list:
        assert "experiment_0" in str(call.kwargs["save_dir"])


@patch("hypertorch.train.trainer.L.Trainer")
@patch("hypertorch.train.trainer.CSVLogger")
@patch("hypertorch.train.trainer.MarkdownTableLogger")
@patch("hypertorch.train.trainer.LaTexTableLogger")
def test_init_defaults_to_experiment_0_when_no_existing_experiment(
    mock_latex_logger_cls,
    mock_md_logger_cls,
    mock_csv_logger_cls,
    mock_trainer_cls,
    mock_model_configs,
    tmp_path,
):
    MultiModelTrainer(
        mock_model_configs,
        default_root_dir=str(tmp_path),
    )

    for call in mock_csv_logger_cls.call_args_list:
        assert "experiment_0" in str(call.kwargs["save_dir"])

    for call in mock_md_logger_cls.call_args_list:
        assert "experiment_0" in str(call.kwargs["save_dir"])

    for call in mock_latex_logger_cls.call_args_list:
        assert "experiment_0" in str(call.kwargs["save_dir"])


@patch("hypertorch.train.trainer.L.Trainer")
@patch("hypertorch.train.trainer.CSVLogger")
@patch("hypertorch.train.trainer.MarkdownTableLogger")
@patch("hypertorch.train.trainer.LaTexTableLogger")
def test_init_always_create_default_csv_logger_per_model(
    mock_latex_logger_cls,
    mock_md_logger_cls,
    mock_csv_logger_cls,
    mock_trainer_cls,
    mock_model_configs,
    tmp_path,
):
    MultiModelTrainer(
        mock_model_configs,
        default_root_dir=str(tmp_path),
        experiment_name="experiment_0",
    )

    assert mock_csv_logger_cls.call_count == len(mock_model_configs)

    model_names = [config.name for config in mock_model_configs]
    model_versions = [f"version_{config.version}" for config in mock_model_configs]

    for call in mock_csv_logger_cls.call_args_list:
        assert call.kwargs["name"] in model_names
        assert call.kwargs["version"] in model_versions


@patch("hypertorch.train.trainer.L.Trainer")
def test_init_passes_custom_logger_to_all_models(mock_trainer_cls, mock_model_configs):
    custom_logger = MagicMock()
    MultiModelTrainer(mock_model_configs, logger=custom_logger)

    for call_args in mock_trainer_cls.call_args_list:
        assert call_args.kwargs["logger"] is custom_logger


@patch("hypertorch.train.trainer.L.Trainer")
@patch("hypertorch.train.trainer.CSVLogger")
@patch("hypertorch.train.trainer.MarkdownTableLogger")
@patch("hypertorch.train.trainer.LaTexTableLogger")
def test_init_adds_per_model_checkpoint_callbacks(
    mock_latex_logger_cls,
    mock_md_logger_cls,
    mock_csv_logger_cls,
    mock_trainer_cls,
    mock_model_configs,
    tmp_path,
):
    MultiModelTrainer(
        mock_model_configs,
        default_root_dir=tmp_path,
        experiment_name="checkpoint_test",
        enable_checkpointing=True,
    )

    for config, call_args in zip(mock_model_configs, mock_trainer_cls.call_args_list, strict=True):
        callbacks = call_args.kwargs["callbacks"]
        checkpoint_callbacks = [
            callback for callback in callbacks if isinstance(callback, ModelCheckpoint)
        ]

        assert len(checkpoint_callbacks) == 1
        assert checkpoint_callbacks[0].dirpath == str(
            tmp_path / "checkpoint_test" / config.name / f"version_{config.version}" / "checkpoints"
        )


@patch("hypertorch.train.trainer.L.Trainer")
@patch("hypertorch.train.trainer.CSVLogger")
@patch("hypertorch.train.trainer.MarkdownTableLogger")
@patch("hypertorch.train.trainer.LaTexTableLogger")
def test_init_passes_checkpoint_callback_kwargs_to_default_checkpoint_callback(
    mock_latex_logger_cls,
    mock_md_logger_cls,
    mock_csv_logger_cls,
    mock_trainer_cls,
    mock_model_configs,
    tmp_path,
):
    MultiModelTrainer(
        mock_model_configs,
        default_root_dir=tmp_path,
        experiment_name="checkpoint_kwargs_test",
        enable_checkpointing=True,
        checkpoint_callback_kwargs={
            "filename": "custom-{epoch}",
            "save_last": True,
            "save_weights_only": True,
        },
    )

    for call_args in mock_trainer_cls.call_args_list:
        checkpoint_callbacks = [
            callback
            for callback in call_args.kwargs["callbacks"]
            if isinstance(callback, ModelCheckpoint)
        ]

        assert len(checkpoint_callbacks) == 1
        assert checkpoint_callbacks[0].filename == "custom-{epoch}"
        assert checkpoint_callbacks[0].save_last is True
        assert checkpoint_callbacks[0].save_weights_only is True


@patch("hypertorch.train.trainer.L.Trainer")
@patch("hypertorch.train.trainer.CSVLogger")
@patch("hypertorch.train.trainer.MarkdownTableLogger")
@patch("hypertorch.train.trainer.LaTexTableLogger")
def test_init_uses_checkpoint_callback_kwargs_dirpath_for_default_checkpoint_callback(
    mock_latex_logger_cls,
    mock_md_logger_cls,
    mock_csv_logger_cls,
    mock_trainer_cls,
    mock_model_configs,
    tmp_path,
):
    checkpoint_callback_kwargs = {
        "dirpath": tmp_path / "custom_checkpoints",
        "save_last": True,
    }

    MultiModelTrainer(
        mock_model_configs,
        default_root_dir=tmp_path,
        experiment_name="checkpoint_kwargs_test",
        enable_checkpointing=True,
        checkpoint_callback_kwargs=checkpoint_callback_kwargs,
    )

    for call_args in mock_trainer_cls.call_args_list:
        checkpoint_callbacks = [
            callback
            for callback in call_args.kwargs["callbacks"]
            if isinstance(callback, ModelCheckpoint)
        ]

        assert len(checkpoint_callbacks) == 1
        assert checkpoint_callbacks[0].dirpath == str(checkpoint_callback_kwargs["dirpath"])
        assert checkpoint_callbacks[0].save_last is True

    assert checkpoint_callback_kwargs["dirpath"] == tmp_path / "custom_checkpoints"


@patch("hypertorch.train.trainer.L.Trainer")
@patch("hypertorch.train.trainer.CSVLogger")
@patch("hypertorch.train.trainer.MarkdownTableLogger")
@patch("hypertorch.train.trainer.LaTexTableLogger")
def test_init_sets_missing_checkpoint_callback_dirpath_per_model(
    mock_latex_logger_cls,
    mock_md_logger_cls,
    mock_csv_logger_cls,
    mock_trainer_cls,
    mock_model_configs,
    tmp_path,
):
    MultiModelTrainer(
        mock_model_configs,
        default_root_dir=tmp_path,
        experiment_name="checkpoint_test",
        enable_checkpointing=True,
        callbacks=[ModelCheckpoint()],
    )

    checkpoint_dirs = []
    for call_args in mock_trainer_cls.call_args_list:
        callbacks = call_args.kwargs["callbacks"]
        checkpoint_callbacks = [
            callback for callback in callbacks if isinstance(callback, ModelCheckpoint)
        ]
        checkpoint_dirs.append(checkpoint_callbacks[0].dirpath)

    assert checkpoint_dirs == [
        str(
            tmp_path / "checkpoint_test" / config.name / f"version_{config.version}" / "checkpoints"
        )
        for config in mock_model_configs
    ]
    assert len(set(checkpoint_dirs)) == len(mock_model_configs)


@patch("hypertorch.train.trainer.L.Trainer")
@patch("hypertorch.train.trainer.CSVLogger")
@patch("hypertorch.train.trainer.MarkdownTableLogger")
@patch("hypertorch.train.trainer.LaTexTableLogger")
def test_init_adds_model_index_to_duplicate_checkpoint_dirs(
    mock_latex_logger_cls,
    mock_md_logger_cls,
    mock_csv_logger_cls,
    mock_trainer_cls,
    mock_model_configs,
    tmp_path,
):
    for config in mock_model_configs:
        config.name = "duplicate"
        config.version = "0"
        config.full_model_name = lambda: "duplicate:0"

    MultiModelTrainer(
        mock_model_configs,
        default_root_dir=tmp_path,
        experiment_name="checkpoint_test",
        enable_checkpointing=True,
    )

    checkpoint_dirs = []
    for call_args in mock_trainer_cls.call_args_list:
        checkpoint_callbacks = [
            callback
            for callback in call_args.kwargs["callbacks"]
            if isinstance(callback, ModelCheckpoint)
        ]
        checkpoint_dirs.append(checkpoint_callbacks[0].dirpath)

    assert checkpoint_dirs == [
        str(
            tmp_path
            / "checkpoint_test"
            / "duplicate"
            / "version_0"
            / f"model_{model_index}"
            / "checkpoints"
        )
        for model_index in range(len(mock_model_configs))
    ]


@patch("hypertorch.train.trainer.L.Trainer")
@patch("hypertorch.train.trainer.CSVLogger")
@patch("hypertorch.train.trainer.MarkdownTableLogger")
@patch("hypertorch.train.trainer.LaTexTableLogger")
def test_init_does_not_add_checkpoint_callback_when_checkpointing_disabled(
    mock_latex_logger_cls,
    mock_md_logger_cls,
    mock_csv_logger_cls,
    mock_trainer_cls,
    mock_model_configs,
):
    callback = Callback()

    MultiModelTrainer(
        mock_model_configs,
        enable_checkpointing=False,
        callbacks=[callback],
    )

    for call_args in mock_trainer_cls.call_args_list:
        callbacks = call_args.kwargs["callbacks"]
        assert len(callbacks) == 1
        assert isinstance(callbacks[0], Callback)
        assert not isinstance(callbacks[0], ModelCheckpoint)


@patch("hypertorch.train.trainer.L.Trainer")
@patch("hypertorch.train.trainer.CSVLogger")
@patch("hypertorch.train.trainer.MarkdownTableLogger")
@patch("hypertorch.train.trainer.LaTexTableLogger")
def test_init_keeps_existing_checkpoint_callback_dirpath(
    mock_latex_logger_cls,
    mock_md_logger_cls,
    mock_csv_logger_cls,
    mock_trainer_cls,
    mock_model_configs,
    tmp_path,
):
    checkpoint_dir = tmp_path / "custom_checkpoints"

    MultiModelTrainer(
        mock_model_configs,
        enable_checkpointing=True,
        callbacks=[ModelCheckpoint(dirpath=checkpoint_dir)],
    )

    for call_args in mock_trainer_cls.call_args_list:
        checkpoint_callbacks = [
            callback
            for callback in call_args.kwargs["callbacks"]
            if isinstance(callback, ModelCheckpoint)
        ]
        assert len(checkpoint_callbacks) == 1
        assert checkpoint_callbacks[0].dirpath == str(checkpoint_dir)


@pytest.mark.parametrize(
    "callbacks",
    [
        pytest.param([Callback()], id="list_of_callbacks"),
        pytest.param(Callback(), id="single_callback"),
    ],
)
@patch("hypertorch.train.trainer.L.Trainer")
@patch("hypertorch.train.trainer.CSVLogger")
@patch("hypertorch.train.trainer.MarkdownTableLogger")
@patch("hypertorch.train.trainer.LaTexTableLogger")
def test_init_accepts_different_callback_types_with_checkpointing_enabled(
    mock_latex_logger_cls,
    mock_md_logger_cls,
    mock_csv_logger_cls,
    mock_trainer_cls,
    mock_model_configs,
    callbacks,
):
    MultiModelTrainer(
        mock_model_configs,
        enable_checkpointing=True,
        callbacks=callbacks,
    )

    for call_args in mock_trainer_cls.call_args_list:
        callbacks = call_args.kwargs["callbacks"]
        assert len(callbacks) == 2
        assert isinstance(callbacks[0], Callback)
        assert not isinstance(callbacks[0], ModelCheckpoint)
        assert isinstance(callbacks[1], ModelCheckpoint)


@patch("hypertorch.train.trainer.L.Trainer")
@patch("hypertorch.train.trainer.CSVLogger")
@patch("hypertorch.train.trainer.MarkdownTableLogger")
@patch("hypertorch.train.trainer.LaTexTableLogger")
def test_init_each_model_gets_distinct_logger(
    mock_latex_logger_cls,
    mock_md_logger_cls,
    mock_csv_logger_cls,
    mock_trainer_cls,
    mock_model_configs,
    tmp_path,
):
    MultiModelTrainer(
        mock_model_configs,
        default_root_dir=str(tmp_path),
        experiment_name="test_experiment",
    )

    csv_logger_names = [call.kwargs["name"] for call in mock_csv_logger_cls.call_args_list]

    for i in range(1, len(csv_logger_names)):
        assert csv_logger_names[i] != csv_logger_names[i - 1]
        assert csv_logger_names[i] in mock_model_configs[i].name

    md_logger_names = [call.kwargs["model_name"] for call in mock_md_logger_cls.call_args_list]

    for i in range(1, len(md_logger_names)):
        assert md_logger_names[i] != md_logger_names[i - 1]
        assert md_logger_names[i] == mock_model_configs[i].full_model_name()

    latex_logger_names = [
        call.kwargs["model_name"] for call in mock_latex_logger_cls.call_args_list
    ]

    for i in range(1, len(latex_logger_names)):
        assert latex_logger_names[i] != latex_logger_names[i - 1]
        assert latex_logger_names[i] == mock_model_configs[i].full_model_name()


@patch("hypertorch.train.trainer.L.Trainer")
@patch(
    "hypertorch.train.trainer.MultiModelTrainer._MultiModelTrainer__is_tensorboard_available",
    return_value=True,
)
@patch("lightning.pytorch.loggers.TensorBoardLogger", create=True)
@patch("hypertorch.train.trainer.CSVLogger")
@patch("hypertorch.train.trainer.MarkdownTableLogger")
@patch("hypertorch.train.trainer.LaTexTableLogger")
def test_init_creates_tensorboard_logger_when_available(
    mock_latex_logger_cls,
    mock_md_logger_cls,
    mock_csv_logger_cls,
    mock_tb_logger_cls,
    mock_is_tb_available,
    mock_trainer_cls,
    mock_model_configs,
    tmp_path,
):
    MultiModelTrainer(
        mock_model_configs,
        default_root_dir=str(tmp_path),
        experiment_name="experiment_0",
    )

    assert mock_tb_logger_cls.call_count == len(mock_model_configs)

    for call_args in mock_trainer_cls.call_args_list:
        logger_arg = call_args.kwargs["logger"]
        assert isinstance(logger_arg, list)
        assert len(logger_arg) == 4
        assert logger_arg[0] is mock_csv_logger_cls.return_value
        assert logger_arg[1] is mock_md_logger_cls.return_value
        assert logger_arg[2] is mock_latex_logger_cls.return_value
        assert logger_arg[3] is mock_tb_logger_cls.return_value


@patch("hypertorch.train.trainer.L.Trainer")
@patch(
    "hypertorch.train.trainer.MultiModelTrainer._MultiModelTrainer__is_tensorboard_available",
    return_value=False,
)
@patch("hypertorch.train.trainer.CSVLogger")
@patch("hypertorch.train.trainer.MarkdownTableLogger")
@patch("hypertorch.train.trainer.LaTexTableLogger")
def test_init_does_not_create_tensorboard_logger_when_not_available(
    mock_latex_logger_cls,
    mock_md_logger_cls,
    mock_csv_logger_cls,
    mock_is_tb_available,
    mock_trainer_cls,
    mock_model_configs,
    tmp_path,
):
    MultiModelTrainer(
        mock_model_configs,
        default_root_dir=str(tmp_path),
        experiment_name="experiment_0",
    )

    for call_args in mock_trainer_cls.call_args_list:
        logger_arg = call_args.kwargs["logger"]
        assert isinstance(logger_arg, list)
        assert len(logger_arg) == 3
        assert logger_arg[0] is mock_csv_logger_cls.return_value
        assert logger_arg[1] is mock_md_logger_cls.return_value
        assert logger_arg[2] is mock_latex_logger_cls.return_value


@patch("hypertorch.train.trainer.L.Trainer")
@patch(
    "hypertorch.train.trainer.MultiModelTrainer._MultiModelTrainer__is_tensorboard_available",
    return_value=True,
)
@patch("lightning.pytorch.loggers.TensorBoardLogger", create=True)
@patch("hypertorch.train.trainer.CSVLogger")
@patch("hypertorch.train.trainer.MarkdownTableLogger")
@patch("hypertorch.train.trainer.LaTexTableLogger")
def test_tensorboard_logger_matches_csv_logger_organization(
    mock_latex_logger_cls,
    mock_md_logger_cls,
    mock_csv_logger_cls,
    mock_tb_logger_cls,
    mock_is_tb_available,
    mock_trainer_cls,
    mock_model_configs,
    tmp_path,
):
    MultiModelTrainer(
        mock_model_configs,
        default_root_dir=str(tmp_path),
        experiment_name="experiment_0",
    )

    for i in range(len(mock_model_configs)):
        csv_call_kwargs = mock_csv_logger_cls.call_args_list[i].kwargs
        tb_call_kwargs = mock_tb_logger_cls.call_args_list[i].kwargs
        assert tb_call_kwargs["save_dir"] == csv_call_kwargs["save_dir"]
        assert tb_call_kwargs["name"] == csv_call_kwargs["name"]
        assert tb_call_kwargs["version"] == csv_call_kwargs["version"]


@patch("hypertorch.train.trainer.L.Trainer")
@patch(
    "hypertorch.train.trainer.MultiModelTrainer._MultiModelTrainer__is_tensorboard_available",
    return_value=True,
)
@patch("hypertorch.train.trainer.subprocess.Popen")
@patch("lightning.pytorch.loggers.TensorBoardLogger", create=True)
@patch("hypertorch.train.trainer.CSVLogger")
@patch("hypertorch.train.trainer.MarkdownTableLogger")
@patch("hypertorch.train.trainer.LaTexTableLogger")
@patch("shutil.which", return_value="tensorboard")
def test_init_starts_tensorboard_when_auto_start_tensorboard_true(
    shutil_which,
    mock_latex_logger_cls,
    mock_md_logger_cls,
    mock_csv_logger_cls,
    mock_tb_logger_cls,
    mock_popen,
    mock_is_tb_available,
    mock_trainer_cls,
    mock_model_configs,
    tmp_path,
):
    MultiModelTrainer(
        mock_model_configs,
        default_root_dir=str(tmp_path),
        experiment_name="experiment_0",
        auto_start_tensorboard=True,
    )

    mock_popen.assert_called_once()
    popen_cmd = mock_popen.call_args[0][0]
    assert "tensorboard" in popen_cmd
    assert "--logdir" in popen_cmd


@patch("hypertorch.train.trainer.L.Trainer")
@patch(
    "hypertorch.train.trainer.MultiModelTrainer._MultiModelTrainer__is_tensorboard_available",
    return_value=False,
)
@patch("hypertorch.train.trainer.subprocess.Popen")
@patch("hypertorch.train.trainer.CSVLogger")
@patch("hypertorch.train.trainer.MarkdownTableLogger")
@patch("hypertorch.train.trainer.LaTexTableLogger")
def test_init_does_not_start_tensorboard_when_not_available_and_auto_start_tensorboard_true(
    mock_latex_logger_cls,
    mock_md_logger_cls,
    mock_csv_logger_cls,
    mock_popen,
    mock_is_tb_available,
    mock_trainer_cls,
    mock_model_configs,
    tmp_path,
):
    with pytest.warns(UserWarning, match="TensorBoard is not available"):
        MultiModelTrainer(
            mock_model_configs,
            default_root_dir=str(tmp_path),
            experiment_name="experiment_0",
            auto_start_tensorboard=True,
        )

    mock_popen.assert_not_called()


@patch("hypertorch.train.trainer.L.Trainer")
@patch(
    "hypertorch.train.trainer.MultiModelTrainer._MultiModelTrainer__is_tensorboard_available",
    return_value=True,
)
@patch("hypertorch.train.trainer.subprocess.Popen")
@patch("lightning.pytorch.loggers.TensorBoardLogger", create=True)
@patch("hypertorch.train.trainer.CSVLogger")
@patch("hypertorch.train.trainer.MarkdownTableLogger")
@patch("hypertorch.train.trainer.LaTexTableLogger")
def test_init_does_not_start_tensorboard_when_auto_start_tensorboard_false(
    mock_latex_logger_cls,
    mock_md_logger_cls,
    mock_csv_logger_cls,
    mock_tb_logger_cls,
    mock_popen,
    mock_is_tb_available,
    mock_trainer_cls,
    mock_model_configs,
    tmp_path,
):
    MultiModelTrainer(
        mock_model_configs,
        default_root_dir=str(tmp_path),
        experiment_name="experiment_0",
        auto_start_tensorboard=False,
    )

    mock_popen.assert_not_called()


@patch("hypertorch.train.trainer.L.Trainer")
@patch(
    "hypertorch.train.trainer.MultiModelTrainer._MultiModelTrainer__is_tensorboard_available",
    return_value=True,
)
@patch("hypertorch.train.trainer.subprocess.Popen", side_effect=OSError("tensorboard not found"))
@patch("lightning.pytorch.loggers.TensorBoardLogger", create=True)
@patch("hypertorch.train.trainer.CSVLogger")
@patch("hypertorch.train.trainer.MarkdownTableLogger")
@patch("hypertorch.train.trainer.LaTexTableLogger")
@patch("shutil.which", return_value="tensorboard")
def test_start_tensorboard_warns_and_returns_none_on_failure(
    shutil_which,
    mock_latex_logger_cls,
    mock_md_logger_cls,
    mock_csv_logger_cls,
    mock_tb_logger_cls,
    mock_popen,
    mock_is_tb_available,
    mock_trainer_cls,
    mock_model_configs,
    tmp_path,
):
    with pytest.warns(UserWarning, match="Proceeding without starting TensorBoard as it failed"):
        MultiModelTrainer(
            mock_model_configs,
            default_root_dir=str(tmp_path),
            experiment_name="experiment_0",
            auto_start_tensorboard=True,
        )


@patch("hypertorch.train.trainer.L.Trainer")
@patch(
    "hypertorch.train.trainer.MultiModelTrainer._MultiModelTrainer__is_tensorboard_available",
    return_value=True,
)
@patch("hypertorch.train.trainer.subprocess.Popen")
@patch("lightning.pytorch.loggers.TensorBoardLogger", create=True)
@patch("hypertorch.train.trainer.CSVLogger")
@patch("hypertorch.train.trainer.MarkdownTableLogger")
@patch("hypertorch.train.trainer.LaTexTableLogger")
@patch("shutil.which", return_value="tensorboard")
def test_finalize_terminates_tensorboard_process(
    shutil_which,
    mock_latex_logger_cls,
    mock_md_logger_cls,
    mock_csv_logger_cls,
    mock_tb_logger_cls,
    mock_popen,
    mock_is_tb_available,
    mock_trainer_cls,
    mock_model_configs,
    tmp_path,
):
    trainer = MultiModelTrainer(
        mock_model_configs,
        default_root_dir=str(tmp_path),
        experiment_name="experiment_0",
        auto_start_tensorboard=True,
    )

    trainer.finalize()

    mock_popen.return_value.terminate.assert_called_once()


@patch("hypertorch.train.trainer.L.Trainer")
@patch("hypertorch.train.trainer.CSVLogger")
@patch("hypertorch.train.trainer.MarkdownTableLogger")
@patch("hypertorch.train.trainer.LaTexTableLogger")
def test_wait_does_nothing_when_no_tensorboard_process(
    mock_latex_logger_cls,
    mock_md_logger_cls,
    mock_csv_logger_cls,
    mock_trainer_cls,
    mock_model_configs,
):
    trainer = MultiModelTrainer(mock_model_configs)

    with patch("builtins.input") as mock_input:
        trainer.wait()
        mock_input.assert_not_called()


@patch("hypertorch.train.trainer.L.Trainer")
@patch(
    "hypertorch.train.trainer.MultiModelTrainer._MultiModelTrainer__is_tensorboard_available",
    return_value=True,
)
@patch("hypertorch.train.trainer.subprocess.Popen")
@patch("lightning.pytorch.loggers.TensorBoardLogger", create=True)
@patch("hypertorch.train.trainer.CSVLogger")
@patch("hypertorch.train.trainer.MarkdownTableLogger")
@patch("hypertorch.train.trainer.LaTexTableLogger")
@patch("shutil.which", return_value="tensorboard")
@patch("builtins.input", return_value="")  # Simulate user pressing Enter
def test_wait_prompts_user_when_tensorboard_process_is_running(
    mock_builtins_input,
    shutil_which,
    mock_latex_logger_cls,
    mock_md_logger_cls,
    mock_csv_logger_cls,
    mock_tb_logger_cls,
    mock_popen,
    mock_is_tb_available,
    mock_trainer_cls,
    mock_model_configs,
    tmp_path,
):
    trainer = MultiModelTrainer(
        mock_model_configs,
        default_root_dir=str(tmp_path),
        experiment_name="experiment_0",
        auto_start_tensorboard=True,
    )
    trainer.wait()

    mock_builtins_input.assert_called_once()


@patch("hypertorch.train.trainer.L.Trainer")
@patch(
    "hypertorch.train.trainer.MultiModelTrainer._MultiModelTrainer__is_tensorboard_available",
    return_value=True,
)
@patch("hypertorch.train.trainer.subprocess.Popen")
@patch("lightning.pytorch.loggers.TensorBoardLogger", create=True)
@patch("hypertorch.train.trainer.CSVLogger")
@patch("hypertorch.train.trainer.MarkdownTableLogger")
@patch("hypertorch.train.trainer.LaTexTableLogger")
@patch("shutil.which", return_value="tensorboard")
@patch("builtins.input", return_value="")  # Simulate user pressing Enter
def test_finalize_calls_wait_when_auto_wait_true(
    mock_builtins_input,
    shutil_which,
    mock_latex_logger_cls,
    mock_md_logger_cls,
    mock_csv_logger_cls,
    mock_tb_logger_cls,
    mock_popen,
    mock_is_tb_available,
    mock_trainer_cls,
    mock_model_configs,
    tmp_path,
):
    trainer = MultiModelTrainer(
        mock_model_configs,
        default_root_dir=str(tmp_path),
        experiment_name="experiment_0",
    )
    trainer.finalize()

    mock_builtins_input.assert_called_once()


@patch("hypertorch.train.trainer.L.Trainer")
@patch(
    "hypertorch.train.trainer.MultiModelTrainer._MultiModelTrainer__is_tensorboard_available",
    return_value=True,
)
@patch("hypertorch.train.trainer.subprocess.Popen")
@patch("lightning.pytorch.loggers.TensorBoardLogger", create=True)
@patch("hypertorch.train.trainer.CSVLogger")
@patch("hypertorch.train.trainer.MarkdownTableLogger")
@patch("hypertorch.train.trainer.LaTexTableLogger")
@patch("builtins.input", return_value="")  # Simulate user pressing Enter
def test_finalize_does_not_call_wait_when_auto_wait_false(
    mock_builtins_input,
    mock_latex_logger_cls,
    mock_md_logger_cls,
    mock_csv_logger_cls,
    mock_tb_logger_cls,
    mock_popen,
    mock_is_tb_available,
    mock_trainer_cls,
    mock_model_configs,
    tmp_path,
):
    trainer = MultiModelTrainer(
        mock_model_configs,
        default_root_dir=str(tmp_path),
        experiment_name="experiment_0",
        auto_start_tensorboard=True,
        auto_wait=False,
    )
    trainer.finalize()

    mock_builtins_input.assert_not_called()


@pytest.mark.parametrize(
    "interrupt",
    [
        pytest.param(KeyboardInterrupt, id="KeyboardInterrupt"),
        pytest.param(EOFError, id="EOFError"),
    ],
)
@patch("hypertorch.train.trainer.L.Trainer")
@patch(
    "hypertorch.train.trainer.MultiModelTrainer._MultiModelTrainer__is_tensorboard_available",
    return_value=True,
)
@patch("hypertorch.train.trainer.subprocess.Popen")
@patch("lightning.pytorch.loggers.TensorBoardLogger", create=True)
@patch("hypertorch.train.trainer.CSVLogger")
@patch("hypertorch.train.trainer.MarkdownTableLogger")
@patch("hypertorch.train.trainer.LaTexTableLogger")
@patch("shutil.which", return_value="tensorboard")
def test_finalize_handles_input_interrupts(
    shutil_which,
    mock_latex_logger_cls,
    mock_md_logger_cls,
    mock_csv_logger_cls,
    mock_tb_logger_cls,
    mock_popen,
    mock_is_tb_available,
    mock_trainer_cls,
    mock_model_configs,
    tmp_path,
    interrupt,
):
    with patch("builtins.input", side_effect=interrupt):
        trainer = MultiModelTrainer(
            mock_model_configs,
            default_root_dir=str(tmp_path),
            experiment_name="experiment_0",
            auto_start_tensorboard=True,
            auto_wait=True,
        )
        trainer.finalize()

    mock_popen.return_value.terminate.assert_called_once()


@patch("hypertorch.train.trainer.L.Trainer")
@patch("hypertorch.train.trainer.CSVLogger")
@patch("hypertorch.train.trainer.MarkdownTableLogger")
@patch("hypertorch.train.trainer.LaTexTableLogger")
def test_init_always_create_default_markdown_logger_per_model(
    mock_markdown_logger_cls,
    mock_latex_logger_cls,
    mock_csv_logger_cls,
    mock_trainer_cls,
    mock_model_configs,
    tmp_path,
):
    MultiModelTrainer(
        mock_model_configs,
        default_root_dir=str(tmp_path),
        experiment_name="experiment_0",
    )

    assert mock_markdown_logger_cls.call_count == len(mock_model_configs)

    full_model_names = [config.full_model_name() for config in mock_model_configs]

    for call in mock_markdown_logger_cls.call_args_list:
        assert call.kwargs["model_name"] in full_model_names
        assert call.kwargs["experiment_name"] == "experiment_0"
