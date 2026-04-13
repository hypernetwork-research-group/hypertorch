import pytest

from unittest.mock import MagicMock, patch
from hyperbench.train import MultiModelTrainer, MarkdownTableLogger
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
        model_config.train_dataloader = None
        model_config.val_dataloader = None
        model_config.test_dataloader = None

        model_configs.append(model_config)

    return model_configs


@patch("hyperbench.train.trainer.L.Trainer")
@patch("hyperbench.train.trainer.CSVLogger")
def test_trainer_initialization(mock_csv_logger_cls, mock_trainer_cls, mock_model_configs):
    multi_model_trainer = MultiModelTrainer(mock_model_configs)

    assert len(multi_model_trainer.model_configs) == len(mock_model_configs)
    for config in multi_model_trainer.model_configs:
        assert config.trainer is not None


@patch("hyperbench.train.trainer.L.Trainer")
@patch("hyperbench.train.trainer.CSVLogger")
def test_trainer_initialization_with_initialized_trainer(
    mock_csv_logger_cls, mock_trainer_cls, mock_model_configs
):
    mock_model_configs[0].trainer = mock_trainer_cls

    multi_model_trainer = MultiModelTrainer(mock_model_configs)

    assert len(multi_model_trainer.model_configs) == len(mock_model_configs)
    for config in multi_model_trainer.model_configs:
        assert config.trainer is not None


@patch("hyperbench.train.trainer.L.Trainer")
@patch("hyperbench.train.trainer.CSVLogger")
def test_context_manager_enter_returns_self(
    mock_csv_logger_cls, mock_trainer_cls, mock_model_configs
):
    multi_model_trainer = MultiModelTrainer(mock_model_configs)
    with multi_model_trainer as trainer:
        assert trainer is multi_model_trainer


@patch("hyperbench.train.trainer.L.Trainer")
@patch("hyperbench.train.trainer.CSVLogger")
@patch("hyperbench.train.trainer.MultiModelTrainer.finalize")
def test_context_manager_exit_calls_finalize(
    mock_finalize, mock_csv_logger_cls, mock_trainer_cls, mock_model_configs
):
    multi_model_trainer = MultiModelTrainer(mock_model_configs)
    with multi_model_trainer:
        pass

    mock_finalize.assert_called_once()


@patch("hyperbench.train.trainer.L.Trainer")
@patch("hyperbench.train.trainer.CSVLogger")
@patch("hyperbench.train.trainer.MultiModelTrainer.finalize", side_effect=Exception("error"))
def test_del_suppresses_exception_from_finalize(
    mock_finalize, mock_csv_logger_cls, mock_trainer_cls, mock_model_configs
):
    MultiModelTrainer(mock_model_configs).__del__()


@patch("hyperbench.train.trainer.L.Trainer")
@patch("hyperbench.train.trainer.CSVLogger")
def test_models_property_returns_models(mock_csv_logger_cls, mock_trainer_cls, mock_model_configs):
    multi_model_trainer = MultiModelTrainer(mock_model_configs)
    models = multi_model_trainer.models

    assert len(models) == len(mock_model_configs)


@patch("hyperbench.train.trainer.L.Trainer")
def test_models_property_returns_empty_when_no_models(mock_trainer_cls):
    multi_model_trainer = MultiModelTrainer([])
    models = multi_model_trainer.models

    assert len(models) == 0


@patch("hyperbench.train.trainer.L.Trainer")
@patch("hyperbench.train.trainer.CSVLogger")
def test_model_returns_model_when_correct_name_and_no_version(
    mock_csv_logger_cls, mock_trainer_cls, mock_model_configs
):
    mock_model_configs[0].version = "default"
    mock_model_configs[0].model.version = "default"

    multi_model_trainer = MultiModelTrainer(mock_model_configs)
    found = multi_model_trainer.model(name="model0")

    assert found is not None
    assert found.name == "model0"
    assert found.version == "default"


@patch("hyperbench.train.trainer.L.Trainer")
@patch("hyperbench.train.trainer.CSVLogger")
def test_model_returns_None_when_incorrect_name_and_no_version(
    mock_csv_logger_cls, mock_trainer_cls, mock_model_configs
):
    multi_model_trainer = MultiModelTrainer(mock_model_configs)
    found = multi_model_trainer.model(name="nonexistent")

    assert found is None


@patch("hyperbench.train.trainer.L.Trainer")
@patch("hyperbench.train.trainer.CSVLogger")
def test_model_returns_model_when_correct_name_and_version(
    mock_csv_logger_cls, mock_trainer_cls, mock_model_configs
):
    multi_model_trainer = MultiModelTrainer(mock_model_configs)
    found = multi_model_trainer.model(name="model0", version="0")

    assert found is not None
    assert found.name == "model0"
    assert found.version == "0"


@patch("hyperbench.train.trainer.L.Trainer")
@patch("hyperbench.train.trainer.CSVLogger")
def test_model_returns_None_when_incorrect_name_and_version(
    mock_csv_logger_cls, mock_trainer_cls, mock_model_configs
):
    multi_model_trainer = MultiModelTrainer(mock_model_configs)
    not_found = multi_model_trainer.model(name="nonexistent", version="100")

    assert not_found is None


@patch("hyperbench.train.trainer.L.Trainer")
@patch("hyperbench.train.trainer.CSVLogger")
def test_model_returns_None_when_incorrect_name_and_correct_version(
    mock_csv_logger_cls, mock_trainer_cls, mock_model_configs
):
    multi_model_trainer = MultiModelTrainer(mock_model_configs)
    not_found = multi_model_trainer.model(name="nonexistent", version="0")

    assert not_found is None


@patch(
    "hyperbench.train.trainer.L.Trainer",
    side_effect=lambda *args, **kwargs: new_mock_trainer(),
)
@patch("hyperbench.train.trainer.CSVLogger")
def test_fit_all_calls_fit(mock_csv_logger_cls, mock_trainer_cls, mock_model_configs):
    multi_model_trainer = MultiModelTrainer(mock_model_configs)

    multi_model_trainer.fit_all(verbose=False)
    for config in mock_model_configs:
        config.trainer.fit.assert_called_once()


@patch("hyperbench.train.trainer.L.Trainer")
def test_fit_all_with_no_models(mock_trainer_cls):
    multi_model_trainer = MultiModelTrainer([])

    with pytest.raises(ValueError, match="No models to fit."):
        multi_model_trainer.fit_all(verbose=False)


@patch("hyperbench.train.trainer.L.Trainer", return_value=None)
@patch("hyperbench.train.trainer.CSVLogger")
def test_fit_all_raises_when_None_trainer(
    mock_csv_logger_cls, mock_trainer_cls, mock_model_configs
):
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
@patch("hyperbench.train.trainer.CSVLogger")
def test_fit_all_with_verbose_true_prints(
    mock_csv_logger_cls, mock_trainer_cls, mock_model_configs, capsys
):
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
@patch("hyperbench.train.trainer.CSVLogger")
def test_fit_all_skips_non_trainable_model(
    mock_csv_logger_cls, mock_trainer_cls, mock_model_configs
):
    mock_model_configs[0].is_trainable = False
    multi_model_trainer = MultiModelTrainer(mock_model_configs)

    multi_model_trainer.fit_all(verbose=False)

    mock_model_configs[0].trainer.fit.assert_not_called()
    mock_model_configs[1].trainer.fit.assert_called_once()


@patch(
    "hyperbench.train.trainer.L.Trainer",
    side_effect=lambda *args, **kwargs: new_mock_trainer(),
)
@patch("hyperbench.train.trainer.CSVLogger")
def test_fit_all_skips_non_trainable_model_with_verbose_prints(
    mock_csv_logger_cls, mock_trainer_cls, mock_model_configs, capsys
):
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
@patch("hyperbench.train.trainer.CSVLogger")
def test_test_all_calls_test_and_returns_results(
    mock_csv_logger_cls, mock_trainer_cls, mock_model_configs
):
    multi_model_trainer = MultiModelTrainer(mock_model_configs)

    results = multi_model_trainer.test_all(verbose=False)

    assert all("acc" in v for v in results.values())

    for config in mock_model_configs:
        config.trainer.test.assert_called_once()


@patch("hyperbench.train.trainer.L.Trainer")
def test_test_all_with_no_models(mock_trainer_cls):
    multi_model_trainer = MultiModelTrainer([])

    with pytest.raises(ValueError, match="No models to test."):
        multi_model_trainer.test_all(verbose=False)


@patch("hyperbench.train.trainer.L.Trainer", return_value=None)
@patch("hyperbench.train.trainer.CSVLogger")
def test_test_all_raises_when_None_trainer(
    mock_csv_logger_cls, mock_trainer_cls, mock_model_configs
):
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
@patch("hyperbench.train.trainer.CSVLogger")
def test_test_all_with_verbose_true_prints(
    mock_csv_logger_cls, mock_trainer_cls, mock_model_configs, capsys
):
    multi_model_trainer = MultiModelTrainer(mock_model_configs)

    multi_model_trainer.test_all(verbose=True)

    for config in mock_model_configs:
        config.trainer.test.assert_called_once()

    captured_out = capsys.readouterr().out
    logs = [line for line in captured_out.splitlines() if "Test model" in line]
    assert len(logs) == len(mock_model_configs)


@patch("hyperbench.train.trainer.L.Trainer")
@patch("hyperbench.train.trainer.CSVLogger")
def test_init_auto_increments_experiment_name(
    mock_csv_logger_cls, mock_trainer_cls, mock_model_configs, tmp_path
):
    (tmp_path / "experiment_0").mkdir()
    (tmp_path / "experiment_1").mkdir()

    MultiModelTrainer(
        mock_model_configs,
        default_root_dir=str(tmp_path),
    )

    for call in mock_csv_logger_cls.call_args_list:
        assert "experiment_2" in str(call.kwargs["save_dir"])


@patch("hyperbench.train.trainer.L.Trainer")
@patch("hyperbench.train.trainer.CSVLogger")
def test_init_defaults_to_experiment_0_when_default_root_dir_does_not_exist(
    mock_csv_logger_cls, mock_trainer_cls, mock_model_configs, tmp_path
):
    MultiModelTrainer(
        mock_model_configs,
        default_root_dir=str(tmp_path / "nonexistent_dir"),
    )

    for call in mock_csv_logger_cls.call_args_list:
        assert "experiment_0" in str(call.kwargs["save_dir"])


@patch("hyperbench.train.trainer.L.Trainer")
@patch("hyperbench.train.trainer.CSVLogger")
def test_init_defaults_to_experiment_0_when_no_existing_experiment(
    mock_csv_logger_cls, mock_trainer_cls, mock_model_configs, tmp_path
):
    MultiModelTrainer(
        mock_model_configs,
        default_root_dir=str(tmp_path),
    )

    for call in mock_csv_logger_cls.call_args_list:
        assert "experiment_0" in str(call.kwargs["save_dir"])


@patch("hyperbench.train.trainer.L.Trainer")
@patch("hyperbench.train.trainer.CSVLogger")
def test_init_always_create_default_csv_logger_per_model(
    mock_csv_logger_cls, mock_trainer_cls, mock_model_configs, tmp_path
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


@patch("hyperbench.train.trainer.L.Trainer")
def test_init_passes_custom_logger_to_all_models(mock_trainer_cls, mock_model_configs):
    custom_logger = MagicMock()
    MultiModelTrainer(mock_model_configs, logger=custom_logger)

    for call_args in mock_trainer_cls.call_args_list:
        assert call_args.kwargs["logger"] is custom_logger


@patch("hyperbench.train.trainer.L.Trainer")
@patch("hyperbench.train.trainer.CSVLogger")
@patch("hyperbench.train.trainer.MarkdownTableLogger")
def test_init_each_model_gets_distinct_logger(
    mock_md_logger_cls, mock_csv_logger_cls, mock_trainer_cls, mock_model_configs, tmp_path
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


@patch("hyperbench.train.trainer.L.Trainer")
@patch(
    "hyperbench.train.trainer.MultiModelTrainer._MultiModelTrainer__is_tensorboard_available",
    return_value=True,
)
@patch("lightning.pytorch.loggers.TensorBoardLogger", create=True)
@patch("hyperbench.train.trainer.CSVLogger")
@patch("hyperbench.train.trainer.MarkdownTableLogger")
def test_init_creates_tensorboard_logger_when_available(
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
        assert len(logger_arg) == 3
        assert logger_arg[0] is mock_csv_logger_cls.return_value
        assert logger_arg[1] is mock_md_logger_cls.return_value
        assert logger_arg[2] is mock_tb_logger_cls.return_value


@patch("hyperbench.train.trainer.L.Trainer")
@patch(
    "hyperbench.train.trainer.MultiModelTrainer._MultiModelTrainer__is_tensorboard_available",
    return_value=False,
)
@patch("hyperbench.train.trainer.CSVLogger")
@patch("hyperbench.train.trainer.MarkdownTableLogger")
def test_init_does_not_create_tensorboard_logger_when_not_available(
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
        assert len(logger_arg) == 2
        assert logger_arg[0] is mock_csv_logger_cls.return_value
        assert logger_arg[1] is mock_md_logger_cls.return_value


@patch("hyperbench.train.trainer.L.Trainer")
@patch(
    "hyperbench.train.trainer.MultiModelTrainer._MultiModelTrainer__is_tensorboard_available",
    return_value=True,
)
@patch("lightning.pytorch.loggers.TensorBoardLogger", create=True)
@patch("hyperbench.train.trainer.CSVLogger")
def test_tensorboard_logger_matches_csv_logger_organization(
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


@patch("hyperbench.train.trainer.L.Trainer")
@patch(
    "hyperbench.train.trainer.MultiModelTrainer._MultiModelTrainer__is_tensorboard_available",
    return_value=True,
)
@patch("hyperbench.train.trainer.subprocess.Popen")
@patch("lightning.pytorch.loggers.TensorBoardLogger", create=True)
@patch("hyperbench.train.trainer.CSVLogger")
def test_init_starts_tensorboard_when_auto_start_tensorboard_true(
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


@patch("hyperbench.train.trainer.L.Trainer")
@patch(
    "hyperbench.train.trainer.MultiModelTrainer._MultiModelTrainer__is_tensorboard_available",
    return_value=False,
)
@patch("hyperbench.train.trainer.subprocess.Popen")
@patch("hyperbench.train.trainer.CSVLogger")
def test_init_does_not_start_tensorboard_when_not_available_and_auto_start_tensorboard_true(
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


@patch("hyperbench.train.trainer.L.Trainer")
@patch(
    "hyperbench.train.trainer.MultiModelTrainer._MultiModelTrainer__is_tensorboard_available",
    return_value=True,
)
@patch("hyperbench.train.trainer.subprocess.Popen")
@patch("lightning.pytorch.loggers.TensorBoardLogger", create=True)
@patch("hyperbench.train.trainer.CSVLogger")
def test_init_does_not_start_tensorboard_when_auto_start_tensorboard_false(
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


@patch("hyperbench.train.trainer.L.Trainer")
@patch(
    "hyperbench.train.trainer.MultiModelTrainer._MultiModelTrainer__is_tensorboard_available",
    return_value=True,
)
@patch("hyperbench.train.trainer.subprocess.Popen", side_effect=OSError("tensorboard not found"))
@patch("lightning.pytorch.loggers.TensorBoardLogger", create=True)
@patch("hyperbench.train.trainer.CSVLogger")
def test_start_tensorboard_warns_and_returns_none_on_failure(
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


@patch("hyperbench.train.trainer.L.Trainer")
@patch(
    "hyperbench.train.trainer.MultiModelTrainer._MultiModelTrainer__is_tensorboard_available",
    return_value=True,
)
@patch("hyperbench.train.trainer.subprocess.Popen")
@patch("lightning.pytorch.loggers.TensorBoardLogger", create=True)
@patch("hyperbench.train.trainer.CSVLogger")
def test_finalize_terminates_tensorboard_process(
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


@patch("hyperbench.train.trainer.L.Trainer")
@patch("hyperbench.train.trainer.CSVLogger")
def test_wait_does_nothing_when_no_tensorboard_process(
    mock_csv_logger_cls, mock_trainer_cls, mock_model_configs
):
    trainer = MultiModelTrainer(mock_model_configs)

    with patch("builtins.input") as mock_input:
        trainer.wait()
        mock_input.assert_not_called()


@patch("hyperbench.train.trainer.L.Trainer")
@patch(
    "hyperbench.train.trainer.MultiModelTrainer._MultiModelTrainer__is_tensorboard_available",
    return_value=True,
)
@patch("hyperbench.train.trainer.subprocess.Popen")
@patch("lightning.pytorch.loggers.TensorBoardLogger", create=True)
@patch("hyperbench.train.trainer.CSVLogger")
@patch("builtins.input", return_value="")  # Simulate user pressing Enter
def test_wait_prompts_user_when_tensorboard_process_is_running(
    mock_builtins_input,
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


@patch("hyperbench.train.trainer.L.Trainer")
@patch(
    "hyperbench.train.trainer.MultiModelTrainer._MultiModelTrainer__is_tensorboard_available",
    return_value=True,
)
@patch("hyperbench.train.trainer.subprocess.Popen")
@patch("lightning.pytorch.loggers.TensorBoardLogger", create=True)
@patch("hyperbench.train.trainer.CSVLogger")
@patch("builtins.input", return_value="")  # Simulate user pressing Enter
def test_finalize_calls_wait_when_auto_wait_true(
    mock_builtins_input,
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
        auto_wait=True,
    )
    trainer.finalize()

    mock_builtins_input.assert_called_once()


@patch("hyperbench.train.trainer.L.Trainer")
@patch(
    "hyperbench.train.trainer.MultiModelTrainer._MultiModelTrainer__is_tensorboard_available",
    return_value=True,
)
@patch("hyperbench.train.trainer.subprocess.Popen")
@patch("lightning.pytorch.loggers.TensorBoardLogger", create=True)
@patch("hyperbench.train.trainer.CSVLogger")
@patch("builtins.input", return_value="")  # Simulate user pressing Enter
def test_finalize_does_not_call_wait_when_auto_wait_false(
    mock_builtins_input,
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
@patch("hyperbench.train.trainer.L.Trainer")
@patch(
    "hyperbench.train.trainer.MultiModelTrainer._MultiModelTrainer__is_tensorboard_available",
    return_value=True,
)
@patch("hyperbench.train.trainer.subprocess.Popen")
@patch("lightning.pytorch.loggers.TensorBoardLogger", create=True)
@patch("hyperbench.train.trainer.CSVLogger")
def test_finalize_handles_input_interrupts(
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


@patch("hyperbench.train.trainer.L.Trainer")
@patch("hyperbench.train.trainer.MarkdownTableLogger")
def test_init_always_create_default_markdown_logger_per_model(
    mock_markdown_logger_cls, mock_trainer_cls, mock_model_configs, tmp_path
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
