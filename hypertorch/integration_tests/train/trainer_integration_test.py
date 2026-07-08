import pytest
import lightning as L
import torch

from pathlib import Path
from torch import Tensor, nn, optim
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, TensorDataset
from hypertorch.train import MultiModelTrainer
from hypertorch.types import ModelConfig

from hypertorch.integration_tests.common import (
    extract_state_dict,
    zero_models_parameters,
)


EXPERIMENT_NAME = "trainer_fit_test_integration_test"


class TinyBinaryClassifier(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(2, 1)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x).squeeze(-1)

    def training_step(self, batch: tuple[Tensor, Tensor], _: int) -> Tensor:
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], _: int) -> None:
        x, y = batch
        loss = self.loss(self(x), y)
        self.log("val/loss", loss)

    def test_step(self, batch: tuple[Tensor, Tensor], _: int) -> None:
        x, y = batch
        loss = self.loss(self(x), y)
        self.log("test/loss", loss)

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.SGD(self.parameters(), lr=0.01)


class EpochStartTrackingTinyBinaryClassifier(TinyBinaryClassifier):
    def __init__(self, epoch_starts: list[int], fail_at_epoch_start: int | None = None) -> None:
        super().__init__()
        self.epoch_starts = epoch_starts
        self.fail_at_epoch_start = fail_at_epoch_start

    def on_train_epoch_start(self) -> None:
        current_epoch = int(self.trainer.current_epoch)
        self.epoch_starts.append(current_epoch)
        if self.fail_at_epoch_start == current_epoch:
            raise RuntimeError(f"Intentional training failure at epoch {current_epoch}.")


@pytest.fixture
def mock_tiny_dataloader() -> DataLoader:
    x = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    y = torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=torch.float32)
    return DataLoader(TensorDataset(x, y), batch_size=2)


@pytest.mark.integration
@pytest.mark.parametrize(
    "num_models",
    [
        pytest.param(1, id="one_model"),
        pytest.param(2, id="multiple_models"),
    ],
)
def test_fit_and_test_all_trains_and_evaluates_models(
    tmp_path,
    mock_tiny_dataloader,
    num_models,
):
    model_configs = __distinct_model_configs(num_models)
    initial_state_dicts = [extract_state_dict(config.model) for config in model_configs]

    trainer = MultiModelTrainer(
        model_configs=model_configs,
        default_root_dir=tmp_path,
        experiment_name=EXPERIMENT_NAME,
        max_epochs=1,
        accelerator="auto",
        devices=1,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=1,
    )

    trainer.fit_all(
        train_dataloader=mock_tiny_dataloader,
        val_dataloader=mock_tiny_dataloader,
        verbose=False,
    )
    results = trainer.test_all(
        dataloader=mock_tiny_dataloader,
        verbose=False,
        verbose_loop=False,
    )

    expected_result_keys = {config.full_model_name() for config in model_configs}
    assert set(results) == expected_result_keys
    assert all("test/loss" in result for result in results.values())

    for config, initial_state_dict in zip(model_configs, initial_state_dicts, strict=True):
        state_dict_after_train = extract_state_dict(config.model)
        assert any(
            not torch.equal(parameter, initial_state_dict[parameter_name])
            for parameter_name, parameter in state_dict_after_train.items()
        )

    __assert_default_logger_outputs(tmp_path / EXPERIMENT_NAME, model_configs)


@pytest.mark.integration
def test_custom_logger_is_used_instead_of_default_loggers(
    tmp_path,
    mock_tiny_dataloader,
):
    custom_logger = CSVLogger(
        save_dir=tmp_path / "custom_logs",
        name="provided_logger",
        version="run_0",
    )

    trainer = MultiModelTrainer(
        model_configs=__distinct_model_configs(num_models=1),
        default_root_dir=tmp_path,
        experiment_name=EXPERIMENT_NAME,
        max_epochs=1,
        accelerator="auto",
        devices=1,
        logger=custom_logger,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=1,
    )

    trainer.fit_all(
        train_dataloader=mock_tiny_dataloader,
        val_dataloader=mock_tiny_dataloader,
        verbose=False,
    )
    trainer.test_all(
        dataloader=mock_tiny_dataloader,
        verbose=False,
        verbose_loop=False,
    )

    custom_metrics_csv = tmp_path / "custom_logs" / "provided_logger" / "run_0" / "metrics.csv"
    assert custom_metrics_csv.exists()

    custom_metrics_content = custom_metrics_csv.read_text()
    assert "train/loss" in custom_metrics_content
    assert "val/loss" in custom_metrics_content
    assert "test/loss" in custom_metrics_content

    assert not (tmp_path / EXPERIMENT_NAME / "comparison").exists()
    assert not (tmp_path / EXPERIMENT_NAME / "tiny" / "version_0" / "metrics.csv").exists()


@pytest.mark.integration
def test_checkpoints_can_be_loaded_and_used(
    tmp_path,
    mock_tiny_dataloader,
):
    model_configs = __duplicate_model_configs()
    trainer = MultiModelTrainer(
        model_configs=model_configs,
        default_root_dir=tmp_path,
        experiment_name=EXPERIMENT_NAME,
        max_epochs=1,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=1,
        callbacks=[ModelCheckpoint(save_last=True)],
    )

    trainer.fit_all(
        train_dataloader=mock_tiny_dataloader,
        val_dataloader=mock_tiny_dataloader,
        verbose=False,
    )

    checkpoint_state_dicts = [
        torch.load(checkpoint_dir / "last.ckpt", map_location="cpu")["state_dict"]
        for checkpoint_dir in __checkpoint_dirs(tmp_path / EXPERIMENT_NAME)
    ]

    zero_models_parameters(model_configs)

    results = trainer.test_all(
        dataloader=mock_tiny_dataloader,
        ckpt_path="last",
        verbose=False,
        verbose_loop=False,
    )

    assert set(results) == {"tiny:duplicate"}
    for config, checkpoint_state_dict in zip(model_configs, checkpoint_state_dicts, strict=True):
        model_state_dict = config.model.state_dict()
        for parameter_name, checkpoint_parameter in checkpoint_state_dict.items():
            assert torch.equal(model_state_dict[parameter_name], checkpoint_parameter)


@pytest.mark.integration
def test_checkpoints_can_be_loaded_and_used_with_different_trainers(
    tmp_path,
    mock_tiny_dataloader,
):
    model_configs = __distinct_model_configs(num_models=1)
    trainer = MultiModelTrainer(
        model_configs=model_configs,
        default_root_dir=tmp_path,
        experiment_name=EXPERIMENT_NAME,
        max_epochs=1,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=1,
        callbacks=[ModelCheckpoint(save_last=True)],
    )

    trainer.fit_all(
        train_dataloader=mock_tiny_dataloader,
        val_dataloader=mock_tiny_dataloader,
        verbose=False,
    )

    assert model_configs[0].trainer is not None
    checkpoint_callback = model_configs[0].trainer.checkpoint_callback
    assert isinstance(checkpoint_callback, ModelCheckpoint)
    checkpoint_path = Path(checkpoint_callback.last_model_path)
    checkpoint_state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]

    load_model_configs = __distinct_model_configs(num_models=1)
    load_trainer = MultiModelTrainer(
        model_configs=load_model_configs,
        default_root_dir=tmp_path,
        experiment_name=EXPERIMENT_NAME,
        max_epochs=1,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=1,
        callbacks=[ModelCheckpoint(save_last=True)],
    )

    results = load_trainer.test_all(
        dataloader=mock_tiny_dataloader,
        ckpt_path=checkpoint_path,
        verbose=False,
        verbose_loop=False,
    )

    assert set(results) == {"tiny:0"}

    model_state_dict = load_model_configs[0].model.state_dict()
    for parameter_name, checkpoint_parameter in checkpoint_state_dict.items():
        assert torch.equal(model_state_dict[parameter_name], checkpoint_parameter)


@pytest.mark.integration
def test_retry_uses_existing_model_checkpoints_after_partial_multi_model_failure(
    tmp_path,
    mock_tiny_dataloader,
):
    max_epochs = 5
    failure_epoch = 3

    first_initial_epoch_starts: list[int] = []
    second_initial_epoch_starts: list[int] = []
    initial_model_configs = [
        ModelConfig(
            name="resumable",
            version="first",
            model=EpochStartTrackingTinyBinaryClassifier(epoch_starts=first_initial_epoch_starts),
        ),
        ModelConfig(
            name="resumable",
            version="second",
            model=EpochStartTrackingTinyBinaryClassifier(
                epoch_starts=second_initial_epoch_starts,
                fail_at_epoch_start=failure_epoch,
            ),
        ),
    ]

    initial_trainer = MultiModelTrainer(
        model_configs=initial_model_configs,
        default_root_dir=tmp_path,
        experiment_name=EXPERIMENT_NAME,
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=1,
        callbacks=[ModelCheckpoint(save_last=True)],
    )

    with pytest.raises(
        RuntimeError,
        match=f"Intentional training failure at epoch {failure_epoch}.",
    ):
        initial_trainer.fit_all(
            train_dataloader=mock_tiny_dataloader,
            val_dataloader=mock_tiny_dataloader,
            verbose=False,
        )

    experiment_dir = tmp_path / EXPERIMENT_NAME
    first_model_checkpoint = __last_checkpoint_path(experiment_dir, "resumable", "first")
    second_model_checkpoint = __last_checkpoint_path(experiment_dir, "resumable", "second")
    first_resume_epoch = torch.load(first_model_checkpoint, map_location="cpu")["epoch"] + 1
    second_resume_epoch = torch.load(second_model_checkpoint, map_location="cpu")["epoch"] + 1

    assert first_initial_epoch_starts == [0, 1, 2, 3, 4]
    assert second_initial_epoch_starts == [0, 1, 2, 3]
    assert first_resume_epoch == max_epochs
    assert second_resume_epoch == failure_epoch

    first_retry_epoch_starts: list[int] = []
    second_retry_epoch_starts: list[int] = []
    retry_model_configs = [
        ModelConfig(
            name="resumable",
            version="first",
            model=EpochStartTrackingTinyBinaryClassifier(first_retry_epoch_starts),
        ),
        ModelConfig(
            name="resumable",
            version="second",
            model=EpochStartTrackingTinyBinaryClassifier(second_retry_epoch_starts),
        ),
    ]

    retry_trainer = MultiModelTrainer(
        model_configs=retry_model_configs,
        default_root_dir=tmp_path,
        experiment_name=EXPERIMENT_NAME,
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=1,
        callbacks=[ModelCheckpoint(save_last=True)],
    )

    with pytest.warns(UserWarning, match="Checkpoint directory .* exists and is not empty"):
        retry_trainer.fit_all(
            train_dataloader=mock_tiny_dataloader,
            val_dataloader=mock_tiny_dataloader,
            ckpt_path="last",
            verbose=False,
        )

    assert first_retry_epoch_starts == []
    assert second_retry_epoch_starts == [3, 4]


@pytest.mark.integration
def test_best_checkpoints_can_be_loaded_and_used(
    tmp_path,
    mock_tiny_dataloader,
):
    model_configs = __duplicate_model_configs()
    trainer = MultiModelTrainer(
        model_configs=model_configs,
        default_root_dir=tmp_path,
        experiment_name=EXPERIMENT_NAME,
        max_epochs=1,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=1,
        # save_top_k=1 to ensure that only the checkpoint with the best val/loss is saved
        callbacks=[ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1)],
    )

    trainer.fit_all(
        train_dataloader=mock_tiny_dataloader,
        val_dataloader=mock_tiny_dataloader,
        verbose=False,
    )

    best_checkpoint_paths = []
    for checkpoint_dir in __checkpoint_dirs(tmp_path / EXPERIMENT_NAME):
        # Load all checkpoints as we know there's only one
        # due to save_top_k=1 and it's the best checkpoint
        checkpoint_paths = list(checkpoint_dir.glob("*.ckpt"))
        assert len(checkpoint_paths) == 1
        best_checkpoint_paths.append(checkpoint_paths[0])

    checkpoint_state_dicts = [
        torch.load(checkpoint_path, map_location="cpu")["state_dict"]
        for checkpoint_path in best_checkpoint_paths
    ]

    for checkpoint_path in best_checkpoint_paths:
        assert checkpoint_path.exists()

    zero_models_parameters(model_configs)

    results = trainer.test_all(
        dataloader=mock_tiny_dataloader,
        ckpt_path="best",
        verbose=False,
        verbose_loop=False,
    )

    assert set(results) == {"tiny:duplicate"}
    for config, checkpoint_state_dict in zip(model_configs, checkpoint_state_dicts, strict=True):
        model_state_dict = config.model.state_dict()
        for parameter_name, checkpoint_parameter in checkpoint_state_dict.items():
            assert torch.equal(model_state_dict[parameter_name], checkpoint_parameter)


@pytest.mark.integration
def test_checkpointing_is_isolated_for_duplicate_model_names(tmp_path, mock_tiny_dataloader):
    trainer = MultiModelTrainer(
        model_configs=__duplicate_model_configs(),
        default_root_dir=tmp_path,
        experiment_name=EXPERIMENT_NAME,
        max_epochs=1,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=1,
    )

    trainer.fit_all(
        train_dataloader=mock_tiny_dataloader,
        val_dataloader=mock_tiny_dataloader,
        verbose=False,
    )

    for checkpoint_dir in __checkpoint_dirs(tmp_path / EXPERIMENT_NAME):
        assert checkpoint_dir.exists()
        assert list(checkpoint_dir.glob("*.ckpt"))


@pytest.mark.integration
def test_user_checkpoint_callback_without_dirpath_is_isolated_per_duplicate_model(
    tmp_path,
    mock_tiny_dataloader,
):
    trainer = MultiModelTrainer(
        model_configs=__duplicate_model_configs(),
        default_root_dir=tmp_path,
        experiment_name=EXPERIMENT_NAME,
        max_epochs=1,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=1,
        callbacks=[ModelCheckpoint(save_last=True)],
    )

    trainer.fit_all(
        train_dataloader=mock_tiny_dataloader,
        val_dataloader=mock_tiny_dataloader,
        verbose=False,
    )

    for checkpoint_dir in __checkpoint_dirs(tmp_path / EXPERIMENT_NAME):
        assert (checkpoint_dir / "last.ckpt").exists()


@pytest.mark.integration
def test_user_checkpoint_callback_with_custom_dirpath_is_respected(
    tmp_path,
    mock_tiny_dataloader,
):
    custom_checkpoint_dir = tmp_path / "custom_checkpoints"
    trainer = MultiModelTrainer(
        model_configs=__distinct_model_configs(num_models=1),
        default_root_dir=tmp_path,
        experiment_name=EXPERIMENT_NAME,
        max_epochs=1,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=1,
        callbacks=[ModelCheckpoint(dirpath=custom_checkpoint_dir, save_last=True)],
    )

    trainer.fit_all(
        train_dataloader=mock_tiny_dataloader,
        val_dataloader=mock_tiny_dataloader,
        verbose=False,
    )

    assert (custom_checkpoint_dir / "last.ckpt").exists()
    assert not (tmp_path / EXPERIMENT_NAME / "tiny" / "version_0" / "checkpoints").exists()


@pytest.mark.integration
def test_checkpoint_callback_kwargs_configure_default_checkpoint_callback(
    tmp_path,
    mock_tiny_dataloader,
):
    custom_checkpoint_dir = tmp_path / "checkpoint_kwargs"
    trainer = MultiModelTrainer(
        model_configs=__distinct_model_configs(num_models=1),
        default_root_dir=tmp_path,
        experiment_name=EXPERIMENT_NAME,
        max_epochs=1,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=1,
        checkpoint_callback_kwargs={
            "dirpath": custom_checkpoint_dir,
            "filename": "weights-only-{epoch}",
            "save_weights_only": True,
        },
    )

    trainer.fit_all(
        train_dataloader=mock_tiny_dataloader,
        val_dataloader=mock_tiny_dataloader,
        verbose=False,
    )

    checkpoint_paths = list(custom_checkpoint_dir.glob("weights-only-*.ckpt"))
    assert len(checkpoint_paths) == 1

    checkpoint = torch.load(checkpoint_paths[0], map_location="cpu")
    assert "state_dict" in checkpoint
    assert "optimizer_states" not in checkpoint
    assert not (tmp_path / EXPERIMENT_NAME / "tiny" / "version_0" / "checkpoints").exists()


def __checkpoint_dirs(experiment_dir: Path) -> list[Path]:
    return [
        experiment_dir / "tiny" / "version_duplicate" / "model_0" / "checkpoints",
        experiment_dir / "tiny" / "version_duplicate" / "model_1" / "checkpoints",
    ]


def __distinct_model_configs(num_models: int = 2) -> list[ModelConfig]:
    return [
        ModelConfig(
            name="tiny",
            version=str(model_index),
            model=TinyBinaryClassifier(),
        )
        for model_index in range(num_models)
    ]


def __duplicate_model_configs(num_models: int = 2) -> list[ModelConfig]:
    return [
        ModelConfig(
            name="tiny",
            version="duplicate",
            model=TinyBinaryClassifier(),
        )
        for _ in range(num_models)
    ]


def __last_checkpoint_path(
    experiment_dir: Path,
    model_name: str,
    model_version: str,
) -> Path:
    return experiment_dir / model_name / f"version_{model_version}" / "checkpoints" / "last.ckpt"


def __assert_default_logger_outputs(experiment_dir: Path, model_configs: list[ModelConfig]) -> None:
    for config in model_configs:
        metrics_csv = experiment_dir / config.name / f"version_{config.version}" / "metrics.csv"
        assert metrics_csv.exists()

        metrics_content = metrics_csv.read_text()
        assert "train/loss" in metrics_content
        assert "val/loss" in metrics_content
        assert "test/loss" in metrics_content

    comparison_dir = experiment_dir / "comparison"
    for result_file in [
        comparison_dir / "overall.md",
        comparison_dir / "overall.tex",
        comparison_dir / "test.tex",
        comparison_dir / "test.md",
    ]:
        assert result_file.exists()

        content = result_file.read_text()
        assert "test/loss" in content

        for config in model_configs:
            assert config.full_model_name() in content

    for result_file in [
        comparison_dir / "overall.md",
        comparison_dir / "overall.tex",
        comparison_dir / "train.md",
        comparison_dir / "train.tex",
    ]:
        assert result_file.exists()

        content = result_file.read_text()
        assert "train/loss" in content

        for config in model_configs:
            assert config.full_model_name() in content

    for result_file in [
        comparison_dir / "overall.md",
        comparison_dir / "overall.tex",
        comparison_dir / "val.md",
        comparison_dir / "val.tex",
    ]:
        assert result_file.exists()

        content = result_file.read_text()
        assert "val/loss" in content

        for config in model_configs:
            assert config.full_model_name() in content
