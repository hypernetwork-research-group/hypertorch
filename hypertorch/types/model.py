from __future__ import annotations

import lightning as L

from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias
from collections.abc import Mapping

if TYPE_CHECKING:
    from hypertorch.data import DataLoader


CkptStrategy: TypeAlias = str | Path
"""Checkpoint selection strategy (``"best"`` or ``"last"``) or checkpoint path."""


TestResult: TypeAlias = Mapping[str, float]
"""Mapping from metric names to scalar test results."""


class ModelConfig:
    """
    A class representing the configuration of a model for training.

    Attributes:
        name: The name of the model.
        version: The version of the model.
        model: a LightningModule instance.
        is_trainable: Whether the model is trainable.
        trainer: A Trainer instance used for `fit_all` and, by default, `test_all`.
        test_trainer: Optional Trainer instance used by ``test_all``. When omitted,
            ``test_all`` falls back to ``trainer``.
        train_dataloader: Optional per-model train dataloader. When set, ``fit_all``
            uses this instead of the shared ``train_dataloader`` argument.
        val_dataloader: Optional per-model validation dataloader. When set, ``fit_all``
            uses this instead of the shared ``val_dataloader`` argument.
        test_dataloader: Optional per-model test dataloader. When set, ``test_all``
            uses this instead of the shared ``dataloader`` argument.
        trainer_kwargs: Optional additional per-model keyword arguments passed to the
            Lightning Trainer. These values override ``MultiModelTrainer`` keyword arguments
            with the same name. Possible keys include:
                accelerator: Optional per-model Trainer accelerator override.
                devices: Optional per-model Trainer devices override for training.
                test_devices: Optional per-model devices override for automatically-created
                    test trainers.
                strategy: Optional per-model Trainer strategy override.
                num_nodes: Optional per-model Trainer node count override.
                precision: Optional per-model Trainer precision override.
                max_epochs: Optional per-model maximum epoch override.
                min_epochs: Optional per-model minimum epoch override.
                max_steps: Optional per-model maximum step override.
                min_steps: Optional per-model minimum step override.
                check_val_every_n_epoch: Optional per-model validation frequency override.
                logger: Optional per-model Trainer logger override.
                default_root_dir: Optional per-model Trainer default root directory override.
                enable_autolog_hparams: Optional per-model Trainer hyperparameter logging override.
                log_every_n_steps: Optional per-model logging interval override.
                profiler: Optional per-model Trainer profiler override.
                fast_dev_run: Optional per-model Trainer fast development run override.
                enable_checkpointing: Optional per-model Trainer checkpointing override.
                enable_progress_bar: Optional per-model Trainer progress bar override.
                enable_model_summary: Optional per-model Trainer model summary override.
                callbacks: Optional per-model Trainer callbacks override.
                checkpoint_callback_kwargs: Optional per-model keyword arguments for the default
                    ``ModelCheckpoint`` callback.
    """

    def __init__(
        self,
        name: str,
        model: L.LightningModule,
        version: str = "default",
        is_trainable: bool = True,
        trainer: L.Trainer | None = None,
        test_trainer: L.Trainer | None = None,
        train_dataloader: DataLoader | None = None,
        val_dataloader: DataLoader | None = None,
        test_dataloader: DataLoader | None = None,
        trainer_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Initialize the model configuration.

        Args:
            name: The name of the model.
            version: The version of the model.
            model: a LightningModule instance.
            is_trainable: Whether the model is trainable.
            trainer: A Trainer instance used for `fit_all` and, by default, `test_all`.
            test_trainer: Optional Trainer instance used by ``test_all``. When omitted,
                ``test_all`` falls back to ``trainer``.
            train_dataloader: Optional per-model train dataloader. When set, ``fit_all``
                uses this instead of the shared ``train_dataloader`` argument.
            val_dataloader: Optional per-model validation dataloader. When set, ``fit_all``
                uses this instead of the shared ``val_dataloader`` argument.
            test_dataloader: Optional per-model test dataloader. When set, ``test_all``
                uses this instead of the shared ``dataloader`` argument.
            trainer_kwargs: Optional additional per-model keyword arguments passed to the
                Lightning Trainer. These values override ``MultiModelTrainer`` keyword arguments
                with the same name. Possible keys include:
                    accelerator: Optional per-model Trainer accelerator override.
                    devices: Optional per-model Trainer devices override for training.
                    test_devices: Optional per-model devices override for automatically-created
                        test trainers.
                    strategy: Optional per-model Trainer strategy override.
                    num_nodes: Optional per-model Trainer node count override.
                    precision: Optional per-model Trainer precision override.
                    max_epochs: Optional per-model maximum epoch override.
                    min_epochs: Optional per-model minimum epoch override.
                    max_steps: Optional per-model maximum step override.
                    min_steps: Optional per-model minimum step override.
                    check_val_every_n_epoch: Optional per-model validation frequency override.
                    logger: Optional per-model Trainer logger override.
                    default_root_dir: Optional per-model Trainer default root directory override.
                    enable_autolog_hparams: Optional per-model Trainer hyperparameter
                        logging override.
                    log_every_n_steps: Optional per-model logging interval override.
                    profiler: Optional per-model Trainer profiler override.
                    fast_dev_run: Optional per-model Trainer fast development run override.
                    enable_checkpointing: Optional per-model Trainer checkpointing override.
                    enable_progress_bar: Optional per-model Trainer progress bar override.
                    enable_model_summary: Optional per-model Trainer model summary override.
                    callbacks: Optional per-model Trainer callbacks override.
                    checkpoint_callback_kwargs: Optional per-model keyword arguments for the default
                        ``ModelCheckpoint`` callback.
        """
        self.name: str = name
        self.version: str = version
        self.model: L.LightningModule = model
        self.is_trainable: bool = is_trainable
        self.trainer: L.Trainer | None = trainer
        self.test_trainer: L.Trainer | None = test_trainer
        self.train_dataloader: DataLoader | None = train_dataloader
        self.val_dataloader: DataLoader | None = val_dataloader
        self.test_dataloader: DataLoader | None = test_dataloader
        self.trainer_kwargs: Mapping[str, Any] | None = trainer_kwargs

    def full_model_name(self) -> str:
        """
        Return the combined model name and version.

        Returns:
            full_model_name: Name formatted as ``"{name}:{version}"``.
        """
        return f"{self.name}:{self.version}"
