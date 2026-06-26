from __future__ import annotations

import lightning as L

from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias
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

    def full_model_name(self) -> str:
        """
        Return the combined model name and version.

        Returns:
            full_model_name: Name formatted as ``"{name}:{version}"``.
        """
        return f"{self.name}:{self.version}"
