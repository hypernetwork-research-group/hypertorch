from __future__ import annotations

import lightning as L

from typing import TYPE_CHECKING, Literal, TypeAlias
from collections.abc import Mapping

if TYPE_CHECKING:
    from hyperbench.data import DataLoader


CkptStrategy: TypeAlias = Literal["best", "last"]
TestResult: TypeAlias = Mapping[str, float]


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
        self.name = name
        self.version = version
        self.model = model
        self.is_trainable = is_trainable
        self.trainer = trainer
        self.test_trainer = test_trainer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    def full_model_name(self) -> str:
        return f"{self.name}:{self.version}"
