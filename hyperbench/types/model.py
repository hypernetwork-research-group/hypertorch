import lightning as L

from typing import Literal, Mapping, Optional, TypeAlias


CkptStrategy: TypeAlias = Literal["best", "last"]
TestResult: TypeAlias = Mapping[str, float]


class ModelConfig:
    """
    A class representing the configuration of a model for the MultiModelTrainer trainer.

    Args:
        name: The name of the model.
        version: The version of the model.
        model: a LightningModule instance.
        trainer: a Trainer instance.
        is_trainable: Whether the model is trainable.
    """

    def __init__(
        self,
        name: str,
        model: L.LightningModule,
        version: str = "default",
        is_trainable: bool = True,
        trainer: Optional[L.Trainer] = None,
    ) -> None:
        self.name = name
        self.version = version
        self.model = model
        self.is_trainable = is_trainable
        self.trainer = trainer

    def full_model_name(self) -> str:
        return f"{self.name}:{self.version}"
