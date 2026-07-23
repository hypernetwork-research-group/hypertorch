from abc import ABC, abstractmethod
from lightning.pytorch.loggers import Logger


class ExperimentSharedLogger(Logger, ABC):
    """
    Extends `lightning.pytorch.loggers.Logger` to provide a base class
    for experiment loggers that share state across instances in one process.
    """

    @abstractmethod
    def clear(self, experiment_name: str) -> None:
        """
        Clear the internal state of the logger for a specific experiment.

        Args:
            experiment_name: The name of the experiment to clear.
        """
        raise NotImplementedError("Subclasses must implement the clear method.")

    @abstractmethod
    def destroy(self) -> None:
        """
        Destroy the internal shared state of the logger.

        Caution: This method should be used with care, as it will clear all shared
            state across experiments in the current process. This means that any metrics
            or data logged by other experiments will be lost as well. Use this method only
            when you are certain that you want to clear all shared state, and not just the
            state for a specific experiment. In that case, use the `clear` methods instead.
        """
        raise NotImplementedError("Subclasses must implement the destroy method.")
