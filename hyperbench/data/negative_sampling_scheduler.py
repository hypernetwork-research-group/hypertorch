from typing import Any, Literal, TypeAlias
from hyperbench.types import HData
from hyperbench.data import NegativeSampler


NegativeSamplingSchedule: TypeAlias = Literal[
    "first_epoch",  # Only at epoch 0, cached for all subsequent epochs
    "every_n_epochs",  # Every N epochs (N provided separately)
    "every_epoch",  # Negatives generated every epoch
]


class NegativeSamplingScheduler:
    """
    Manages when to perform negative sampling during training based on a specified schedule.

    This class allows for flexible scheduling of negative sampling, enabling it to be performed at
    different frequencies (e.g., every epoch, every N epochs, or only at the first epoch). The
    scheduler maintains a cache of the most recently sampled negatives, which can be reused across
    epochs if the schedule does not require resampling. This helps to optimize training by avoiding
    unnecessary sampling when the schedule dictates that negatives should only be generated at
    certain intervals.

    Args:
        negative_sampler: An instance of a ``NegativeSampler`` that defines how to sample negatives.
        negative_sampling_schedule: Literal string specifying the schedule for sampling negatives.
        negative_sampling_every_n: An integer specifying the interval for sampling negatives
            when the schedule is set to ``"every_n_epochs"``. This parameter is ignored
            for other schedules.

    """

    def __init__(
        self,
        negative_sampler: NegativeSampler,
        negative_sampling_schedule: NegativeSamplingSchedule = "every_epoch",
        negative_sampling_every_n: int = 1,
    ) -> None:
        self.negative_sampler = negative_sampler
        self.negative_sampling_schedule = negative_sampling_schedule
        self.negative_sampling_every_n = negative_sampling_every_n

        self.__cached_negative_samples: HData | None = None

    @property
    def config(self) -> dict[str, Any]:
        """
        Returns the configuration of the negative sampling scheduler as a dictionary.
        """
        return {
            "negative_sampler": self.negative_sampler,
            "negative_sampling_schedule": self.negative_sampling_schedule,
            "negative_sampling_every_n": self.negative_sampling_every_n,
        }

    def should_sample(self, epoch: int) -> bool:
        """
        Whether to resample negatives for the current epoch.

        Args:
            epoch: The current epoch number, used to determine if sampling should occur based on the schedule.

        Returns:
            should_sample: True if negatives should be resampled for the current epoch, False otherwise.

        """
        if epoch < 0:
            raise ValueError(f"Epoch must be non-negative, got {epoch}.")

        match self.negative_sampling_schedule:
            case "every_n_epochs":
                if self.negative_sampling_every_n <= 0:
                    raise ValueError(
                        f"negative_sampling_every_n must be positive, got {self.negative_sampling_every_n}."
                    )
                return epoch % self.negative_sampling_every_n == 0
            case "first_epoch":
                return epoch == 0
            case "every_epoch":
                return True
            case _:
                raise ValueError(
                    f"Unsupported negative sampling schedule: {self.negative_sampling_schedule!r}."
                )

    def sample(self, batch: HData, epoch: int) -> HData:
        """
        Sample fresh negatives if the schedule requires it, otherwise return cache.

        Args:
            batch: The current batch of data for which to sample negatives.
            epoch: The current epoch number, used to determine if sampling should occur based on the schedule.

        Returns:
            negatives: A batch of negative samples, either freshly sampled or from cache.

        """
        if self.should_sample(epoch):
            self.__cached_negative_samples = self.negative_sampler.sample(batch)

        if self.__cached_negative_samples is None:
            raise ValueError(
                "Asked to sample negatives but no scheduling happen, "
                f"check that the configuration is correct: {self.config}"
            )

        return self.__cached_negative_samples
