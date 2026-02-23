from enum import Enum
from typing import Any, Dict, Optional

from hyperbench.train.negative_sampler import NegativeSampler
from hyperbench.types.hdata import HData


class NegativeSamplingSchedule(Enum):
    """When to run negative sampling during training."""

    FIRST_EPOCH = "first_epoch"  # Only at epoch 0, cached for all subsequent epochs
    EVERY_N_EPOCHS = "every_n_epochs"  # Every N epochs (N provided separately)
    EVERY_EPOCH = "every_epoch"  # Negatives generated every epoch


class NegativeSamplingScheduler:
    def __init__(
        self,
        negative_sampler: NegativeSampler,
        negative_sampling_schedule: NegativeSamplingSchedule = NegativeSamplingSchedule.EVERY_EPOCH,
        negative_sampling_every_n: int = 1,
    ) -> None:
        self.negative_sampler = negative_sampler
        self.negative_sampling_schedule = negative_sampling_schedule
        self.negative_sampling_every_n = negative_sampling_every_n

        self.__cached_negative_samples: Optional[HData] = None

    @property
    def config(self) -> Dict[str, Any]:
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
            True if negatives should be resampled for the current epoch, False otherwise.
        """
        match self.negative_sampling_schedule:
            case NegativeSamplingSchedule.EVERY_EPOCH:
                return True
            case NegativeSamplingSchedule.EVERY_N_EPOCHS:
                return epoch % self.negative_sampling_every_n == 0
            case NegativeSamplingSchedule.FIRST_EPOCH:
                return epoch == 0
            case _:
                raise ValueError(
                    f"Unsupported negative sampling schedule: {self.negative_sampling_schedule}"
                )

    def sample(self, batch: HData, epoch: int) -> HData:
        """
        Sample fresh negatives if the schedule requires it, otherwise return cache.

        Args:
            batch: The current batch of data for which to sample negatives.
            epoch: The current epoch number, used to determine if sampling should occur based on the schedule.

        Returns:
            A batch of negative samples, either freshly sampled or from cache.
        """
        if self.should_sample(epoch):
            self.__cached_negative_samples = self.negative_sampler.sample(batch)

        if self.__cached_negative_samples is None:
            raise ValueError(
                "Asked to sample negatives but no scheduling happen, "
                f"check that the configuration is correct: {self.config}"
            )

        return self.__cached_negative_samples
