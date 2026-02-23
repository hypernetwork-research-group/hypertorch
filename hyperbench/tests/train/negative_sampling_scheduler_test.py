import pytest
import torch

from unittest.mock import MagicMock
from hyperbench.train import NegativeSampler, NegativeSamplingSchedule, NegativeSamplingScheduler
from hyperbench.types import HData


@pytest.fixture
def mock_batch():
    return HData(
        x=torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        hyperedge_index=torch.tensor([[0, 1, 2], [0, 0, 1]]),
        num_nodes=3,
        num_hyperedges=2,
    )


@pytest.fixture
def mock_negative_hdata():
    return HData(
        x=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        hyperedge_index=torch.tensor([[0, 1], [0, 0]]),
        num_nodes=2,
        num_hyperedges=1,
    )


@pytest.fixture
def mock_sampler(mock_negative_hdata):
    sampler = MagicMock(spec=NegativeSampler)
    sampler.sample.return_value = mock_negative_hdata
    return sampler


def test_config_returns_scheduler_parameters(mock_sampler):
    schedule = NegativeSamplingSchedule.EVERY_N_EPOCHS
    scheduler = NegativeSamplingScheduler(
        negative_sampler=mock_sampler,
        negative_sampling_schedule=schedule,
        negative_sampling_every_n=3,
    )

    config = scheduler.config
    assert config["negative_sampler"] is mock_sampler
    assert config["negative_sampling_schedule"] == schedule
    assert config["negative_sampling_every_n"] == 3


def test_sample_caches_result_across_non_sampling_epochs(mock_sampler, mock_batch):
    scheduler = NegativeSamplingScheduler(
        negative_sampler=mock_sampler,
        negative_sampling_schedule=NegativeSamplingSchedule.FIRST_EPOCH,
    )

    # Epoch 0: should sample
    result_epoch_0 = scheduler.sample(mock_batch, epoch=0)
    # Epoch 1: should return cache
    result_epoch_1 = scheduler.sample(mock_batch, epoch=1)

    assert result_epoch_0 is result_epoch_1
    mock_sampler.sample.assert_called_once()


def test_sample_delegates_to_negative_sampler(mock_sampler, mock_batch, mock_negative_hdata):
    scheduler = NegativeSamplingScheduler(
        negative_sampler=mock_sampler,
        negative_sampling_schedule=NegativeSamplingSchedule.EVERY_EPOCH,
    )

    result = scheduler.sample(mock_batch, epoch=0)

    mock_sampler.sample.assert_called_once_with(mock_batch)
    assert result is mock_negative_hdata


def test_sample_raises_when_cache_is_empty(mock_sampler, mock_batch):
    scheduler = NegativeSamplingScheduler(
        negative_sampler=mock_sampler,
        negative_sampling_schedule=NegativeSamplingSchedule.EVERY_N_EPOCHS,
        negative_sampling_every_n=5,
    )

    with pytest.raises(ValueError, match="Asked to sample negatives but no scheduling happen"):
        # Epoch 1 is not a multiple of 5 and cache is empty
        scheduler.sample(mock_batch, epoch=1)


def test_sample_resamples_on_every_n_epoch(mock_sampler, mock_batch):
    scheduler = NegativeSamplingScheduler(
        negative_sampler=mock_sampler,
        negative_sampling_schedule=NegativeSamplingSchedule.EVERY_N_EPOCHS,
        negative_sampling_every_n=3,
    )

    scheduler.sample(mock_batch, epoch=0)
    scheduler.sample(mock_batch, epoch=1)
    scheduler.sample(mock_batch, epoch=2)
    scheduler.sample(mock_batch, epoch=3)

    # Should have sampled at epoch 0 and epoch 3 (multiples of 3)
    assert mock_sampler.sample.call_count == 2


@pytest.mark.parametrize(
    "epoch, expected_should_sample",
    [
        pytest.param(0, True, id="epoch_0"),
        pytest.param(1, True, id="epoch_1"),
        pytest.param(5, True, id="epoch_5"),
        pytest.param(100, True, id="epoch_100"),
    ],
)
def test_should_sample_every_epoch(mock_sampler, epoch, expected_should_sample):
    scheduler = NegativeSamplingScheduler(
        negative_sampler=mock_sampler,
        negative_sampling_schedule=NegativeSamplingSchedule.EVERY_EPOCH,
    )

    assert scheduler.should_sample(epoch) == expected_should_sample


@pytest.mark.parametrize(
    "epoch, every_n, expected_should_sample",
    [
        pytest.param(0, 3, True, id="epoch_0_every_3"),
        pytest.param(1, 3, False, id="epoch_1_every_3"),
        pytest.param(2, 3, False, id="epoch_2_every_3"),
        pytest.param(3, 3, True, id="epoch_3_every_3"),
        pytest.param(6, 3, True, id="epoch_6_every_3"),
        pytest.param(0, 1, True, id="epoch_0_every_1"),
        pytest.param(5, 1, True, id="epoch_5_every_1"),
        pytest.param(4, 5, False, id="epoch_4_every_5"),
        pytest.param(5, 5, True, id="epoch_5_every_5"),
    ],
)
def test_should_sample_every_n_epochs(mock_sampler, epoch, every_n, expected_should_sample):
    scheduler = NegativeSamplingScheduler(
        negative_sampler=mock_sampler,
        negative_sampling_schedule=NegativeSamplingSchedule.EVERY_N_EPOCHS,
        negative_sampling_every_n=every_n,
    )

    assert scheduler.should_sample(epoch) == expected_should_sample


@pytest.mark.parametrize(
    "epoch, expected_should_sample",
    [
        pytest.param(0, True, id="epoch_0"),
        pytest.param(1, False, id="epoch_1"),
        pytest.param(2, False, id="epoch_2"),
        pytest.param(100, False, id="epoch_100"),
    ],
)
def test_should_sample_first_epoch(mock_sampler, epoch, expected_should_sample):
    scheduler = NegativeSamplingScheduler(
        negative_sampler=mock_sampler,
        negative_sampling_schedule=NegativeSamplingSchedule.FIRST_EPOCH,
    )

    assert scheduler.should_sample(epoch) == expected_should_sample


def test_should_sample_raises_on_unsupported_schedule(mock_sampler):
    scheduler = NegativeSamplingScheduler(negative_sampler=mock_sampler)
    scheduler.negative_sampling_schedule = MagicMock(__str__=lambda _: "UNSUPPORTED_SCHEDULE")

    with pytest.raises(
        ValueError, match="Unsupported negative sampling schedule: UNSUPPORTED_SCHEDULE"
    ):
        scheduler.should_sample(epoch=0)
