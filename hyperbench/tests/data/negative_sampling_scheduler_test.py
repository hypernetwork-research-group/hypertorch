import pytest
import torch
import re

from typing import Any, cast
from unittest.mock import MagicMock
from hyperbench.data import NegativeSampler, NegativeSamplingSchedule, NegativeSamplingScheduler
from hyperbench.types import HData


@pytest.fixture
def mock_batch():
    return HData(
        x=torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float),
        hyperedge_index=torch.tensor([[0, 1, 2], [0, 0, 1]], dtype=torch.long),
        num_nodes=3,
        num_hyperedges=2,
    )


@pytest.fixture
def mock_negative_hdata():
    return HData(
        x=torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float),
        hyperedge_index=torch.tensor([[0, 1], [0, 0]], dtype=torch.long),
        num_nodes=2,
        num_hyperedges=1,
    )


@pytest.fixture
def mock_sampler(mock_negative_hdata):
    sampler = MagicMock(spec=NegativeSampler)
    sampler.sample.return_value = mock_negative_hdata
    return sampler


def test_config_returns_scheduler_parameters(mock_sampler):
    schedule: NegativeSamplingSchedule = "every_n_epochs"
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
        negative_sampling_schedule="first_epoch",
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
        negative_sampling_schedule="every_epoch",
    )

    result = scheduler.sample(mock_batch, epoch=0)

    mock_sampler.sample.assert_called_once_with(mock_batch)
    assert result is mock_negative_hdata


def test_sample_raises_when_cache_is_empty(mock_sampler, mock_batch):
    scheduler = NegativeSamplingScheduler(
        negative_sampler=mock_sampler,
        negative_sampling_schedule="every_n_epochs",
        negative_sampling_every_n=5,
    )

    with pytest.raises(ValueError, match="Asked to sample negatives but no scheduling happen"):
        # Epoch 1 is not a multiple of 5 and cache is empty
        scheduler.sample(mock_batch, epoch=1)


@pytest.mark.parametrize(
    "num_epoch, expected_call_count",
    [
        pytest.param(0, 1, id="num_epoch_0_multiple_of_n=3"),
        pytest.param(1, 1, id="num_epoch_1_not_multiple_of_n=3"),
        pytest.param(2, 1, id="num_epoch_2_not_multiple_of_n=3"),
        pytest.param(3, 2, id="num_epoch_3_multiple_of_n=3"),
        pytest.param(120, 41, id="num_epoch_120_multiple_of_n=3"),
        pytest.param(121, 41, id="num_epoch_121_not_multiple_of_n=3"),
        pytest.param(122, 41, id="num_epoch_122_not_multiple_of_n=3"),
        pytest.param(123, 42, id="num_epoch_123_multiple_of_n=3"),
        pytest.param(1000, 334, id="num_epoch_1000_multiple_of_n=3"),
        pytest.param(1001, 334, id="num_epoch_1001_not_multiple_of_n=3"),
        pytest.param(1002, 335, id="num_epoch_1002_multiple_of_n=3"),
        pytest.param(1003, 335, id="num_epoch_1003_not_multiple_of_n=3"),
    ],
)
def test_sample_resamples_on_every_n_epoch(
    num_epoch,
    expected_call_count,
    mock_sampler,
    mock_batch,
):
    scheduler = NegativeSamplingScheduler(
        negative_sampler=mock_sampler,
        negative_sampling_schedule="every_n_epochs",
        negative_sampling_every_n=3,
    )

    for epoch in range(num_epoch + 1):
        scheduler.sample(mock_batch, epoch)

    # Example: epoch=0, should sample once
    #          epoch=3, should sample twice (at epochs 0 and 3)
    #          epoch=4, should sample twice (at epochs 0 and 3)
    #          epoch=6, should sample three times (at epochs 0, 3, and 6) and so on
    assert mock_sampler.sample.call_count == expected_call_count


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
        negative_sampling_schedule="every_epoch",
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
        negative_sampling_schedule="first_epoch",
    )

    assert scheduler.should_sample(epoch) == expected_should_sample


def test_should_sample_rejects_invalid_epoch(mock_sampler):
    scheduler = NegativeSamplingScheduler(negative_sampler=mock_sampler)

    with pytest.raises(ValueError, match=re.escape("Epoch must be non-negative, got -1.")):
        scheduler.should_sample(epoch=-1)


def test_should_sample_rejects_unsupported_schedule(mock_sampler):
    scheduler = NegativeSamplingScheduler(
        negative_sampler=mock_sampler,
        negative_sampling_schedule=cast(Any, "sometimes"),
    )

    with pytest.raises(
        ValueError,
        match=re.escape("Unsupported negative sampling schedule: 'sometimes'."),
    ):
        scheduler.should_sample(epoch=0)


@pytest.mark.parametrize(
    "every_n, expected_exception, expected_message",
    [
        pytest.param(
            0, ValueError, "negative_sampling_every_n must be positive, got 0.", id="zero"
        ),
        pytest.param(
            -1, ValueError, "negative_sampling_every_n must be positive, got -1.", id="negative"
        ),
    ],
)
def test_should_sample_rejects_invalid_every_n(
    mock_sampler, every_n, expected_exception, expected_message
):
    scheduler = NegativeSamplingScheduler(
        negative_sampler=mock_sampler,
        negative_sampling_schedule="every_n_epochs",
        negative_sampling_every_n=cast(Any, every_n),
    )

    with pytest.raises(expected_exception, match=re.escape(expected_message)):
        scheduler.should_sample(epoch=0)
