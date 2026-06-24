import torch

from hypertorch.utils import create_seeded_torch_generator


def test_create_seeded_torch_generator_returns_none_without_seed():
    generator = create_seeded_torch_generator(device=torch.device("cpu"), seed=None)

    assert generator is None


def test_create_seeded_torch_generator_returns_reproducible_generator_with_seed():
    generator_a = create_seeded_torch_generator(device=torch.device("cpu"), seed=123)
    generator_b = create_seeded_torch_generator(device=torch.device("cpu"), seed=123)

    assert generator_a is not None
    assert generator_b is not None
    assert torch.equal(
        torch.randperm(5, generator=generator_a, dtype=torch.long),
        torch.randperm(5, generator=generator_b, dtype=torch.long),
    )
