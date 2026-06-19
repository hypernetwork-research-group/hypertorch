import torch

from torch import Generator


def create_seeded_torch_generator(
    device: torch.device,
    seed: int | None,
) -> Generator | None:
    """
    Create a seeded torch generator when a seed is provided.

    Args:
        device: Device where the generator should be created.
        seed: Optional seed for deterministic random operations.

    Returns:
        generator: A seeded torch.Generator when ``seed`` is provided, otherwise ``None``.
    """
    if seed is None:
        return None
    generator = Generator(device=device)
    generator.manual_seed(seed)
    return generator
