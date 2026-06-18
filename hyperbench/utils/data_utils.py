import math
import torch

from collections.abc import Sequence
from torch import Tensor


LATEX_CHARACTER_ESCAPE_TABLE: dict[str, str] = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "\r": " ",
    "\n": " ",
    "\t": " ",
}

MARKDOWN_CHARACTER_ESCAPE_TABLE: dict[str, str] = {
    "\\": r"\\",
    "|": r"\|",
    "`": r"\`",
    "*": r"\*",
    "=": r"\=",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "[": r"\[",
    "]": r"\]",
    "(": r"\(",
    ")": r"\)",
    "#": r"\#",
    "+": r"\+",
    "-": r"\-",
    ".": r"\.",
    "!": r"\!",
    "~": r"\~",
    "$": r"\$",
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "\r": " ",
    "\n": " ",
    "\t": " ",
}


def clone_optional_tensor(tensor: Tensor | None) -> Tensor | None:
    """
    Clone a tensor when it is provided.

    Args:
        tensor: Optional tensor to clone.

    Returns:
        tensor: A cloned tensor, or ``None`` when no tensor is provided.
    """
    return tensor.clone() if tensor is not None else None


def empty_nodefeatures() -> Tensor:
    """
    Create an empty node feature tensor.

    Returns:
        features: Empty floating-point tensor of shape ``(0, 0)``.
    """
    return torch.empty((0, 0), dtype=torch.float)


def empty_hyperedgeindex() -> Tensor:
    """
    Create an empty hyperedge index tensor.

    Returns:
        hyperedge_index: Empty long tensor of shape ``(2, 0)``.
    """
    return torch.empty((2, 0), dtype=torch.long)


def empty_edgeattr(num_edges: int) -> Tensor:
    """
    Create an empty edge attribute tensor for a fixed number of edges.

    Args:
        num_edges: Number of edge rows to allocate.

    Returns:
        edge_attr: Empty floating-point tensor of shape ``(num_edges, 0)``.
    """
    return torch.empty((num_edges, 0), dtype=torch.float)


def escape(text: str, escaped_characters_table: dict[str, str]) -> str:
    """
    Escape characters in text according to the provided escape table.

    Args:
        text: The input string to escape.
        escaped_characters_table: A dictionary mapping characters to their escaped versions.

    Returns:
        escaped_text: The escaped string based on the escape table.
    """
    return text.translate(str.maketrans(escaped_characters_table))


def to_non_empty_edgeattr(edge_attr: Tensor | None) -> Tensor:
    """
    Convert optional edge attributes to a tensor with an edge dimension.

    Args:
        edge_attr: Optional edge attribute tensor.

    Returns:
        edge_attr: The provided tensor, or an empty tensor with the inferred edge count.
    """
    num_edges = edge_attr.size(0) if edge_attr is not None else 0
    return empty_edgeattr(num_edges) if edge_attr is None else edge_attr


def to_0based_ids(original_ids: Tensor, ids_to_rebase: Tensor | None = None) -> Tensor:
    """
    Remap IDs to contiguous 0-based indices.

    If ``ids_to_rebase`` is provided, only IDs present in it are kept and remapped.
    If ``ids_to_rebase`` is not provided, all unique IDs in ``original_ids`` are remapped.

    Examples:
        >>> to_0based_ids(torch.tensor([1, 3, 3, 7]), torch.tensor([3, 7]))
        ... -> tensor([0, 0, 1])  # 1 is excluded, 3 -> 0, 7 -> 1

        >>> to_0based_ids(torch.tensor([5, 3, 5, 8]))
        ... -> tensor([1, 0, 1, 2])  # 3 -> 0, 5 -> 1, 8 -> 2

    Args:
        original_ids: Tensor of original IDs.
        ids_to_rebase: Optional tensor of IDs to keep and remap. If None, all unique IDs are used.

    Returns:
        ids: Tensor of 0-based IDs.
    """
    if ids_to_rebase is None:
        sorted_unique_original_ids = original_ids.unique(sorted=True)
        return torch.searchsorted(sorted_unique_original_ids, original_ids)

    keep_mask = torch.isin(original_ids, ids_to_rebase)
    ids_to_keep = original_ids[keep_mask]
    sorted_unique_ids_to_rebase = ids_to_rebase.unique(sorted=True)
    return torch.searchsorted(sorted_unique_ids_to_rebase, ids_to_keep)


def validate_is_between(
    name: str,
    value: int | float,
    min_value: int | float,
    max_value: int | float,
) -> None:
    """
    Validate that a numeric value is finite and lies within inclusive bounds.

    Args:
        name: Name of the validated value.
        value: Numeric value to validate.
        min_value: Inclusive lower bound.
        max_value: Inclusive upper bound.

    Raises:
        ValueError: If the bounds are invalid or the value is outside them.
    """
    if min_value > max_value:
        raise ValueError(
            f"Invalid bounds for {name!r}: 'min_value' ({min_value}) cannot "
            f"be greater than 'max_value' ({max_value})."
        )
    if not math.isfinite(value) or value < min_value or value > max_value:
        raise ValueError(
            f"{name!r} must be between {min_value} and {max_value} inclusive, got {value}."
        )


def validate_is_finite(name: str, value: int | float) -> None:
    """
    Validate that a numeric value is finite.

    Args:
        name: Name of the validated value.
        value: Numeric value to validate.

    Raises:
        ValueError: If the value is not finite.
    """
    if not math.isfinite(value):
        raise ValueError(f"{name!r} must be finite, got {value}.")


def validate_is_finite_when_provided(name: str, value: int | float | None) -> None:
    """
    Validate that an optional numeric value is finite when provided.

    Args:
        name: Name of the validated value.
        value: Optional numeric value to validate.

    Raises:
        ValueError: If the provided value is not finite.
    """
    if value is not None and not math.isfinite(value):
        raise ValueError(f"{name!r} must be finite when provided, got {value}.")


def validate_is_non_negative(name: str, value: int | float) -> None:
    """
    Validate that a numeric value is non-negative.

    Args:
        name: Name of the validated value.
        value: Numeric value to validate.

    Raises:
        ValueError: If the value is negative.
    """
    if value < 0:
        raise ValueError(f"{name!r} must be non-negative, got {value}.")


def validate_is_positive(name: str, value: int | float) -> None:
    """
    Validate that a numeric value is positive.

    Args:
        name: Name of the validated value.
        value: Numeric value to validate.

    Raises:
        ValueError: If the value is not positive.
    """
    if value <= 0:
        raise ValueError(f"{name!r} must be positive, got {value}.")


def validate_is_non_empty(name: str, value: Sequence) -> None:
    """
    Validate that a sequence is not empty.

    Args:
        name: Name of the validated sequence.
        value: Sequence to validate.

    Raises:
        ValueError: If the sequence is empty.
    """
    if len(value) < 1:
        raise ValueError(f"{name!r} cannot be empty.")


def validate_ratios(ratios: list[int | float]) -> None:
    """
    Validate split ratios.

    Args:
        ratios: Ratios that must be positive, finite, non-empty, and sum to one.

    Raises:
        ValueError: If any ratio is invalid or the ratios do not sum to one.
    """
    validate_is_non_empty("ratios", ratios)

    for ratio in ratios:
        validate_is_finite("ratios", ratio)
        validate_is_positive("ratios", ratio)

    # Allow small imprecision in sum of ratios, but raise error if it's significant
    # Example: ratios = [0.8, 0.1, 0.1] -> sum = 1.0 (valid)
    #          ratios = [0.8, 0.1, 0.05] -> sum = 0.95 (invalid, raises ValueError)
    #          (valid, allows small imprecision)
    #          ratios = [0.8, 0.1, 0.1, 0.0000001] -> sum = 1.0000001
    ratio_sum = float(sum(ratios))
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"'ratios' must sum to 1.0, got {ratio_sum}.")
