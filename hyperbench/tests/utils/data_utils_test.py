import pytest
import torch
import re

from hyperbench.utils import (
    LATEX_CHARACTER_ESCAPE_TABLE,
    MARKDOWN_CHARACTER_ESCAPE_TABLE,
    clone_optional_tensor,
    empty_nodefeatures,
    empty_hyperedgeindex,
    empty_edgeattr,
    escape,
    to_non_empty_edgeattr,
    to_0based_ids,
    validate_is_between,
    validate_is_finite,
    validate_is_finite_when_provided,
    validate_is_non_empty,
    validate_is_non_negative,
    validate_is_positive,
    validate_ratios,
)


def test_clone_optional_tensor_with_none():
    result = clone_optional_tensor(None)

    assert result is None


def test_clone_optional_tensor_with_tensor_preserves_values():
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float)

    result = clone_optional_tensor(tensor)

    assert result is not None
    assert torch.equal(result, tensor)


def test_clone_optional_tensor_with_tensor_does_not_share_storage():
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float)

    result = clone_optional_tensor(tensor)
    assert result is not None

    result[0, 0] = 99.0
    print(tensor, result)
    assert tensor[0, 0] == 1.0


def test_empty_hyperedgeindex():
    result = empty_hyperedgeindex()

    assert result.shape == (2, 0)
    assert result.dtype == torch.long


def test_empty_edgeattr_zero_edges():
    result = empty_edgeattr(num_edges=0)

    assert result.shape == (0, 0)
    assert result.dtype == torch.float


def test_empty_edgeattr_single_edge():
    result = empty_edgeattr(num_edges=1)

    assert result.shape == (1, 0)
    assert result.dtype == torch.float


def test_empty_edgeattr_with_edges():
    result = empty_edgeattr(num_edges=5)

    assert result.shape == (5, 0)
    assert result.dtype == torch.float


@pytest.mark.parametrize(
    "text, table, expected_escaped_text",
    [
        pytest.param(
            r"Special characters: \ & % $ # _ { } ~ ^",
            LATEX_CHARACTER_ESCAPE_TABLE,
            r"Special characters: \textbackslash{} \& \% \$ \# \_ \{ \} "
            r"\textasciitilde{} \textasciicircum{}",
            id="with_special_characters_latex",
        ),
        pytest.param(
            "Newlines and tabs:\nLine 2\tTabbed",
            LATEX_CHARACTER_ESCAPE_TABLE,
            "Newlines and tabs: Line 2 Tabbed",
            id="with_newlines_and_tabs_latex",
        ),
        pytest.param(
            "No special characters here",
            LATEX_CHARACTER_ESCAPE_TABLE,
            "No special characters here",
            id="without_special_characters_latex",
        ),
        pytest.param(
            r"Special characters: \ | ` * = _ { } [ ] ( ) # + - . ! ~ $ & < >",
            MARKDOWN_CHARACTER_ESCAPE_TABLE,
            r"Special characters: \\ \| \` \* \= \_ \{ \} \[ \] \( \) "
            r"\# \+ \- \. \! \~ \$ &amp; &lt; &gt;",
            id="with_special_characters_markdown",
        ),
        pytest.param(
            "No special characters here",
            MARKDOWN_CHARACTER_ESCAPE_TABLE,
            "No special characters here",
            id="without_special_characters_markdown",
        ),
        pytest.param(
            "Newlines and tabs:\nLine 2\tTabbed",
            MARKDOWN_CHARACTER_ESCAPE_TABLE,
            "Newlines and tabs: Line 2 Tabbed",
            id="with_newlines_and_tabs_markdown",
        ),
    ],
)
def test_escape_with_provided_table(text, table, expected_escaped_text):
    escaped_text = escape(text, table)

    assert escaped_text == expected_escaped_text


def test_to_non_empty_edgeattr_with_none():
    result = to_non_empty_edgeattr(edge_attr=None)

    assert result.shape == (0, 0)
    assert result.dtype == torch.float


def test_to_non_empty_edgeattr_with_tensor():
    edge_attr = torch.tensor([[0.5], [0.7], [0.9]], dtype=torch.float)
    result = to_non_empty_edgeattr(edge_attr)

    assert torch.equal(result, edge_attr)
    assert result.shape == (3, 1)
    assert result.dtype == torch.float


def test_to_non_empty_edgeattr_with_empty_tensor():
    edge_attr = torch.empty((0, 3), dtype=torch.float)
    result = to_non_empty_edgeattr(edge_attr)

    assert torch.equal(result, edge_attr)
    assert result.shape == (0, 3)
    assert result.dtype == torch.float


def test_to_non_empty_edgeattr_with_multi_dimensional():
    edge_attr = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float)
    result = to_non_empty_edgeattr(edge_attr)

    assert torch.equal(result, edge_attr)
    assert result.shape == (2, 3)
    assert result.dtype == torch.float


def test_empty_nodefeatures():
    result = empty_nodefeatures()

    assert result.shape == (0, 0)
    assert result.dtype == torch.float


@pytest.mark.parametrize(
    "original_ids, ids_to_rebase, expected_result",
    [
        pytest.param(
            torch.tensor([1, 3, 3, 7], dtype=torch.long),
            torch.tensor([3, 7], dtype=torch.long),
            torch.tensor([0, 0, 1], dtype=torch.long),
            id="with_ids_to_rebase",
        ),
        pytest.param(
            torch.tensor([1, 3, 3, 7], dtype=torch.long),
            torch.tensor([1, 3, 7], dtype=torch.long),
            torch.tensor([0, 1, 1, 2], dtype=torch.long),
            id="with_ids_to_rebase_all",
        ),
        pytest.param(
            torch.tensor([5, 3, 5, 8], dtype=torch.long),
            None,
            torch.tensor([1, 0, 1, 2], dtype=torch.long),
            id="without_ids_to_rebase",
        ),
    ],
)
def test_to_0based_ids(original_ids, ids_to_rebase, expected_result):
    result = to_0based_ids(original_ids, ids_to_rebase)

    assert torch.equal(result, expected_result)


@pytest.mark.parametrize(
    "value, min_value, max_value",
    [
        pytest.param(0, 0, 1, id="lower_bound"),
        pytest.param(0.5, 0, 1, id="middle_float"),
        pytest.param(1, 0, 1, id="upper_bound"),
    ],
)
def test_validate_is_between_accepts_values_within_inclusive_bounds(value, min_value, max_value):
    validate_is_between("value", value, min_value, max_value)


@pytest.mark.parametrize(
    "value, min_value, max_value, match",
    [
        pytest.param(
            -0.1, 0, 1, "'value' must be between 0 and 1 inclusive, got -0.1.", id="below_min"
        ),
        pytest.param(
            1.1, 0, 1, "'value' must be between 0 and 1 inclusive, got 1.1.", id="above_max"
        ),
        pytest.param(
            float("inf"), 0, 1, "'value' must be between 0 and 1 inclusive, got inf.", id="infinite"
        ),
        pytest.param(
            float("nan"), 0, 1, "'value' must be between 0 and 1 inclusive, got nan.", id="nan"
        ),
    ],
)
def test_validate_is_between_rejects_values_outside_bounds_or_non_finite(
    value, min_value, max_value, match
):
    with pytest.raises(ValueError, match=match):
        validate_is_between("value", value, min_value, max_value)


def test_validate_is_between_rejects_invalid_bounds():
    with pytest.raises(
        ValueError,
        match=re.escape("'min_value' (2) cannot be greater than 'max_value' (1)"),
    ):
        validate_is_between("value", 1, 2, 1)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative_int"),
        pytest.param(1.5, id="positive_float"),
    ],
)
def test_validate_is_finite_accepts_finite_values(value):
    validate_is_finite("value", value)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(float("inf"), id="positive_infinity"),
        pytest.param(float("-inf"), id="negative_infinity"),
        pytest.param(float("nan"), id="nan"),
    ],
)
def test_validate_is_finite_rejects_non_finite_values(value):
    with pytest.raises(ValueError, match="'value' must be finite"):
        validate_is_finite("value", value)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(None, id="none"),
        pytest.param(0, id="zero"),
        pytest.param(2.5, id="float"),
    ],
)
def test_validate_is_finite_when_provided_accepts_none_and_finite_values(value):
    validate_is_finite_when_provided("value", value)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(float("inf"), id="positive_infinity"),
        pytest.param(float("-inf"), id="negative_infinity"),
        pytest.param(float("nan"), id="nan"),
    ],
)
def test_validate_is_finite_when_provided_rejects_non_finite_values(value):
    with pytest.raises(ValueError, match="'value' must be finite when provided"):
        validate_is_finite_when_provided("value", value)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(0, id="zero"),
        pytest.param(3, id="positive_int"),
        pytest.param(0.5, id="positive_float"),
        pytest.param(1e-10, id="small_positive_float"),
    ],
)
def test_validate_is_non_negative_accepts_zero_and_positive_values(value):
    validate_is_non_negative("value", value)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(-1, id="negative_int"),
        pytest.param(-0.1, id="negative_float"),
        pytest.param(-1e-10, id="small_negative_float"),
    ],
)
def test_validate_is_non_negative_rejects_negative_values(value):
    with pytest.raises(ValueError, match="'value' must be non-negative"):
        validate_is_non_negative("value", value)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(1, id="positive_int"),
        pytest.param(0.1, id="positive_float"),
        pytest.param(1e-10, id="small_positive_float"),
    ],
)
def test_validate_is_positive_accepts_positive_values(value):
    validate_is_positive("value", value)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(0, id="zero"),
        pytest.param(-1e-10, id="small_negative_float"),
        pytest.param(-1, id="negative_int"),
    ],
)
def test_validate_is_positive_rejects_zero_and_negative_values(value):
    with pytest.raises(ValueError, match="'value' must be positive"):
        validate_is_positive("value", value)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param([1], id="list"),
        pytest.param((1,), id="tuple"),
        pytest.param("x", id="string"),
    ],
)
def test_validate_is_non_empty_accepts_non_empty_sequences(value):
    validate_is_non_empty("value", value)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param([], id="list"),
        pytest.param((), id="tuple"),
        pytest.param("", id="string"),
    ],
)
def test_validate_is_non_empty_rejects_empty_sequences(value):
    with pytest.raises(ValueError, match=re.escape("'value' cannot be empty.")):
        validate_is_non_empty("value", value)


@pytest.mark.parametrize(
    "ratios",
    [
        pytest.param([1], id="single"),
        pytest.param([0.8, 0.1, 0.1], id="multiple"),
        pytest.param([0.8, 0.1, 0.1000001], id="within_tolerance"),
    ],
)
def test_validate_ratios_accepts_positive_finite_ratios_that_sum_to_one(ratios):
    validate_ratios(ratios)


@pytest.mark.parametrize(
    "ratios, expected_error_message",
    [
        pytest.param([], re.escape("'ratios' cannot be empty."), id="empty"),
        pytest.param(
            [0.5, 0.5, float("inf")], re.escape("'ratios' must be finite"), id="infinite_ratio"
        ),
        pytest.param(
            [0.5, 0.5, float("nan")], re.escape("'ratios' must be finite"), id="nan_ratio"
        ),
        pytest.param([0.5, 0.5, 0], re.escape("'ratios' must be positive"), id="zero_ratio"),
        pytest.param(
            [0.5, 0.6], re.escape("'ratios' must sum to 1.0, got 1.1."), id="sum_too_high"
        ),
        pytest.param([0.4, 0.5], re.escape("'ratios' must sum to 1.0, got 0.9."), id="sum_too_low"),
    ],
)
def test_validate_ratios_rejects_invalid_ratios(ratios, expected_error_message):
    with pytest.raises(ValueError, match=expected_error_message):
        validate_ratios(ratios)
