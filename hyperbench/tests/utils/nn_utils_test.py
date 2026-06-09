import pytest
import torch

from hyperbench.utils import (
    INPUT_LAYER,
    is_input_layer,
    is_layer,
    maxmin_scatter,
    validate_floating_tensor_dtype,
    validate_long_tensor_dtype,
)


@pytest.mark.parametrize(
    "layer_idx, desired_layer, expected",
    [
        pytest.param(0, 0, True, id="same_layer"),
        pytest.param(1, 0, False, id="different_layer"),
        pytest.param(3, 3, True, id="higher_same_layer"),
    ],
)
def test_is_layer(layer_idx, desired_layer, expected):
    assert is_layer(layer_idx, desired_layer) == expected


@pytest.mark.parametrize(
    "layer_idx, expected",
    [
        pytest.param(0, True, id="input_layer"),
        pytest.param(1, False, id="not_input_layer"),
        pytest.param(INPUT_LAYER, True, id="using_constant"),
    ],
)
def test_is_input_layer(layer_idx, expected):
    assert is_input_layer(layer_idx) == expected


def test_maxmin_scatter_computes_channelwise_range_by_group():
    src = torch.tensor(
        [
            [1.0, 4.0],
            [3.0, 1.0],
            [-2.0, 7.0],
            [5.0, -1.0],
            [5.0, 8.0],
        ],
        dtype=torch.float,
    )
    # index[k] says which output group receives src[k].
    # Example: index[1] == 0 means src[1] = [3, 1] contributes to output row 0.
    index = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)

    # dim=0 scatters rows into grouped output rows, preserving the feature/channel dimension.
    result = maxmin_scatter(src=src, index=index, dim=0)

    # Group 0 receives [1, 4] and [3, 1], so its per-channel range is
    # [max(1, 3) - min(1, 3), max(4, 1) - min(4, 1)] = [2, 3].
    # Group 1 receives [-2, 7], [5, -1], and [5, 8], so its range is
    # [max(-2, 5, 5) - min(-2, 5, 5), max(7, -1, 8) - min(7, -1, 8)] = [7, 9].
    expected = torch.tensor(
        [
            [2.0, 3.0],
            [7.0, 9.0],
        ],
        dtype=torch.float,
    )
    assert torch.allclose(result, expected)


def test_maxmin_scatter_respects_explicit_dim_size():
    src = torch.tensor(
        [
            [1.0, 4.0],
            [3.0, 1.0],
            [-2.0, 7.0],
        ],
        dtype=torch.float,
    )

    # index[k] says which output group receives src[k].
    # Example:
    # - index[1] == 0 means src[1] = [3, 1] contributes to output row 0.
    # - index[2] == 2 means src[2] = [-2, 7] contributes to output row 2.
    # Missing group ids indicate that those groups receive no source rows, so group 1 and group 3 are empty.
    index = torch.tensor([0, 0, 2], dtype=torch.long)

    # dim_size=4 forces four output rows even though max(index) would only imply three rows.
    result = maxmin_scatter(src=src, index=index, dim=0, dim_size=4)

    # Group 0 receives [1, 4] and [3, 1], so its range is [2, 3].
    # Group 2 receives only row [-2, 7], so max(-2) - min(-2) and max(7) - min(7) are both 0 and the range is [0, 0].
    # Empty groups 1 and 3 follow torch_geometric.scatter's neutral empty output,
    # so max and min both become [0, 0], and max - min is also [0, 0].
    expected = torch.tensor(
        [
            [2.0, 3.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=torch.float,
    )

    assert torch.allclose(result, expected)


def test_maxmin_scatter_supports_nonzero_scatter_dimension():
    src = torch.tensor(
        [
            [1.0, 5.0, 3.0],
            [4.0, 2.0, 8.0],
        ],
        dtype=torch.float,
    )

    # With dim=1, index[k] says which output column group receives source column k.
    # Example: index[2] == 0 means the third source column contributes to output column 0.
    # Source columns 0 and 2 are grouped together, while source column 1 is alone in group 1.
    index = torch.tensor([0, 1, 0], dtype=torch.long)

    # dim_size=2 keeps exactly two output column groups: group 0 and group 1.
    result = maxmin_scatter(src=src, index=index, dim=1, dim_size=2)

    # Row 0: group 0 receives 1 and 3, so max(1, 3) - min(1, 3) = 2.
    # Row 1: group 0 receives 4 and 8, so max(4, 8) - min(4, 8) = 4.
    # Row 0: group 1 receives only 5, so max(5) - min(5) = 0.
    # Row 1: group 1 receives only 2, so max(2) - min(2) = 0.
    expected = torch.tensor(
        [
            [2.0, 0.0],
            [4.0, 0.0],
        ],
        dtype=torch.float,
    )
    assert torch.allclose(result, expected)


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float16, id="float16"),
        pytest.param(torch.float32, id="float32"),
        pytest.param(torch.float64, id="float64"),
    ],
)
def test_validate_floating_tensor_dtype_accepts_floating_dtype(dtype):
    tensor = torch.tensor([1.0], dtype=dtype)

    validate_floating_tensor_dtype("t", tensor)


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.long, id="long"),
        pytest.param(torch.int, id="int"),
        pytest.param(torch.bool, id="bool"),
    ],
)
def test_validate_floating_tensor_dtype_rejects_non_floating_dtype(dtype):
    tensor = torch.tensor([1], dtype=dtype)

    with pytest.raises(
        ValueError,
        match=rf"'t' must have a floating-point dtype, got {dtype}\.",
    ):
        validate_floating_tensor_dtype("t", tensor)


def test_validate_long_tensor_dtype_accepts_long_dtype():
    tensor = torch.tensor([1], dtype=torch.long)

    validate_long_tensor_dtype("t", tensor)


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.int32, id="int32"),
        pytest.param(torch.float32, id="float32"),
        pytest.param(torch.int, id="int"),
    ],
)
def test_validate_long_tensor_dtype_rejects_non_long_dtype(dtype):
    tensor = torch.tensor([1], dtype=dtype)

    with pytest.raises(
        ValueError,
        match=rf"'t' must have dtype torch.long, got {dtype}\.",
    ):
        validate_long_tensor_dtype("t", tensor)
