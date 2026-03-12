import pytest

from hyperbench.utils import is_layer, is_input_layer, INPUT_LAYER


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
