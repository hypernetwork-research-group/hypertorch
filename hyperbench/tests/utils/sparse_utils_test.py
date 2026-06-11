import pytest
import re as regex
import torch

from hyperbench.utils import sparse_dropout


@pytest.fixture
def mock_indices():
    return torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)


@pytest.fixture
def mock_values():
    return torch.tensor([1.0, 2.0, 3.0], dtype=torch.float)


def test_dropout_zero_probability(mock_indices, mock_values):
    sparse_tensor = torch.sparse_coo_tensor(
        mock_indices, mock_values, (3, 3), dtype=mock_values.dtype
    )

    result = sparse_dropout(sparse_tensor, dropout_prob=0.0)

    assert torch.allclose(result.coalesce().values(), mock_values)
    assert torch.equal(result.coalesce().indices(), sparse_tensor.coalesce().indices())


def test_dropout_full_probability(mock_indices, mock_values):
    sparse_tensor = torch.sparse_coo_tensor(
        mock_indices, mock_values, (3, 3), dtype=mock_values.dtype
    )

    result = sparse_dropout(sparse_tensor, dropout_prob=1.0)

    # All values should be zero when fill_value is 0
    assert torch.allclose(
        result.coalesce().values(), torch.zeros_like(mock_values, dtype=mock_values.dtype)
    )


@pytest.mark.parametrize("invalid_prob", [-0.5, 1.5])
def test_dropout_invalid_probability_out_of_range(mock_indices, mock_values, invalid_prob):
    sparse_tensor = torch.sparse_coo_tensor(
        mock_indices, mock_values, (2, 2), dtype=mock_values.dtype
    )

    with pytest.raises(
        ValueError,
        match=regex.escape("Dropout probability must be in the range [0, 1]"),
    ):
        sparse_dropout(sparse_tensor, dropout_prob=invalid_prob)


def test_dropout_preserves_indices():
    indices = torch.tensor([[0, 1, 2, 0], [0, 1, 2, 2]], dtype=torch.long)
    values = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float)
    sparse_tensor = torch.sparse_coo_tensor(indices, values, (3, 3), dtype=values.dtype)

    result = sparse_dropout(sparse_tensor, dropout_prob=0.5)

    # Indices should remain the same (in coalesced form)
    assert torch.equal(result.coalesce().indices(), sparse_tensor.coalesce().indices())


def test_dropout_preserves_shape():
    shape = (5, 10)  # Shape of the tensor if it were dense
    indices = torch.tensor([[0, 2, 4], [1, 5, 9]], dtype=torch.long)
    values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float)
    sparse_tensor = torch.sparse_coo_tensor(indices, values, shape, dtype=values.dtype)

    result = sparse_dropout(sparse_tensor, dropout_prob=0.5)

    assert result.size() == shape


def test_dropout_preserves_dtype():
    indices = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    values = torch.tensor([1.0, 2.0], dtype=torch.float32)
    sparse_tensor = torch.sparse_coo_tensor(indices, values, (2, 2), dtype=torch.float32)

    result = sparse_dropout(sparse_tensor, dropout_prob=0.5)

    assert result.dtype == sparse_tensor.dtype


def test_dropout_with_fill_value_zero(mock_indices):
    values = torch.tensor([5.0, 10.0, 15.0], dtype=torch.float)
    sparse_tensor = torch.sparse_coo_tensor(mock_indices, values, (3, 3), dtype=values.dtype)

    result = sparse_dropout(sparse_tensor, dropout_prob=0.5, fill_value=0.0)

    coalesced = result.coalesce()

    # Values should be either original or zero
    for val in coalesced.values():
        assert val in [0.0, 5.0, 10.0, 15.0]


def test_dropout_with_nonzero_fill_value(mock_indices):
    values = torch.tensor([5.0, 10.0, 15.0], dtype=torch.float)
    sparse_tensor = torch.sparse_coo_tensor(mock_indices, values, (3, 3), dtype=values.dtype)
    fill_value = 99.0

    result = sparse_dropout(sparse_tensor, dropout_prob=0.5, fill_value=fill_value)

    coalesced = result.coalesce()

    # Values should be either original or fill_value
    for val in coalesced.values():
        assert val in [5.0, 10.0, 15.0, fill_value]


def test_dropout_with_negative_values():
    indices = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
    values = torch.tensor([-1.0, -5.0, -10.0], dtype=torch.float)
    sparse_tensor = torch.sparse_coo_tensor(indices, values, (3, 3), dtype=values.dtype)

    result = sparse_dropout(sparse_tensor, dropout_prob=0.5)

    # Should handle negative values correctly
    assert result.size() == sparse_tensor.size()
    assert result.coalesce().values().dtype == values.dtype


def test_dropout_preserves_cpu_device():
    device = torch.device("cpu")

    indices = torch.tensor([[0, 1], [0, 1]], device=device, dtype=torch.long)
    values = torch.tensor([1.0, 2.0], device=device, dtype=torch.float)
    sparse_tensor = torch.sparse_coo_tensor(
        indices, values, (2, 2), device=device, dtype=values.dtype
    )

    result = sparse_dropout(sparse_tensor, dropout_prob=0.5)

    assert result.device == sparse_tensor.device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Cuda not available")
def test_dropout_preserves_cuda_device():
    device = torch.device("cuda")

    indices = torch.tensor([[0, 1], [0, 1]], device=device, dtype=torch.long)
    values = torch.tensor([1.0, 2.0], device=device, dtype=torch.float)
    sparse_tensor = torch.sparse_coo_tensor(
        indices, values, (2, 2), device=device, dtype=values.dtype
    )

    result = sparse_dropout(sparse_tensor, dropout_prob=0.5)

    assert result.device == sparse_tensor.device


@pytest.mark.skipif(not torch.mps.is_available(), reason="MPS not available")
def test_dropout_preserves_mps_device():
    device = torch.device("mps")

    indices = torch.tensor([[0, 1], [0, 1]], device=device, dtype=torch.long)
    values = torch.tensor([1.0, 2.0], device=device, dtype=torch.float)
    sparse_tensor = torch.sparse_coo_tensor(
        indices, values, (2, 2), device=device, dtype=values.dtype
    )

    result = sparse_dropout(sparse_tensor, dropout_prob=0.5)

    assert result.device == sparse_tensor.device


def test_dropout_fill_value_with_full_dropout():
    indices = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    values = torch.tensor([1.0, 2.0], dtype=torch.float)
    sparse_tensor = torch.sparse_coo_tensor(indices, values, (2, 2), dtype=values.dtype)
    fill_value = 7.0

    result = sparse_dropout(sparse_tensor, dropout_prob=1.0, fill_value=fill_value)

    # All values should be the fill_value
    assert torch.allclose(
        result.coalesce().values(), torch.full_like(values, fill_value, dtype=values.dtype)
    )


def test_dropout_with_unsorted_indices():
    # Create a sparse tensor with unsorted/duplicate indices
    indices = torch.tensor([[2, 0, 1, 0], [2, 0, 1, 0]], dtype=torch.long)
    values = torch.tensor([3.0, 1.0, 2.0, 4.0], dtype=torch.float)
    sparse_tensor = torch.sparse_coo_tensor(indices, values, (3, 3), dtype=values.dtype)

    result = sparse_dropout(sparse_tensor, dropout_prob=0.5)

    coalesced = result.coalesce()

    # Should coalesce without errors
    assert coalesced is not None

    # Original had 3 unique indices (0,0), (1,1), (2,2)
    assert len(coalesced.indices()[0]) == 3
    assert len(coalesced.indices()[1]) == 3


def test_dropout_single_element():
    indices = torch.tensor([[0], [0]], dtype=torch.long)
    values = torch.tensor([42.0], dtype=torch.float)
    sparse_tensor = torch.sparse_coo_tensor(indices, values, (1, 1), dtype=values.dtype)

    result = sparse_dropout(sparse_tensor, dropout_prob=0.5)

    coalesced = result.coalesce()

    # Result should contain one value, either 0 or the original value
    assert len(coalesced.values()) == 1
    assert coalesced.values().item() in [0.0, 42.0]


def test_dropout_large_sparse_matrix():
    size = 1000
    num_nonzero_elements = 500
    rows = torch.randint(0, size, (num_nonzero_elements,), dtype=torch.long)
    cols = torch.randint(0, size, (num_nonzero_elements,), dtype=torch.long)

    indices = torch.stack([rows, cols])
    values = torch.randn(num_nonzero_elements, dtype=torch.float)
    sparse_tensor = torch.sparse_coo_tensor(indices, values, (size, size), dtype=values.dtype)

    result = sparse_dropout(sparse_tensor, dropout_prob=0.2)

    assert result.size() == sparse_tensor.size()


def test_dropout_returns_new_tensor(mock_indices, mock_values):
    sparse_tensor = torch.sparse_coo_tensor(
        mock_indices, mock_values, (2, 2), dtype=mock_values.dtype
    )

    result = sparse_dropout(sparse_tensor, dropout_prob=0.0)

    # Even with 0 dropout, the returned tensor should be a different object
    assert result is not sparse_tensor


def test_dropout_statistical_property_moderate_rate():
    # Create a larger sparse tensor for statistical testing
    num_elements = 1000

    indices = torch.tensor([list(range(num_elements)), list(range(num_elements))], dtype=torch.long)
    values = torch.ones(num_elements, dtype=torch.float)
    sparse_tensor = torch.sparse_coo_tensor(
        indices, values, (num_elements, num_elements), dtype=values.dtype
    )

    dropout_prob = 0.3
    keep_prob = 1 - dropout_prob  # Keep ~70% of elements

    result = sparse_dropout(sparse_tensor, dropout_prob=dropout_prob, fill_value=0.0)
    result_values = result.coalesce().values()

    # Count non-zero values (kept elements)
    kept_count = (result_values != 0).sum(dtype=torch.long).item()
    actual_keep_prob = kept_count / num_elements

    # Allow 10% tolerance for statistical variance
    tolerance = 0.1
    assert abs(actual_keep_prob - keep_prob) < tolerance
