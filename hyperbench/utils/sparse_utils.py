import torch

from torch import Tensor


def sparse_dropout(
    sparse_tensor: Tensor,
    dropout_prob: float,
    fill_value: float = 0.0,
) -> Tensor:
    """Dropout function for sparse matrix.

    Returns a new sparse matrix with the same shape as the input sparse matrix,
    but with some elements dropped out.

    Args:
        sparse_tensor: The sparse matrix with format ``torch.sparse_coo_tensor``.
        dropout_prob: Probability of an element to be dropped.
        fill_value: The fill value for dropped elements. Defaults to ``0.0``.

    Returns:
        A new sparse matrix with the same shape as the input sparse matrix, but with some elements dropped out.
    """
    device = sparse_tensor.device

    # Sparse tensors may be unsorted indices or have duplicate entries
    # 'coalesce()' will sum duplicates and sort indices to have a consistent format for dropout
    sparse_tensor = sparse_tensor.coalesce()

    if dropout_prob > 1 or dropout_prob < 0:
        raise ValueError("Dropout probability must be in the range [0, 1]")

    # Nothing to drop, return the original sparse tensor
    if dropout_prob == 0:
        return sparse_tensor

    values = sparse_tensor.values()
    indices = sparse_tensor.indices()

    keep_prob = 1 - dropout_prob

    # Generate a binary mask matching the shape of values for elements to keep
    # 'torch.bernoulli()' samples 1 with probability keep_prob and 0 with probability dropout_prob
    # Example: values = [0.5, 1.2, 3.4], keep_prob = 0.8
    #          -> keep_mask might be [1, 0, 1], meaning we keep the 1st and 3rd elements, drop the 2nd
    keep_mask = torch.bernoulli(torch.full_like(values, keep_prob)).to(device)

    if fill_value == 0.0:
        # If fill_value is 0, just zero out the dropped elements,
        # as keep_mask will be 0 for dropped elements and 1 for kept elements
        # Example: values = [0.5, 1.2, 3.4], keep_mask = [1, 0, 1], fill_value = 0.0
        #          -> new_values = [0.5*1, 1.2*0, 3.4*1] = [0.5, 0.0, 3.4]
        new_values = values * keep_mask
    else:
        # If fill_value is non-zero, we must fill the dropped elements with the specified fill_value instead of zero
        # 'torch.logical_not(keep_mask)' identifies dropped elements where mask is 0 and
        # Example: values = [0.5, 1.2, 3.4], keep_mask = [1, 0, 1], fill_value = 9.9
        #          -> values_to_fill_mask = [0, 1, 0]
        #          -> fill_values = [0*9.9, 1*9.9, 0*9.9] = [0.0, 9.9, 0.0]
        #          -> new_values = [0.5*1 + 0.0, 1.2*0 + 9.9, 3.4*1 + 0.0] = [0.5, 9.9, 3.4]
        values_to_fill_mask = torch.logical_not(keep_mask)
        fill_values = values_to_fill_mask * fill_value
        new_values = values * keep_mask + fill_values

    # Reuse the original indices and shape to preserve spasity but change values
    dropout_sparse_tensor = torch.sparse_coo_tensor(
        indices=indices,
        values=new_values,
        size=sparse_tensor.size(),
        dtype=sparse_tensor.dtype,
        device=device,
    )

    return dropout_sparse_tensor
