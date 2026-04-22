import torch

from abc import ABC
from torch import Tensor
from typing import Literal, Optional, TypeAlias
from hyperbench.types import EdgeIndex, HyperedgeIndex
import random

EnrichmentMode: TypeAlias = Literal["concatenate", "replace"]


class Enricher(ABC):
    """
    Args:
        cache_dir: Directory for saving/loading cached features. If ``None``, caching is disabled.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
    ):
        self.cache_dir = cache_dir

    def enrich(self, hyperedge_index: Tensor) -> Tensor:
        raise NotImplementedError("Subclasses must implement the enrich method.")


class NodeEnricher(Enricher, ABC):
    """
    Base class for node enrichers.
    """

    pass


class HyperedgeEnricher(Enricher, ABC):
    """
    Base class for hyperedge enrichers.
    """

    pass


class HyperedgeAttrsEnricher(HyperedgeEnricher):
    """
    Base class for enrichers that generate hyperedge attributes (features).
    Args:
    - cache_dir: Directory for saving/loading cached features. If ``None``, caching is disabled.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
    ):
        super().__init__(cache_dir=cache_dir)

    def enrich(self, hyperedge_index: Tensor) -> Tensor:

        # add a feature of 1.0 for each hyperedge, which can be used as a baseline or for methods that require hyperedge features.
        hyperedge_attrs = torch.ones(size=(hyperedge_index.size(1),), device=hyperedge_index.device)

        return hyperedge_attrs


class HyperedgeWeightsEnricher(HyperedgeEnricher):
    """
    Generates hyperedge weights based on the number of nodes in each hyperedge.
    Args:
    - cache_dir: Directory for saving/loading cached features. If ``None``, caching is disabled.
    - alpha: Scaling factor for the random component added to weights. Must be between 0.0 and 1.0.
    - beta: If provided, the random component is alpha * beta. If None, no random component is added.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        alpha: float = 1.0,
        beta: Optional[float] = None,
    ):
        super().__init__(cache_dir=cache_dir)
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError("Alpha must be between 0.0 and 1.0.")

        self.alpha = alpha
        self.beta = beta

    def enrich(self, hyperedge_index: Tensor) -> Tensor:
        """
        Compute edge weights as the number of nodes in each hyperedge.

        Args:
            hyperedge_index: Hyperedge index tensor of shape ``(2, num_hyperedges)``.
            alpha: Scaling factor for the random component added to weights.
            beta: If provided, the random component is alpha * beta.
        Returns:
            Tensor of shape ``(num_hyperedges,)`` containing the weight of each hyperedge.
        """
        # Count the number of nodes in each hyperedge by counting occurrences of each hyperedge index.
        # Example: if hyperedge_index[1] = [0, 0, 1, 1, 1], then we have 2 nodes in hyperedge 0 and 3 nodes in hyperedge 1.
        num_hyperedges = int(hyperedge_index[1].max().item()) + 1
        weights = torch.bincount(hyperedge_index[1], minlength=num_hyperedges).float()

        random_alpha = random.uniform(0, self.alpha)
        if self.beta is not None:
            weights += random_alpha * self.beta
        return weights


class LaplacianPositionalEncodingEnricher(NodeEnricher):
    def __init__(
        self,
        num_features: int,
        cache_dir: Optional[str] = None,
    ):
        super().__init__(cache_dir=cache_dir)
        self.num_features = num_features

    def enrich(self, hyperedge_index: Tensor) -> Tensor:
        """
        Compute Laplacian Positional Encoding: the k smallest non-trivial eigenvectors
        of the symmetric normalized Laplacian L = I - D^{-1/2} A D^{-1/2}.

        The first eigenvector (constant, eigenvalue ~0) is skipped.
        The next num_features eigenvectors are returned as positional features.

        Args:
            hyperedge_index: Hyperedge index tensor of shape ``(2, num_hyperedges)``.

        Returns:
            Tensor of shape ``(num_nodes, num_features)``.
        """
        edge_index = HyperedgeIndex(hyperedge_index).reduce_to_edge_index_on_clique_expansion()
        edge_index_wrapper = EdgeIndex(edge_index)
        laplacian_matrix = edge_index_wrapper.get_sparse_normalized_laplacian()
        laplacian_matrix_dense = (
            laplacian_matrix.to_dense()  # torch.linalg.eigh only works on dense tensors
        )

        # Compute eigenvalues and eigenvectors of the symmetric Laplacian.
        # torch.linalg.eigh returns them sorted in ascending order of eigenvalue.
        # The smallest eigenvalue is ~0 with a constant eigenvector (all entries equal),
        # which carries no positional information and will be skipped.
        # Example: eigenvalues ~ [0, 1, 2],
        #          eigenvectors ~ [[0.577, -0.707, 0.408],
        #                          [0.577,  0.000, -0.816],
        #                          [0.577,  0.707,  0.408]]
        # Column 0 (eigenvalue ~0) is the trivial constant vector, all entries ~0.577.
        # eigenvectors shape is ``(num_nodes, num_nodes)``, each column is an eigenvector.
        with torch.no_grad():
            _, eigenvectors = torch.linalg.eigh(laplacian_matrix_dense)

        # We skip the first (trivial) eigenvector, so at most num_nodes - 1 are usable.
        # Example: 3 nodes -> 2 available non-trivial eigenvectors
        num_nodes = int(eigenvectors.size(0))
        num_nontrivial_eigenvectors = num_nodes - 1

        # If we have enough eigenvectors, slice columns 1 through num_features (inclusive).
        # Each row will be the positional encoding for that node.
        # Example: num_features = 2, eigenvectors.shape = (3, 3)
        #          -> return columns 1 and 2
        #             shape (3, 2)  # (num_nodes, num_features)
        if num_nontrivial_eigenvectors >= self.num_features:
            return eigenvectors[:, 1 : self.num_features + 1]

        # If the graph has fewer usable eigenvectors than requested
        # (e.g., num_features = 5 but only 2 available), we create a zero-padded tensor and fill what we have.
        # Example: num_nontrivial_eigenvectors = 2, num_features = 5
        #          -> shape (3, 5)  # columns 0-1 filled, 2-4 are zeros.
        x = torch.zeros(size=(num_nodes, self.num_features), device=edge_index.device)
        x[:, :num_nontrivial_eigenvectors] = eigenvectors[:, 1:]
        return x
