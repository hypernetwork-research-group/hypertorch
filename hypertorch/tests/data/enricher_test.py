import pytest
import torch
import re

from pathlib import Path
from collections.abc import Callable
from unittest.mock import patch
from torch import Tensor
from hypertorch.data import (
    ABHyperedgeWeightsEnricher,
    FillValueHyperedgeAttrsEnricher,
    HyperedgeAttrsEnricher,
    HyperedgeEnricher,
    HyperedgeWeightsEnricher,
    LaplacianPositionalEncodingEnricher,
    Node2VecEnricher,
    NodeEnricher,
    VilLainEnricher,
    VilLainHyperedgeAttrsEnricher,
)
from hypertorch.data.enricher import Enricher, _VilLainTrainer
from hypertorch.tests.mock.mock import new_mock_pyg_node2vec, new_mock_villain


@pytest.fixture
def mock_two_hyperedge_index() -> Tensor:
    return torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]], dtype=torch.long)


@pytest.fixture
def mock_clique_hyperedge_index() -> Tensor:
    return torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long)


@pytest.mark.parametrize(
    "base_class",
    [
        pytest.param(Enricher, id="base_enricher"),
        pytest.param(HyperedgeEnricher, id="hyperedge_enricher"),
        pytest.param(HyperedgeAttrsEnricher, id="hyperedge_attrs_enricher"),
        pytest.param(HyperedgeWeightsEnricher, id="hyperedge_weights_enricher"),
        pytest.param(NodeEnricher, id="node_enricher"),
    ],
)
def test_abstract_enricher_base_classes_cannot_be_instantiated(base_class) -> None:
    with pytest.raises(TypeError):
        base_class()


@pytest.mark.parametrize(
    ("fill_value", "expected"),
    [
        pytest.param(1.0, [[1.0], [1.0]], id="default_like_value"),
        pytest.param(2.5, [[2.5], [2.5]], id="custom_value"),
    ],
)
def test_fill_value_hyperedge_attrs_enricher_returns_fixed_attrs(
    mock_two_hyperedge_index: Tensor,
    tmp_path: Path,
    fill_value: float,
    expected: list[list[float]],
) -> None:
    cache_dir = tmp_path / "hypertorch-cache"
    enricher = FillValueHyperedgeAttrsEnricher(
        cache_dir=str(cache_dir),
        fill_value=fill_value,
    )

    result = enricher.enrich(mock_two_hyperedge_index)

    assert enricher.cache_dir == str(cache_dir)
    assert result.shape == (2, 1)
    assert result.dtype == torch.float
    assert result.device == mock_two_hyperedge_index.device
    assert torch.equal(result, torch.tensor(expected, dtype=torch.float))


def test_fill_value_hyperedge_attrs_enricher_returns_empty_attrs_for_empty_input() -> None:
    hyperedge_index = torch.zeros((2, 0), dtype=torch.long)
    enricher = FillValueHyperedgeAttrsEnricher(fill_value=3.0)

    result = enricher.enrich(hyperedge_index)

    assert result.shape == (0, 1)
    assert result.dtype == torch.float
    assert result.device == hyperedge_index.device


@pytest.mark.parametrize(
    "alpha",
    [
        pytest.param(-0.1, id="below_zero"),
        pytest.param(1.1, id="above_one"),
    ],
)
def test_ab_hyperedge_weights_enricher_rejects_invalid_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match=re.escape("'alpha' must be between 0.0 and 1.0")):
        ABHyperedgeWeightsEnricher(alpha=alpha)


def test_ab_hyperedge_weights_enricher_rejects_non_finite_beta() -> None:
    with pytest.raises(ValueError, match=re.escape("'beta' must be finite when provided")):
        ABHyperedgeWeightsEnricher(beta=float("inf"))


def test_ab_hyperedge_weights_enricher_counts_nodes_per_hyperedge(
    mock_two_hyperedge_index: Tensor,
) -> None:
    # With beta=None, the weight should be the number of nodes per hyperedge
    enricher = ABHyperedgeWeightsEnricher(alpha=0.8)

    result = enricher.enrich(mock_two_hyperedge_index)

    assert torch.equal(result, torch.tensor([2.0, 2.0], dtype=torch.float))


@pytest.mark.parametrize(
    "beta, expected",
    [
        pytest.param(-0.1, [1.975, 1.975], id="negative"),
        pytest.param(0.0, [2.0, 2.0], id="zero"),
        pytest.param(1.0, [2.25, 2.25], id="one"),
    ],
)
def test_ab_hyperedge_weights_enricher_adds_beta_scaled_random_component(
    mock_two_hyperedge_index: Tensor,
    beta: float,
    expected: list[float],
) -> None:
    enricher = ABHyperedgeWeightsEnricher(alpha=0.8, beta=beta)

    with patch("hypertorch.data.enricher.random.uniform", return_value=0.25) as mock_uniform:
        result = enricher.enrich(mock_two_hyperedge_index)

    mock_uniform.assert_called_once_with(0, 0.8)
    assert torch.equal(result, torch.tensor(expected, dtype=torch.float32))


def test_node2vec_enricher_rejects_context_larger_than_walk_length() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape("Expected walk_length >= context_size, got walk_length=2, context_size=3."),
    ):
        Node2VecEnricher(num_features=4, walk_length=2, context_size=3)


@pytest.mark.parametrize(
    ("build_invalid_enricher", "expected_message"),
    [
        pytest.param(
            lambda: Node2VecEnricher(num_features=0),
            "'num_features' must be positive",
            id="features",
        ),
        pytest.param(
            lambda: Node2VecEnricher(num_features=3, walk_length=0),
            "'walk_length' must be positive",
            id="walk_length",
        ),
        pytest.param(
            lambda: Node2VecEnricher(num_features=3, context_size=0),
            "'context_size' must be positive",
            id="context_size",
        ),
        pytest.param(
            lambda: Node2VecEnricher(num_features=3, num_walks_per_node=0),
            "'num_walks_per_node' must be positive",
            id="walks_per_node",
        ),
        pytest.param(
            lambda: Node2VecEnricher(num_features=3, p=0.0),
            "'p' must be positive",
            id="p",
        ),
        pytest.param(
            lambda: Node2VecEnricher(num_features=3, q=0.0),
            "'q' must be positive",
            id="q",
        ),
        pytest.param(
            lambda: Node2VecEnricher(num_features=3, num_negative_samples=0),
            "'num_negative_samples' must be positive",
            id="negative_samples",
        ),
        pytest.param(
            lambda: Node2VecEnricher(num_features=3, num_nodes=-1),
            "'num_nodes' must be non-negative",
            id="num_nodes",
        ),
        pytest.param(
            lambda: Node2VecEnricher(num_features=3, num_epochs=0),
            "'num_epochs' must be positive",
            id="epochs",
        ),
        pytest.param(
            lambda: Node2VecEnricher(num_features=3, learning_rate=float("nan")),
            "'learning_rate' must be finite",
            id="learning_rate_finite",
        ),
        pytest.param(
            lambda: Node2VecEnricher(num_features=3, batch_size=0),
            "'batch_size' must be positive",
            id="batch_size",
        ),
    ],
)
def test_node2vec_enricher_rejects_invalid_params(
    build_invalid_enricher: Callable[[], object],
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=re.escape(expected_message)):
        build_invalid_enricher()


def test_node2vec_enricher_returns_empty_features_when_no_nodes() -> None:
    hyperedge_index = torch.zeros((2, 0), dtype=torch.long)
    enricher = Node2VecEnricher(num_features=3)

    with pytest.warns(UserWarning, match="Found no nodes. Returning empty node features."):
        result = enricher.enrich(hyperedge_index)

    assert result.shape == (0, 3)
    assert result.dtype == torch.float
    assert result.device == hyperedge_index.device


def test_node2vec_enricher_returns_zero_features_when_clique_has_no_non_selfloop_edges() -> None:
    hyperedge_index = torch.tensor([[0], [0]], dtype=torch.long)
    enricher = Node2VecEnricher(num_features=3)

    with pytest.warns(
        UserWarning,
        match=re.escape(
            "Clique expansion produced no non-self-loop edges. Returning zero node features."
        ),
    ):
        result = enricher.enrich(hyperedge_index)

    assert torch.equal(result, torch.zeros((1, 3), dtype=torch.float))
    assert result.dtype == torch.float


@pytest.mark.parametrize(
    ("verbose", "expected_output"),
    [
        pytest.param(False, "", id="quiet"),
        pytest.param(
            True,
            "Reducing hypergraph to graph via clique_expansion...\n"
            "Training Node2Vec model for 2 epochs...\n"
            "Epoch 1/2\r"
            "Epoch 2/2\r"
            "Training complete. Generating node embeddings...\n",
            id="verbose",
        ),
    ],
)
def test_node2vec_enricher_trains_model_and_enrichers_correctly(
    mock_two_hyperedge_index: Tensor,
    capsys: pytest.CaptureFixture[str],
    verbose: bool,
    expected_output: str,
) -> None:
    model = new_mock_pyg_node2vec()

    enricher = Node2VecEnricher(
        num_features=2,
        walk_length=4,
        context_size=2,
        num_walks_per_node=3,
        p=0.5,
        q=2.0,
        num_negative_samples=4,
        num_nodes=4,
        num_epochs=2,
        learning_rate=0.05,
        batch_size=16,
        sparse=False,
        verbose=verbose,
    )

    with patch("hypertorch.data.enricher.PyGNode2Vec", return_value=model) as mock_node2vec:
        result = enricher.enrich(mock_two_hyperedge_index)

    captured = capsys.readouterr()

    kwargs = mock_node2vec.call_args.kwargs
    assert kwargs["embedding_dim"] == 2
    assert kwargs["walk_length"] == 4
    assert kwargs["context_size"] == 2
    assert kwargs["walks_per_node"] == 3
    assert kwargs["p"] == 0.5
    assert kwargs["q"] == 2.0
    assert kwargs["num_negative_samples"] == 4
    assert kwargs["num_nodes"] == 4
    assert kwargs["sparse"] is False

    model.loader.assert_called_once_with(batch_size=16, shuffle=True)

    assert result.shape == (4, 2)
    assert result.requires_grad is False
    assert result.device == mock_two_hyperedge_index.device
    assert captured.out == expected_output


@pytest.mark.parametrize(
    ("num_features", "expected_shape"),
    [
        pytest.param(1, (3, 1), id="single_feature"),
        pytest.param(4, (3, 4), id="padded_features"),
    ],
)
def test_laplacian_positional_encoding_enricher_enriches_correctly(
    mock_clique_hyperedge_index: Tensor,
    num_features: int,
    expected_shape: tuple[int, int],
) -> None:
    result = LaplacianPositionalEncodingEnricher(num_features=num_features).enrich(
        mock_clique_hyperedge_index
    )

    assert result.shape == expected_shape
    assert result.dtype == torch.float
    assert result.device == mock_clique_hyperedge_index.device
    assert result.requires_grad is False


def test_laplacian_positional_encoding_enricher_zero_pads_missing_eigenvectors(
    mock_clique_hyperedge_index: Tensor,
) -> None:
    # A 3-node clique yields 3 Laplacian eigenvectors in total.
    # The enricher skips the first trivial eigenvector, so only 2 non-trivial remain.
    # Requesting 4 features should return shape (3, 4), where the last 2 features are zero padding.
    result = LaplacianPositionalEncodingEnricher(num_features=4).enrich(mock_clique_hyperedge_index)

    # Example: result: [[v1_0, v2_0, 0, 0],
    #                   [v1_1, v2_1, 0, 0],
    #                   [v1_2, v2_2, 0, 0]]
    assert torch.allclose(result[:, 2:], torch.zeros((3, 2), dtype=result.dtype))

    # The first two columns are the two usable non-trivial eigenvectors.
    # Example: features = [[a, b],
    #                      [c, d],
    #                      [e, f]],
    #           -> Then feature.T @ feature should result in: [[1, 0],
    #                                                          [0, 1]]
    #              which is exactly the orthonormality property expected
    #              from eigenvectors returned by torch.linalg.eigh
    assert torch.allclose(
        torch.matmul(result[:, :2].T, result[:, :2]),
        torch.eye(2, dtype=torch.float),
        atol=1e-6,
    )


def test_laplacian_positional_encoding_enricher_uses_explicit_num_nodes() -> None:
    hyperedge_index = torch.tensor([[0, 1], [0, 0]], dtype=torch.long)

    result = LaplacianPositionalEncodingEnricher(num_features=2, num_nodes=4).enrich(
        hyperedge_index
    )

    assert result.shape == (4, 2)


@pytest.mark.parametrize(
    ("build_invalid_enricher", "expected_message"),
    [
        pytest.param(
            lambda: LaplacianPositionalEncodingEnricher(num_features=0),
            "'num_features' must be positive",
            id="features",
        ),
        pytest.param(
            lambda: LaplacianPositionalEncodingEnricher(num_features=3, num_nodes=-1),
            "'num_nodes' must be non-negative",
            id="num_nodes",
        ),
    ],
)
def test_laplacian_positional_encoding_enricher_rejects_invalid_semantic_params(
    build_invalid_enricher: Callable[[], object],
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=re.escape(expected_message)):
        build_invalid_enricher()


def test_villain_trainer_resolves_explicit_and_inferred_counts(
    mock_two_hyperedge_index: Tensor,
) -> None:
    trainer = _VilLainTrainer(num_features=3, num_nodes=6, num_hyperedges=5)

    assert trainer._num_nodes(mock_two_hyperedge_index) == 6
    assert trainer._num_hyperedges(mock_two_hyperedge_index) == 5
    assert trainer._empty_features(mock_two_hyperedge_index).shape == (0, 3)
    assert trainer._empty_features(mock_two_hyperedge_index).dtype == torch.float


def test_villain_trainer_falls_back_to_inferred_counts(
    mock_two_hyperedge_index: Tensor,
) -> None:
    trainer = _VilLainTrainer(num_features=3)

    assert trainer._num_nodes(mock_two_hyperedge_index) == 4
    assert trainer._num_hyperedges(mock_two_hyperedge_index) == 2


@pytest.mark.parametrize(
    ("build_invalid_enricher", "expected_message"),
    [
        pytest.param(
            lambda: VilLainEnricher(num_features=0),
            "'num_features' must be positive",
            id="features",
        ),
        pytest.param(
            lambda: VilLainEnricher(num_features=3, num_nodes=-1),
            "'num_nodes' must be non-negative",
            id="num_nodes",
        ),
        pytest.param(
            lambda: VilLainEnricher(num_features=3, num_hyperedges=-1),
            "'num_hyperedges' must be non-negative",
            id="num_hyperedges",
        ),
        pytest.param(
            lambda: VilLainEnricher(num_features=3, labels_per_subspace=1),
            "'labels_per_subspace' must be at least 2",
            id="labels_per_subspace",
        ),
        pytest.param(
            lambda: VilLainEnricher(num_features=3, training_steps=0),
            "'training_steps' must be positive",
            id="training_steps",
        ),
        pytest.param(
            lambda: VilLainEnricher(num_features=3, generation_steps=0),
            "'generation_steps' must be positive",
            id="generation_steps",
        ),
        pytest.param(
            lambda: VilLainEnricher(num_features=3, tau=float("nan")),
            "'tau' must be finite",
            id="tau_finite",
        ),
        pytest.param(
            lambda: VilLainEnricher(num_features=3, tau=0.0),
            "'tau' must be positive",
            id="tau_positive",
        ),
        pytest.param(
            lambda: VilLainEnricher(num_features=3, eps=float("nan")),
            "'eps' must be finite",
            id="eps_finite",
        ),
        pytest.param(
            lambda: VilLainEnricher(num_features=3, eps=0.0),
            "'eps' must be positive",
            id="eps_positive",
        ),
        pytest.param(
            lambda: VilLainEnricher(num_features=3, num_epochs=0),
            "'num_epochs' must be positive",
            id="num_epochs",
        ),
        pytest.param(
            lambda: VilLainEnricher(num_features=3, learning_rate=float("nan")),
            "'learning_rate' must be finite",
            id="learning_rate_finite",
        ),
        pytest.param(
            lambda: VilLainEnricher(num_features=3, learning_rate=0.0),
            "'learning_rate' must be positive",
            id="learning_rate_positive",
        ),
        pytest.param(
            lambda: VilLainEnricher(num_features=3, weight_decay=float("nan")),
            "'weight_decay' must be finite",
            id="weight_decay_finite",
        ),
        pytest.param(
            lambda: VilLainEnricher(num_features=3, weight_decay=-0.1),
            "'weight_decay' must be non-negative",
            id="weight_decay_non_negative",
        ),
    ],
)
def test_villain_node_enricher_rejects_invalid_params(
    build_invalid_enricher: Callable[[], object],
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=re.escape(expected_message)):
        build_invalid_enricher()


def test_villain_node_enricher_returns_empty_features_when_no_nodes() -> None:
    hyperedge_index = torch.zeros((2, 0), dtype=torch.long)
    enricher = VilLainEnricher(num_features=3)

    with pytest.warns(UserWarning, match="Found no nodes. Returning empty node features."):
        result = enricher.enrich(hyperedge_index)

    assert result.shape == (0, 3)
    assert result.dtype == torch.float
    assert result.device == hyperedge_index.device


@pytest.mark.parametrize(
    ("build_invalid_enricher", "expected_message"),
    [
        pytest.param(
            lambda: VilLainHyperedgeAttrsEnricher(num_features=0),
            "'num_features' must be positive",
            id="features",
        ),
        pytest.param(
            lambda: VilLainHyperedgeAttrsEnricher(num_features=3, num_nodes=-1),
            "'num_nodes' must be non-negative",
            id="num_nodes",
        ),
        pytest.param(
            lambda: VilLainHyperedgeAttrsEnricher(num_features=3, num_hyperedges=-1),
            "'num_hyperedges' must be non-negative",
            id="num_hyperedges",
        ),
        pytest.param(
            lambda: VilLainHyperedgeAttrsEnricher(num_features=3, labels_per_subspace=1),
            "'labels_per_subspace' must be at least 2",
            id="labels_per_subspace",
        ),
        pytest.param(
            lambda: VilLainHyperedgeAttrsEnricher(num_features=3, training_steps=0),
            "'training_steps' must be positive",
            id="training_steps",
        ),
        pytest.param(
            lambda: VilLainHyperedgeAttrsEnricher(num_features=3, generation_steps=0),
            "'generation_steps' must be positive",
            id="generation_steps",
        ),
        pytest.param(
            lambda: VilLainHyperedgeAttrsEnricher(num_features=3, tau=float("nan")),
            "'tau' must be finite",
            id="tau_finite",
        ),
        pytest.param(
            lambda: VilLainHyperedgeAttrsEnricher(num_features=3, tau=0.0),
            "'tau' must be positive",
            id="tau_positive",
        ),
        pytest.param(
            lambda: VilLainHyperedgeAttrsEnricher(num_features=3, eps=float("nan")),
            "'eps' must be finite",
            id="eps_finite",
        ),
        pytest.param(
            lambda: VilLainHyperedgeAttrsEnricher(num_features=3, eps=0.0),
            "'eps' must be positive",
            id="eps_positive",
        ),
        pytest.param(
            lambda: VilLainHyperedgeAttrsEnricher(num_features=3, num_epochs=0),
            "'num_epochs' must be positive",
            id="num_epochs",
        ),
        pytest.param(
            lambda: VilLainHyperedgeAttrsEnricher(num_features=3, learning_rate=float("nan")),
            "'learning_rate' must be finite",
            id="learning_rate_finite",
        ),
        pytest.param(
            lambda: VilLainHyperedgeAttrsEnricher(num_features=3, learning_rate=0.0),
            "'learning_rate' must be positive",
            id="learning_rate_positive",
        ),
        pytest.param(
            lambda: VilLainHyperedgeAttrsEnricher(num_features=3, weight_decay=float("nan")),
            "'weight_decay' must be finite",
            id="weight_decay_finite",
        ),
        pytest.param(
            lambda: VilLainHyperedgeAttrsEnricher(num_features=3, weight_decay=-0.1),
            "'weight_decay' must be non-negative",
            id="weight_decay_non_negative",
        ),
    ],
)
def test_villain_hyperedge_attrs_enricher_rejects_invalid_params(
    build_invalid_enricher: Callable[[], object],
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=re.escape(expected_message)):
        build_invalid_enricher()


def test_villain_hyperedge_attrs_enricher_returns_empty_attrs_when_no_hyperedges() -> None:
    hyperedge_index = torch.zeros((2, 0), dtype=torch.long)
    enricher = VilLainHyperedgeAttrsEnricher(num_features=3)

    with pytest.warns(
        UserWarning,
        match="Found no hyperedges. Returning empty hyperedge attributes.",
    ):
        result = enricher.enrich(hyperedge_index)

    assert result.shape == (0, 3)
    assert result.device == hyperedge_index.device


@pytest.mark.parametrize(
    ("num_nodes", "num_hyperedges", "expected_num_hyperedges"),
    [
        pytest.param(0, 0, 2, id="infer_hyperedge_count"),
        pytest.param(0, 7, 7, id="explicit_hyperedge_count"),
        pytest.param(6, 0, 2, id="explicit_node_count_unused_in_node_embedding_call"),
        pytest.param(6, 7, 7, id="explicit_node_and_hyperedge_count"),
    ],
)
def test_villain_node_enricher_uses_trained_model_for_node_embeddings(
    mock_two_hyperedge_index: Tensor,
    num_nodes: int,
    num_hyperedges: int,
    expected_num_hyperedges: int,
) -> None:
    model = new_mock_villain()

    enricher = VilLainEnricher(
        num_features=2,
        num_nodes=num_nodes,
        num_hyperedges=num_hyperedges,
    )

    with patch.object(enricher, "_train", return_value=model) as mock_train:
        result = enricher.enrich(mock_two_hyperedge_index)

    mock_train.assert_called_once_with(mock_two_hyperedge_index)
    model.eval.assert_called_once_with()
    model.node_embeddings.assert_called_once_with(
        hyperedge_index=mock_two_hyperedge_index,
        num_hyperedges=expected_num_hyperedges,
    )
    assert result.shape == (4, 2)
    assert result.requires_grad is False
    assert result.device == mock_two_hyperedge_index.device


@pytest.mark.parametrize(
    ("num_nodes", "num_hyperedges", "expected_num_hyperedges"),
    [
        pytest.param(0, 0, 2, id="infer_hyperedge_count"),
        pytest.param(0, 5, 5, id="explicit_hyperedge_count"),
        pytest.param(6, 0, 2, id="explicit_node_count_unused_in_hyperedge_embedding_call"),
        pytest.param(6, 5, 5, id="explicit_node_and_hyperedge_count"),
    ],
)
def test_villain_hyperedge_attrs_enricher_uses_trained_model_for_hyperedge_embeddings(
    mock_two_hyperedge_index: Tensor,
    num_nodes: int,
    num_hyperedges: int,
    expected_num_hyperedges: int,
) -> None:
    model = new_mock_villain()

    enricher = VilLainHyperedgeAttrsEnricher(
        num_features=3,
        num_nodes=num_nodes,
        num_hyperedges=num_hyperedges,
    )

    with patch.object(enricher, "_train", return_value=model) as mock_train:
        result = enricher.enrich(mock_two_hyperedge_index)

    mock_train.assert_called_once_with(mock_two_hyperedge_index)
    model.eval.assert_called_once_with()
    model.hyperedge_embeddings.assert_called_once_with(
        hyperedge_index=mock_two_hyperedge_index,
        num_hyperedges=expected_num_hyperedges,
    )
    assert result.shape == (2, 3)
    assert result.requires_grad is False
    assert result.device == mock_two_hyperedge_index.device


@pytest.mark.parametrize(
    ("verbose", "expected_output"),
    [
        pytest.param(False, "", id="quiet"),
        pytest.param(
            True,
            "Training VilLain model for 2 epochs...\nEpoch 1/2\rEpoch 2/2\r",
            id="verbose",
        ),
    ],
)
def test_villain_trainer_uses_verbose_option_correctly(
    mock_two_hyperedge_index: Tensor,
    capsys: pytest.CaptureFixture[str],
    verbose: bool,
    expected_output: str,
) -> None:
    model = new_mock_villain()

    trainer = _VilLainTrainer(
        num_features=4,
        num_nodes=6,
        num_hyperedges=5,
        labels_per_subspace=3,
        training_steps=2,
        generation_steps=8,
        tau=0.7,
        eps=1e-5,
        num_epochs=2,
        learning_rate=0.02,
        weight_decay=0.01,
        verbose=verbose,
    )

    with patch("hypertorch.data.enricher.VilLain", return_value=model):
        _ = trainer._train(mock_two_hyperedge_index)

    captured = capsys.readouterr()

    assert captured.out == expected_output


def test_lpe_enricher_raises_value_error_for_non_finite_laplacian(
    mock_two_hyperedge_index: Tensor,
) -> None:
    enricher = LaplacianPositionalEncodingEnricher(num_features=2)

    non_finite_laplacian = torch.sparse_coo_tensor(
        indices=torch.tensor([[0], [0]], dtype=torch.long),
        values=torch.tensor([float("inf")], dtype=torch.float),
        size=(1, 1),
    ).coalesce()

    with (
        patch(
            "hypertorch.data.enricher.EdgeIndex.get_sparse_normalized_laplacian",
            return_value=non_finite_laplacian,
        ),
        pytest.raises(
            ValueError, match=re.escape("The normalized Laplacian contains non-finite values.")
        ),
    ):
        _ = enricher.enrich(mock_two_hyperedge_index)
