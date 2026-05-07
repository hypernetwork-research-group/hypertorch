import pytest
import torch

from unittest.mock import patch, MagicMock
from hyperbench.data import AlgebraDataset, Dataset, HIFLoader, SamplingStrategy
from hyperbench.nn import NodeEnricher, HyperedgeEnricher
from hyperbench.types import HData


@pytest.fixture
def mock_hdata() -> HData:
    x = torch.ones((3, 1), dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1, 2], [0, 0, 1]], dtype=torch.long)
    return HData(x=x, hyperedge_index=hyperedge_index)


@pytest.fixture
def mock_hdata_with_hyperedge_attr() -> HData:
    x = torch.ones((3, 1), dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1, 2], [0, 0, 1]], dtype=torch.long)
    hyperedge_attr = torch.ones((3, 1), dtype=torch.float)
    return HData(x=x, hyperedge_index=hyperedge_index, hyperedge_attr=hyperedge_attr)


@pytest.fixture
def mock_hdata_with_hyperedge_weights() -> HData:
    x = torch.ones((3, 1), dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1, 2], [0, 0, 1]], dtype=torch.long)
    hyperedge_weights = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float)
    return HData(x=x, hyperedge_index=hyperedge_index, hyperedge_weights=hyperedge_weights)


@pytest.fixture
def mock_hdata_sample_hypergraph() -> HData:
    x = torch.ones((2, 1), dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    return HData(x=x, hyperedge_index=hyperedge_index)


@pytest.fixture
def mock_hdata_simple_hypergraph() -> HData:
    x = torch.ones((2, 1), dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    return HData(x=x, hyperedge_index=hyperedge_index)


@pytest.fixture
def mock_hdata_three_node_weighted_hypergraph() -> HData:
    x = torch.ones((3, 1), dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1, 2], [0, 0, 1]], dtype=torch.long)
    hyperedge_weights = torch.tensor([[1.0], [2.0]], dtype=torch.float)
    return HData(x=x, hyperedge_index=hyperedge_index, hyperedge_weights=hyperedge_weights)


@pytest.fixture
def mock_hdata_four_node_hypergraph() -> HData:
    x = torch.ones((4, 1), dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]], dtype=torch.long)
    return HData(x=x, hyperedge_index=hyperedge_index)


@pytest.fixture
def mock_hdata_no_edge_attr_hypergraph() -> HData:
    x = torch.ones((2, 1), dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1], [0, 0]], dtype=torch.long)
    return HData(x=x, hyperedge_index=hyperedge_index)


@pytest.fixture
def mock_hdata_multiple_edges_attr_hypergraph() -> HData:
    x = torch.ones((4, 1), dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 2]], dtype=torch.long)
    hyperedge_weights = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float)
    hyperedge_attr = torch.ones((4, 1), dtype=torch.float)
    return HData(
        x=x,
        hyperedge_index=hyperedge_index,
        hyperedge_weights=hyperedge_weights,
        hyperedge_attr=hyperedge_attr,
    )


@pytest.fixture
def mock_hdata_no_incidences() -> HData:
    x = torch.ones((2, 1), dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    return HData(x=x, hyperedge_index=hyperedge_index)


@pytest.fixture
def mock_hdata_with_two_edge_attributes() -> HData:
    x = torch.ones((3, 1), dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1, 2], [0, 0, 1]], dtype=torch.long)
    hyperedge_weights = torch.tensor([[1.0, 2.0], [3.0, 0.1]], dtype=torch.float)
    return HData(x=x, hyperedge_index=hyperedge_index, hyperedge_weights=hyperedge_weights)


@pytest.fixture
def mock_hdata_random_ids() -> HData:
    x = torch.ones((3, 1), dtype=torch.float)
    hyperedge_index = torch.tensor([[0, 1, 2], [0, 0, 1]], dtype=torch.long)
    return HData(x=x, hyperedge_index=hyperedge_index)


def test_preloaded_dataset_loads_hdata_when_hdata_is_none():
    mock_hdata = MagicMock(spec=HData)
    with patch.object(HIFLoader, "load_by_name", return_value=mock_hdata) as mock_load:
        dataset = AlgebraDataset(hdata=None)

    assert dataset.hdata == mock_hdata
    mock_load.assert_called_once_with(
        "algebra", hf_sha="2bb641461e00c103fb5ef4fe6a30aad42500fc21", save_on_disk=True
    )


@pytest.mark.parametrize(
    "strategy, expected_len",
    [
        pytest.param(SamplingStrategy.NODE, 4, id="node_strategy"),
        pytest.param(SamplingStrategy.HYPEREDGE, 2, id="hyperedge_strategy"),
    ],
)
def test_dataset_is_available_with_all_strategies(
    strategy, expected_len, mock_hdata_four_node_hypergraph
):

    with patch.object(HIFLoader, "load_by_name", return_value=mock_hdata_four_node_hypergraph):
        dataset = AlgebraDataset(sampling_strategy=strategy)

        assert dataset.DATASET_NAME == "algebra"
        assert len(dataset) == expected_len


def test_dataset_process_no_incidences(mock_hdata_no_incidences):
    with patch.object(HIFLoader, "load_by_name", return_value=mock_hdata_no_incidences):
        dataset = AlgebraDataset()

        assert dataset.hdata is not None
        assert dataset.hdata.x.shape[0] == 2
        assert dataset.hdata.hyperedge_index.shape[0] == 2
        assert dataset.hdata.hyperedge_index.shape[1] == 2
        assert dataset.hdata.hyperedge_attr is None


def test_dataset_process_with_edge_attributes(mock_hdata_with_two_edge_attributes):
    with patch.object(HIFLoader, "load_by_name", return_value=mock_hdata_with_two_edge_attributes):
        dataset = AlgebraDataset()

    assert dataset.hdata is not None
    assert dataset.hdata.x.shape[0] == 3
    assert dataset.hdata.hyperedge_index.shape[0] == 2
    assert dataset.hdata.hyperedge_index.shape[1] == 3
    assert dataset.hdata.hyperedge_attr is None
    assert dataset.hdata.hyperedge_weights is not None
    assert dataset.hdata.hyperedge_weights.shape == (2, 2)
    assert torch.allclose(dataset.hdata.hyperedge_weights[0], torch.tensor([1.0, 2.0]))
    assert torch.allclose(dataset.hdata.hyperedge_weights[1], torch.tensor([3.0, 0.1]))


def test_dataset_process_without_edge_attributes(mock_hdata_no_edge_attr_hypergraph):
    with patch.object(HIFLoader, "load_by_name", return_value=mock_hdata_no_edge_attr_hypergraph):
        dataset = AlgebraDataset()

    assert dataset.hdata is not None
    assert dataset.hdata.hyperedge_index.shape[0] == 2
    assert dataset.hdata.hyperedge_index.shape[1] == 2
    assert dataset.hdata.hyperedge_attr is None


def test_dataset_process_hyperedge_index_in_correct_format(mock_hdata_four_node_hypergraph):
    with patch.object(HIFLoader, "load_by_name", return_value=mock_hdata_four_node_hypergraph):
        dataset = AlgebraDataset()

    assert dataset.hdata.hyperedge_index.shape == (2, 4)
    assert torch.allclose(dataset.hdata.hyperedge_index[0], torch.tensor([0, 1, 2, 3]))
    assert torch.allclose(dataset.hdata.hyperedge_index[1], torch.tensor([0, 0, 1, 1]))


def test_dataset_process_random_ids(mock_hdata_random_ids):
    with patch.object(HIFLoader, "load_by_name", return_value=mock_hdata_random_ids):
        dataset = AlgebraDataset()

    assert dataset.hdata.hyperedge_index.shape == (2, 3)
    assert torch.allclose(dataset.hdata.hyperedge_index[0], torch.tensor([0, 1, 2]))
    assert torch.allclose(dataset.hdata.hyperedge_index[1], torch.tensor([0, 0, 1]))
    assert dataset.hdata.hyperedge_attr is None


@pytest.mark.parametrize(
    "strategy",
    [
        pytest.param(SamplingStrategy.NODE, id="node_strategy"),
        pytest.param(SamplingStrategy.HYPEREDGE, id="hyperedge_strategy"),
    ],
)
def test_getitem_index_list_empty(mock_hdata_simple_hypergraph, strategy):
    with patch.object(HIFLoader, "load_by_name", return_value=mock_hdata_simple_hypergraph):
        dataset = AlgebraDataset(sampling_strategy=strategy)

    with pytest.raises(ValueError, match="Index list cannot be empty."):
        dataset[[]]


@pytest.mark.parametrize(
    "strategy, index_list, expected_message",
    [
        pytest.param(
            SamplingStrategy.NODE,
            [0, 1, 2, 3, 4],
            r"Index list length \(5\) cannot exceed the number of sampleable items \(4\)\.",
            id="node_strategy",
        ),
        pytest.param(
            SamplingStrategy.HYPEREDGE,
            [0, 1, 2],
            r"Index list length \(3\) cannot exceed the number of sampleable items \(2\)\.",
            id="hyperedge_strategy",
        ),
    ],
)
def test_getitem_raises_when_index_list_larger_than_max(
    mock_hdata_four_node_hypergraph, strategy, index_list, expected_message
):
    with patch.object(HIFLoader, "load_by_name", return_value=mock_hdata_four_node_hypergraph):
        dataset = AlgebraDataset(sampling_strategy=strategy)

    with pytest.raises(ValueError, match=expected_message):
        dataset[index_list]


@pytest.mark.parametrize(
    "strategy, index, expected_message",
    [
        pytest.param(
            SamplingStrategy.NODE, 4, r"Node ID 4 is out of bounds \(0, 3\)\.", id="node_strategy"
        ),
        pytest.param(
            SamplingStrategy.HYPEREDGE,
            2,
            r"Hyperedge ID 2 is out of bounds \(0, 1\)\.",
            id="hyperedge_strategy",
        ),
    ],
)
def test_getitem_raises_when_index_out_of_bounds(
    mock_hdata_four_node_hypergraph, strategy, index, expected_message
):
    with patch.object(HIFLoader, "load_by_name", return_value=mock_hdata_four_node_hypergraph):
        dataset = AlgebraDataset(sampling_strategy=strategy)

    with pytest.raises(IndexError, match=expected_message):
        dataset[index]


@pytest.mark.parametrize(
    "strategy, index, expected_shape, expected_num_hyperedges",
    [
        # When node 1 is selected, we get hyperedge 0 with nodes 0 and 1 -> 2 incidences, 1 hyperedge
        pytest.param(SamplingStrategy.NODE, 1, (2, 1), 1, id="node_strategy"),
        # When hyperedge 0 is selected, we get nodes 0 and 1 -> 2 incidences, 1 hyperedge
        pytest.param(SamplingStrategy.HYPEREDGE, 0, (2, 1), 1, id="hyperedge_strategy"),
    ],
)
def test_getitem_single_index(
    mock_hdata_sample_hypergraph, strategy, index, expected_shape, expected_num_hyperedges
):
    with patch.object(HIFLoader, "load_by_name", return_value=mock_hdata_sample_hypergraph):
        dataset = AlgebraDataset(sampling_strategy=strategy)

    data = dataset[index]

    assert data.hyperedge_index.shape == expected_shape
    assert data.num_hyperedges == expected_num_hyperedges


@pytest.mark.parametrize(
    "strategy, index, expected_shape, expected_num_hyperedges",
    [
        # When nodes (0, 2, 3) -> hyperedge 0 (nodes 0, 1) + hyperedge 1 (nodes 2, 3) -> 4 incidences, 2 hyperedges
        pytest.param(SamplingStrategy.NODE, [0, 2, 3], (2, 4), 2, id="node_strategy"),
        # When hyperedge 0 (nodes 0, 1) + hyperedge 1 (nodes 2, 3) -> 4 incidences, 2 hyperedges
        pytest.param(SamplingStrategy.HYPEREDGE, [0, 1], (2, 4), 2, id="hyperedge_strategy"),
    ],
)
def test_getitem_when_list_index_provided(
    mock_hdata_four_node_hypergraph, strategy, index, expected_shape, expected_num_hyperedges
):
    with patch.object(HIFLoader, "load_by_name", return_value=mock_hdata_four_node_hypergraph):
        dataset = AlgebraDataset(sampling_strategy=strategy)

    data = dataset[index]

    assert data.hyperedge_index.shape == expected_shape
    assert data.num_hyperedges == expected_num_hyperedges


@pytest.mark.parametrize(
    "strategy",
    [
        pytest.param(SamplingStrategy.NODE, id="node_strategy"),
        pytest.param(SamplingStrategy.HYPEREDGE, id="hyperedge_strategy"),
    ],
)
def test_getitem_with_edge_attr(mock_hdata_three_node_weighted_hypergraph, strategy):
    with patch.object(
        HIFLoader, "load_by_name", return_value=mock_hdata_three_node_weighted_hypergraph
    ):
        dataset = AlgebraDataset(sampling_strategy=strategy)

    data = dataset[0]

    assert data.hyperedge_index.shape == (2, 2)
    assert data.num_hyperedges == 1
    assert data.hyperedge_attr is None


@pytest.mark.parametrize(
    "strategy",
    [
        pytest.param(SamplingStrategy.NODE, id="node_strategy"),
        pytest.param(SamplingStrategy.HYPEREDGE, id="hyperedge_strategy"),
    ],
)
def test_getitem_without_edge_attr(mock_hdata_no_edge_attr_hypergraph, strategy):
    with patch.object(HIFLoader, "load_by_name", return_value=mock_hdata_no_edge_attr_hypergraph):
        dataset = AlgebraDataset(sampling_strategy=strategy)

    data = dataset[0]
    assert data.hyperedge_attr is None


@pytest.mark.parametrize(
    "strategy, index",
    [
        # When nodes 0,2 -> hyperedge 0 (nodes 0, 1) + hyperedge 1 (node 2) -> 2 hyperedges
        pytest.param(SamplingStrategy.NODE, [0, 2], id="node_strategy"),
        # When hyperedge 0 (nodes 0, 1) + hyperedge 1 (node 2) -> 2 hyperedges
        pytest.param(SamplingStrategy.HYPEREDGE, [0, 1], id="hyperedge_strategy"),
    ],
)
def test_getitem_with_multiple_edges_attr(
    mock_hdata_multiple_edges_attr_hypergraph, strategy, index
):
    with patch.object(
        HIFLoader, "load_by_name", return_value=mock_hdata_multiple_edges_attr_hypergraph
    ):
        dataset = AlgebraDataset(sampling_strategy=strategy)

    data = dataset[index]
    assert data.num_hyperedges == 2

    # Even though the original hypergraph has edge attributes, __getitem__ should return hyperedge_attr as None
    # as the hyperedge attributes are handled by the loader's collate function during batching
    assert data.hyperedge_attr is None


@pytest.mark.parametrize(
    "strategy, expected_len",
    [
        # mock_hdata: 3 nodes, 2 hyperedges
        pytest.param(SamplingStrategy.NODE, 3, id="node_strategy"),
        pytest.param(SamplingStrategy.HYPEREDGE, 2, id="hyperedge_strategy"),
    ],
)
def test_from_hdata(strategy, expected_len, mock_hdata):
    dataset = Dataset.from_hdata(mock_hdata, sampling_strategy=strategy)

    assert dataset.hdata is mock_hdata
    assert len(dataset) == expected_len


@pytest.mark.parametrize(
    "strategy",
    [
        pytest.param(SamplingStrategy.HYPEREDGE, id="hyperedge_strategy"),
        pytest.param(SamplingStrategy.NODE, id="node_strategy"),
    ],
)
def test_from_url(strategy, mock_hdata):
    url = "https://example.com/sample.json.zst"

    with patch.object(HIFLoader, "load_from_url", return_value=mock_hdata) as mock_load_from_url:
        dataset = Dataset.from_url(url=url, sampling_strategy=strategy, save_on_disk=True)

    mock_load_from_url.assert_called_once_with(url=url, save_on_disk=True)
    assert dataset.hdata is mock_hdata
    assert dataset.sampling_strategy == strategy


@pytest.mark.parametrize(
    "strategy",
    [
        pytest.param(SamplingStrategy.HYPEREDGE, id="hyperedge_strategy"),
        pytest.param(SamplingStrategy.NODE, id="node_strategy"),
    ],
)
def test_from_path(strategy, mock_hdata):
    filepath = "/abc/sample.json.zst"

    with patch.object(HIFLoader, "load_from_path", return_value=mock_hdata) as mock_load_from_path:
        dataset = Dataset.from_path(filepath=filepath, sampling_strategy=strategy)

    mock_load_from_path.assert_called_once_with(filepath=filepath)
    assert dataset.hdata is mock_hdata
    assert dataset.sampling_strategy == strategy


def test_enrich_node_features_replace(mock_hdata):
    dataset = Dataset.from_hdata(mock_hdata)

    enricher = MagicMock(spec=NodeEnricher)
    enriched_x = torch.randn(3, 4)
    enricher.enrich.return_value = enriched_x

    dataset.enrich_node_features(enricher)

    enricher.enrich.assert_called_once_with(mock_hdata.hyperedge_index)
    assert torch.equal(dataset.hdata.x, enriched_x)


def test_enrich_node_features_concatenate(mock_hdata):
    dataset = Dataset.from_hdata(mock_hdata)
    original_x = dataset.hdata.x.clone()

    enricher = MagicMock(spec=NodeEnricher)
    enriched_x = torch.randn(3, 4)
    enricher.enrich.return_value = enriched_x

    dataset.enrich_node_features(enricher, enrichment_mode="concatenate")

    enricher.enrich.assert_called_once_with(mock_hdata.hyperedge_index)
    expected_x = torch.cat([original_x, enriched_x], dim=1)
    assert torch.equal(dataset.hdata.x, expected_x)
    assert dataset.hdata.x.shape == (3, 5)  # 1 original + 4 enriched


def test_enrich_hyperedge_attr_replace(mock_hdata):
    dataset = Dataset.from_hdata(mock_hdata)

    enricher = MagicMock(spec=HyperedgeEnricher)
    enriched_x = torch.randn(3, 4)
    enricher.enrich.return_value = enriched_x

    dataset.enrich_hyperedge_attr(enricher)

    enricher.enrich.assert_called_once_with(mock_hdata.hyperedge_index)
    hyperedge_attr = dataset.hdata.hyperedge_attr
    assert hyperedge_attr is not None
    assert torch.equal(hyperedge_attr, enriched_x)


def test_enrich_hyperedge_attr_concatenate(mock_hdata_with_hyperedge_attr):
    dataset = Dataset.from_hdata(mock_hdata_with_hyperedge_attr)
    original_hyperedge_attr = dataset.hdata.hyperedge_attr
    assert original_hyperedge_attr is not None
    original_hyperedge_attr = original_hyperedge_attr.clone()

    enricher = MagicMock(spec=HyperedgeEnricher)
    enriched_x = torch.randn(3, 4)
    enricher.enrich.return_value = enriched_x

    dataset.enrich_hyperedge_attr(enricher, enrichment_mode="concatenate")

    enricher.enrich.assert_called_once_with(mock_hdata_with_hyperedge_attr.hyperedge_index)
    expected_x = torch.cat([original_hyperedge_attr, enriched_x], dim=1)
    hyperedge_attr = dataset.hdata.hyperedge_attr
    assert hyperedge_attr is not None
    assert torch.equal(hyperedge_attr, expected_x)
    assert hyperedge_attr.shape == (3, 5)  # 1 original + 4 enriched


def test_enrich_hyperedge_weights_replace(mock_hdata):
    dataset = Dataset.from_hdata(mock_hdata)

    enricher = MagicMock(spec=HyperedgeEnricher)
    enriched_weights = torch.randn(3)
    enricher.enrich.return_value = enriched_weights

    dataset.enrich_hyperedge_weights(enricher)

    enricher.enrich.assert_called_once_with(mock_hdata.hyperedge_index)
    hyperedge_weights = dataset.hdata.hyperedge_weights
    assert hyperedge_weights is not None
    assert torch.equal(hyperedge_weights, enriched_weights)


def test_enrich_hyperedge_weights_concatenate(mock_hdata_with_hyperedge_weights):
    dataset = Dataset.from_hdata(mock_hdata_with_hyperedge_weights)
    original_weights = dataset.hdata.hyperedge_weights
    assert original_weights is not None
    original_weights = original_weights.clone()

    enricher = MagicMock(spec=HyperedgeEnricher)
    enriched_weights = torch.randn(3)
    enricher.enrich.return_value = enriched_weights

    dataset.enrich_hyperedge_weights(enricher, enrichment_mode="concatenate")

    enricher.enrich.assert_called_once_with(mock_hdata_with_hyperedge_weights.hyperedge_index)
    expected_weights = torch.cat([original_weights, enriched_weights], dim=0)
    hyperedge_weights = dataset.hdata.hyperedge_weights
    assert hyperedge_weights is not None
    assert torch.equal(hyperedge_weights, expected_weights)
    assert hyperedge_weights.shape == (6,)  # 3 original + 3 enriched


@pytest.mark.parametrize(
    "hyperedge_index, k, expected_hyperedge_index",
    [
        pytest.param(
            torch.tensor([[0, 1, 2], [0, 0, 0]]),
            4,
            torch.zeros((2, 0), dtype=torch.long),
            id="single_hyperedge_below_k_removed",
        ),
        pytest.param(
            torch.tensor([[0, 1, 2], [0, 0, 0]]),
            3,
            torch.tensor([[0, 1, 2], [0, 0, 0]]),
            id="single_hyperedge_at_exact_k_kept",
        ),
        pytest.param(
            torch.tensor([[0, 1, 2, 3, 4], [0, 0, 0, 1, 1]]),
            3,
            torch.tensor([[0, 1, 2], [0, 0, 0]]),
            id="two_hyperedges_first_kept_second_removed",
        ),
        pytest.param(
            torch.tensor([[0, 1, 2, 3, 4, 5], [0, 0, 0, 1, 1, 1]]),
            3,
            torch.tensor([[0, 1, 2, 3, 4, 5], [0, 0, 0, 1, 1, 1]]),
            id="two_hyperedges_both_kept",
        ),
        pytest.param(
            torch.tensor([[0, 1, 2, 3, 4, 5], [0, 0, 1, 1, 2, 2]]),
            3,
            torch.zeros((2, 0), dtype=torch.long),
            id="three_hyperedges_all_removed",
        ),
    ],
)
def test_remove_hyperedges_with_fewer_than_k_nodes(hyperedge_index, k, expected_hyperedge_index):
    num_nodes = hyperedge_index[0].max().item() + 1 if hyperedge_index.shape[1] > 0 else 0
    x = torch.ones((num_nodes, 1), dtype=torch.float)
    hdata = HData(x=x, hyperedge_index=hyperedge_index)
    dataset = Dataset.from_hdata(hdata)

    dataset.remove_hyperedges_with_fewer_than_k_nodes(k)

    expected_num_nodes = expected_hyperedge_index[0].unique().shape[0]
    expected_num_hyperedges = expected_hyperedge_index[1].unique().shape[0]

    assert torch.equal(dataset.hdata.hyperedge_index, expected_hyperedge_index)
    assert dataset.hdata.x.shape[0] == expected_num_nodes
    assert dataset.hdata.y.shape[0] == expected_num_hyperedges


def test_split_with_equal_ratios(mock_hdata_four_node_hypergraph):
    with patch.object(HIFLoader, "load_by_name", return_value=mock_hdata_four_node_hypergraph):
        dataset = AlgebraDataset()

    splits = dataset.split([0.5, 0.5])

    assert len(splits) == 2
    assert (
        splits[0].hdata.num_hyperedges + splits[1].hdata.num_hyperedges
        == dataset.hdata.num_hyperedges
    )
    for split in splits:
        assert split.hdata.x is not None
        assert split.hdata.num_nodes > 0
        assert split.hdata.num_hyperedges > 0


def test_split_three_way(mock_hdata_multiple_edges_attr_hypergraph):
    with patch.object(
        HIFLoader, "load_by_name", return_value=mock_hdata_multiple_edges_attr_hypergraph
    ):
        dataset = AlgebraDataset()

    splits = dataset.split([0.5, 0.25, 0.25])
    total_edges = sum(split.hdata.num_hyperedges for split in splits)

    assert len(splits) == 3
    assert total_edges == dataset.hdata.num_hyperedges

    for split in splits:
        assert split.hdata.x is not None
        assert split.hdata.num_nodes > 0
        assert split.hdata.num_hyperedges > 0


def test_split_raises_when_ratios_do_not_sum_to_one(mock_hdata_four_node_hypergraph):
    with patch.object(HIFLoader, "load_by_name", return_value=mock_hdata_four_node_hypergraph):
        dataset = AlgebraDataset()

    with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
        dataset.split([0.8, 0.1, 0.05])


def test_split_with_shuffle_produces_deterministic_results_when_seed_provided(
    mock_hdata_four_node_hypergraph,
):
    with patch.object(HIFLoader, "load_by_name", return_value=mock_hdata_four_node_hypergraph):
        dataset = AlgebraDataset()

    splits_a = dataset.split([0.5, 0.5], shuffle=True, seed=42)
    splits_b = dataset.split([0.5, 0.5], shuffle=True, seed=42)

    assert torch.equal(splits_a[0].hdata.hyperedge_index, splits_b[0].hdata.hyperedge_index)
    assert torch.equal(splits_a[1].hdata.hyperedge_index, splits_b[1].hdata.hyperedge_index)


def test_split_with_shuffle_when_no_seed_provided(
    mock_hdata_four_node_hypergraph,
):
    with patch.object(HIFLoader, "load_by_name", return_value=mock_hdata_four_node_hypergraph):
        dataset = AlgebraDataset()

    splits = dataset.split([0.5, 0.5], shuffle=True)
    total_edges = sum(split.hdata.num_hyperedges for split in splits)

    assert len(splits) == 2
    assert total_edges == dataset.hdata.num_hyperedges

    for split in splits:
        assert split.hdata.x is not None
        assert split.hdata.num_nodes > 0
        assert split.hdata.num_hyperedges > 0


def test_split_preserves_edge_attr(mock_hdata_multiple_edges_attr_hypergraph):
    with patch.object(
        HIFLoader, "load_by_name", return_value=mock_hdata_multiple_edges_attr_hypergraph
    ):
        dataset = AlgebraDataset()

    splits = dataset.split([0.5, 0.5])

    for split in splits:
        assert split.hdata.hyperedge_attr is not None
        assert split.hdata.hyperedge_attr.shape[0] == split.hdata.num_hyperedges


def test_split_without_edge_attr(mock_hdata_no_edge_attr_hypergraph):
    with patch.object(HIFLoader, "load_by_name", return_value=mock_hdata_no_edge_attr_hypergraph):
        dataset = AlgebraDataset()

    splits = dataset.split([0.5, 0.5])

    for split in splits:
        assert split.hdata.hyperedge_attr is None


def test_to_device(mock_hdata):
    device = torch.device("cpu")

    dataset = Dataset.from_hdata(mock_hdata)

    result = dataset.to(device)

    assert result is dataset
    assert dataset.hdata.device == device


def test_default_sampling_strategy_is_hyperedge(mock_hdata_four_node_hypergraph):
    with patch.object(HIFLoader, "load_by_name", return_value=mock_hdata_four_node_hypergraph):
        dataset = AlgebraDataset()

    # Default strategy is HYPEREDGE, so len should be num_hyperedges (2), not num_nodes (4)
    assert dataset.sampling_strategy == SamplingStrategy.HYPEREDGE
    assert len(dataset) == 2


def test_explicit_node_sampling_strategy(mock_hdata_four_node_hypergraph):
    with patch.object(HIFLoader, "load_by_name", return_value=mock_hdata_four_node_hypergraph):
        dataset = AlgebraDataset(sampling_strategy=SamplingStrategy.NODE)

    # NODE strategy, so len should be num_nodes (4), not num_hyperedges (2)
    assert dataset.sampling_strategy == SamplingStrategy.NODE
    assert len(dataset) == 4


@pytest.mark.parametrize(
    "strategy",
    [
        pytest.param(SamplingStrategy.NODE, id="node_strategy"),
        pytest.param(SamplingStrategy.HYPEREDGE, id="hyperedge_strategy"),
    ],
)
def test_split_preserves_sampling_strategy(mock_hdata_four_node_hypergraph, strategy):
    with patch.object(HIFLoader, "load_by_name", return_value=mock_hdata_four_node_hypergraph):
        dataset = AlgebraDataset(sampling_strategy=strategy)

    splits = dataset.split([0.5, 0.5])

    for split in splits:
        assert split.sampling_strategy == strategy


def test_from_hdata_with_explicit_strategy(mock_hdata):
    dataset = Dataset.from_hdata(mock_hdata, sampling_strategy=SamplingStrategy.NODE)

    assert dataset.sampling_strategy == SamplingStrategy.NODE
    assert len(dataset) == 3  # mock_hdata has 3 nodes


def test_update_from_hdata_returns_new_dataset(mock_hdata):
    dataset = Dataset(hdata=mock_hdata)
    new_x = torch.ones((2, 1), dtype=torch.float)
    new_hyperedge_index = torch.tensor([[0, 1], [0, 0]], dtype=torch.long)
    new_hdata = HData(x=new_x, hyperedge_index=new_hyperedge_index)

    result = dataset.update_from_hdata(new_hdata)

    assert result is not dataset
    assert result.hdata is new_hdata
    assert dataset.hdata is mock_hdata


def test_update_from_hdata_stores_provided_hdata(mock_hdata):
    dataset = Dataset(hdata=mock_hdata)
    new_x = torch.ones((2, 1), dtype=torch.float)
    new_hyperedge_index = torch.tensor([[0, 1], [0, 0]], dtype=torch.long)
    new_hdata = HData(x=new_x, hyperedge_index=new_hyperedge_index)

    result = dataset.update_from_hdata(new_hdata)

    assert result.hdata is new_hdata


@pytest.mark.parametrize(
    "strategy, expected_len",
    [
        pytest.param(SamplingStrategy.NODE, 4, id="node_strategy"),
        pytest.param(SamplingStrategy.HYPEREDGE, 3, id="hyperedge_strategy"),
    ],
)
def test_update_from_hdata_inherits_sampling_strategy(mock_hdata, strategy, expected_len):
    dataset = Dataset(hdata=mock_hdata, sampling_strategy=strategy)
    new_x = torch.ones((4, 1), dtype=torch.float)
    new_hyperedge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 2]], dtype=torch.long)
    new_hdata = HData(x=new_x, hyperedge_index=new_hyperedge_index)

    result = dataset.update_from_hdata(new_hdata)

    assert result.sampling_strategy == strategy
    assert len(result) == expected_len


def test_update_from_hdata_preserves_subclass_type(mock_hdata):
    dataset = AlgebraDataset(hdata=mock_hdata)
    new_x = torch.ones((2, 1), dtype=torch.float)
    new_hyperedge_index = torch.tensor([[0, 1], [0, 0]], dtype=torch.long)
    new_hdata = HData(x=new_x, hyperedge_index=new_hyperedge_index)

    result = dataset.update_from_hdata(new_hdata)

    assert type(result) is AlgebraDataset


@pytest.fixture
def mock_hdata_stats():
    x = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
        ],
        dtype=torch.float,
    )
    hyperedge_index = torch.tensor(
        [
            [0, 1, 2, 2, 3],
            [0, 0, 0, 1, 1],
        ]
    )
    return HData(x=x, hyperedge_index=hyperedge_index)


def test_dataset_stats_computation(mock_hdata_stats):
    expected_stats = {
        "shape_x": torch.Size([4, 4]),
        "shape_hyperedge_attr": None,
        "shape_hyperedge_weights": None,
        "num_nodes": 4,
        "num_hyperedges": 2,
        "avg_degree_node_raw": 1.25,
        "avg_degree_node": 1,
        "avg_degree_hyperedge_raw": 2.5,
        "avg_degree_hyperedge": 2,
        "node_degree_max": 2,
        "hyperedge_degree_max": 3,
        "node_degree_median": 1,
        "hyperedge_degree_median": 2,
        "distribution_node_degree": [1, 1, 2, 1],
        "distribution_hyperedge_size": [3, 2],
        "distribution_node_degree_hist": {1: 3, 2: 1},
        "distribution_hyperedge_size_hist": {2: 1, 3: 1},
    }

    dataset = Dataset.from_hdata(mock_hdata_stats)

    stats = dataset.stats()
    assert stats == expected_stats


def test_enrich_node_features_from_dataset():
    source_dataset = Dataset.from_hdata(
        HData(
            x=torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]),
            hyperedge_index=torch.tensor([[0, 1, 2], [0, 0, 1]]),
            global_node_ids=torch.tensor([100, 200, 300]),
        )
    )
    target_dataset = Dataset.from_hdata(
        HData(
            x=torch.tensor([[0.0], [0.0]]),
            hyperedge_index=torch.tensor([[0, 1], [0, 0]]),
            global_node_ids=torch.tensor([300, 100]),
        )
    )

    target_dataset.enrich_node_features_from(source_dataset)

    assert torch.equal(target_dataset.hdata.x, torch.tensor([[3.0, 30.0], [1.0, 10.0]]))


def test_enrich_node_features_from_propagates_hdata_validation_errors():
    source_dataset = Dataset.from_hdata(
        HData(
            x=torch.tensor([[1.0], [2.0]]),
            hyperedge_index=torch.tensor([[0, 1], [0, 0]]),
            global_node_ids=torch.tensor([10, 20]),
        )
    )
    target_dataset = Dataset.from_hdata(
        HData(
            x=torch.tensor([[0.0]]),
            hyperedge_index=torch.tensor([[0], [0]]),
            global_node_ids=torch.tensor([10]),
        )
    )
    target_dataset.hdata.global_node_ids = None

    with pytest.raises(
        ValueError,
        match="Both HData instances must define global_node_ids to align node features.",
    ):
        target_dataset.enrich_node_features_from(source_dataset)


def test_enrich_node_features_from_dataset_with_fill_value():
    source_dataset = Dataset.from_hdata(
        HData(
            x=torch.tensor([[1.0, 10.0], [2.0, 20.0]]),
            hyperedge_index=torch.tensor([[0, 1], [0, 0]]),
            global_node_ids=torch.tensor([10, 20]),
        )
    )
    target_dataset = Dataset.from_hdata(
        HData(
            x=torch.tensor([[0.0], [0.0]]),
            hyperedge_index=torch.tensor([[0, 1], [0, 0]]),
            global_node_ids=torch.tensor([10, 30]),
        )
    )

    target_dataset.enrich_node_features_from(
        source_dataset,
        node_space_setting="inductive",
        fill_value=[7.0, 8.0],
    )

    assert torch.equal(target_dataset.hdata.x, torch.tensor([[1.0, 10.0], [7.0, 8.0]]))


def test_split_transductive_default_preserves_first_split_node_space():
    hdata = HData(
        x=torch.arange(4, dtype=torch.float).unsqueeze(1),
        hyperedge_index=torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]),
        global_node_ids=torch.tensor([100, 200, 300, 400]),
    )
    dataset = Dataset.from_hdata(hdata)

    train_dataset, test_dataset = dataset.split([0.75, 0.25])

    assert train_dataset.hdata.num_nodes == dataset.hdata.num_nodes
    assert torch.equal(train_dataset.hdata.x, dataset.hdata.x)
    assert test_dataset.hdata.num_nodes == 1


def test_split_transductive_all_preserves_all_split_node_spaces():
    hdata = HData(
        x=torch.arange(4, dtype=torch.float).unsqueeze(1),
        hyperedge_index=torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]),
        global_node_ids=torch.tensor([100, 200, 300, 400]),
    )
    dataset = Dataset.from_hdata(hdata)

    train_dataset, test_dataset = dataset.split(
        [0.75, 0.25],
        node_space_setting="transductive",
        assign_node_space_to="all",
    )

    assert train_dataset.hdata.num_nodes == dataset.hdata.num_nodes
    assert test_dataset.hdata.num_nodes == dataset.hdata.num_nodes
    assert torch.equal(train_dataset.hdata.x, dataset.hdata.x)
    assert torch.equal(test_dataset.hdata.x, dataset.hdata.x)


def test_split_raises_when_node_space_provided_with_transductive_disabled():
    hdata = HData(
        x=torch.arange(4, dtype=torch.float).unsqueeze(1),
        hyperedge_index=torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]),
        global_node_ids=torch.tensor([100, 200, 300, 400]),
    )
    dataset = Dataset.from_hdata(hdata)

    with pytest.raises(
        ValueError,
        match="assign_node_space_to can only be provided when node_space_setting='transductive'.",
    ):
        dataset.split(
            [0.75, 0.25],
            node_space_setting="inductive",
            assign_node_space_to="first",
        )


def test_nested_transductive_split_supports_train_feature_reuse():
    hdata = HData(
        x=torch.arange(4, dtype=torch.float).unsqueeze(1),
        hyperedge_index=torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]),
        global_node_ids=torch.tensor([100, 200, 300, 400]),
    )
    dataset = Dataset.from_hdata(hdata)

    train_dataset, test_dataset = dataset.split(
        [0.75, 0.25],
        node_space_setting="transductive",
    )
    train_dataset, val_dataset = train_dataset.split(
        [2 / 3, 1 / 3],
        node_space_setting="transductive",
    )

    enricher = MagicMock(spec=NodeEnricher)
    enricher.enrich.return_value = torch.tensor(
        [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0], [40.0, 41.0]]
    )
    train_dataset.enrich_node_features(enricher, enrichment_mode="replace")
    val_dataset.enrich_node_features_from(train_dataset)
    test_dataset.enrich_node_features_from(train_dataset)

    assert torch.equal(val_dataset.hdata.x, torch.tensor([[30.0, 31.0]]))
    assert torch.equal(test_dataset.hdata.x, torch.tensor([[40.0, 41.0]]))


def test_transform_node_attrs_adds_padding_zero_when_attr_keys_padding(mock_hdata):

    with patch.object(HIFLoader, "load_by_name", return_value=mock_hdata):

        class TestDataset(Dataset):
            DATASET_NAME = "TEST"

        _ = TestDataset()

        # Test with attr_keys - should pad missing attributes with 0.0
        attrs = {"weight": 1.5}
        result = Dataset.transform_node_attrs(attrs, attr_keys=["score", "weight", "age"])
        assert torch.allclose(
            result, torch.tensor([0.0, 1.5, 0.0])
        )  # score=0.0, weight=1.5, age=0.0

        # Test with all attributes present
        attrs = {"weight": 1.5, "score": 0.8, "age": 25.0}
        result = Dataset.transform_node_attrs(attrs, attr_keys=["age", "score", "weight"])
        assert torch.allclose(
            result, torch.tensor([25.0, 0.8, 1.5])
        )  # age=25.0, score=0.8, weight=1.5

        # Test without attr_keys - maintains insertion order
        attrs = {"weight": 1.5, "score": 0.8}
        result = Dataset.transform_node_attrs(attrs)
        assert torch.allclose(result, torch.tensor([1.5, 0.8]))  # weight, score (insertion order)


def test_transform_hyperedge_attrs_adds_padding_zero_when_attr_keys_padding(mock_hdata):

    with patch.object(HIFLoader, "load_by_name", return_value=mock_hdata):

        class TestDataset(Dataset):
            DATASET_NAME = "TEST"

        _ = TestDataset()
        # Test with attr_keys - should pad missing attributes with 0.0
        attrs = {"weight": 1.5}
        result = Dataset.transform_hyperedge_attrs(
            attrs, attr_keys=["capacity", "weight", "length"]
        )
        assert torch.allclose(
            result, torch.tensor([0.0, 1.5, 0.0])
        )  # capacity=0.0, weight=1.5, length=0.0
        # Test with all attributes present
        attrs = {"weight": 1.5, "capacity": 10.0, "length": 5.0}
        result = Dataset.transform_hyperedge_attrs(
            attrs, attr_keys=["length", "capacity", "weight"]
        )
        assert torch.allclose(
            result, torch.tensor([5.0, 10.0, 1.5])
        )  # length=5.0, capacity=10.0, weight=1.5
        # Test without attr_keys - maintains insertion order
        attrs = {"weight": 1.5, "capacity": 10.0}
        result = Dataset.transform_hyperedge_attrs(attrs)
        assert torch.allclose(
            result, torch.tensor([1.5, 10.0])
        )  # capacity, weight (insertion order)
