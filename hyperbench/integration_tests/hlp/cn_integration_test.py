import pytest
from hyperbench.integration_tests.common import (
    common_metrics,
    loaders,
    train_test_loop,
    split_dataset,
    enrich_datasets,
    add_negatives,
)
from hyperbench.hlp import CommonNeighborsHlpModule
from hyperbench.types import ModelConfig
from hyperbench.data import SamplingStrategy

NUM_FEATURES = 8


@pytest.mark.integration
@pytest.mark.parametrize(
    "sampling_strategy, full, batch_size",
    [
        pytest.param(SamplingStrategy.HYPEREDGE, False, 128, id="hyperedge_batch_128"),
        pytest.param(SamplingStrategy.NODE, False, 128, id="node_batch_128"),
        pytest.param(SamplingStrategy.HYPEREDGE, True, 1, id="hyperedge_full"),
        pytest.param(SamplingStrategy.NODE, True, 1, id="node_full"),
    ],
)
def test_model_cn(tmp_path, sampling_strategy, full, batch_size, request):
    test_id = request.node.callspec.id
    num_features = NUM_FEATURES
    metrics = common_metrics()

    train_dataset, val_dataset, test_dataset = split_dataset(sampling_strategy)

    train_hyperedge_index = train_dataset.hdata.hyperedge_index

    train_dataset, val_dataset, test_dataset = add_negatives(
        train_dataset, val_dataset, test_dataset
    )

    enrich_datasets(train_dataset, val_dataset, test_dataset, num_features=num_features)

    train_loader, val_loader, test_loader = loaders(
        train_dataset, val_dataset, test_dataset, batch_size=batch_size, sample_full_hypergraph=full
    )

    mean_cn_module = CommonNeighborsHlpModule(
        train_hyperedge_index=train_hyperedge_index,
        aggregation="mean",
        metrics=metrics,
    )

    configs = [
        ModelConfig(
            name="common_neighbors",
            version="mean",
            model=mean_cn_module,
            is_trainable=False,
        ),
    ]

    train_test_loop(
        configs,
        path=tmp_path,
        experiment_name=f"cn_integration_test_{test_id}",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    assert (tmp_path / f"cn_integration_test_{test_id}" / "comparison" / "overall.tex").exists()
    assert (tmp_path / f"cn_integration_test_{test_id}" / "comparison" / "test.tex").exists()
    assert (tmp_path / f"cn_integration_test_{test_id}" / "comparison" / "results.md").exists()
