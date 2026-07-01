import pytest

from hypertorch.hlp import CommonNeighborsHlpModule
from hypertorch.types import ModelConfig
from hypertorch.data import RandomNegativeSampler, SamplingStrategyEnum
from hypertorch.integration_tests.common import (
    hlp_metrics,
    loaders,
    train_test_loop,
    split_dataset,
    enrich_datasets,
    model_configs_with_single_model,
)

NUM_FEATURES = 8


@pytest.mark.integration
@pytest.mark.parametrize(
    "sampling_strategy, full, batch_size",
    [
        pytest.param(SamplingStrategyEnum.HYPEREDGE, False, 128, id="hyperedge_batch_128"),
        pytest.param(SamplingStrategyEnum.NODE, False, 128, id="node_batch_128"),
        pytest.param(SamplingStrategyEnum.HYPEREDGE, True, 1, id="hyperedge_full"),
        pytest.param(SamplingStrategyEnum.NODE, True, 1, id="node_full"),
    ],
)
def test_model_common_neighbors(tmp_path, sampling_strategy, full, batch_size, request):
    test_id = request.node.callspec.id

    train_dataset, val_dataset, test_dataset = split_dataset(sampling_strategy)

    negative_sampler = RandomNegativeSampler(
        num_negative_samples=int(test_dataset.hdata.num_hyperedges * 0.6),
        num_nodes_per_sample=int(test_dataset.stats()["avg_degree_hyperedge"]),
    )
    test_dataset = test_dataset.add_negative_samples(negative_sampler, seed=42)

    enrich_datasets(train_dataset, val_dataset, test_dataset, num_features=NUM_FEATURES)

    train_loader, val_loader, test_loader = loaders(
        train_dataset, val_dataset, test_dataset, batch_size=batch_size, sample_full_hypergraph=full
    )

    common_neighbors = CommonNeighborsHlpModule(
        train_hdata=train_dataset.hdata,
        aggregation="mean",
        metrics=hlp_metrics(),
    )

    configs = [
        ModelConfig(
            name="common_neighbors",
            version="hlp",
            model=common_neighbors,
            is_trainable=False,
        ),
    ]

    configs = model_configs_with_single_model(
        name="common_neighbors",
        version="hlp",
        model=common_neighbors,
        is_trainable=False,
    )

    train_test_loop(
        configs=configs,
        path=tmp_path,
        experiment_name=f"cn_hlp_integration_test_{test_id}",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    assert (tmp_path / f"cn_hlp_integration_test_{test_id}" / "comparison" / "overall.tex").exists()
    assert (tmp_path / f"cn_hlp_integration_test_{test_id}" / "comparison" / "overall.md").exists()
    assert (tmp_path / f"cn_hlp_integration_test_{test_id}" / "comparison" / "test.tex").exists()
    assert (tmp_path / f"cn_hlp_integration_test_{test_id}" / "comparison" / "test.md").exists()
    assert (tmp_path / f"cn_hlp_integration_test_{test_id}" / "comparison" / "train.md").exists()
    assert (tmp_path / f"cn_hlp_integration_test_{test_id}" / "comparison" / "train.tex").exists()
    assert (tmp_path / f"cn_hlp_integration_test_{test_id}" / "comparison" / "val.md").exists()
    assert (tmp_path / f"cn_hlp_integration_test_{test_id}" / "comparison" / "val.tex").exists()
