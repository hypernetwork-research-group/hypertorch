import pytest

from hypertorch.data import SamplingStrategyEnum
from hypertorch.hlp import NHPHlpModule
from hypertorch.integration_tests.common import (
    hlp_metrics,
    loaders,
    train_test_loop,
    split_dataset,
    enrich_datasets,
    add_negatives,
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
def test_model_nhp(tmp_path, sampling_strategy, full, batch_size, request):
    test_id = request.node.callspec.id
    num_features = NUM_FEATURES
    metrics = hlp_metrics()

    train_dataset, val_dataset, test_dataset = split_dataset(sampling_strategy)

    train_dataset, val_dataset, test_dataset = add_negatives(
        train_dataset, val_dataset, test_dataset
    )

    enrich_datasets(
        train_dataset,
        val_dataset,
        test_dataset,
        num_features=num_features,
    )

    train_loader, val_loader, test_loader = loaders(
        train_dataset, val_dataset, test_dataset, batch_size=batch_size, sample_full_hypergraph=full
    )

    maxmin_nhp_module = NHPHlpModule(
        encoder_config={
            "in_channels": num_features,
            "hidden_channels": 8,
            "aggregation": "maxmin",
        },
        lr=0.001,
        weight_decay=5e-4,
        metrics=metrics,
    )

    configs = model_configs_with_single_model(
        train_loader,
        val_loader,
        test_loader,
        name="nhp",
        version="maxmin",
        model=maxmin_nhp_module,
    )

    train_test_loop(configs, path=tmp_path, experiment_name=f"nhp_integration_test_{test_id}")

    assert (tmp_path / f"nhp_integration_test_{test_id}" / "comparison" / "overall.tex").exists()
    assert (tmp_path / f"nhp_integration_test_{test_id}" / "comparison" / "overall.md").exists()
    assert (tmp_path / f"nhp_integration_test_{test_id}" / "comparison" / "test.tex").exists()
    assert (tmp_path / f"nhp_integration_test_{test_id}" / "comparison" / "test.md").exists()
    assert (tmp_path / f"nhp_integration_test_{test_id}" / "comparison" / "train.md").exists()
    assert (tmp_path / f"nhp_integration_test_{test_id}" / "comparison" / "train.tex").exists()
    assert (tmp_path / f"nhp_integration_test_{test_id}" / "comparison" / "val.md").exists()
    assert (tmp_path / f"nhp_integration_test_{test_id}" / "comparison" / "val.tex").exists()
