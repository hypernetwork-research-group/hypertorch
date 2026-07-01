import pytest

from hypertorch.hlp import MLPHlpModule
from hypertorch.data import SamplingStrategyEnum
from hypertorch.integration_tests.common import (
    hlp_metrics,
    loaders,
    model_configs_with_single_model,
    train_test_loop,
    split_dataset,
    enrich_datasets,
    add_negatives,
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
def test_model_mlp(tmp_path, sampling_strategy, full, batch_size, request):
    test_id = request.node.callspec.id

    train_dataset, val_dataset, test_dataset = split_dataset(sampling_strategy)

    train_dataset, val_dataset, test_dataset = add_negatives(
        train_dataset, val_dataset, test_dataset
    )

    enrich_datasets(train_dataset, val_dataset, test_dataset, num_features=NUM_FEATURES)

    train_loader, val_loader, test_loader = loaders(
        train_dataset, val_dataset, test_dataset, batch_size=batch_size, sample_full_hypergraph=full
    )

    mlp = MLPHlpModule(
        encoder_config={
            "in_channels": NUM_FEATURES,
            "out_channels": NUM_FEATURES,
            "hidden_channels": 8,
            "num_layers": 3,
            "drop_rate": 0.3,
        },
        aggregation="mean",
        metrics=hlp_metrics(),
    )

    configs = model_configs_with_single_model(
        name="mlp",
        version="hlp",
        model=mlp,
    )

    train_test_loop(
        configs=configs,
        path=tmp_path,
        experiment_name=f"mlp_hlp_integration_test_{test_id}",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    assert (
        tmp_path / f"mlp_hlp_integration_test_{test_id}" / "comparison" / "overall.tex"
    ).exists()
    assert (tmp_path / f"mlp_hlp_integration_test_{test_id}" / "comparison" / "overall.md").exists()
    assert (tmp_path / f"mlp_hlp_integration_test_{test_id}" / "comparison" / "test.tex").exists()
    assert (tmp_path / f"mlp_hlp_integration_test_{test_id}" / "comparison" / "test.md").exists()
    assert (tmp_path / f"mlp_hlp_integration_test_{test_id}" / "comparison" / "train.md").exists()
    assert (tmp_path / f"mlp_hlp_integration_test_{test_id}" / "comparison" / "train.tex").exists()
    assert (tmp_path / f"mlp_hlp_integration_test_{test_id}" / "comparison" / "val.md").exists()
    assert (tmp_path / f"mlp_hlp_integration_test_{test_id}" / "comparison" / "val.tex").exists()
