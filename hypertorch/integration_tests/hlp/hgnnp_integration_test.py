import pytest

from hypertorch.hlp import HGNNPHlpModule
from hypertorch.data import SamplingStrategy
from hypertorch.integration_tests.common import (
    common_metrics,
    enrich_datasets,
    model_configs_with_single_model,
    split_dataset,
    train_test_loop,
    add_negatives,
    loaders,
)

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
def test_model_hgnnp_batch(tmp_path, sampling_strategy, full, batch_size, request):
    test_id = request.node.callspec.id

    num_features = NUM_FEATURES

    metrics = common_metrics()

    train_dataset, val_dataset, test_dataset = split_dataset(sampling_strategy)

    train_dataset, val_dataset, test_dataset = add_negatives(
        train_dataset, val_dataset, test_dataset
    )

    enrich_datasets(train_dataset, val_dataset, test_dataset, num_features=num_features)

    train_loader, val_loader, test_loader = loaders(
        train_dataset, val_dataset, test_dataset, batch_size=batch_size, sample_full_hypergraph=full
    )

    mean_hgnnp_module = HGNNPHlpModule(
        encoder_config={
            "in_channels": num_features,
            "hidden_channels": 8,
            "out_channels": 8,
            "bias": True,
            "use_batch_normalization": False,
            "drop_rate": 0.5,
        },
        aggregation="mean",
        lr=0.01,
        weight_decay=5e-4,
        metrics=metrics,
    )

    configs = model_configs_with_single_model(
        train_loader,
        val_loader,
        test_loader,
        name="hgnnp",
        version="mean",
        model=mean_hgnnp_module,
    )

    train_test_loop(configs, path=tmp_path, experiment_name=f"hgnnp_integration_test_{test_id}")

    assert (tmp_path / f"hgnnp_integration_test_{test_id}" / "comparison" / "overall.tex").exists()
    assert (tmp_path / f"hgnnp_integration_test_{test_id}" / "comparison" / "overall.md").exists()
    assert (tmp_path / f"hgnnp_integration_test_{test_id}" / "comparison" / "test.tex").exists()
    assert (tmp_path / f"hgnnp_integration_test_{test_id}" / "comparison" / "test.md").exists()
    assert (tmp_path / f"hgnnp_integration_test_{test_id}" / "comparison" / "train.md").exists()
    assert (tmp_path / f"hgnnp_integration_test_{test_id}" / "comparison" / "train.tex").exists()
    assert (tmp_path / f"hgnnp_integration_test_{test_id}" / "comparison" / "val.md").exists()
    assert (tmp_path / f"hgnnp_integration_test_{test_id}" / "comparison" / "val.tex").exists()
