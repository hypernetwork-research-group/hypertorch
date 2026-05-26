import pytest
from hyperbench.integration_tests.common import (
    common_metrics,
    enrich_datasets,
    model_configs,
    multi_model_trainer,
    splits_dataset,
    add_negatives,
    loaders,
)
from hyperbench.hlp import HNHNHlpModule
from hyperbench.data import SamplingStrategy

NUM_FEATURES = 8


@pytest.mark.integration
@pytest.mark.parametrize(
    "sampling_strategy, batch, batch_size",
    [
        pytest.param(SamplingStrategy.HYPEREDGE, True, 128, id="hyperedge_batch_128"),
        pytest.param(SamplingStrategy.NODE, True, 128, id="node_batch_128"),
    ],
)
def test_model_hnhn_batch(tmp_path, sampling_strategy, batch, batch_size, request):
    test_id = request.node.callspec.id
    num_features = NUM_FEATURES
    metrics = common_metrics()

    train_dataset, val_dataset, test_dataset = splits_dataset(sampling_strategy)
    train_dataset, val_dataset, test_dataset = add_negatives(
        train_dataset, val_dataset, test_dataset
    )

    enrich_datasets(train_dataset, val_dataset, test_dataset, num_features=num_features)

    train_loader, val_loader, test_loader = loaders(
        train_dataset, val_dataset, test_dataset, batch=batch, batch_size=batch_size
    )

    mean_hnhn_module = HNHNHlpModule(
        encoder_config={
            "in_channels": num_features,
            "hidden_channels": 16,
            "out_channels": 16,
            "bias": True,
            "use_batch_normalization": False,
            "drop_rate": 0.3,
        },
        aggregation="mean",
        lr=0.04,
        weight_decay=5e-4,
        scheduler_step_size=100,
        scheduler_gamma=0.51,
        metrics=metrics,
    )

    configs = model_configs(
        train_loader,
        val_loader,
        test_loader,
        name="hnhn",
        version="mean",
        model=mean_hnhn_module,
    )

    multi_model_trainer(configs, path=tmp_path, experiment_name=f"hnhn_integration_test_{test_id}")

    assert (tmp_path / f"hnhn_integration_test_{test_id}" / "comparison" / "overall.tex").exists()
    assert (tmp_path / f"hnhn_integration_test_{test_id}" / "comparison" / "test.tex").exists()
    assert (tmp_path / f"hnhn_integration_test_{test_id}" / "comparison" / "results.md").exists()


@pytest.mark.integration
@pytest.mark.parametrize(
    "sampling_strategy",
    [
        pytest.param(SamplingStrategy.HYPEREDGE, id="hyperedge_batch"),
        pytest.param(SamplingStrategy.NODE, id="node_batch"),
    ],
)
def test_model_hnhn_full(tmp_path, sampling_strategy, request):
    test_id = request.node.callspec.id
    num_features = NUM_FEATURES
    metrics = common_metrics()

    train_dataset, val_dataset, test_dataset = splits_dataset(sampling_strategy)
    train_dataset, val_dataset, test_dataset = add_negatives(
        train_dataset, val_dataset, test_dataset
    )

    enrich_datasets(train_dataset, val_dataset, test_dataset, num_features=num_features)

    train_loader, val_loader, test_loader = loaders(train_dataset, val_dataset, test_dataset)

    mean_hnhn_module = HNHNHlpModule(
        encoder_config={
            "in_channels": num_features,
            "hidden_channels": 16,
            "out_channels": 16,
            "bias": True,
            "use_batch_normalization": False,
            "drop_rate": 0.3,
        },
        aggregation="mean",
        lr=0.04,
        weight_decay=5e-4,
        scheduler_step_size=100,
        scheduler_gamma=0.51,
        metrics=metrics,
    )

    configs = model_configs(
        train_loader,
        val_loader,
        test_loader,
        name="hnhn",
        version="mean",
        model=mean_hnhn_module,
    )

    multi_model_trainer(configs, path=tmp_path, experiment_name=f"hnhn_integration_test_{test_id}")

    assert (tmp_path / f"hnhn_integration_test_{test_id}" / "comparison" / "overall.tex").exists()
    assert (tmp_path / f"hnhn_integration_test_{test_id}" / "comparison" / "test.tex").exists()
    assert (tmp_path / f"hnhn_integration_test_{test_id}" / "comparison" / "results.md").exists()
