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
from hyperbench.hlp import HGNNPHlpModule
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
def test_model_hgnnp_batch(tmp_path, sampling_strategy, batch, batch_size, request):
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

    mean_hgnnp_module = HGNNPHlpModule(
        encoder_config={
            "in_channels": num_features,
            "hidden_channels": 16,
            "out_channels": 16,
            "bias": True,
            "use_batch_normalization": False,
            "drop_rate": 0.5,
        },
        aggregation="mean",
        lr=0.01,
        weight_decay=5e-4,
        metrics=metrics,
    )

    configs = model_configs(
        train_loader,
        val_loader,
        test_loader,
        name="hgnnp",
        version="mean",
        model=mean_hgnnp_module,
    )

    multi_model_trainer(configs, path=tmp_path, experiment_name=f"hgnnp_integration_test_{test_id}")

    assert (tmp_path / f"hgnnp_integration_test_{test_id}" / "comparison" / "overall.tex").exists()
    assert (tmp_path / f"hgnnp_integration_test_{test_id}" / "comparison" / "test.tex").exists()
    assert (tmp_path / f"hgnnp_integration_test_{test_id}" / "comparison" / "results.md").exists()


@pytest.mark.integration
@pytest.mark.parametrize(
    "sampling_strategy",
    [
        pytest.param(SamplingStrategy.HYPEREDGE, id="hyperedge_batch"),
        pytest.param(SamplingStrategy.NODE, id="node_batch"),
    ],
)
def test_model_hgnnp_full(tmp_path, sampling_strategy, request):
    test_id = request.node.callspec.id

    num_features = NUM_FEATURES

    metrics = common_metrics()

    train_dataset, val_dataset, test_dataset = splits_dataset(sampling_strategy)

    train_dataset, val_dataset, test_dataset = add_negatives(
        train_dataset, val_dataset, test_dataset
    )

    enrich_datasets(train_dataset, val_dataset, test_dataset, num_features=num_features)

    train_loader, val_loader, test_loader = loaders(train_dataset, val_dataset, test_dataset)

    mean_hgnnp_module = HGNNPHlpModule(
        encoder_config={
            "in_channels": num_features,
            "hidden_channels": 16,
            "out_channels": 16,
            "bias": True,
            "use_batch_normalization": False,
            "drop_rate": 0.5,
        },
        aggregation="mean",
        lr=0.01,
        weight_decay=5e-4,
        metrics=metrics,
    )

    configs = model_configs(
        train_loader,
        val_loader,
        test_loader,
        name="hgnnp",
        version="mean",
        model=mean_hgnnp_module,
    )

    multi_model_trainer(configs, path=tmp_path, experiment_name=f"hgnnp_integration_test_{test_id}")

    assert (tmp_path / f"hgnnp_integration_test_{test_id}" / "comparison" / "overall.tex").exists()
    assert (tmp_path / f"hgnnp_integration_test_{test_id}" / "comparison" / "test.tex").exists()
    assert (tmp_path / f"hgnnp_integration_test_{test_id}" / "comparison" / "results.md").exists()
