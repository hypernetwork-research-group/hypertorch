import pytest
from hyperbench.integration_tests.common import (
    common_metrics,
    enrich_datasets,
    model_configs,
    multi_model_trainer,
    splits_dataset,
    add_negatives,
    loaders,
    add_model_configs,
)
from hyperbench.hlp import HyperGCNHlpModule
from hyperbench.data import SamplingStrategy

NUM_FEATURES = 8


@pytest.mark.integration
@pytest.mark.parametrize(
    "sampling_strategy, batch, batch_size, mediator",
    [
        pytest.param(
            SamplingStrategy.HYPEREDGE, True, 128, False, id="no_mediator_hyperedge_batch_128"
        ),
        pytest.param(
            SamplingStrategy.HYPEREDGE, True, 64, False, id="no_mediator_hyperedge_batch_64"
        ),
        pytest.param(SamplingStrategy.NODE, True, 128, False, id="no_mediator_node_batch_128"),
        pytest.param(SamplingStrategy.NODE, True, 64, False, id="no_mediator_node_batch_64"),
        pytest.param(
            SamplingStrategy.HYPEREDGE, True, 128, True, id="with_mediator_hyperedge_batch_128"
        ),
        pytest.param(
            SamplingStrategy.HYPEREDGE, True, 64, True, id="with_mediator_hyperedge_batch_64"
        ),
        pytest.param(SamplingStrategy.NODE, True, 128, True, id="with_mediator_node_batch_128"),
        pytest.param(SamplingStrategy.NODE, True, 64, True, id="with_mediator_node_batch_64"),
    ],
)
def test_model_hypergcn_batch(tmp_path, sampling_strategy, batch, batch_size, mediator, request):
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

    mean_hypergcn_no_mediator_module = HyperGCNHlpModule(
        encoder_config={
            "in_channels": num_features,
            "hidden_channels": 16,
            "out_channels": 16,
            "bias": True,
            "use_batch_normalization": False,
            "drop_rate": 0.5,
            "use_mediator": False,
            "fast": False,
        },
        aggregation="mean",
        lr=0.01,
        weight_decay=5e-4,
        metrics=metrics,
    )
    if mediator:
        mean_hypergcn_with_mediator_module = HyperGCNHlpModule(
            encoder_config={
                "in_channels": num_features,
                "hidden_channels": 16,
                "out_channels": 16,
                "bias": True,
                "use_batch_normalization": False,
                "drop_rate": 0.5,
                "use_mediator": True,
                "fast": False,
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
        name="hypergcn",
        version="mean",
        model=mean_hypergcn_no_mediator_module,
    )

    if mediator:
        configs = add_model_configs(
            configs,
            train_loader,
            val_loader,
            test_loader,
            name="hypergcn",
            version="mean_with_mediator",
            model=mean_hypergcn_with_mediator_module,
        )

    multi_model_trainer(
        configs, path=tmp_path, experiment_name=f"hypergcn_integration_test_{test_id}"
    )

    assert (
        tmp_path / f"hypergcn_integration_test_{test_id}" / "comparison" / "overall.tex"
    ).exists()
    assert (tmp_path / f"hypergcn_integration_test_{test_id}" / "comparison" / "test.tex").exists()
    assert (
        tmp_path / f"hypergcn_integration_test_{test_id}" / "comparison" / "results.md"
    ).exists()


@pytest.mark.integration
@pytest.mark.integration
@pytest.mark.parametrize(
    "sampling_strategy, mediator",
    [
        pytest.param(SamplingStrategy.HYPEREDGE, False, id="hyperedge_full_no_mediator"),
        pytest.param(SamplingStrategy.HYPEREDGE, True, id="hyperedge_full_with_mediator"),
        pytest.param(SamplingStrategy.NODE, False, id="node_full_no_mediator"),
        pytest.param(SamplingStrategy.NODE, True, id="node_full_with_mediator"),
    ],
)
def test_model_hypergcn_full(tmp_path, sampling_strategy, mediator, request):
    test_id = request.node.callspec.id
    num_features = NUM_FEATURES
    metrics = common_metrics()

    train_dataset, val_dataset, test_dataset = splits_dataset(sampling_strategy)
    train_dataset, val_dataset, test_dataset = add_negatives(
        train_dataset, val_dataset, test_dataset
    )

    enrich_datasets(train_dataset, val_dataset, test_dataset, num_features=num_features)

    train_loader, val_loader, test_loader = loaders(train_dataset, val_dataset, test_dataset)

    mean_hypergcn_no_mediator_module = HyperGCNHlpModule(
        encoder_config={
            "in_channels": num_features,
            "hidden_channels": 16,
            "out_channels": 16,
            "bias": True,
            "use_batch_normalization": False,
            "drop_rate": 0.5,
            "use_mediator": False,
            "fast": False,
        },
        aggregation="mean",
        lr=0.01,
        weight_decay=5e-4,
        metrics=metrics,
    )

    if mediator:
        mean_hypergcn_with_mediator_module = HyperGCNHlpModule(
            encoder_config={
                "in_channels": num_features,
                "hidden_channels": 16,
                "out_channels": 16,
                "bias": True,
                "use_batch_normalization": False,
                "drop_rate": 0.5,
                "use_mediator": True,
                "fast": False,
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
        name="hypergcn",
        version="mean",
        model=mean_hypergcn_no_mediator_module,
    )

    if mediator:
        configs = add_model_configs(
            configs,
            train_loader,
            val_loader,
            test_loader,
            name="hypergcn",
            version="mean_with_mediator",
            model=mean_hypergcn_with_mediator_module,
        )

    multi_model_trainer(
        configs, path=tmp_path, experiment_name=f"hypergcn_integration_test_{test_id}"
    )

    assert (
        tmp_path / f"hypergcn_integration_test_{test_id}" / "comparison" / "overall.tex"
    ).exists()
    assert (tmp_path / f"hypergcn_integration_test_{test_id}" / "comparison" / "test.tex").exists()
    assert (
        tmp_path / f"hypergcn_integration_test_{test_id}" / "comparison" / "results.md"
    ).exists()
