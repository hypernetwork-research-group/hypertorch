import pytest
from hyperbench.integration_tests.common import (
    common_standard_metrics,
    datasets_enrichers,
    model_configs,
    multi_model_trainer,
    splits_dataset,
    add_negatives,
    loaders,
    add_model_configs,
)
from hyperbench.hlp import HyperGCNHlpModule

NUM_FEATURES = 8


@pytest.mark.integration
def test_model_hypergcn():
    num_features = NUM_FEATURES
    metrics = common_standard_metrics()

    train_dataset, val_dataset, test_dataset = splits_dataset()
    train_dataset, val_dataset, test_dataset = add_negatives(
        train_dataset, val_dataset, test_dataset
    )

    datasets_enrichers(train_dataset, val_dataset, test_dataset, num_features=num_features)

    train_loader_full_hypergraph, val_loader_full_hypergraph, test_loader_full_hypergraph = loaders(
        train_dataset, val_dataset, test_dataset
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
        train_loader_full_hypergraph,
        val_loader_full_hypergraph,
        test_loader_full_hypergraph,
        name="hypergcn",
        version="mean",
        module=mean_hypergcn_no_mediator_module,
    )

    configs = add_model_configs(
        configs,
        train_loader_full_hypergraph,
        val_loader_full_hypergraph,
        test_loader_full_hypergraph,
        name="hypergcn",
        version="mean_with_mediator",
        module=mean_hypergcn_with_mediator_module,
    )

    multi_model_trainer(configs)
