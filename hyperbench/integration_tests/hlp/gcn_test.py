import pytest
from hyperbench.integration_tests.common import (
    common_standard_metrics,
    datasets_enrichers,
    model_configs,
    multi_model_trainer,
    splits_dataset,
    add_negatives,
    loaders,
)
from hyperbench.hlp import GCNHlpModule

NUM_FEATURES = 8


@pytest.mark.integration
def test_model_gcn():
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

    mean_gcn_module = GCNHlpModule(
        encoder_config={
            "in_channels": num_features,
            "hidden_channels": 16,
            "out_channels": 16,
            "num_layers": 2,
            "drop_rate": 0.1,
            "bias": True,
            "improved": False,
            "add_self_loops": True,
            "normalize": True,
            "cached": False,
            "graph_reduction_strategy": "clique_expansion",
        },
        aggregation="mean",
        lr=0.001,
        weight_decay=5e-4,
        metrics=metrics,
    )

    configs = model_configs(
        train_loader_full_hypergraph,
        val_loader_full_hypergraph,
        test_loader_full_hypergraph,
        name="gcn",
        version="mean",
        module=mean_gcn_module,
    )

    multi_model_trainer(configs)
