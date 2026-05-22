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
from hyperbench.hlp import HGNNHlpModule

NUM_FEATURES = 32


@pytest.mark.integration
def test_hgnn_model():
    num_features = NUM_FEATURES
    metrics = common_standard_metrics()

    train_dataset, val_dataset, test_dataset = splits_dataset()
    train_dataset, val_dataset, test_dataset = add_negatives(
        train_dataset, val_dataset, test_dataset
    )

    datasets_enrichers(train_dataset, val_dataset, test_dataset)

    train_loader_full_hypergraph, val_loader_full_hypergraph, test_loader_full_hypergraph = loaders(
        train_dataset, val_dataset, test_dataset
    )

    mean_hgnn_module = HGNNHlpModule(
        encoder_config={
            "in_channels": num_features,
            "hidden_channels": 16,
            "out_channels": 16,
            "bias": True,
            "use_batch_normalization": False,
            "drop_rate": 0.5,
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
        name="hgnn",
        version="mean",
        module=mean_hgnn_module,
    )

    multi_model_trainer(configs)
