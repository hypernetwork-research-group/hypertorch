import pytest
from hyperbench.integration_tests.common import (
    common_standard_metrics,
    datasets_enrichers,
    splits_dataset,
    add_negatives,
    loaders,
    model_configs,
    add_model_configs,
    multi_model_trainer,
)
from hyperbench.data import Node2VecEnricher
from hyperbench.hlp import NHPHlpModule

pytestmark = pytest.mark.filterwarnings(
    "ignore:Failing to pass a value to the 'type_params' parameter of 'typing._eval_type' is deprecated.*:DeprecationWarning"
)

NUM_FEATURES = 8


@pytest.mark.integration
def test_model_nhp():
    num_features = NUM_FEATURES
    metrics = common_standard_metrics()

    train_dataset, val_dataset, test_dataset = splits_dataset()

    train_dataset, val_dataset, test_dataset = add_negatives(
        train_dataset, val_dataset, test_dataset
    )

    node2vec_enricher = Node2VecEnricher(
        num_features=num_features,
        context_size=10,
        walk_length=20,
        num_walks_per_node=10,
        num_negative_samples=1,
        num_nodes=train_dataset.hdata.num_nodes,
        num_epochs=10,
        learning_rate=0.01,
        batch_size=128,
        sparse=False,
    )

    datasets_enrichers(
        train_dataset,
        val_dataset,
        test_dataset,
        num_features=num_features,
        enricher=node2vec_enricher,
    )

    train_loader_batch_hypergraph, val_loader_batch_hypergraph, test_loader_batch_hypergraph = (
        loaders(train_dataset, val_dataset, test_dataset, batch=True, batch_size=64)
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

    mean_nhp_module = NHPHlpModule(
        encoder_config={
            "in_channels": num_features,
            "hidden_channels": 8,
            "aggregation": "mean",
        },
        lr=0.001,
        weight_decay=5e-4,
        metrics=metrics,
    )

    configs = model_configs(
        train_loader_batch_hypergraph,
        val_loader_batch_hypergraph,
        test_loader_batch_hypergraph,
        name="nhp",
        version="maxmin",
        module=maxmin_nhp_module,
    )
    configs = add_model_configs(
        configs,
        train_loader_batch_hypergraph,
        val_loader_batch_hypergraph,
        test_loader_batch_hypergraph,
        name="nhp",
        version="mean",
        module=mean_nhp_module,
    )

    multi_model_trainer(configs)
