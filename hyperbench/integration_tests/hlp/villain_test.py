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
from hyperbench.hlp import VilLainHlpModule

NUM_FEATURES = 8


@pytest.mark.integration
def test_model_villain():
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

    node_villain_module = VilLainHlpModule(
        encoder_config={
            "num_nodes": train_dataset.hdata.num_nodes,
            "embedding_dim": 128,
            "labels_per_subspace": 8,
            "training_steps": 4,
            "generation_steps": 128,
            "tau": 1.0,
            "eps": 1e-10,
            "villain_loss_weight": 1.0,
        },
        embedding_mode="node",
        aggregation="maxmin",
        lr=0.01,
        weight_decay=0.0,
        metrics=metrics,
    )

    hyperedge_villain_module = VilLainHlpModule(
        encoder_config={
            "num_nodes": train_dataset.hdata.num_nodes,
            "embedding_dim": 128,
            "labels_per_subspace": 8,
            "training_steps": 4,
            "generation_steps": 28,
            "tau": 1.0,
            "eps": 1e-10,
            "villain_loss_weight": 1.0,
        },
        embedding_mode="hyperedge",
        lr=0.01,
        weight_decay=0.0,
        metrics=metrics,
    )

    configs = model_configs(
        train_loader_full_hypergraph,
        val_loader_full_hypergraph,
        test_loader_full_hypergraph,
        name="villain",
        version="node_maxmin",
        module=node_villain_module,
    )

    configs = add_model_configs(
        configs,
        train_loader_full_hypergraph,
        val_loader_full_hypergraph,
        test_loader_full_hypergraph,
        name="villain",
        version="hyperedge",
        module=hyperedge_villain_module,
    )

    multi_model_trainer(configs)
