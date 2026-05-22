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
from hyperbench.nn import Node2VecEnricher
from hyperbench.hlp import Node2VecSLPHlpModule

pytestmark = pytest.mark.filterwarnings(
    "ignore:Failing to pass a value to the 'type_params' parameter of 'typing._eval_type' is deprecated.*:DeprecationWarning"
)

NUM_FEATURES = 8


@pytest.mark.integration
def test_model_node2vecslp():
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
        loaders(train_dataset, val_dataset, test_dataset, batch=True, batch_size=128)
    )

    precomputed_node2vecslp_module = Node2VecSLPHlpModule(
        encoder_config={
            "mode": "precomputed",
            "num_features": num_features,
            "node2vec_config": {},
        },
        aggregation="mean",
        lr=0.001,
        weight_decay=0.0,
        metrics=metrics,
    )

    train_hyperedge_index = train_dataset.hdata.hyperedge_index
    joint_node2vecslp_module = Node2VecSLPHlpModule(
        encoder_config={
            "mode": "joint",
            "num_features": num_features,
            "node2vec_config": {
                "context_size": 10,
                "walk_length": 20,
                "num_walks_per_node": 10,
                "p": 1.0,
                "q": 1.0,
                "num_negative_samples": 1,
                "train_hyperedge_index": train_hyperedge_index,
                "num_nodes": train_dataset.hdata.num_nodes,
                "graph_reduction_strategy": "clique_expansion",
                "random_walk_batch_size": 128,
                # We count the node2vec loss as 40% of the total loss (the rest is the SLP loss)
                "node2vec_loss_weight": 0.4,
            },
        },
        aggregation="mean",
        lr=0.001,
        weight_decay=0.0,
        metrics=metrics,
    )

    configs = model_configs(
        train_loader_batch_hypergraph,
        val_loader_batch_hypergraph,
        test_loader_batch_hypergraph,
        name="node2vecslp",
        version="precomputed",
        module=precomputed_node2vecslp_module,
    )
    configs = add_model_configs(
        configs,
        train_loader_batch_hypergraph,
        val_loader_batch_hypergraph,
        test_loader_batch_hypergraph,
        name="node2vecslp",
        version="joint",
        module=joint_node2vecslp_module,
    )

    multi_model_trainer(configs)
