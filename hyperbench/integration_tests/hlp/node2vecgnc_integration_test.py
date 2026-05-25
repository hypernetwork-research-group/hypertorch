import pytest
from hyperbench.integration_tests.common import (
    common_metrics,
    enrich_datasets,
    splits_dataset,
    add_negatives,
    loaders,
    model_configs,
    add_model_configs,
    multi_model_trainer,
)
from hyperbench.data import Node2VecEnricher, SamplingStrategy
from hyperbench.hlp import Node2VecGCNHlpModule, Node2VecGCNHlpConfig

NUM_FEATURES = 8


@pytest.mark.integration
@pytest.mark.parametrize("sampling_strategy", [SamplingStrategy.HYPEREDGE, SamplingStrategy.NODE])
def test_model_node2vecgcn(tmp_path, sampling_strategy):
    num_features = NUM_FEATURES
    metrics = common_metrics()

    train_dataset, val_dataset, test_dataset = splits_dataset(sampling_strategy)

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

    enrich_datasets(
        train_dataset,
        val_dataset,
        test_dataset,
        num_features=num_features,
        enricher=node2vec_enricher,
    )

    train_loader, val_loader, test_loader = loaders(
        train_dataset, val_dataset, test_dataset, batch=True, batch_size=128
    )

    gcn_config: Node2VecGCNHlpConfig = {
        "out_channels": num_features,
        "hidden_channels": num_features,
        "num_layers": 2,
        "drop_rate": 0.1,
        "bias": True,
        "improved": False,
        "add_self_loops": True,
        "normalize": True,
        "cached": False,
        "graph_reduction_strategy": "clique_expansion",
    }

    precomputed_node2vecgcn_module = Node2VecGCNHlpModule(
        encoder_config={
            "mode": "precomputed",
            "num_features": num_features,
            "node2vec_config": {},
            "gcn_config": gcn_config,
        },
        aggregation="mean",
        lr=0.001,
        weight_decay=0.0,
        metrics=metrics,
    )

    train_hyperedge_index = train_dataset.hdata.hyperedge_index
    joint_node2vecgcn_module = Node2VecGCNHlpModule(
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
            "gcn_config": gcn_config,
        },
        aggregation="mean",
        lr=0.001,
        weight_decay=0.0,
        metrics=metrics,
    )

    configs = model_configs(
        train_loader,
        val_loader,
        test_loader,
        name="node2vecgcn",
        version="precomputed",
        model=precomputed_node2vecgcn_module,
    )
    configs = add_model_configs(
        configs,
        train_loader,
        val_loader,
        test_loader,
        name="node2vecgcn",
        version="joint",
        model=joint_node2vecgcn_module,
    )

    multi_model_trainer(configs, path=tmp_path, experiment_name="node2vecgcn_integration_test")

    assert (tmp_path / "node2vecgcn_integration_test" / "comparison" / "overall.tex").exists()
    assert (tmp_path / "node2vecgcn_integration_test" / "comparison" / "test.tex").exists()
    assert (tmp_path / "node2vecgcn_integration_test" / "comparison" / "results.md").exists()
