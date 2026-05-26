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
from hyperbench.hlp import Node2VecSLPHlpModule

NUM_FEATURES = 8


@pytest.mark.integration
@pytest.mark.parametrize(
    "sampling_strategy, batch, batch_size",
    [
        pytest.param(SamplingStrategy.HYPEREDGE, True, 128, id="hyperedge_batch_128"),
        pytest.param(SamplingStrategy.NODE, True, 128, id="node_batch_128"),
    ],
)
def test_model_node2vecslp_batch(tmp_path, sampling_strategy, batch, batch_size, request):
    test_id = request.node.callspec.id
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
        train_dataset, val_dataset, test_dataset, batch=batch, batch_size=batch_size
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
        train_loader,
        val_loader,
        test_loader,
        name="node2vecslp",
        version="precomputed",
        model=precomputed_node2vecslp_module,
    )
    configs = add_model_configs(
        configs,
        train_loader,
        val_loader,
        test_loader,
        name="node2vecslp",
        version="joint",
        model=joint_node2vecslp_module,
    )

    multi_model_trainer(
        configs=configs,
        path=tmp_path,
        experiment_name=f"node2vecslp_integration_test_{test_id}",
    )

    assert (
        tmp_path / f"node2vecslp_integration_test_{test_id}" / "comparison" / "overall.tex"
    ).exists()
    assert (
        tmp_path / f"node2vecslp_integration_test_{test_id}" / "comparison" / "test.tex"
    ).exists()
    assert (
        tmp_path / f"node2vecslp_integration_test_{test_id}" / "comparison" / "results.md"
    ).exists()


@pytest.mark.integration
@pytest.mark.parametrize(
    "sampling_strategy",
    [
        pytest.param(SamplingStrategy.HYPEREDGE, id="hyperedge"),
        pytest.param(SamplingStrategy.NODE, id="node"),
    ],
)
def test_model_node2vecslp_full(tmp_path, sampling_strategy, request):
    test_id = request.node.callspec.id
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

    train_loader, val_loader, test_loader = loaders(train_dataset, val_dataset, test_dataset)

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
        train_loader,
        val_loader,
        test_loader,
        name="node2vecslp",
        version="precomputed",
        model=precomputed_node2vecslp_module,
    )
    configs = add_model_configs(
        configs,
        train_loader,
        val_loader,
        test_loader,
        name="node2vecslp",
        version="joint",
        model=joint_node2vecslp_module,
    )

    multi_model_trainer(
        configs=configs,
        path=tmp_path,
        experiment_name=f"node2vecslp_integration_test_{test_id}",
    )

    assert (
        tmp_path / f"node2vecslp_integration_test_{test_id}" / "comparison" / "overall.tex"
    ).exists()
    assert (
        tmp_path / f"node2vecslp_integration_test_{test_id}" / "comparison" / "test.tex"
    ).exists()
    assert (
        tmp_path / f"node2vecslp_integration_test_{test_id}" / "comparison" / "results.md"
    ).exists()
