import pytest

from hyperbench.data import SamplingStrategy
from hyperbench.hlp import Node2VecGCNHlpModule, Node2VecGCNHlpConfig
from hyperbench.integration_tests.common import (
    common_metrics,
    loaders,
    train_test_loop,
    split_dataset,
    enrich_datasets,
    add_negatives,
    model_configs_with_single_model,
)

NUM_FEATURES = 8


@pytest.mark.integration
@pytest.mark.parametrize(
    "sampling_strategy, full, batch_size",
    [
        pytest.param(SamplingStrategy.HYPEREDGE, False, 128, id="hyperedge_batch_128"),
        pytest.param(SamplingStrategy.NODE, False, 128, id="node_batch_128"),
        pytest.param(SamplingStrategy.HYPEREDGE, True, 1, id="hyperedge_full"),
        pytest.param(SamplingStrategy.NODE, True, 1, id="node_full"),
    ],
)
def test_model_node2vecgcn_precomputed(tmp_path, sampling_strategy, full, batch_size, request):
    test_id = request.node.callspec.id
    num_features = NUM_FEATURES
    metrics = common_metrics()

    train_dataset, val_dataset, test_dataset = split_dataset(sampling_strategy)

    train_dataset, val_dataset, test_dataset = add_negatives(
        train_dataset, val_dataset, test_dataset
    )

    enrich_datasets(
        train_dataset,
        val_dataset,
        test_dataset,
        num_features=num_features,
    )

    train_loader, val_loader, test_loader = loaders(
        train_dataset, val_dataset, test_dataset, batch_size=batch_size, sample_full_hypergraph=full
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

    configs = model_configs_with_single_model(
        train_loader,
        val_loader,
        test_loader,
        name="node2vecgcn",
        version="precomputed",
        model=precomputed_node2vecgcn_module,
    )

    train_test_loop(
        configs,
        path=tmp_path,
        experiment_name=f"node2vecgcn_integration_test_{test_id}",
    )

    assert (
        tmp_path / f"node2vecgcn_integration_test_{test_id}" / "comparison" / "overall.tex"
    ).exists()
    assert (
        tmp_path / f"node2vecgcn_integration_test_{test_id}" / "comparison" / "test.tex"
    ).exists()
    assert (
        tmp_path / f"node2vecgcn_integration_test_{test_id}" / "comparison" / "results.md"
    ).exists()


@pytest.mark.integration
@pytest.mark.parametrize(
    "sampling_strategy, full, batch_size",
    [
        pytest.param(SamplingStrategy.HYPEREDGE, False, 128, id="hyperedge_batch_128"),
        pytest.param(SamplingStrategy.NODE, False, 128, id="node_batch_128"),
        pytest.param(SamplingStrategy.HYPEREDGE, True, 1, id="hyperedge_full"),
        pytest.param(SamplingStrategy.NODE, True, 1, id="node_full"),
    ],
)
def test_model_node2vecgcn_joint(tmp_path, sampling_strategy, full, batch_size, request):
    test_id = request.node.callspec.id
    num_features = NUM_FEATURES
    metrics = common_metrics()

    train_dataset, val_dataset, test_dataset = split_dataset(sampling_strategy)

    train_dataset, val_dataset, test_dataset = add_negatives(
        train_dataset, val_dataset, test_dataset
    )

    enrich_datasets(
        train_dataset,
        val_dataset,
        test_dataset,
        num_features=num_features,
    )

    train_loader, val_loader, test_loader = loaders(
        train_dataset, val_dataset, test_dataset, batch_size=batch_size, sample_full_hypergraph=full
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

    configs = model_configs_with_single_model(
        train_loader,
        val_loader,
        test_loader,
        name="node2vecgcn",
        version="joint",
        model=joint_node2vecgcn_module,
    )

    train_test_loop(
        configs,
        path=tmp_path,
        experiment_name=f"node2vecgcn_integration_test_{test_id}",
    )

    assert (
        tmp_path / f"node2vecgcn_integration_test_{test_id}" / "comparison" / "overall.tex"
    ).exists()
    assert (
        tmp_path / f"node2vecgcn_integration_test_{test_id}" / "comparison" / "test.tex"
    ).exists()
    assert (
        tmp_path / f"node2vecgcn_integration_test_{test_id}" / "comparison" / "results.md"
    ).exists()
