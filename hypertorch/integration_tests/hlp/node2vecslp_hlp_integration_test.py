import pytest

from hypertorch.data import SamplingStrategyEnum
from hypertorch.hlp import Node2VecSLPHlpModule
from hypertorch.integration_tests.common import (
    hlp_metrics,
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
        pytest.param(SamplingStrategyEnum.HYPEREDGE, False, 128, id="hyperedge_batch_128"),
        pytest.param(SamplingStrategyEnum.NODE, False, 128, id="node_batch_128"),
        pytest.param(SamplingStrategyEnum.HYPEREDGE, True, 1, id="hyperedge_full"),
        pytest.param(SamplingStrategyEnum.NODE, True, 1, id="node_full"),
    ],
)
def test_model_node2vecslp_precomputed(tmp_path, sampling_strategy, full, batch_size, request):
    test_id = request.node.callspec.id
    num_features = NUM_FEATURES
    metrics = hlp_metrics()

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

    configs = model_configs_with_single_model(
        train_loader,
        val_loader,
        test_loader,
        name="node2vecslp",
        version="precomputed",
        model=precomputed_node2vecslp_module,
    )

    train_test_loop(
        configs=configs,
        path=tmp_path,
        experiment_name=f"node2vecslp_precomputed_integration_test_{test_id}",
    )

    assert (
        tmp_path
        / f"node2vecslp_precomputed_integration_test_{test_id}"
        / "comparison"
        / "overall.tex"
    ).exists()
    assert (
        tmp_path
        / f"node2vecslp_precomputed_integration_test_{test_id}"
        / "comparison"
        / "overall.md"
    ).exists()
    assert (
        tmp_path / f"node2vecslp_precomputed_integration_test_{test_id}" / "comparison" / "test.tex"
    ).exists()
    assert (
        tmp_path / f"node2vecslp_precomputed_integration_test_{test_id}" / "comparison" / "test.md"
    ).exists()
    assert (
        tmp_path / f"node2vecslp_precomputed_integration_test_{test_id}" / "comparison" / "train.md"
    ).exists()
    assert (
        tmp_path
        / f"node2vecslp_precomputed_integration_test_{test_id}"
        / "comparison"
        / "train.tex"
    ).exists()
    assert (
        tmp_path / f"node2vecslp_precomputed_integration_test_{test_id}" / "comparison" / "val.md"
    ).exists()
    assert (
        tmp_path / f"node2vecslp_precomputed_integration_test_{test_id}" / "comparison" / "val.tex"
    ).exists()


@pytest.mark.integration
@pytest.mark.parametrize(
    "sampling_strategy, full, batch_size",
    [
        pytest.param(SamplingStrategyEnum.HYPEREDGE, False, 128, id="hyperedge_batch_128"),
        pytest.param(SamplingStrategyEnum.NODE, False, 128, id="node_batch_128"),
        pytest.param(SamplingStrategyEnum.HYPEREDGE, True, 1, id="hyperedge_full"),
        pytest.param(SamplingStrategyEnum.NODE, True, 1, id="node_full"),
    ],
)
def test_model_node2vecslp_joint(tmp_path, sampling_strategy, full, batch_size, request):
    test_id = request.node.callspec.id
    num_features = NUM_FEATURES
    metrics = hlp_metrics()

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

    train_hyperedge_index = train_dataset.hdata.hyperedge_index
    joint_node2vecslp_module = Node2VecSLPHlpModule(
        encoder_config={
            "mode": "joint",
            "num_features": num_features,
            "node2vec_config": {
                "context_size": 2,
                "walk_length": 5,
                "num_walks_per_node": 2,
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

    configs = model_configs_with_single_model(
        train_loader,
        val_loader,
        test_loader,
        name="node2vecslp",
        version="joint",
        model=joint_node2vecslp_module,
    )

    train_test_loop(
        configs=configs,
        path=tmp_path,
        experiment_name=f"node2vecslp_joint_integration_test_{test_id}",
    )

    assert (
        tmp_path / f"node2vecslp_joint_integration_test_{test_id}" / "comparison" / "overall.tex"
    ).exists()
    assert (
        tmp_path / f"node2vecslp_joint_integration_test_{test_id}" / "comparison" / "overall.md"
    ).exists()
    assert (
        tmp_path / f"node2vecslp_joint_integration_test_{test_id}" / "comparison" / "test.tex"
    ).exists()
    assert (
        tmp_path / f"node2vecslp_joint_integration_test_{test_id}" / "comparison" / "test.md"
    ).exists()
    assert (
        tmp_path / f"node2vecslp_joint_integration_test_{test_id}" / "comparison" / "train.md"
    ).exists()
    assert (
        tmp_path / f"node2vecslp_joint_integration_test_{test_id}" / "comparison" / "train.tex"
    ).exists()
    assert (
        tmp_path / f"node2vecslp_joint_integration_test_{test_id}" / "comparison" / "val.md"
    ).exists()
    assert (
        tmp_path / f"node2vecslp_joint_integration_test_{test_id}" / "comparison" / "val.tex"
    ).exists()
