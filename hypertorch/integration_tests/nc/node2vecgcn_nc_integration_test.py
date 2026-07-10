import pytest

from hypertorch.data import SamplingStrategyEnum
from hypertorch.integration_tests.common import (
    enrich_datasets,
    loaders,
    model_configs_with_single_model,
    nc_metrics,
    split_dataset,
    train_test_loop,
)
from hypertorch.nc import Node2VecGCNEncoderConfig, Node2VecGCNNcModule
from hypertorch.types import TaskEnum


NUM_CLASSES = 3
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
def test_model_node2vecgcn_precomputed(tmp_path, sampling_strategy, full, batch_size, request):
    test_id = request.node.callspec.id

    train_dataset, val_dataset, test_dataset = split_dataset(
        sampling_strategy,
        task=TaskEnum.NODE_CLASSIFICATION,
        num_classes=NUM_CLASSES,
    )

    enrich_datasets(train_dataset, val_dataset, test_dataset, num_features=NUM_FEATURES)

    train_loader, val_loader, test_loader = loaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=batch_size,
        sample_full_hypergraph=full,
    )

    classifier_config: Node2VecGCNEncoderConfig = {
        "mode": "precomputed",
        "num_features": NUM_FEATURES,
        "node2vec_config": {},
        "gcn_config": {
            "out_channels": NUM_CLASSES,
            "hidden_channels": NUM_FEATURES,
            "num_layers": 2,
            "drop_rate": 0.1,
            "bias": True,
            "improved": False,
            "add_self_loops": True,
            "normalize": True,
            "cached": False,
            "graph_reduction_strategy": "clique_expansion",
            "num_nodes": train_dataset.hdata.num_nodes,
        },
    }

    precomputed_node2vecgcn = Node2VecGCNNcModule(
        classifier_config=classifier_config,
        lr=0.001,
        weight_decay=0.0,
        metrics=nc_metrics(num_classes=NUM_CLASSES),
    )

    configs = model_configs_with_single_model(
        name="node2vecgcn-precomputed",
        version="nc",
        model=precomputed_node2vecgcn,
    )

    train_test_loop(
        configs=configs,
        path=tmp_path,
        experiment_name=f"node2vecgcn_precomputed_nc_integration_test_{test_id}",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    comparison_path = (
        tmp_path / f"node2vecgcn_precomputed_nc_integration_test_{test_id}" / "comparison"
    )
    assert (comparison_path / "overall.tex").exists()
    assert (comparison_path / "overall.md").exists()
    assert (comparison_path / "test.tex").exists()
    assert (comparison_path / "test.md").exists()
    assert (comparison_path / "train.md").exists()
    assert (comparison_path / "train.tex").exists()
    assert (comparison_path / "val.md").exists()
    assert (comparison_path / "val.tex").exists()


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
def test_model_node2vecgcn_joint(tmp_path, sampling_strategy, full, batch_size, request):
    test_id = request.node.callspec.id

    train_dataset, val_dataset, test_dataset = split_dataset(
        sampling_strategy,
        task=TaskEnum.NODE_CLASSIFICATION,
        num_classes=NUM_CLASSES,
    )

    enrich_datasets(train_dataset, val_dataset, test_dataset, num_features=NUM_FEATURES)

    train_loader, val_loader, test_loader = loaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=batch_size,
        sample_full_hypergraph=full,
    )

    classifier_config: Node2VecGCNEncoderConfig = {
        "mode": "joint",
        "num_features": NUM_FEATURES,
        "node2vec_config": {
            "context_size": 2,
            "walk_length": 5,
            "num_walks_per_node": 2,
            "p": 1.0,
            "q": 1.0,
            "num_negative_samples": 1,
            "train_hyperedge_index": train_dataset.hdata.hyperedge_index,
            "num_nodes": train_dataset.hdata.num_nodes,
            "graph_reduction_strategy": "clique_expansion",
            "random_walk_batch_size": 128,
            "node2vec_loss_weight": 0.4,
        },
        "gcn_config": {
            "out_channels": NUM_CLASSES,
            "hidden_channels": NUM_FEATURES,
            "num_layers": 2,
            "drop_rate": 0.1,
            "bias": True,
            "improved": False,
            "add_self_loops": True,
            "normalize": True,
            "cached": False,
            "graph_reduction_strategy": "clique_expansion",
            "num_nodes": train_dataset.hdata.num_nodes,
        },
    }

    joint_node2vecgcn = Node2VecGCNNcModule(
        classifier_config=classifier_config,
        lr=0.001,
        weight_decay=0.0,
        metrics=nc_metrics(num_classes=NUM_CLASSES),
    )

    configs = model_configs_with_single_model(
        name="node2vecgcn-joint",
        version="nc",
        model=joint_node2vecgcn,
    )

    train_test_loop(
        configs=configs,
        path=tmp_path,
        experiment_name=f"node2vecgcn_joint_nc_integration_test_{test_id}",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    comparison_path = tmp_path / f"node2vecgcn_joint_nc_integration_test_{test_id}" / "comparison"
    assert (comparison_path / "overall.tex").exists()
    assert (comparison_path / "overall.md").exists()
    assert (comparison_path / "test.tex").exists()
    assert (comparison_path / "test.md").exists()
    assert (comparison_path / "train.md").exists()
    assert (comparison_path / "train.tex").exists()
    assert (comparison_path / "val.md").exists()
    assert (comparison_path / "val.tex").exists()
