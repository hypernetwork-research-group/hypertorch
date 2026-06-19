import pytest

from hypertorch.hlp import VilLainHlpModule
from hypertorch.data import SamplingStrategy
from hypertorch.integration_tests.common import (
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
def test_model_villain_node(tmp_path, sampling_strategy, full, batch_size, request):
    test_id = request.node.callspec.id
    num_features = NUM_FEATURES
    metrics = common_metrics()

    train_dataset, val_dataset, test_dataset = split_dataset(sampling_strategy)
    train_dataset, val_dataset, test_dataset = add_negatives(
        train_dataset, val_dataset, test_dataset
    )

    enrich_datasets(train_dataset, val_dataset, test_dataset, num_features=num_features)

    train_loader, val_loader, test_loader = loaders(
        train_dataset, val_dataset, test_dataset, batch_size=batch_size, sample_full_hypergraph=full
    )

    node_villain_module = VilLainHlpModule(
        encoder_config={
            "num_nodes": train_dataset.hdata.num_nodes,
            "embedding_dim": 64,
            "labels_per_subspace": 2,
            "training_steps": 2,
            "generation_steps": 4,
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

    configs = model_configs_with_single_model(
        train_loader,
        val_loader,
        test_loader,
        name="villain",
        version="node_maxmin",
        model=node_villain_module,
    )

    train_test_loop(
        configs, path=tmp_path, experiment_name=f"villain_node_integration_test_{test_id}"
    )

    assert (
        tmp_path / f"villain_node_integration_test_{test_id}" / "comparison" / "overall.tex"
    ).exists()
    assert (
        tmp_path / f"villain_node_integration_test_{test_id}" / "comparison" / "overall.md"
    ).exists()
    assert (
        tmp_path / f"villain_node_integration_test_{test_id}" / "comparison" / "test.tex"
    ).exists()
    assert (
        tmp_path / f"villain_node_integration_test_{test_id}" / "comparison" / "test.md"
    ).exists()
    assert (
        tmp_path / f"villain_node_integration_test_{test_id}" / "comparison" / "train.md"
    ).exists()
    assert (
        tmp_path / f"villain_node_integration_test_{test_id}" / "comparison" / "train.tex"
    ).exists()
    assert (
        tmp_path / f"villain_node_integration_test_{test_id}" / "comparison" / "val.md"
    ).exists()
    assert (
        tmp_path / f"villain_node_integration_test_{test_id}" / "comparison" / "val.tex"
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
def test_model_villain_hyperedge(tmp_path, sampling_strategy, full, batch_size, request):
    test_id = request.node.callspec.id
    num_features = NUM_FEATURES
    metrics = common_metrics()

    train_dataset, val_dataset, test_dataset = split_dataset(sampling_strategy)
    train_dataset, val_dataset, test_dataset = add_negatives(
        train_dataset, val_dataset, test_dataset
    )

    enrich_datasets(train_dataset, val_dataset, test_dataset, num_features=num_features)

    train_loader, val_loader, test_loader = loaders(
        train_dataset, val_dataset, test_dataset, batch_size=batch_size, sample_full_hypergraph=full
    )

    hyperedge_villain_module = VilLainHlpModule(
        encoder_config={
            "num_nodes": train_dataset.hdata.num_nodes,
            "embedding_dim": 64,
            "labels_per_subspace": 2,
            "training_steps": 2,
            "generation_steps": 4,
            "tau": 1.0,
            "eps": 1e-10,
            "villain_loss_weight": 1.0,
        },
        embedding_mode="hyperedge",
        lr=0.01,
        weight_decay=0.0,
        metrics=metrics,
    )

    configs = model_configs_with_single_model(
        train_loader,
        val_loader,
        test_loader,
        name="villain",
        version="hyperedge_maxmin",
        model=hyperedge_villain_module,
    )

    train_test_loop(
        configs, path=tmp_path, experiment_name=f"villain_hyperedge_integration_test_{test_id}"
    )

    assert (
        tmp_path / f"villain_hyperedge_integration_test_{test_id}" / "comparison" / "overall.tex"
    ).exists()
    assert (
        tmp_path / f"villain_hyperedge_integration_test_{test_id}" / "comparison" / "overall.md"
    ).exists()
    assert (
        tmp_path / f"villain_hyperedge_integration_test_{test_id}" / "comparison" / "test.tex"
    ).exists()
    assert (
        tmp_path / f"villain_hyperedge_integration_test_{test_id}" / "comparison" / "test.md"
    ).exists()
    assert (
        tmp_path / f"villain_hyperedge_integration_test_{test_id}" / "comparison" / "train.md"
    ).exists()
    assert (
        tmp_path / f"villain_hyperedge_integration_test_{test_id}" / "comparison" / "train.tex"
    ).exists()
    assert (
        tmp_path / f"villain_hyperedge_integration_test_{test_id}" / "comparison" / "val.md"
    ).exists()
    assert (
        tmp_path / f"villain_hyperedge_integration_test_{test_id}" / "comparison" / "val.tex"
    ).exists()
