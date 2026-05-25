import pytest
from hyperbench.integration_tests.common import (
    common_metrics,
    enrich_datasets,
    model_configs,
    multi_model_trainer,
    splits_dataset,
    add_negatives,
    loaders,
    add_model_configs,
)
from hyperbench.hlp import VilLainHlpModule
from hyperbench.data import SamplingStrategy

NUM_FEATURES = 8


@pytest.mark.integration
@pytest.mark.parametrize("sampling_strategy", [SamplingStrategy.HYPEREDGE, SamplingStrategy.NODE])
def test_model_villain(tmp_path, sampling_strategy):
    num_features = NUM_FEATURES
    metrics = common_metrics()

    train_dataset, val_dataset, test_dataset = splits_dataset(sampling_strategy)
    train_dataset, val_dataset, test_dataset = add_negatives(
        train_dataset, val_dataset, test_dataset
    )

    enrich_datasets(train_dataset, val_dataset, test_dataset, num_features=num_features)

    train_loader, val_loader, test_loader = loaders(train_dataset, val_dataset, test_dataset)

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
        train_loader,
        val_loader,
        test_loader,
        name="villain",
        version="node_maxmin",
        model=node_villain_module,
    )

    configs = add_model_configs(
        configs,
        train_loader,
        val_loader,
        test_loader,
        name="villain",
        version="hyperedge",
        model=hyperedge_villain_module,
    )

    multi_model_trainer(configs, path=tmp_path, experiment_name="villain_integration_test")

    assert (tmp_path / "villain_integration_test" / "comparison" / "overall.tex").exists()
    assert (tmp_path / "villain_integration_test" / "comparison" / "test.tex").exists()
    assert (tmp_path / "villain_integration_test" / "comparison" / "results.md").exists()
