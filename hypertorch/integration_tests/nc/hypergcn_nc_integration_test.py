import pytest

from hypertorch.data import SamplingStrategyEnum
from hypertorch.types import TaskEnum
from hypertorch.integration_tests.common import (
    SEED,
    enrich_datasets,
    loaders,
    model_configs_with_single_model,
    nc_metrics,
    split_dataset,
    train_test_loop,
)
from hypertorch.nc import HyperGCNNcModule


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
def test_model_hypergcn_with_mediator(tmp_path, sampling_strategy, full, batch_size, request):
    test_id = request.node.callspec.id
    num_features = NUM_FEATURES
    metrics = nc_metrics(num_classes=NUM_CLASSES)

    train_dataset, val_dataset, test_dataset = split_dataset(
        sampling_strategy,
        task=TaskEnum.NODE_CLASSIFICATION,
        num_classes=NUM_CLASSES,
    )

    enrich_datasets(train_dataset, val_dataset, test_dataset, num_features=num_features)

    train_loader, val_loader, test_loader = loaders(
        train_dataset, val_dataset, test_dataset, batch_size=batch_size, sample_full_hypergraph=full
    )

    hypergcn_nc_module = HyperGCNNcModule(
        classifier_config={
            "in_channels": num_features,
            "out_channels": NUM_CLASSES,
            "hidden_channels": 8,
            "drop_rate": 0.3,
            "use_mediator": True,
            "fast": False,
            "seed": SEED,
        },
        metrics=metrics,
    )

    configs = model_configs_with_single_model(
        train_loader,
        val_loader,
        test_loader,
        name="hypergcn",
        version="nc-with-mediator",
        model=hypergcn_nc_module,
    )

    train_test_loop(
        configs, path=tmp_path, experiment_name=f"hypergcn_mediator_nc_integration_{test_id}"
    )

    comparison_path = tmp_path / f"hypergcn_mediator_nc_integration_{test_id}" / "comparison"
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
def test_model_hypergcn_no_mediator(tmp_path, sampling_strategy, full, batch_size, request):
    test_id = request.node.callspec.id
    num_features = NUM_FEATURES
    metrics = nc_metrics(num_classes=NUM_CLASSES)

    train_dataset, val_dataset, test_dataset = split_dataset(
        sampling_strategy,
        task=TaskEnum.NODE_CLASSIFICATION,
        num_classes=NUM_CLASSES,
    )

    enrich_datasets(train_dataset, val_dataset, test_dataset, num_features=num_features)

    train_loader, val_loader, test_loader = loaders(
        train_dataset, val_dataset, test_dataset, batch_size=batch_size, sample_full_hypergraph=full
    )

    hypergcn_nc_module = HyperGCNNcModule(
        classifier_config={
            "in_channels": num_features,
            "out_channels": NUM_CLASSES,
            "hidden_channels": 8,
            "drop_rate": 0.3,
            "use_mediator": False,
            "fast": False,
            "seed": SEED,
        },
        metrics=metrics,
    )

    configs = model_configs_with_single_model(
        train_loader,
        val_loader,
        test_loader,
        name="hypergcn",
        version="nc-no-mediator",
        model=hypergcn_nc_module,
    )

    train_test_loop(
        configs, path=tmp_path, experiment_name=f"hypergcn_no_mediator_nc_integration_{test_id}"
    )

    comparison_path = tmp_path / f"hypergcn_no_mediator_nc_integration_{test_id}" / "comparison"
    assert (comparison_path / "overall.tex").exists()
    assert (comparison_path / "overall.md").exists()
    assert (comparison_path / "test.tex").exists()
    assert (comparison_path / "test.md").exists()
    assert (comparison_path / "train.md").exists()
    assert (comparison_path / "train.tex").exists()
    assert (comparison_path / "val.md").exists()
    assert (comparison_path / "val.tex").exists()
