import pytest

from hypertorch.data import SamplingStrategyEnum
from hypertorch.types import TaskEnum
from hypertorch.integration_tests.common import (
    enrich_datasets,
    loaders,
    model_configs_with_single_model,
    nc_metrics,
    split_dataset,
    train_test_loop,
)
from hypertorch.nc import HNHNClassifier


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
def test_model_hnhn(tmp_path, sampling_strategy, full, batch_size, request):
    test_id = request.node.callspec.id

    train_dataset, val_dataset, test_dataset = split_dataset(
        sampling_strategy,
        task=TaskEnum.NODE_CLASSIFICATION,
        num_classes=NUM_CLASSES,
    )

    enrich_datasets(train_dataset, val_dataset, test_dataset, num_features=NUM_FEATURES)

    train_loader, val_loader, test_loader = loaders(
        train_dataset, val_dataset, test_dataset, batch_size=batch_size, sample_full_hypergraph=full
    )

    hnhn = HNHNClassifier(
        classifier_config={
            "in_channels": NUM_FEATURES,
            "hidden_channels": 8,
            "out_channels": NUM_CLASSES,
            "bias": True,
            "use_batch_normalization": False,
            "drop_rate": 0.5,
        },
        lr=0.001,
        weight_decay=5e-4,
        metrics=nc_metrics(num_classes=NUM_CLASSES),
    )

    configs = model_configs_with_single_model(
        name="hnhn",
        version="nc",
        model=hnhn,
    )

    train_test_loop(
        configs=configs,
        path=tmp_path,
        experiment_name=f"hnhn_nc_integration_test_{test_id}",
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    comparison_path = tmp_path / f"hnhn_nc_integration_test_{test_id}" / "comparison"
    assert (comparison_path / "overall.tex").exists()
    assert (comparison_path / "overall.md").exists()
    assert (comparison_path / "test.tex").exists()
    assert (comparison_path / "test.md").exists()
    assert (comparison_path / "train.md").exists()
    assert (comparison_path / "train.tex").exists()
    assert (comparison_path / "val.md").exists()
    assert (comparison_path / "val.tex").exists()
