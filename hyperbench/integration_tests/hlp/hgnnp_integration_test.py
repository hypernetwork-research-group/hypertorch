import pytest
from hyperbench.integration_tests.common import (
    common_standard_metrics,
    enrich_datasets,
    model_configs,
    multi_model_trainer,
    splits_dataset,
    add_negatives,
    loaders,
)
from hyperbench.hlp import HGNNPHlpModule

NUM_FEATURES = 8


@pytest.mark.integration
def test_model_hgnnp():

    num_features = NUM_FEATURES

    metrics = common_standard_metrics()

    train_dataset, val_dataset, test_dataset = splits_dataset()

    train_dataset, val_dataset, test_dataset = add_negatives(
        train_dataset, val_dataset, test_dataset
    )

    enrich_datasets(train_dataset, val_dataset, test_dataset, num_features=num_features)

    train_loader, val_loader, test_loader = loaders(train_dataset, val_dataset, test_dataset)

    mean_hgnnp_module = HGNNPHlpModule(
        encoder_config={
            "in_channels": num_features,
            "hidden_channels": 16,
            "out_channels": 16,
            "bias": True,
            "use_batch_normalization": False,
            "drop_rate": 0.5,
        },
        aggregation="mean",
        lr=0.01,
        weight_decay=5e-4,
        metrics=metrics,
    )

    configs = model_configs(
        train_loader,
        val_loader,
        test_loader,
        name="hgnnp",
        version="mean",
        model=mean_hgnnp_module,
    )

    multi_model_trainer(configs)
