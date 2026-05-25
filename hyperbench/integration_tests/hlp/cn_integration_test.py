import pytest
from hyperbench.integration_tests.common import (
    common_metrics,
    enrich_datasets,
    splits_dataset,
    add_negatives,
    loaders,
)
from hyperbench.hlp import CommonNeighborsHlpModule
from hyperbench.types import ModelConfig
from hyperbench.train import MultiModelTrainer
from hyperbench.data import SamplingStrategy

NUM_FEATURES = 8


@pytest.mark.integration
@pytest.mark.parametrize("sampling_strategy", [SamplingStrategy.HYPEREDGE, SamplingStrategy.NODE])
def test_model_cn(tmp_path, sampling_strategy):
    num_features = NUM_FEATURES
    metrics = common_metrics()

    train_dataset, val_dataset, test_dataset = splits_dataset(sampling_strategy)

    train_hyperedge_index = train_dataset.hdata.hyperedge_index

    train_dataset, val_dataset, test_dataset = add_negatives(
        train_dataset, val_dataset, test_dataset
    )

    enrich_datasets(train_dataset, val_dataset, test_dataset, num_features=num_features)

    train_loader, val_loader, test_loader = loaders(
        train_dataset, val_dataset, test_dataset, batch=True, batch_size=128
    )

    mean_cn_module = CommonNeighborsHlpModule(
        train_hyperedge_index=train_hyperedge_index,
        aggregation="mean",
        metrics=metrics,
    )

    configs = [
        ModelConfig(
            name="common_neighbors",
            version="mean",
            model=mean_cn_module,
            is_trainable=False,
        ),
    ]

    with MultiModelTrainer(
        model_configs=configs,
        max_epochs=3,
        accelerator="auto",
        log_every_n_steps=1,
        enable_checkpointing=False,
        auto_start_tensorboard=False,
        auto_wait=True,
        default_root_dir=tmp_path,
        experiment_name="cn_integration_test",
    ) as trainer:
        trainer.fit_all(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            verbose=True,
        )
        trainer.test_all(dataloader=test_loader, verbose=True)

    assert (tmp_path / "cn_integration_test" / "comparison" / "overall.tex").exists()
    assert (tmp_path / "cn_integration_test" / "comparison" / "test.tex").exists()
    assert (tmp_path / "cn_integration_test" / "comparison" / "results.md").exists()
