import pytest
from hyperbench.integration_tests.common import (
    common_standard_metrics,
    datasets_enrichers,
    splits_dataset,
    add_negatives,
    loaders,
)
from hyperbench.hlp import CommonNeighborsHlpModule, MLPHlpModule
from hyperbench.types import ModelConfig
from hyperbench.train import MultiModelTrainer

pytestmark = pytest.mark.filterwarnings(
    "ignore:Failing to pass a value to the 'type_params' parameter of 'typing._eval_type' is deprecated.*:DeprecationWarning"
)

NUM_FEATURES = 8


@pytest.mark.integration
def test_model_mlp():
    num_features = NUM_FEATURES
    metrics = common_standard_metrics()

    train_dataset, val_dataset, test_dataset = splits_dataset()

    train_hyperedge_index = train_dataset.hdata.hyperedge_index

    train_dataset, val_dataset, test_dataset = add_negatives(
        train_dataset, val_dataset, test_dataset
    )

    datasets_enrichers(train_dataset, val_dataset, test_dataset, num_features=num_features)

    train_loader_batch_hypergraph, val_loader_batch_hypergraph, test_loader_batch_hypergraph = (
        loaders(train_dataset, val_dataset, test_dataset, batch=True, batch_size=128)
    )

    mean_cn_module = CommonNeighborsHlpModule(
        train_hyperedge_index=train_hyperedge_index,
        aggregation="mean",
        metrics=metrics,
    )

    mean_mlp_module = MLPHlpModule(
        encoder_config={
            "in_channels": num_features,
            "out_channels": num_features,
            "hidden_channels": 64,
            "num_layers": 3,
            "drop_rate": 0.3,
        },
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
        ModelConfig(name="mlp", version="mean", model=mean_mlp_module),
    ]

    with MultiModelTrainer(
        model_configs=configs,
        max_epochs=5,
        accelerator="auto",
        log_every_n_steps=10,
        enable_checkpointing=False,
        auto_start_tensorboard=False,
        auto_wait=True,
    ) as trainer:
        trainer.fit_all(
            train_dataloader=train_loader_batch_hypergraph,
            val_dataloader=val_loader_batch_hypergraph,
            verbose=True,
        )
        trainer.test_all(dataloader=test_loader_batch_hypergraph, verbose=True)
