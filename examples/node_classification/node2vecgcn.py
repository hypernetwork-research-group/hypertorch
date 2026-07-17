from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAUROC, MulticlassAccuracy, MulticlassF1Score

from hypertorch.data import AlgebraDataset, DataLoader, Node2VecEnricher
from hypertorch.node_classification import (
    Node2VecGCNNCConfig,
    Node2VecGCNClassifierConfig,
    Node2VecGCNClassifier,
)
from hypertorch.train import MultiModelTrainer
from hypertorch.types import ModelConfig
from hypertorch.utils import node_labels_from_node_degrees


if __name__ == "__main__":
    verbose = False
    num_workers = 8
    num_features = 32
    num_classes = 3
    metrics = MetricCollection(
        {
            "auc": MulticlassAUROC(num_classes=num_classes),
            "accuracy": MulticlassAccuracy(num_classes=num_classes),
            "f1": MulticlassF1Score(num_classes=num_classes),
        }
    )

    print("Loading and preparing dataset...")

    dataset = AlgebraDataset(sampling_strategy="node", task="node-classification")
    dataset.hdata.y = node_labels_from_node_degrees(
        node_incidences=dataset.hdata.hyperedge_index[0],
        num_nodes=dataset.hdata.num_nodes,
        num_classes=num_classes,
    )

    train_dataset, val_dataset, test_dataset = dataset.split(
        ratios=[0.7, 0.1, 0.2],
        node_space_setting="transductive",
        shuffle=True,
        seed=42,
    )

    print("Computing Node2Vec embeddings from the train graph...")

    node2vec_enricher = Node2VecEnricher(
        num_features=num_features,
        context_size=10,
        walk_length=20,
        num_walks_per_node=10,
        num_negative_samples=1,
        num_nodes=dataset.hdata.num_nodes,
        num_epochs=10,
        learning_rate=0.01,
        batch_size=128,
        sparse=False,
        verbose=verbose,
    )
    train_dataset.enrich_node_features(
        enricher=node2vec_enricher,
        enrichment_mode="replace",
    )
    val_dataset.enrich_node_features_from(train_dataset)
    test_dataset.enrich_node_features_from(train_dataset)

    print("Creating dataloaders...")

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_dataset,
        sample_full_hypergraph=True,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
    )

    gcn_config: Node2VecGCNNCConfig = {
        "out_channels": num_classes,
        "hidden_channels": num_features,
        "num_layers": 2,
        "drop_rate": 0.1,
        "bias": True,
        "improved": False,
        "add_self_loops": True,
        "normalize": True,
        "cached": False,
        "graph_reduction_strategy": "clique_expansion",
        "num_nodes": dataset.hdata.num_nodes,
    }
    precomputed_config: Node2VecGCNClassifierConfig = {
        "mode": "precomputed",
        "num_features": num_features,
        "node2vec_config": {},
        "gcn_config": gcn_config,
    }

    node2vecgcn_precomputed = Node2VecGCNClassifier(
        classifier_config=precomputed_config,
        lr=0.001,
        weight_decay=0.0,
        metrics=metrics,
    )

    node2vecgcn_joint = Node2VecGCNClassifier(
        classifier_config={
            "mode": "joint",
            "num_features": num_features,
            "node2vec_config": {
                "context_size": 10,
                "walk_length": 20,
                "num_walks_per_node": 10,
                "p": 1.0,
                "q": 1.0,
                "num_negative_samples": 1,
                "train_hyperedge_index": train_dataset.hdata.hyperedge_index,
                "num_nodes": dataset.hdata.num_nodes,
                "graph_reduction_strategy": "clique_expansion",
                "random_walk_batch_size": 128,
                "node2vec_loss_weight": 0.4,
            },
            "gcn_config": gcn_config,
        },
        lr=0.001,
        weight_decay=0.0,
        metrics=metrics,
    )

    configs = [
        ModelConfig(
            name="node2vecgcn-precomputed",
            version="node-classification",
            model=node2vecgcn_precomputed,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
        ),
        ModelConfig(
            name="node2vecgcn-joint",
            version="node-classification",
            model=node2vecgcn_joint,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
        ),
    ]

    print("Starting training and evaluation...")

    with MultiModelTrainer(
        model_configs=configs,
        max_epochs=60,
        accelerator="auto",
        log_every_n_steps=1,
        enable_checkpointing=False,
        devices=1,
        test_devices=1,
        enable_model_summary=True,
    ) as trainer:
        trainer.fit_all(train_dataloader=train_loader, val_dataloader=val_loader, verbose=True)
        trainer.test_all(dataloader=test_loader, verbose=True)

    print("Complete!")
