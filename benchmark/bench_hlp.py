from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAccuracy,
    BinaryAveragePrecision,
    BinaryPrecision,
    BinaryRecall,
)
from hypertorch.train import MultiModelTrainer
from hypertorch.data import DataLoader

from common import (
    load_gcn,
    load_common_neighbors,
    load_hgnn,
    load_hgnnp,
    load_hypergcn_no_mediator,
    load_hypergcn_with_mediator,
    load_mlp,
    load_nhp,
    load_n2v_joint,
    load_villain_node,
    load_villain_hyperedge,
    parse_arguments,
    prepare,
    merge_all_results,
)

if __name__ == "__main__":
    args = parse_arguments()

    run = args.run
    num_workers = args.num_workers
    num_features = args.num_features
    seed = args.seed
    k_nodes = args.k_nodes
    test_set_negative_ratio = args.test_set_negative_ratio
    split_ratios = args.split_ratios
    datasets = args.datasets

    print("Loading and preparing datasets...")
    prepared_datasets = {}
    for r in range(run):
        for dataset_name in datasets:
            picked_seed = seed[r]  # ogni run usa lo stesso seed
            train_dataset, val_dataset, test_dataset, num_nodes, num_features = prepare(
                dataset_name=dataset_name,
                k_nodes=k_nodes,
                split_ratios=split_ratios,
                seed=picked_seed,
                test_set_negative_ratio=test_set_negative_ratio,
            )
            prepared_datasets[dataset_name] = (
                train_dataset,
                val_dataset,
                test_dataset,
                num_nodes,
                num_features,
            )

            print(f"Running benchmark for dataset: {dataset_name}")

            data_loader = DataLoader.from_datasets(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                batch_size=64,
                num_workers=num_workers,
                persistent_workers=True,
            )

            print(f"Run {r + 1}/{run} for dataset: {dataset_name}")

            metrics = MetricCollection(
                {
                    "auc": BinaryAUROC(),
                    "accuracy": BinaryAccuracy(),
                    "avg_precision": BinaryAveragePrecision(),
                    "precision": BinaryPrecision(),
                    "recall": BinaryRecall(),
                }
            )

            list_model = [
                "gcn",
                "common_neighbors",
                "hgnn",
                "hgnnp",
                "hypergcn_no_mediator",
                "hypergcn_with_mediator",
                "mlp",
                "nhp",
                "villain_node",
                "villain_hyperedge",
                "node2vec",
            ]

            max_epochs = 100
            for model in list_model:
                if model == "gcn":
                    config = load_gcn(
                        metrics=metrics,
                        num_features=num_features,
                        train_loader=data_loader.train_dataloader(),
                        val_loader=data_loader.val_dataloader(),
                        test_loader=data_loader.test_dataloader(),
                        num_nodes=num_nodes,
                        num_run=r,
                        max_epochs=max_epochs,
                    )
                elif model == "common_neighbors":
                    config = load_common_neighbors(
                        metrics=metrics,
                        num_features=num_features,
                        train_dataset=train_dataset,
                        test_loader=data_loader.test_dataloader(),
                        num_nodes=num_nodes,
                        num_run=r,
                        max_epochs=max_epochs,
                    )
                elif model == "hgnn":
                    config = load_hgnn(
                        metrics=metrics,
                        num_features=num_features,
                        train_loader=data_loader.train_dataloader(),
                        val_loader=data_loader.val_dataloader(),
                        test_loader=data_loader.test_dataloader(),
                        num_nodes=num_nodes,
                        num_run=r,
                        max_epochs=max_epochs,
                    )
                elif model == "hgnnp":
                    config = load_hgnnp(
                        metrics=metrics,
                        num_features=num_features,
                        train_loader=data_loader.train_dataloader(),
                        val_loader=data_loader.val_dataloader(),
                        test_loader=data_loader.test_dataloader(),
                        num_nodes=num_nodes,
                        num_run=r,
                        max_epochs=max_epochs,
                    )
                elif model == "hypergcn_no_mediator":
                    config = load_hypergcn_no_mediator(
                        metrics=metrics,
                        num_features=num_features,
                        train_loader=data_loader.train_dataloader(),
                        val_loader=data_loader.val_dataloader(),
                        test_loader=data_loader.test_dataloader(),
                        num_nodes=num_nodes,
                        num_run=r,
                        max_epochs=max_epochs,
                    )
                elif model == "hypergcn_with_mediator":
                    config = load_hypergcn_with_mediator(
                        metrics=metrics,
                        num_features=num_features,
                        train_loader=data_loader.train_dataloader(),
                        val_loader=data_loader.val_dataloader(),
                        test_loader=data_loader.test_dataloader(),
                        num_nodes=num_nodes,
                        num_run=r,
                        max_epochs=max_epochs,
                    )
                elif model == "mlp":
                    config = load_mlp(
                        metrics=metrics,
                        num_features=num_features,
                        train_loader=data_loader.train_dataloader(),
                        val_loader=data_loader.val_dataloader(),
                        test_loader=data_loader.test_dataloader(),
                        num_nodes=num_nodes,
                        num_run=r,
                        max_epochs=max_epochs,
                    )
                elif model == "nhp":
                    config = load_nhp(
                        metrics=metrics,
                        num_features=num_features,
                        train_loader=data_loader.train_dataloader(),
                        val_loader=data_loader.val_dataloader(),
                        test_loader=data_loader.test_dataloader(),
                        num_nodes=num_nodes,
                        num_run=r,
                        max_epochs=max_epochs,
                    )
                elif model == "villain_node":
                    config = load_villain_node(
                        metrics=metrics,
                        num_features=num_features,
                        train_loader=data_loader.train_dataloader(),
                        val_loader=data_loader.val_dataloader(),
                        test_loader=data_loader.test_dataloader(),
                        num_nodes=num_nodes,
                        num_run=r,
                        max_epochs=max_epochs,
                    )
                elif model == "villain_hyperedge":
                    config = load_villain_hyperedge(
                        metrics=metrics,
                        num_features=num_features,
                        train_loader=data_loader.train_dataloader(),
                        val_loader=data_loader.val_dataloader(),
                        test_loader=data_loader.test_dataloader(),
                        num_nodes=num_nodes,
                        num_run=r,
                        max_epochs=max_epochs,
                    )
                elif model == "node2vec":
                    config = load_n2v_joint(
                        metrics=metrics,
                        num_features=num_features,
                        train_loader=data_loader.train_dataloader(),
                        val_loader=data_loader.val_dataloader(),
                        test_loader=data_loader.test_dataloader(),
                        num_nodes=num_nodes,
                        num_run=r,
                        train_hyperedge_index=train_dataset.hdata.hyperedge_index,
                        max_epochs=max_epochs,
                    )
                # model = config[0].model
                print("Starting training and evaluation...")

                with MultiModelTrainer(
                    model_configs=config,
                    default_root_dir=f"benchmark/results/{dataset_name}/{config[0].name}",
                ) as trainer:
                    trainer.fit_all(
                        train_dataloader=data_loader.train_dataloader(),
                        val_dataloader=data_loader.val_dataloader(),
                        verbose=True,
                    )
                    trainer.test_all(dataloader=data_loader.test_dataloader(), verbose=True)

                del config

        del prepared_datasets[dataset_name]  # free memory
    print("Merging all results into a single CSV file...")
    merge_all_results(dir_path="benchmark/results", output_file="merged_results.csv")
