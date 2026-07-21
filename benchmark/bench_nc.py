import pandas as pd
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAUROC, MulticlassAccuracy, MulticlassF1Score

from hypertorch.train import MultiModelTrainer
from hypertorch.data import DataLoader

from common_nc import (
    load_gcn,
    load_common_neighbors,
    load_hgnn,
    load_hgnnp,
    load_hypergcn_no_mediator,
    load_hypergcn_with_mediator,
    load_mlp,
    load_hnhn,
    # load_nhp,
    load_n2v_joint,
    load_villain_node,
    # load_villain_hyperedge,
    collect_hw_stats_row,
    retrieve_hw_stats,
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
    task = args.task

    print("Running task:", task)

    print("Loading and preparing datasets...")
    prepared_datasets = {}

    hw_stats_df = pd.DataFrame(
        columns=pd.Index(
            [
                "run",
                "dataset",
                "model",
                "cpu_usage_before",
                "ram_usage_before",
                "gpu_usage_before",
                "cpu_usage_after",
                "ram_usage_after",
                "gpu_usage_after",
                "cpu_usage_diff",
                "ram_usage_diff",
                "gpu_usage_diff",
            ]
        )
    )

    for r in range(run):
        for dataset_name in datasets:
            picked_seed = seed[r]
            train_dataset, val_dataset, test_dataset, num_nodes, num_features = prepare(
                dataset_name=dataset_name,
                k_nodes=k_nodes,
                split_ratios=split_ratios,
                seed=picked_seed,
                test_set_negative_ratio=test_set_negative_ratio,
                task=task,
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
                test_dataset=None,
                batch_size=64,
                num_workers=num_workers,
                persistent_workers=True,
            )
            test_loader = DataLoader(
                dataset=test_dataset,
                sample_full_hypergraph=True,
                shuffle=False,
                num_workers=num_workers,
                persistent_workers=True,
            )

            print(f"Run {r + 1}/{run} for dataset: {dataset_name}")
            num_classes = int(train_dataset.hdata.y.max().item() + 1)
            metrics = MetricCollection(
                {
                    "auc": MulticlassAUROC(num_classes=num_classes),
                    "accuracy": MulticlassAccuracy(num_classes=num_classes),
                    "f1": MulticlassF1Score(num_classes=num_classes),
                }
            )

            list_model = [
                # "gcn",
                "common_neighbors",
                # "hgnn",
                # "hgnnp",
                # "hnhn",
                # "hypergcn_no_mediator",
                # "hypergcn_with_mediator",
                # "mlp",
                # "villain_node",
                # "node2vec",
                # "villain_hyperedge",
                # "nhp",
            ]

            for model in list_model:
                if model == "gcn":
                    config = load_gcn(
                        metrics=metrics,
                        num_features=num_features,
                        train_loader=data_loader.train_dataloader(),
                        val_loader=data_loader.val_dataloader(),
                        test_loader=test_loader,
                        num_nodes=num_nodes,
                        num_run=r,
                        max_epochs=60,
                        num_classes=num_classes,
                    )
                elif model == "common_neighbors":
                    config = load_common_neighbors(
                        metrics=metrics,
                        num_features=num_features,
                        train_dataset=train_dataset,
                        test_loader=test_loader,
                        num_nodes=num_nodes,
                        num_run=r,
                        max_epochs=0,
                        num_classes=num_classes,
                    )
                elif model == "hgnn":
                    config = load_hgnn(
                        metrics=metrics,
                        num_features=num_features,
                        train_loader=data_loader.train_dataloader(),
                        val_loader=data_loader.val_dataloader(),
                        test_loader=test_loader,
                        num_nodes=num_nodes,
                        num_run=r,
                        max_epochs=60,
                        num_classes=num_classes,
                    )
                elif model == "hgnnp":
                    config = load_hgnnp(
                        metrics=metrics,
                        num_features=num_features,
                        train_loader=data_loader.train_dataloader(),
                        val_loader=data_loader.val_dataloader(),
                        test_loader=test_loader,
                        num_nodes=num_nodes,
                        num_run=r,
                        max_epochs=60,
                        num_classes=num_classes,
                    )
                elif model == "hnhn":
                    config = load_hnhn(
                        metrics=metrics,
                        num_features=num_features,
                        train_loader=data_loader.train_dataloader(),
                        val_loader=data_loader.val_dataloader(),
                        test_loader=test_loader,
                        num_nodes=num_nodes,
                        num_run=r,
                        max_epochs=200,
                        num_classes=num_classes,
                    )
                elif model == "hypergcn_no_mediator":
                    config = load_hypergcn_no_mediator(
                        metrics=metrics,
                        num_features=num_features,
                        train_loader=data_loader.train_dataloader(),
                        val_loader=data_loader.val_dataloader(),
                        test_loader=test_loader,
                        num_nodes=num_nodes,
                        num_run=r,
                        max_epochs=100,
                        num_classes=num_classes,
                    )
                elif model == "hypergcn_with_mediator":
                    config = load_hypergcn_with_mediator(
                        metrics=metrics,
                        num_features=num_features,
                        train_loader=data_loader.train_dataloader(),
                        val_loader=data_loader.val_dataloader(),
                        test_loader=test_loader,
                        num_nodes=num_nodes,
                        num_run=r,
                        max_epochs=100,
                        num_classes=num_classes,
                    )
                elif model == "mlp":
                    config = load_mlp(
                        metrics=metrics,
                        num_features=num_features,
                        train_loader=data_loader.train_dataloader(),
                        val_loader=data_loader.val_dataloader(),
                        test_loader=test_loader,
                        num_nodes=num_nodes,
                        num_run=r,
                        max_epochs=100,
                        num_classes=num_classes,
                    )
                # elif model == "nhp":
                #     config = load_nhp(
                #         metrics=metrics,
                #         num_features=num_features,
                #         train_loader=data_loader.train_dataloader(),
                #         val_loader=data_loader.val_dataloader(),
                #         test_loader=test_loader,
                #         num_nodes=num_nodes,
                #         num_run=r,
                #         max_epochs=max_epochs,
                #     )
                elif model == "villain_node":
                    config = load_villain_node(
                        metrics=metrics,
                        num_features=num_features,
                        train_loader=data_loader.train_dataloader(),
                        val_loader=data_loader.val_dataloader(),
                        test_loader=test_loader,
                        num_nodes=num_nodes,
                        num_run=r,
                        max_epochs=100,
                        num_classes=num_classes,
                    )
                # elif model == "villain_hyperedge":
                #     config = load_villain_hyperedge(
                #         metrics=metrics,
                #         num_features=num_features,
                #         train_loader=data_loader.train_dataloader(),
                #         val_loader=data_loader.val_dataloader(),
                #         test_loader=test_loader,
                #         num_nodes=num_nodes,
                #         num_run=r,
                #         max_epochs=max_epochs,
                #     )
                elif model == "node2vec":
                    config = load_n2v_joint(
                        metrics=metrics,
                        num_features=num_features,
                        train_loader=data_loader.train_dataloader(),
                        val_loader=data_loader.val_dataloader(),
                        test_loader=test_loader,
                        num_nodes=num_nodes,
                        num_run=r,
                        train_hyperedge_index=train_dataset.hdata.hyperedge_index,
                        max_epochs=60,
                        num_classes=num_classes,
                    )
                print("Starting training and evaluation...")

                with MultiModelTrainer(
                    experiment_name=f"{model}_{r}",
                    model_configs=config,
                    default_root_dir=f"benchmark/results_nc/{dataset_name}/",
                ) as trainer:
                    before_stats = retrieve_hw_stats()

                    trainer.fit_all(
                        train_dataloader=data_loader.train_dataloader(),
                        val_dataloader=data_loader.val_dataloader(),
                        verbose=True,
                    )
                    trainer.test_all(dataloader=test_loader, verbose=True)
                    after_stats = retrieve_hw_stats()
                    hw_stats_df = pd.concat(
                        [
                            hw_stats_df,
                            pd.DataFrame(
                                [
                                    collect_hw_stats_row(
                                        run=r,
                                        dataset=dataset_name,
                                        model=model,
                                        before_stats=before_stats,
                                        after_stats=after_stats,
                                    )
                                ]
                            ),
                        ],
                        ignore_index=True,
                    )
                del config

        del prepared_datasets[dataset_name]  # free memory
    print("Merging all results into a single CSV file...")
    merge_all_results(dir_path="benchmark/results_nc", output_file="merged_results.csv")
    hw_stats_df.to_csv("benchmark/results_nc/hw_usage_stats.csv", index=False)
