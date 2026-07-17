import argparse

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
    prepare,
    merge_all_results,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-features", type=int, default=32)
    parser.add_argument("--seed", type=int, nargs=3, default=[42, 43, 44])
    parser.add_argument("--k-nodes", type=int, default=2)
    parser.add_argument("--test-set-negative-ratio", type=float, default=0.6)
    parser.add_argument("--split-ratios", type=float, nargs=3, default=[0.7, 0.1, 0.2])
    parser.add_argument("--datasets", nargs="+", default=["cora", "citeseer", "pubmed"])
    args = parser.parse_args()

    run = args.run
    num_workers = args.num_workers
    num_features = args.num_features
    seed = args.seed
    k_nodes = args.k_nodes
    test_set_negative_ratio = args.test_set_negative_ratio
    split_ratios = args.split_ratios
    datasets = args.datasets
    print(
        f"Running benchmark with the following parameters:\n"
        f"num_workers: {num_workers}\n"
        f"num_features: {num_features}\n"
        f"seed: {seed}\n"
        f"k_nodes: {k_nodes}\n"
        f"test_set_negative_ratio: {test_set_negative_ratio}\n"
        f"split_ratios: {split_ratios}\n"
        f"datasets: {datasets}\n"
    )

    print("Loading and preparing datasets...")
    prepared_datasets = {}
    for dataset_name in datasets:
        picked_seed = seed[datasets.index(dataset_name) % len(seed)]  # ogni run usa lo stesso seed
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

        print("Creating dataloaders...")
        train_loader = DataLoader(
            train_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=64,
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

        for r in range(run):
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
                "node2vec_joint",
            ]

            for model in list_model:
                if model == "gcn":
                    config = load_gcn(
                        metrics=metrics,
                        num_features=num_features,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        test_loader=test_loader,
                        num_nodes=num_nodes,
                        num_run=r,
                    )
                elif model == "common_neighbors":
                    config = load_common_neighbors(
                        metrics=metrics,
                        num_features=num_features,
                        train_dataset=train_dataset,
                        test_loader=test_loader,
                        num_nodes=num_nodes,
                        num_run=r,
                    )
                elif model == "hgnn":
                    config = load_hgnn(
                        metrics=metrics,
                        num_features=num_features,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        test_loader=test_loader,
                        num_nodes=num_nodes,
                        num_run=r,
                    )
                elif model == "hgnnp":
                    config = load_hgnnp(
                        metrics=metrics,
                        num_features=num_features,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        test_loader=test_loader,
                        num_nodes=num_nodes,
                        num_run=r,
                    )
                elif model == "hypergcn_no_mediator":
                    config = load_hypergcn_no_mediator(
                        metrics=metrics,
                        num_features=num_features,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        test_loader=test_loader,
                        num_nodes=num_nodes,
                        num_run=r,
                    )
                elif model == "hypergcn_with_mediator":
                    config = load_hypergcn_with_mediator(
                        metrics=metrics,
                        num_features=num_features,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        test_loader=test_loader,
                        num_nodes=num_nodes,
                        num_run=r,
                    )
                elif model == "mlp":
                    config = load_mlp(
                        metrics=metrics,
                        num_features=num_features,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        test_loader=test_loader,
                        num_nodes=num_nodes,
                        num_run=r,
                    )
                elif model == "nhp":
                    config = load_nhp(
                        metrics=metrics,
                        num_features=num_features,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        test_loader=test_loader,
                        num_nodes=num_nodes,
                        num_run=r,
                    )
                elif model == "villain_node":
                    config = load_villain_node(
                        metrics=metrics,
                        num_features=num_features,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        test_loader=test_loader,
                        num_nodes=num_nodes,
                        num_run=r,
                    )
                elif model == "villain_hyperedge":
                    config = load_villain_hyperedge(
                        metrics=metrics,
                        num_features=num_features,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        test_loader=test_loader,
                        num_nodes=num_nodes,
                        num_run=r,
                    )
                elif model == "node2vec_joint":
                    config = load_n2v_joint(
                        metrics=metrics,
                        num_features=num_features,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        test_loader=test_loader,
                        num_nodes=num_nodes,
                        num_run=r,
                        train_hyperedge_index=train_dataset.hdata.hyperedge_index,
                    )
                # model = config[0].model
                print("Starting training and evaluation...")

                with MultiModelTrainer(
                    model_configs=config,
                    default_root_dir=f"benchmark/results/{dataset_name}/{config[0].name}",
                ) as trainer:
                    trainer.fit_all(
                        train_dataloader=train_loader,
                        val_dataloader=val_loader,
                        verbose=True,
                    )
                    trainer.test_all(dataloader=test_loader, verbose=True)

                del config
        del prepared_datasets[dataset_name]  # free memory
    print("Merging all results into a single CSV file...")
    merge_all_results(dir_path="benchmark/results", output_file="merged_results.csv")
