import os
import subprocess
import sys
import pytest
import hypertorch.train.trainer as trainer_module

from itertools import count
from pathlib import Path
from textwrap import dedent
from hypertorch.train import MultiModelTrainer
from hypertorch.types import ModelConfig

from hypertorch.integration_tests.common import TinyBinaryClassifier


@pytest.mark.integration
def test_auto_named_trainer_reuses_parent_experiment_dir_in_relaunched_processes(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv("HYPERTORCH_AUTO_EXPERIMENT_DIR_0", raising=False)
    monkeypatch.setattr(trainer_module, "AUTO_EXPERIMENT_INDEX", count())

    parent_trainer = MultiModelTrainer(
        model_configs=[
            ModelConfig(
                name="parent",
                version="0",
                model=TinyBinaryClassifier(),
            )
        ],
        default_root_dir=tmp_path,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    expected_log_dir = (tmp_path / "experiment_0").resolve()

    child_script = dedent(
        """
        import lightning as L
        import os
        from hypertorch.train import MultiModelTrainer
        from hypertorch.types import ModelConfig

        trainer = MultiModelTrainer(
            model_configs=[
                ModelConfig(
                    name="child",
                    version=os.environ["SIMULATED_CHILD_RANK"],
                    model=L.LightningModule(),
                )
            ],
            default_root_dir=os.environ["SIMULATED_LOG_ROOT"],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        print(f"SIMULATED_CHILD_LOG_DIR={trainer.log_dir}")
        """
    )

    child_log_dirs = []
    for child_rank in ("1", "2"):
        child_environment = __inherit_process_environment(
            {
                "SIMULATED_CHILD_RANK": child_rank,
                "SIMULATED_LOG_ROOT": str(tmp_path),
            },
            inherit_auto_experiment_dirs=True,
        )

        try:
            child_process = subprocess.run(
                [sys.executable, "-c", child_script],
                env=child_environment,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            pytest.fail(
                "DDP subprocess failed.\n\n"
                f"Exit code: {e.returncode}\n\n"
                f"STDOUT:\n{e.stdout}\n\n"
                f"STDERR:\n{e.stderr}"
            )

        child_log_dir_line = next(
            line
            for line in child_process.stdout.splitlines()
            if line.startswith("SIMULATED_CHILD_LOG_DIR=")
        )
        child_log_dirs.append(Path(child_log_dir_line.split("=", maxsplit=1)[1]))

    assert parent_trainer.log_dir == expected_log_dir
    assert child_log_dirs == [expected_log_dir, expected_log_dir]
    assert [path.name for path in tmp_path.iterdir()] == ["experiment_0"]


@pytest.mark.integration
def test_multiple_auto_named_trainers_reuse_parent_experiment_dirs_in_relaunched_processes(
    tmp_path,
    monkeypatch,
):
    for trainer_index in range(2):
        monkeypatch.delenv(
            f"HYPERTORCH_AUTO_EXPERIMENT_DIR_{trainer_index}",
            raising=False,
        )
    monkeypatch.setattr(trainer_module, "AUTO_EXPERIMENT_INDEX", count())

    parent_trainers = [
        MultiModelTrainer(
            model_configs=[
                ModelConfig(
                    name=f"parent_{trainer_index}",
                    version="0",
                    model=TinyBinaryClassifier(),
                )
            ],
            default_root_dir=tmp_path,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        for trainer_index in range(2)
    ]
    expected_log_dirs = [
        (tmp_path / f"experiment_{trainer_index}").resolve() for trainer_index in range(2)
    ]

    child_script = dedent(
        """
        import lightning as L
        import os

        from hypertorch.train import MultiModelTrainer
        from hypertorch.types import ModelConfig


        for trainer_index in range(2):
            trainer = MultiModelTrainer(
                model_configs=[
                    ModelConfig(
                        name=f"child_{trainer_index}",
                        version=os.environ["SIMULATED_CHILD_RANK"],
                        model=L.LightningModule(),
                    )
                ],
                default_root_dir=os.environ["SIMULATED_LOG_ROOT"],
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
            )
            print(
                f"SIMULATED_CHILD_LOG_DIR_{trainer_index}={trainer.log_dir}"
            )
        """
    )

    child_log_dirs_by_rank = []
    for child_rank in ("1", "2"):
        child_environment = __inherit_process_environment(
            {
                "SIMULATED_CHILD_RANK": child_rank,
                "SIMULATED_LOG_ROOT": str(tmp_path),
            },
            inherit_auto_experiment_dirs=True,
        )

        try:
            child_process = subprocess.run(
                [sys.executable, "-c", child_script],
                env=child_environment,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            pytest.fail(
                "DDP subprocess failed.\n\n"
                f"Exit code: {e.returncode}\n\n"
                f"STDOUT:\n{e.stdout}\n\n"
                f"STDERR:\n{e.stderr}"
            )

        child_log_dirs = [
            Path(line.split("=", maxsplit=1)[1])
            for line in child_process.stdout.splitlines()
            if line.startswith("SIMULATED_CHILD_LOG_DIR_")
        ]
        child_log_dirs_by_rank.append(child_log_dirs)

    assert [trainer.log_dir for trainer in parent_trainers] == expected_log_dirs
    assert child_log_dirs_by_rank == [expected_log_dirs, expected_log_dirs]

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "experiment_0",
        "experiment_1",
    ]


@pytest.mark.integration
def test_lightning_distributed_training_works_correctly(tmp_path):
    script_path = tmp_path / "cpu_ddp_training.py"
    script_path.write_text(
        dedent(
            """
            import os
            import lightning as L
            import torch
            from torch import Tensor, nn, optim
            from torch.utils.data import DataLoader, TensorDataset
            from hypertorch.train import MultiModelTrainer
            from hypertorch.types import ModelConfig

            class RankDependentLossModel(L.LightningModule):
                def __init__(self) -> None:
                    super().__init__()
                    # DDP and automatic optimization require a gradient-bearing parameter.
                    self.dummy_parameter = nn.Parameter(torch.tensor(0.0))

                def training_step(self, _batch: tuple[Tensor, Tensor], _batch_index: int) -> Tensor:
                    local_loss = 1.0 + (2.0 * self.global_rank)
                    loss = self.dummy_parameter * 0.0 + local_loss
                    self.log(
                        "train/loss",
                        loss,
                        batch_size=2,
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True,
                    )
                    return loss

                def configure_optimizers(self) -> optim.Optimizer:
                    return optim.SGD(self.parameters(), lr=0.01)

            def main() -> None:
                dataloader = DataLoader(
                    TensorDataset(torch.ones(8, 2), torch.zeros(8, 1)),
                    batch_size=2,
                )
                model_configs = [
                    ModelConfig(name="tiny", version="cpu-ddp", model=RankDependentLossModel()),
                ]
                with MultiModelTrainer(
                    model_configs=model_configs,
                    # This is used to provide a unique log directory from the integration test
                    default_root_dir=os.environ["HYPERTORCH_DDP_TEST_ROOT"],
                    accelerator="cpu",
                    devices=2,
                    strategy="ddp",
                    max_epochs=1,
                    enable_checkpointing=False,
                    enable_progress_bar=False,
                    enable_model_summary=False,
                    log_every_n_steps=1,
                ) as trainer:
                    local_rank = os.environ.get("LOCAL_RANK", "0")
                    rank_log_dir_path = (
                        os.environ["HYPERTORCH_DDP_TEST_ROOT"] + f"/rank_{local_rank}.txt"
                    )
                    with open(rank_log_dir_path, "w") as rank_log_dir_file:
                        rank_log_dir_file.write(str(trainer.log_dir))
                    trainer.fit_all(train_dataloader=dataloader, verbose=False)

            if __name__ == "__main__":
                main()
            """
        )
    )

    # For MacOS, we need to set the GLOO_SOCKET_IFNAME environment variable to "lo0"
    # Ref: https://docs.pytorch.org/docs/2.13/distributed.html#choosing-the-network-interface-to-use
    # Start a fresh local job while preserving the system environment required to launch
    # subprocesses. Removing MASTER_PORT prevents reuse of a stale or occupied rendezvous
    # port so Lightning can select a free one. Both ranks rendezvous through the loopback address.
    # On macOS, explicitly binding Gloo to lo0 avoids hosted-runner interface auto-detection
    # selecting an unsuitable adapter and leaving both ranks blocked during process-group
    # initialization.
    child_environment = __inherit_process_environment(
        variables={
            "HYPERTORCH_DDP_TEST_ROOT": str(tmp_path),
            "MASTER_ADDR": "127.0.0.1",
        },
        variables_to_remove=("MASTER_PORT",),
    )
    if sys.platform == "darwin":
        child_environment["GLOO_SOCKET_IFNAME"] = "lo0"

    subprocess.run(
        [sys.executable, str(script_path)],
        check=True,
        env=child_environment,
        timeout=120,
    )

    expected_log_dir = (tmp_path / "experiment_0").resolve()
    rank_log_dir_paths = sorted(tmp_path.glob("rank_*.txt"))

    assert [path.name for path in rank_log_dir_paths] == ["rank_0.txt", "rank_1.txt"]
    assert [Path(path.read_text()) for path in rank_log_dir_paths] == [
        expected_log_dir,
        expected_log_dir,
    ]
    assert [path for path in tmp_path.glob("experiment_*") if path.is_dir()] == [expected_log_dir]
    markdown_results = (expected_log_dir / "comparison" / "overall.md").read_text()
    latex_results = (expected_log_dir / "comparison" / "overall.tex").read_text()

    for results in (markdown_results, latex_results):
        assert "train/loss" in results
        assert "2.0000" in results  # The loss averaged across ranks 0 and 1 should be present
        assert "1.0000" not in results  # The loss from rank 0 should not be present
        assert "3.0000" not in results  # The loss from rank 1 should not be present


def __inherit_process_environment(
    variables: dict[str, str],
    inherit_auto_experiment_dirs: bool = False,
    variables_to_remove: tuple[str, ...] = (),
) -> dict[str, str]:
    environment = os.environ.copy()

    if not inherit_auto_experiment_dirs:
        auto_experiment_dir_prefix = f"{trainer_module.AUTO_EXPERIMENT_DIR_ENV_PREFIX}_"
        environment = {
            name: value
            for name, value in environment.items()
            if not name.startswith(auto_experiment_dir_prefix)
        }

    for variable in variables_to_remove:
        environment.pop(variable, None)

    environment.update(variables)
    return environment
