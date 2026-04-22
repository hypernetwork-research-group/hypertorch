import copy
import importlib.util
import subprocess
import warnings
import lightning as L

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional
from collections.abc import Iterable
from lightning.pytorch.accelerators import Accelerator
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import CSVLogger, Logger
from lightning.pytorch.profilers import Profiler
from lightning.pytorch.strategies import Strategy
from hyperbench.data import DataLoader
from hyperbench.train.markdown_logger import MarkdownTableLogger
from hyperbench.types import CkptStrategy, ModelConfig, TestResult
from hyperbench.train.latex_logger import LaTexTableLogger


class MultiModelTrainer:
    """
    A trainer class to handle training multiple models with individual trainers.

    Args:
        model_configs: A list of ModelConfig objects, each containing a model and its associated trainer (if any).

        experiment_name: Name for this experiment run's log directory. When ``None`` (default),
            auto-increments as ``experiment_0``, ``experiment_1``, etc. under the log root directory.
            Only used when ``logger`` is not provided.

        accelerator: Supports passing different accelerator types ("cpu", "gpu", "tpu", "hpu", "mps", "auto")
            as well as custom accelerator instances.

        devices: The devices to use. Can be set to a positive number (int or str), a sequence of device indices
            (list or str), the value ``-1`` to indicate all available devices should be used, or ``"auto"`` for
            automatic selection based on the chosen accelerator. Defaults to ``"auto"``.

        strategy: Supports different training strategies with aliases as well custom strategies.
            Defaults to ``"auto"``.

        num_nodes: Number of GPU nodes for distributed training.
            Defaults to ``1``.

        precision: Double precision (64, '64' or '64-true'), full precision (32, '32' or '32-true'),
            16bit mixed precision (16, '16', '16-mixed') or bfloat16 mixed precision ('bf16', 'bf16-mixed').
            Can be used on CPU, GPU, TPUs, or HPUs.
            Defaults to ``'32-true'``.

        max_epochs: Stop training once this number of epochs is reached. Disabled by default (None).
            If both max_epochs and max_steps are not specified, defaults to ``max_epochs = 1000``.
            To enable infinite training, set ``max_epochs = -1``.

        min_epochs: Force training for at least these many epochs. Disabled by default (None).

        max_steps: Stop training after this number of steps. Disabled by default (-1). If ``max_steps = -1``
            and ``max_epochs = None``, will default to ``max_epochs = 1000``. To enable infinite training, set
            ``max_epochs`` to ``-1``.

        min_steps: Force training for at least these number of steps. Disabled by default (``None``).

        check_val_every_n_epoch: Perform a validation loop after every `N` training epochs. If ``None``,
            validation will be done solely based on the number of training batches, requiring ``val_check_interval``
            to be an integer value. When used together with a time-based ``val_check_interval`` and
            ``check_val_every_n_epoch`` > 1, validation is aligned to epoch multiples: if the interval elapses
            before the next multiple-N epoch, validation runs at the start of that epoch (after the first batch)
            and the timer resets; if it elapses during a multiple-N epoch, validation runs after the current batch.
            For ``None`` or ``1`` cases, the time-based behavior of ``val_check_interval`` applies without
            additional alignment.
            Defaults to ``1``.

        logger: Logger (or iterable collection of loggers) for experiment tracking. A ``True`` value uses
            the default ``TensorBoardLogger`` if it is installed, otherwise ``CSVLogger``.
            ``False`` will disable logging. If multiple loggers are provided, local files
            (checkpoints, profiler traces, etc.) are saved in the ``log_dir`` of the first logger.
            Defaults to ``True``.

        default_root_dir: Default path for logs and weights when no logger/ckpt_callback passed.
            Defaults to ``os.getcwd()``.
            Can be remote file paths such as `s3://mybucket/path` or 'hdfs://path/'

        enable_autolog_hparams: Whether to log hyperparameters at the start of a run.
            Defaults to ``True``.

        log_every_n_steps: How often to log within steps.
            Defaults to ``50``.

        profiler: To profile individual steps during training and assist in identifying bottlenecks.
            Defaults to ``None``.

        fast_dev_run: Runs n if set to ``n`` (int) else 1 if set to ``True`` batch(es)
            of train, val and test to find any bugs (ie: a sort of unit test).
            Defaults to ``False``.

        enable_checkpointing: If ``True``, enable checkpointing.
            It will configure a default ModelCheckpoint callback if there is no user-defined ModelCheckpoint in
            :paramref:`~hyperbench.train.MultiModelTrainer.callbacks`.
            Defaults to ``True``.

        enable_progress_bar: Whether to enable the progress bar by default.
            Defaults to ``True``.

        enable_model_summary: Whether to enable model summarization by default.
            Defaults to ``True``.

        callbacks: Add a callback or list of callbacks.
            Defaults to ``None``.

        auto_start_tensorboard: When ``True`` and tensorboard is installed, automatically starts
            a TensorBoard server pointing at the experiment log directory.
            Using this option requires that TensorBoard is installed in the environment and moves control
            of the TensorBoard server lifecycle to the trainer, which will automatically terminate the server
            when the trainer is finalized (e.g., at the end of a `with` block or when the object is garbage collected).
            Enable `auto_wait` to keep the server alive after training completes so you can inspect results before the trainer is finalized.
            Defaults to ``False``.

        tensorboard_port: Port for the auto-launched TensorBoard server.
            Defaults to ``6006``.

        auto_wait: When ``True`` and a TensorBoard server is running, automatically calls
            :meth:`wait` inside `finalize` before terminating the server, so the user
            can inspect results before the process is stopped.
            Defaults to ``False``.
    """

    DEFAULT_BASE_LOG_DIR = "hyperbench_logs"
    EXPERIMENT_NAME_PREFIX = "experiment"
    VERSION_NAME_PREFIX = "version"

    def __init__(
        self,
        model_configs: List[ModelConfig],
        experiment_name: Optional[str] = None,
        # args to pass to each Trainer
        accelerator: str | Accelerator = "auto",
        devices: list[int] | str | int = "auto",
        strategy: str | Strategy = "auto",
        num_nodes: int = 1,
        precision: Optional[
            Any  # Any as Lightning accepts multiple types (int, str, Literal, etc.)
        ] = None,
        max_epochs: Optional[int] = None,
        min_epochs: Optional[int] = None,
        max_steps: int = -1,
        min_steps: Optional[int] = None,
        check_val_every_n_epoch: Optional[int] = 1,
        logger: Optional[Logger | Iterable[Logger] | bool] = None,
        default_root_dir: Optional[str | Path] = None,
        enable_autolog_hparams: bool = True,
        log_every_n_steps: Optional[int] = None,
        profiler: Optional[Profiler | str] = None,
        fast_dev_run: int | bool = False,
        enable_checkpointing: bool = True,
        enable_progress_bar: bool = True,
        enable_model_summary: Optional[bool] = None,
        callbacks: Optional[List[Callback] | Callback] = None,
        auto_start_tensorboard: bool = False,
        tensorboard_port: int = 6006,
        auto_wait: bool = False,
        **kwargs,
    ) -> None:
        self.model_configs = model_configs
        self.log_dir = self.__setup_logdir(default_root_dir, experiment_name)

        self.auto_start_tensorboard = auto_start_tensorboard
        self.auto_wait = auto_wait
        self.tensorboard_port = tensorboard_port
        self.__tensorboard_process: Optional[subprocess.Popen] = None

        for model_config in model_configs:
            if model_config.trainer is None:
                model_logger = self.__setup_logger(model_config, logger)

                model_config.trainer = L.Trainer(
                    accelerator=accelerator,
                    devices=devices,
                    strategy=strategy,
                    num_nodes=num_nodes,
                    precision=precision,
                    max_epochs=max_epochs,
                    min_epochs=min_epochs,
                    max_steps=max_steps,
                    min_steps=min_steps,
                    check_val_every_n_epoch=check_val_every_n_epoch,
                    logger=model_logger,
                    default_root_dir=default_root_dir,
                    enable_autolog_hparams=enable_autolog_hparams,
                    log_every_n_steps=log_every_n_steps,
                    profiler=profiler,
                    fast_dev_run=fast_dev_run,
                    enable_checkpointing=enable_checkpointing,
                    enable_progress_bar=enable_progress_bar,
                    enable_model_summary=enable_model_summary,
                    callbacks=copy.deepcopy(callbacks),
                    **kwargs,
                )

        print(f"Initialized trainer(models: {len(model_configs)}, log_dir: {self.log_dir})")
        self.__auto_start_tensorboard_if_enabled()

    def __enter__(self) -> "MultiModelTrainer":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.finalize()

    def __del__(self) -> None:
        try:
            self.finalize()
        except Exception:
            pass

    @property
    def models(self) -> List[L.LightningModule]:
        return [config.model for config in self.model_configs]

    def model(self, name: str, version: str = "default") -> Optional[L.LightningModule]:
        for config in self.model_configs:
            if config.name == name and config.version == version:
                return config.model
        return None

    def fit_all(
        self,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloader: Optional[DataLoader] = None,
        datamodule: Optional[L.LightningDataModule] = None,
        ckpt_path: Optional[CkptStrategy] = None,
        verbose: bool = True,
    ) -> None:
        if len(self.model_configs) < 1:
            raise ValueError("No models to fit.")

        for i, config in enumerate(self.model_configs):
            if not config.is_trainable:
                if verbose:
                    print(
                        f"Skipping training for model {config.full_model_name()} [{i + 1}/{len(self.model_configs)} models] (is_trainable=False)"
                    )
                continue

            if config.trainer is None:
                raise ValueError(f"Trainer not defined for model {config.full_model_name()}.")

            if verbose:
                print(
                    f"Fit model {config.full_model_name()} [{i + 1}/{len(self.model_configs)} models]"
                )

            train_dataloaders = (
                config.train_dataloader if config.train_dataloader is not None else train_dataloader
            )
            val_dataloaders = (
                config.val_dataloader if config.val_dataloader is not None else val_dataloader
            )
            config.trainer.fit(
                model=config.model,
                train_dataloaders=train_dataloaders,
                val_dataloaders=val_dataloaders,
                datamodule=datamodule,
                ckpt_path=ckpt_path,
            )

    def test_all(
        self,
        dataloader: Optional[DataLoader] = None,
        datamodule: Optional[L.LightningDataModule] = None,
        ckpt_path: Optional[CkptStrategy] = None,
        verbose: bool = True,
        verbose_loop: bool = True,
    ) -> Mapping[str, TestResult]:
        if len(self.model_configs) < 1:
            raise ValueError("No models to test.")

        test_results: Dict[str, TestResult] = {}

        for i, config in enumerate(self.model_configs):
            if config.trainer is None:
                raise ValueError(f"Trainer not defined for model {config.full_model_name()}.")

            if verbose:
                print(
                    f"Test model {config.full_model_name()} [{i + 1}/{len(self.model_configs)} models]"
                )

            test_dataloaders = (
                config.test_dataloader if config.test_dataloader is not None else dataloader
            )
            trainer_test_results: List[TestResult] = config.trainer.test(
                model=config.model,
                dataloaders=test_dataloaders,
                datamodule=datamodule,
                ckpt_path=ckpt_path,
                verbose=verbose_loop,
            )

            # In Lightning, test() returns a list of dicts, one per dataloader, but we use a single dataloader
            test_results[config.full_model_name()] = (
                trainer_test_results[0] if len(trainer_test_results) > 0 else {}
            )

        return test_results

    def __auto_start_tensorboard_if_enabled(self) -> None:
        if self.auto_start_tensorboard:
            if self.__is_tensorboard_available():
                self.__tensorboard_process = self.__start_tensorboard_process()
            else:
                warnings.warn(
                    "TensorBoard is not available. "
                    "Install it with `pip install hyperbench[tensorboard]` or `pip install tensorboard`"
                    "to enable auto-start.",
                    category=UserWarning,
                    stacklevel=2,
                )

    def finalize(self) -> None:
        if self.auto_wait:
            self.wait()
        if self.__tensorboard_process is not None:
            self.__tensorboard_process.terminate()
            self.__tensorboard_process = None

    def wait(self) -> None:
        """
        Wait until the user presses Enter, keeping process alive.
        If no process is running, this method does nothing.
        """
        # For now, we only use this for waiting on TensorBoard, but this can be extended
        # to support waiting for other processes or conditions as needed
        if self.__tensorboard_process is None:
            return

        print(f"TensorBoard is running at http://localhost:{self.tensorboard_port}")

        try:
            input("Press Enter to stop...")
        except (KeyboardInterrupt, EOFError):
            pass

    def __is_tensorboard_available(self) -> bool:
        return importlib.util.find_spec("tensorboard") is not None

    def __start_tensorboard_process(self) -> Optional[subprocess.Popen]:
        try:
            process = subprocess.Popen(
                ["tensorboard", "--logdir", self.log_dir, "--port", str(self.tensorboard_port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(
                f"TensorBoard started at http://localhost:{self.tensorboard_port} (logdir={self.log_dir})"
            )
            return process
        except Exception as e:
            warnings.warn(
                f"Proceeding without starting TensorBoard as it failed: {e}",
                category=UserWarning,
                stacklevel=2,
            )
            return None

    def __next_experiment_name(self, save_dir: Path) -> Path:
        if not save_dir.exists():
            return Path(f"{MultiModelTrainer.EXPERIMENT_NAME_PREFIX}_0")

        existing_experiment_names: List[str] = [
            dir.name
            for dir in save_dir.iterdir()
            if dir.is_dir() and dir.name.startswith(MultiModelTrainer.EXPERIMENT_NAME_PREFIX)
        ]
        if len(existing_experiment_names) < 1:
            return Path(f"{MultiModelTrainer.EXPERIMENT_NAME_PREFIX}_0")

        last_experiment_number = max(
            int(experiment_name.split("_")[1])
            for experiment_name in existing_experiment_names
            if experiment_name.split("_")[1].isdigit()
        )
        return Path(f"{MultiModelTrainer.EXPERIMENT_NAME_PREFIX}_{last_experiment_number + 1}")

    def __setup_logdir(
        self,
        default_root_dir: Optional[str | Path],
        experiment_name: Optional[str],
    ) -> Path:
        base_dir = (
            Path(MultiModelTrainer.DEFAULT_BASE_LOG_DIR)
            if default_root_dir is None
            else Path(default_root_dir)
        )
        next_experiment_name = (
            self.__next_experiment_name(base_dir)
            if experiment_name is None
            else Path(experiment_name)
        )
        return base_dir / next_experiment_name

    def __setup_logger(
        self,
        model_config: ModelConfig,
        logger: Optional[Logger | Iterable[Logger] | bool],
    ) -> Optional[Logger | Iterable[Logger] | bool]:
        if logger is not None:
            return logger

        experiment_name = str(self.__next_experiment_name(self.log_dir))

        loggers: List[Logger] = [
            CSVLogger(
                save_dir=self.log_dir,
                name=model_config.name,
                version=f"{MultiModelTrainer.VERSION_NAME_PREFIX}_{model_config.version}",
            ),
            LaTexTableLogger(
                save_dir=self.log_dir,
                model_name=model_config.full_model_name(),
                experiment_name=experiment_name,
                options={
                    "table_caption": f"Results for Experiments",
                    "sort_by": ["des", "asc"],
                    "border": False,
                },
            ),
            MarkdownTableLogger(
                save_dir=self.log_dir,
                model_name=model_config.full_model_name(),
                experiment_name=experiment_name,
            ),
        ]

        if self.__is_tensorboard_available():
            from lightning.pytorch.loggers import TensorBoardLogger

            loggers.append(
                TensorBoardLogger(
                    save_dir=self.log_dir,
                    name=model_config.name,
                    version=f"{MultiModelTrainer.VERSION_NAME_PREFIX}_{model_config.version}",
                ),
            )

        return loggers
