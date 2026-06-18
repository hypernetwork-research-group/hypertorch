from __future__ import annotations

import copy
import importlib.util
import shutil
import subprocess
import warnings
import lightning as L
import torch

from pathlib import Path
from typing import Any
from collections.abc import Iterable, Mapping
from lightning.pytorch.accelerators import Accelerator
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, Logger
from lightning.pytorch.profilers import Profiler
from lightning.pytorch.strategies import Strategy
from hyperbench.data import DataLoader
from hyperbench.types import CkptStrategy, ModelConfig, TestResult
from hyperbench.utils import validate_is_non_empty, validate_is_non_negative

from hyperbench.train.markdown_logger import MarkdownTableLogger
from hyperbench.train.latex_logger import LaTexTableLogger


class MultiModelTrainer:
    """
    A trainer class to handle training multiple models with individual trainers.

    Attributes:
        model_configs: Model configurations managed by the trainer.
        log_dir: Directory used for logs and default checkpoints.
        auto_start_tensorboard: Whether TensorBoard should be started automatically.
        tensorboard_port: Port used by the auto-started TensorBoard server.
        auto_wait: Whether ``finalize`` should wait before stopping TensorBoard.
        __tensorboard_process: Running TensorBoard process, if one was started.
        __checkpoint_callback_kwargs: Default checkpoint callback keyword arguments.
    """

    DEFAULT_BASE_LOG_DIR = "hyperbench_logs"
    EXPERIMENT_NAME_PREFIX = "experiment"
    EXPERIMENT_SEPARATOR = "_"
    FIRST_EXPERIMENT_NUMBER = 0
    VERSION_NAME_PREFIX = "version"

    DEFAULT_BASE_CHECKPOINT_DIR = "checkpoints"

    __UNKNOWN_DEVICE = "unknown"

    def __init__(
        self,
        model_configs: list[ModelConfig],
        experiment_name: str | None = None,
        # args to pass to each Trainer
        accelerator: str | Accelerator = "auto",
        devices: list[int] | str | int = "auto",
        test_devices: list[int] | str | int | None = None,
        strategy: str | Strategy = "auto",
        num_nodes: int = 1,
        precision: Any
        | None = None,  # Any as Lightning accepts multiple types (int, str, Literal, etc.)
        max_epochs: int | None = None,
        min_epochs: int | None = None,
        max_steps: int = -1,
        min_steps: int | None = None,
        check_val_every_n_epoch: int | None = 1,
        logger: Logger | Iterable[Logger] | bool | None = None,
        default_root_dir: str | Path | None = None,
        enable_autolog_hparams: bool = True,
        log_every_n_steps: int | None = None,
        profiler: Profiler | str | None = None,
        fast_dev_run: int | bool = False,
        enable_checkpointing: bool = True,
        enable_progress_bar: bool = True,
        enable_model_summary: bool | None = None,
        callbacks: list[Callback] | Callback | None = None,
        checkpoint_callback_kwargs: dict[str, Any] | None = None,
        auto_start_tensorboard: bool = False,
        tensorboard_port: int = 6006,
        auto_wait: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize trainers for one or more model configurations.

        Args:
            model_configs: A list of ModelConfig objects, each containing a model and its
                associated trainer (if any).
            experiment_name: Name for this experiment run's log directory. When ``None`` (default),
                auto-increments as ``experiment_0``, ``experiment_1``, etc. under
                the log root directory. Only used when ``logger`` is not provided.
            accelerator: Supports passing different accelerator types
                ("cpu", "gpu", "tpu", "hpu", "mps", "auto") as well as custom accelerator instances.
            devices: The devices to use. Can be set to a positive number (int or str), a
                sequence of device indices (list or str), the value ``-1`` to indicate all available
                devices should be used, or ``"auto"`` for automatic selection based on the chosen
                accelerator. Defaults to ``"auto"``.
            test_devices: Optional device configuration for automatically-created test trainers.
                When set, ``test_all`` uses a separate Trainer with the same Trainer parameters as
                training except for ``devices``. This is useful for running distributed training
                but single-device testing, e.g. ``test_devices=1``. Defaults to ``None``,
                which makes testing use the fit trainer unless a
                ``ModelConfig.test_trainer`` is provided.
            strategy: Supports different training strategies with aliases as well custom strategies.
                Defaults to ``"auto"``.
            num_nodes: Number of GPU nodes for distributed training. Defaults to ``1``.
            precision: Double precision (64, '64' or '64-true'),
                full precision (32, '32' or '32-true'), 16bit mixed precision (16, '16', '16-mixed')
                or bfloat16 mixed precision ('bf16', 'bf16-mixed').
                Can be used on CPU, GPU, TPUs, or HPUs. Defaults to ``'32-true'``.
            max_epochs: Stop training once this number of epochs is reached.
                Disabled by default (None). If both max_epochs and max_steps are not specified,
                defaults to ``max_epochs = 1000``. To enable infinite training,
                set ``max_epochs = -1``.
            min_epochs: Force training for at least these many epochs. Disabled by default (None).
            max_steps: Stop training after this number of steps. Disabled by default (-1).
                If ``max_steps = -1`` and ``max_epochs = None``, will default to
                ``max_epochs = 1000``. To enable infinite training, set ``max_epochs`` to ``-1``.
            min_steps: Force training for at least these number of steps.
                Disabled by default (``None``).
            check_val_every_n_epoch: Perform a validation loop after every `N` training epochs.
                If ``None``, validation will be done solely based on the number of training batches,
                requiring ``val_check_interval`` to be an integer value. When used together with a
                time-based ``val_check_interval`` and ``check_val_every_n_epoch`` > 1, validation is
                aligned to epoch multiples: if the interval elapses before the next multiple-N
                epoch, validation runs at the start of that epoch (after the first batch) and the
                timer resets; if it elapses during a multiple-N epoch, validation runs after the
                current batch. For ``None`` or ``1`` cases, the time-based behavior of
                ``val_check_interval`` applies without additional alignment. Defaults to ``1``.
            logger: Logger (or iterable collection of loggers) for experiment tracking. A ``True``
                value uses the default ``TensorBoardLogger`` if it is installed,
                otherwise ``CSVLogger``. ``False`` will disable logging. If multiple loggers are
                provided, local files (checkpoints, profiler traces, etc.) are saved in the
                ``log_dir`` of the first logger. Defaults to ``True``.
            default_root_dir: Default path for logs and weights when no logger/ckpt_callback passed.
                Defaults to ``os.getcwd()``.
                Can be remote file paths such as `s3://mybucket/path` or 'hdfs://path/'
            enable_autolog_hparams: Whether to log hyperparameters at the start of a run.
                Defaults to ``True``.
            log_every_n_steps: How often to log within steps.
                Defaults to ``50``.
            profiler: To profile individual steps during training and assist in
                identifying bottlenecks. Defaults to ``None``.
            fast_dev_run: Runs n if set to ``n`` (int) else 1 if set to ``True`` batch(es)
                of train, val and test to find any bugs (ie: a sort of unit test).
                Defaults to ``False``.
            enable_checkpointing: If ``True``, enable checkpointing.
                It will configure a default ModelCheckpoint callback if there is no user-defined
                    ModelCheckpoint in :paramref:`~hyperbench.train.MultiModelTrainer.callbacks`.
                Defaults to ``True``.
            enable_progress_bar: Whether to enable the progress bar by default.
                Defaults to ``True``.
            enable_model_summary: Whether to enable model summarization by default.
                Defaults to ``True``.
            callbacks: Add a callback or list of callbacks.
                Defaults to ``None``.
            checkpoint_callback_kwargs: Keyword arguments passed to the default
                ``ModelCheckpoint`` callback when checkpointing is enabled and no
                user-defined ``ModelCheckpoint`` is provided. Pass ``dirpath`` to
                override the default checkpoint directory.
                Defaults to ``None``.
            auto_start_tensorboard: When ``True`` and tensorboard is installed, automatically
                starts a TensorBoard server pointing at the experiment log directory.
                Using this option requires that TensorBoard is installed in the environment and
                moves control of the TensorBoard server lifecycle to the trainer, which will
                automatically terminate the server when the trainer is finalized
                (e.g., at the end of a ``with`` block or when the object is garbage collected).
                Enable `auto_wait` to keep the server alive after training completes so you can
                inspect results before the trainer is finalized. Defaults to ``False``.
            tensorboard_port: Port for the auto-launched TensorBoard server.
                Defaults to ``6006``.
            auto_wait: When ``True`` and a TensorBoard server is running, automatically calls
                ``wait`` inside `finalize` before terminating the server, so the user
                can inspect results before the process is stopped.
                Defaults to ``False``.

            kwargs:
                max_time: Maximum wall-clock training time. Passed to Lightning's ``Trainer``.
                limit_train_batches: Fraction or number of training batches to use.
                limit_val_batches: Fraction or number of validation batches to use.
                limit_test_batches: Fraction or number of test batches to use.
                limit_predict_batches: Fraction or number of prediction batches to use.
                overfit_batches: Fraction or number of batches to use for overfitting checks.
                val_check_interval: How often to run validation within a training epoch.
                num_sanity_val_steps: Number of validation batches to run before training starts.
                accumulate_grad_batches: Number of batches over which to accumulate gradients.
                gradient_clip_val: Value used for gradient clipping.
                gradient_clip_algorithm: Gradient clipping algorithm to use.
                deterministic: Whether to enable deterministic operations, or ``"warn"``
                    to warn when deterministic execution is not available.
                benchmark: Whether to enable cuDNN benchmarking.
                inference_mode: Whether validation, test, and prediction loops should use
                    ``torch.inference_mode``.
                use_distributed_sampler: Whether Lightning should inject distributed samplers for
                    distributed training.
                detect_anomaly: Whether to enable PyTorch autograd anomaly detection.
                barebones: Whether to disable features that may affect raw training-loop speed.
                plugins: Lightning plugins or list of plugins to use.
                sync_batchnorm: Whether to synchronize batch normalization across devices.
                reload_dataloaders_every_n_epochs: How often to recreate dataloaders.
                model_registry: Name of the Lightning model registry to use.

        Raises:
            ValueError: If no model configurations are provided or numeric settings are invalid.
        """
        self.auto_wait: bool = auto_wait
        self.__tensorboard_process: subprocess.Popen | None = None
        validate_is_non_negative("tensorboard_port", tensorboard_port)

        self.model_configs: list[ModelConfig] = model_configs
        validate_is_non_empty("model_configs", self.model_configs)

        self.log_dir: Path = self.__logdir(default_root_dir, experiment_name)

        self.auto_start_tensorboard: bool = auto_start_tensorboard
        self.tensorboard_port: int = tensorboard_port
        self.__checkpoint_callback_kwargs: dict[str, Any] = (
            checkpoint_callback_kwargs if checkpoint_callback_kwargs is not None else {}
        )

        full_model_name_counts = self.__full_model_name_counts(model_configs)
        for model_index, model_config in enumerate(model_configs):
            should_create_trainer = model_config.trainer is None
            should_create_test_trainer = (
                model_config.test_trainer is None and test_devices is not None
            )
            should_create_at_least_one_trainer = should_create_trainer or should_create_test_trainer

            if should_create_at_least_one_trainer:
                has_duplicate_full_model_name = (
                    full_model_name_counts[model_config.full_model_name()] > 1
                )
                model_logger = self.__setup_logger(model_config, logger)
                model_callbacks = self.__setup_callbacks(
                    model_config=model_config,
                    model_index=model_index,
                    has_duplicate_full_model_name=has_duplicate_full_model_name,
                    callbacks=callbacks,
                    enable_checkpointing=enable_checkpointing,
                )

                def __trainer_for(
                    trainer_devices: list[int] | str | int,
                    model_logger: Logger | Iterable[Logger] | bool | None,
                    model_callbacks: list[Callback] | Callback | None,
                ) -> L.Trainer:
                    return L.Trainer(
                        accelerator=accelerator,
                        devices=trainer_devices,
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
                        callbacks=copy.deepcopy(model_callbacks),
                        **kwargs,
                    )

                if should_create_trainer:
                    model_config.trainer = __trainer_for(
                        trainer_devices=devices,
                        model_logger=model_logger,
                        model_callbacks=model_callbacks,
                    )

                if should_create_test_trainer:
                    model_config.test_trainer = __trainer_for(
                        trainer_devices=test_devices if test_devices is not None else devices,
                        model_logger=model_logger,
                        model_callbacks=model_callbacks,
                    )

        print(f"Initialized trainer(models: {len(model_configs)}, log_dir: {self.log_dir})")
        self.__auto_start_tensorboard_if_enabled()

    def __enter__(self) -> MultiModelTrainer:
        """
        Enter the trainer context manager.

        Returns:
            trainer: This trainer instance.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """
        Exit the trainer context manager and finalize resources.

        Args:
            exc_type: Exception type from the context, if any, unused.
            exc_val: Exception value from the context, if any, unused.
            exc_tb: Exception traceback from the context, if any, unused.
        """
        self.finalize()

    def __del__(self) -> None:
        """
        Finalize external resources during object cleanup.
        """
        try:
            self.finalize()
        except Exception as e:
            warnings.warn(
                f"Exception occurred during {self.__class__.__name__} cleanup. Error: {e}",
                category=UserWarning,
                stacklevel=2,
            )

    @property
    def distributed_global_rank(self) -> int:
        """
        Return the global rank of the current process.

        Returns:
            global_rank: The global rank of the current process.
        """
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank()
        return 0  # if distributed not available or not initialized, we are in single-process setup

    @property
    def is_global_zero(self) -> bool:
        """
        Return whether the current process is the global zero process.

        Check first if any of the trainers is the global zero, which covers the case where
        a custom strategy that handles distributed differently was provided.
        If no trainer is available, fall back to checking the global rank.

        Returns:
            is_global_zero: ``True`` if the current process is the global zero process,
                ``False`` otherwise.
        """
        model_config = self.model_configs[0]  # It's a process-level property, so we can check any
        trainer = (
            model_config.trainer if model_config.trainer is not None else model_config.test_trainer
        )
        if trainer is not None:
            return trainer.is_global_zero
        return self.distributed_global_rank == 0

    @property
    def models(self) -> list[L.LightningModule]:
        """
        Return all configured Lightning modules.

        Returns:
            models: List of configured models.
        """
        return [config.model for config in self.model_configs]

    def model(self, name: str, version: str = "default") -> L.LightningModule | None:
        """
        Return a configured model by name and version.

        Args:
            name: Model name.
            version: Model version. Defaults to ``"default"``.

        Returns:
            model: Matching model, or ``None`` when no model matches.
        """
        for config in self.model_configs:
            if config.name == name and config.version == version:
                return config.model
        return None

    def fit_all(
        self,
        train_dataloader: DataLoader | None = None,
        val_dataloader: DataLoader | None = None,
        datamodule: L.LightningDataModule | None = None,
        ckpt_path: CkptStrategy | None = None,
        verbose: bool = True,
    ) -> None:
        """
        Fit every trainable configured model.

        Args:
            train_dataloader: Shared training dataloader used when a model has no override.
                Defaults to ``None``, which requires each model to have its own
                training dataloader provided.
            val_dataloader: Shared validation dataloader used when a model has no override.
                Defaults to ``None``, which requires each model to have its own
                validation dataloader provided.
            datamodule: Optional Lightning datamodule.
                Defaults to ``None``, which requires each model to have its own
                test dataloader provided.
            ckpt_path: Optional checkpoint strategy or path.
                Defaults to ``None``, which uses the best checkpoint if available,
                otherwise the last checkpoint.
            verbose: Whether to print progress messages. Defaults to ``True``.

        Raises:
            ValueError: If a trainable model has no trainer.
        """
        for i, config in enumerate(self.model_configs):
            if not config.is_trainable:
                if verbose:
                    print(
                        f"Skipping training for model {config.full_model_name()} "
                        f"[{i + 1}/{len(self.model_configs)} models] (is_trainable=False)"
                    )
                continue

            if config.trainer is None:
                raise ValueError(f"Trainer not defined for model {config.full_model_name()}.")

            if verbose:
                print(
                    f"Fit model {config.full_model_name()} "
                    f"[{i + 1}/{len(self.model_configs)} models] "
                    f"(device: {self.__device(config.trainer)}, "
                    f"log_dir: {config.trainer.log_dir}, "
                    f"ckpt_path: {ckpt_path if ckpt_path is not None else 'None'})"
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
        dataloader: DataLoader | None = None,
        datamodule: L.LightningDataModule | None = None,
        ckpt_path: CkptStrategy | None = None,
        verbose: bool = True,
        verbose_loop: bool = True,
    ) -> Mapping[str, TestResult]:
        """
        Test every configured model.

        Args:
            dataloader: Shared test dataloader used when a model has no override.
                Defaults to ``None``, which requires each model to have
                its own test dataloader provided.
            datamodule: Optional Lightning datamodule. Defaults to ``None``,
                which requires each model to have its own test dataloader provided.
            ckpt_path: Optional checkpoint strategy or path. Defaults to ``None``,
                which uses the best checkpoint if available, otherwise the last checkpoint.
            verbose: Whether to print model-level progress messages. Defaults to ``True``.
            verbose_loop: Whether Lightning should print test-loop output. Defaults to ``True``.

        Returns:
            test_results: Mapping from full model name to Lightning test metrics.

        Raises:
            ValueError: If a model has no trainer available for testing.
        """
        test_results: dict[str, TestResult] = {}
        for i, config in enumerate(self.model_configs):
            test_trainer = (
                config.test_trainer if config.test_trainer is not None else config.trainer
            )
            if test_trainer is None:
                raise ValueError(f"Trainer not defined for model {config.full_model_name()}.")

            if self.__is_separate_single_process_test_trainer(config, test_trainer):
                # Ensure all processes have finished training before testing
                # to avoid warning about sync_dist=True when no distributed is happening in test
                self.__wait_for_distributed()

            if self.__should_skip_test_on_current_rank(config, test_trainer):
                test_results[config.full_model_name()] = {}
                continue

            if verbose:
                print(
                    f"Test model {config.full_model_name()} "
                    f"[{i + 1}/{len(self.model_configs)} models] "
                    f"(device: {self.__device(test_trainer)}, "
                    f"log_dir: {test_trainer.log_dir}, "
                    f"ckpt_path: {ckpt_path if ckpt_path is not None else 'None'})"
                )

            test_dataloaders = (
                config.test_dataloader if config.test_dataloader is not None else dataloader
            )
            trainer_test_results: list[TestResult] = test_trainer.test(
                model=config.model,
                dataloaders=test_dataloaders,
                datamodule=datamodule,
                ckpt_path=ckpt_path,
                verbose=verbose_loop,
            )

            # In Lightning, test() returns a list of dicts, one per dataloader,
            # but we use a single dataloader
            test_results[config.full_model_name()] = (
                trainer_test_results[0] if len(trainer_test_results) > 0 else {}
            )

        return test_results

    def finalize(self) -> None:
        """
        Finalize trainer-managed external resources.
        """
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
            print("Stopping TensorBoard...")

    def __auto_start_tensorboard_if_enabled(self) -> None:
        """
        Start TensorBoard when auto-start is enabled and available.
        """
        if self.auto_start_tensorboard:
            if self.__is_tensorboard_available():
                self.__tensorboard_process = self.__start_tensorboard_process()
            else:
                warnings.warn(
                    "TensorBoard is not available. "
                    "Install it with `pip install hyperbench[tensorboard]` or "
                    "`pip install tensorboard`"
                    "to enable auto-start.",
                    category=UserWarning,
                    stacklevel=2,
                )

    def __is_separate_single_process_test_trainer(
        self,
        model_config: ModelConfig,
        test_trainer: L.Trainer,
    ) -> bool:
        """
        Check whether testing uses a separate single-process trainer.

        Args:
            model_config: Model configuration being tested.
            test_trainer: Trainer selected for testing.

        Returns:
            result: ``True`` when a separate single-process test trainer is being used.
        """
        return (
            model_config.test_trainer is not None
            and test_trainer is not model_config.trainer
            and test_trainer.world_size <= 1
        )

    def __is_tensorboard_available(self) -> bool:
        """
        Check whether TensorBoard is importable.

        Returns:
            available: ``True`` when TensorBoard is installed.
        """
        return importlib.util.find_spec("tensorboard") is not None

    def __start_tensorboard_process(self) -> subprocess.Popen | None:
        """
        Start a TensorBoard subprocess for the trainer log directory.

        Returns:
            process: Started TensorBoard process, or ``None`` when startup is unavailable.
        """
        try:
            tensorboard_executable = shutil.which("tensorboard")
            if tensorboard_executable is None:
                return None

            log_dir = str(self.log_dir)
            tensorboard_port = str(self.tensorboard_port)
            process = subprocess.Popen(
                [tensorboard_executable, "--logdir", log_dir, "--port", tensorboard_port],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"TensorBoard started at http://localhost:{tensorboard_port} (logdir={log_dir})")
            return process
        except Exception as e:
            warnings.warn(
                f"Proceeding without starting TensorBoard as it failed: {e}",
                category=UserWarning,
                stacklevel=2,
            )
            return None

    def __checkpoint_dir(
        self,
        model_config: ModelConfig,
        model_index: int,
        has_duplicate_full_model_name: bool,
    ) -> Path:
        """
        Resolve the checkpoint directory for a model configuration.

        Args:
            model_config: Model configuration that owns the checkpoints.
            model_index: Index of the model in ``model_configs``.
            has_duplicate_full_model_name: Whether the full model name is duplicated.

        Returns:
            checkpoint_dir: Directory for model checkpoints.
        """
        provided_dirpath: str | Path | None = self.__checkpoint_callback_kwargs.get("dirpath")
        if provided_dirpath is not None:
            return Path(provided_dirpath)

        checkpoint_dir = (
            self.log_dir / model_config.name / f"{self.VERSION_NAME_PREFIX}_{model_config.version}"
        )

        if has_duplicate_full_model_name:
            checkpoint_dir /= f"model_{model_index}"

        return checkpoint_dir / self.DEFAULT_BASE_CHECKPOINT_DIR

    def __device(self, trainer: L.Trainer) -> str:
        """
        Return the root device string for a trainer.

        Args:
            trainer: Lightning trainer to inspect.

        Returns:
            device: Root device string, or ``"unknown"`` when unavailable.
        """
        if trainer.strategy is None:
            return self.__UNKNOWN_DEVICE
        strategy = trainer.strategy
        if strategy.root_device is None:
            return self.__UNKNOWN_DEVICE
        return str(strategy.root_device)

    def __full_model_name_counts(self, model_configs: list[ModelConfig]) -> dict[str, int]:
        """
        Count occurrences of full model names.

        Args:
            model_configs: Model configurations to inspect.

        Returns:
            counts: Mapping from full model name to occurrence count.
        """
        full_model_name_counts: dict[str, int] = {}
        for model_config in model_configs:
            full_model_name = model_config.full_model_name()
            full_model_name_counts[full_model_name] = (
                full_model_name_counts.get(full_model_name, 0) + 1
            )
        return full_model_name_counts

    def __next_experiment_name(self, save_dir: Path) -> Path:
        """
        Return the next available experiment directory name.

        Args:
            save_dir: Base directory containing experiment directories.

        Returns:
            experiment_name: Next experiment directory name.
        """
        if not save_dir.exists():
            # Example: EXPERIMENT_NAME_PREFIX = "experiment",
            #          EXPERIMENT_SEPARATOR = "_",
            #          FIRST_EXPERIMENT_NUMBER = 0
            #          -> next_experiment_name = "experiment_0"
            return Path(
                f"{self.EXPERIMENT_NAME_PREFIX}{self.EXPERIMENT_SEPARATOR}{self.FIRST_EXPERIMENT_NUMBER}"
            )

        existing_experiment_names: list[str] = [
            dir.name
            for dir in save_dir.iterdir()
            if dir.is_dir() and dir.name.startswith(self.EXPERIMENT_NAME_PREFIX)
        ]
        if len(existing_experiment_names) < 1:
            return Path(
                f"{self.EXPERIMENT_NAME_PREFIX}{self.EXPERIMENT_SEPARATOR}{self.FIRST_EXPERIMENT_NUMBER}"
            )

        last_experiment_number = max(
            int(experiment_name.split(self.EXPERIMENT_SEPARATOR)[1])
            for experiment_name in existing_experiment_names
            if experiment_name.split(self.EXPERIMENT_SEPARATOR)[1].isdigit()
        )
        return Path(
            f"{self.EXPERIMENT_NAME_PREFIX}{self.EXPERIMENT_SEPARATOR}{last_experiment_number + 1}"
        )

    def __logdir(
        self,
        default_root_dir: str | Path | None,
        experiment_name: str | None,
    ) -> Path:
        """
        Resolve the log directory for this trainer instance.

        Args:
            default_root_dir: Optional root directory for logs.
            experiment_name: Optional explicit experiment directory name.

        Returns:
            log_dir: Resolved log directory.
        """
        base_dir = (
            Path(self.DEFAULT_BASE_LOG_DIR) if default_root_dir is None else Path(default_root_dir)
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
        logger: Logger | Iterable[Logger] | bool | None,
    ) -> Logger | Iterable[Logger] | bool | None:
        """
        Resolve loggers for a model configuration.

        Args:
            model_config: Model configuration being prepared.
            logger: User-provided logger configuration.

        Returns:
            logger: User-provided logger configuration or default HyperBench loggers.
        """
        if logger is not None:
            return logger

        experiment_name = str(self.__next_experiment_name(self.log_dir))

        loggers: list[Logger] = [
            CSVLogger(
                save_dir=self.log_dir,
                name=model_config.name,
                version=f"{self.VERSION_NAME_PREFIX}_{model_config.version}",
            ),
            MarkdownTableLogger(
                save_dir=self.log_dir,
                model_name=model_config.full_model_name(),
                experiment_name=experiment_name,
            ),
            LaTexTableLogger(
                save_dir=self.log_dir,
                model_name=model_config.full_model_name(),
                experiment_name=experiment_name,
                options={
                    "table_caption": "Results for Experiments",
                    "sort_by": ["des", "asc"],
                    "border": False,
                },
            ),
        ]

        if self.__is_tensorboard_available():
            from lightning.pytorch.loggers import TensorBoardLogger

            loggers.append(
                TensorBoardLogger(
                    save_dir=self.log_dir,
                    name=model_config.name,
                    version=f"{self.VERSION_NAME_PREFIX}_{model_config.version}",
                ),
            )

        return loggers

    def __setup_callbacks(
        self,
        model_config: ModelConfig,
        model_index: int,
        has_duplicate_full_model_name: bool,
        callbacks: list[Callback] | Callback | None,
        enable_checkpointing: bool,
    ) -> list[Callback] | Callback | None:
        """
        Resolve callbacks for a model configuration.

        Args:
            model_config: Model configuration being prepared.
            model_index: Index of the model in ``model_configs``.
            has_duplicate_full_model_name: Whether the full model name is duplicated.
            callbacks: User-provided callbacks.
            enable_checkpointing: Whether checkpointing is enabled.

        Returns:
            callbacks: Resolved callbacks for the trainer.
        """
        model_callbacks = copy.deepcopy(callbacks)

        if not enable_checkpointing:
            return model_callbacks

        checkpoint_dir = self.__checkpoint_dir(
            model_config=model_config,
            model_index=model_index,
            has_duplicate_full_model_name=has_duplicate_full_model_name,
        )
        callback_list = self.__to_callback_list(model_callbacks)
        checkpoint_callbacks = [
            callback for callback in callback_list if isinstance(callback, ModelCheckpoint)
        ]

        if len(checkpoint_callbacks) < 1:
            checkpoint_callback_kwargs = copy.deepcopy(self.__checkpoint_callback_kwargs)
            checkpoint_callback_kwargs["dirpath"] = checkpoint_dir
            callback_list.append(ModelCheckpoint(**checkpoint_callback_kwargs))
            return callback_list

        for callback in checkpoint_callbacks:
            if callback.dirpath is None:
                callback.dirpath = str(checkpoint_dir)
        return callback_list

    def __should_skip_test_on_current_rank(
        self,
        model_config: ModelConfig,
        test_trainer: L.Trainer,
    ) -> bool:
        """
        Check whether the current rank should skip testing.

        Args:
            model_config: Model configuration being tested.
            test_trainer: Trainer selected for testing.

        Returns:
            result: ``True`` when this process should skip the test call.
        """
        if not self.__is_separate_single_process_test_trainer(model_config, test_trainer):
            return False

        if model_config.trainer is not None:
            return not model_config.trainer.is_global_zero

        return self.distributed_global_rank != 0

    def __to_callback_list(self, callbacks: list[Callback] | Callback | None) -> list[Callback]:
        """
        Normalize callback configuration to a list.

        Args:
            callbacks: Optional callback, callback list, or ``None``.

        Returns:
            callbacks: Callback list.
        """
        if callbacks is None:
            return []
        if isinstance(callbacks, Callback):
            return [callbacks]
        return callbacks

    def __wait_for_distributed(self) -> None:
        """
        Synchronize and tear down an initialized distributed process group.
        """
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
