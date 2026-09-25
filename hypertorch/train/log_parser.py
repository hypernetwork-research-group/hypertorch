from pathlib import Path
import re
import warnings
import pandas as pd

from hypertorch.types import ParsedMetrics


class LogParser:
    """Finds and parses experiment metric logs into tidy ParsedMetrics containers.

    Args:
        base_logs_dir: Root directory containing experiment run folders.
                       Defaults to 'hypertorch_logs'.
    """

    def __init__(self, base_logs_dir: str | Path = "hypertorch_logs") -> None:
        self._base_logs_dir: Path = Path(base_logs_dir)
        self._latest_experiment_dir: Path | None = None
        self._latest_csv_file: Path | None = None

    @property
    def base_logs_dir(self) -> Path:
        """Root directory where experiment logs are located."""
        return self._base_logs_dir

    @property
    def latest_experiment_dir(self) -> Path:
        """Path to the most recently modified experiment directory containing a CSV."""
        if self._latest_experiment_dir is None:
            self._discover_latest()
        assert self._latest_experiment_dir is not None
        return self._latest_experiment_dir

    @property
    def latest_csv_file(self) -> Path:
        """Path to the most recently modified CSV inside the latest valid experiment directory."""
        if self._latest_csv_file is None:
            self._discover_latest()
        assert self._latest_csv_file is not None
        return self._latest_csv_file

    def refresh(self, updated_base_folder: str | Path | None = None) -> None:
        """Resets and re-discovers the latest experiment directory and CSV file.

        Args:
            updated_base_folder: Optional new base directory to scan.
        """
        if updated_base_folder is not None:
            self._base_logs_dir = Path(updated_base_folder)
        self._latest_experiment_dir = None
        self._latest_csv_file = None

    def parse(self, csv_path: str | Path | None = None) -> ParsedMetrics:
        """Loads and reshapes experiment metrics into a ParsedMetrics container.

        Args:
            csv_path: Optional path to a specific CSV file. If None,
                      automatically finds and parses the latest run.

        Returns:
            A ParsedMetrics instance containing tidy DataFrames per metric.

        Raises:
            ValueError: If the CSV contains no data or columns.
        """
        target_csv = self.latest_csv_file if csv_path is None else csv_path
        raw_df, resolved_path = self._load_csv(target_csv)

        # Resolve experiment directory and experiment name
        if csv_path is None:
            exp_dir = self.latest_experiment_dir
        elif resolved_path.is_relative_to(self._base_logs_dir):
            rel_parts = resolved_path.relative_to(self._base_logs_dir).parts
            exp_dir = (
                self._base_logs_dir / rel_parts[0] if len(rel_parts) > 1 else resolved_path.parent
            )
        else:
            exp_dir = resolved_path.parent
            for parent in resolved_path.parents:
                if re.search(r"experiment_(\d+)", parent.name):
                    exp_dir = parent
                    break

        match = re.search(r"experiment_(\d+)", exp_dir.name)
        exp_name = match.group(1) if match else "0"

        # Identify primary tracking dimension
        x_col = (
            "epoch"
            if "epoch" in raw_df.columns
            else ("step" if "step" in raw_df.columns else raw_df.columns[0])
        )

        tracking_cols = {"epoch", "step", x_col}
        metric_cols = [c for c in raw_df.columns if c not in tracking_cols]

        # Extract base metric names ('val/loss' -> 'loss')
        variables = set()
        for col in metric_cols:
            clean_var = col.split("/", 1)[1] if "/" in col else col
            if clean_var not in tracking_cols:
                variables.add(clean_var)

        parsed = ParsedMetrics(
            x_col=x_col,
            csv_path=resolved_path,
            experiment_dir=exp_dir,
            experiment_name=exp_name,
        )

        # Reshape each metric into a tidy DataFrame
        for var_name in sorted(variables):
            matching_cols = [
                c
                for c in metric_cols
                if c == var_name or ("/" in c and c.split("/", 1)[1] == var_name)
            ]

            melted = raw_df.melt(
                id_vars=[x_col],
                value_vars=matching_cols,
                var_name="split",
                value_name="value",
            ).dropna()

            if melted.empty:
                continue

            melted["split"] = melted["split"].astype(str).str.split("/").str[0]
            parsed.add(var_name, melted)

        return parsed

    def _resolve_and_validate_path(self, path: str | Path) -> Path:
        """Internal helper to resolve relative paths and validate file existence.

        Raises:
            FileNotFoundError: If the target file does not exist.
            ValueError: If the file does not have a .csv extension.
        """
        target_path = Path(path)
        if not target_path.is_absolute() and not target_path.is_relative_to(self._base_logs_dir):
            target_path = self._base_logs_dir / target_path

        if target_path.suffix.lower() != ".csv":
            raise ValueError(f"File '{target_path}' is not a CSV file.")

        if not target_path.is_file():
            raise FileNotFoundError(f"CSV file '{target_path}' does not exist.")

        return target_path

    def _load_csv(self, path: str | Path) -> tuple[pd.DataFrame, Path]:
        """Internal helper to validate and load raw CSV data into a DataFrame.

        Raises:
            ValueError: If the CSV file is completely empty or contains no data rows.
        """
        resolved_path = self._resolve_and_validate_path(path)
        try:
            raw_df = pd.read_csv(resolved_path)
        except pd.errors.EmptyDataError:
            raise ValueError(f"CSV file '{resolved_path}' is completely empty.") from None

        if raw_df.empty:
            warnings.warn(
                f"CSV file '{resolved_path}' contains headers but no data rows.",
                category=UserWarning,
                stacklevel=2,
            )

        return raw_df, resolved_path

    def _discover_latest(self) -> None:
        """Finds and caches the latest experiment directory that contains a CSV file.

        Raises:
            FileNotFoundError: If base_logs_dir does not exist, contains no experiment
                               directories, or no directory contains a CSV.
        """
        if not self._base_logs_dir.exists():
            raise FileNotFoundError(f"Logs root directory '{self._base_logs_dir}' does not exist.")

        experiment_dirs = [p for p in self._base_logs_dir.iterdir() if p.is_dir()]
        if not experiment_dirs:
            raise FileNotFoundError(f"No experiment folders found inside '{self._base_logs_dir}'.")

        # Sort experiment directories from newest to oldest
        experiment_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        for exp_dir in experiment_dirs:
            csv_files = list(exp_dir.rglob("*.csv"))
            if csv_files:
                self._latest_experiment_dir = exp_dir
                self._latest_csv_file = max(csv_files, key=lambda p: p.stat().st_mtime)
                return

        raise FileNotFoundError(
            f"No CSV metric files found inside any experiment folder in '{self._base_logs_dir}'."
        )
