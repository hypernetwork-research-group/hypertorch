from pathlib import Path
import pandas as pd

from .parsed_metrics import ParsedMetrics


class LogParser:
    """Finds and parses experiment metric logs into tidy ParsedMetrics containers.

    Args:
        base_logs_dir: Root directory containing experiment run folders.
                       Defaults to 'hypertorch_logs'.
    """

    def __init__(self, base_logs_dir: str | Path = "hypertorch_logs") -> None:
        self.base_logs_dir = Path(base_logs_dir)

    def find_latest_experiment_dir(self) -> Path:
        """Finds the most recently modified experiment folder inside base_logs_dir.

        Returns:
            Path to the latest experiment directory.

        Raises:
            FileNotFoundError: If there are no subdirectories or if base_logs_dir does not exist.
        """
        if not self.base_logs_dir.exists():
            raise FileNotFoundError(f"Logs root directory '{self.base_logs_dir}' does not exist.")

        experiment_dirs = [p for p in self.base_logs_dir.iterdir() if p.is_dir()]
        if not experiment_dirs:
            raise FileNotFoundError(f"No experiment folders found inside '{self.base_logs_dir}'.")

        experiment_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return experiment_dirs[0]

    def find_latest_metrics_csv(self) -> Path:
        """Finds the most recently modified CSV inside the latest experiment folder.

        Returns:
            Path to the latest metrics CSV file.

        Raises:
            FileNotFoundError: If there is no CSV file in the latest experiment folder.
        """
        latest_exp_dir = self.find_latest_experiment_dir()
        csv_files = list(latest_exp_dir.rglob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV metric files found inside latest experiment folder: '{latest_exp_dir}'"
            )

        csv_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return csv_files[0]

    def _resolve_and_read_csv(self, path: str | Path) -> tuple[pd.DataFrame, Path]:
        """Internal helper to validate paths and load raw CSV data.

        Args:
            path: Relative or absolute path to the CSV file.

        Returns:
            A tuple of (DataFrame, resolved Path).

        Raises:
            ValueError: If the file does not have a .csv extension.
            FileNotFoundError: If the target file does not exist.
        """
        target_path = Path(path)
        if not target_path.is_absolute():
            try:
                if not target_path.is_relative_to(self.base_logs_dir):
                    target_path = self.base_logs_dir / target_path
            except ValueError:
                target_path = self.base_logs_dir / target_path

        if target_path.suffix.lower() != ".csv":
            raise ValueError(f"File '{target_path}' is not a CSV file.")

        if not target_path.is_file():
            raise FileNotFoundError(f"CSV file '{target_path}' does not exist.")

        return pd.read_csv(target_path), target_path

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
        target_csv = self.find_latest_metrics_csv() if csv_path is None else csv_path
        raw_df, resolved_path = self._resolve_and_read_csv(target_csv)

        if raw_df.empty or len(raw_df.columns) == 0:
            raise ValueError(f"CSV file '{resolved_path}' contains no data or columns.")

        #Identify primary tracking dimension
        x_col = (
            "epoch"
            if "epoch" in raw_df.columns
            else ("step" if "step" in raw_df.columns else raw_df.columns[0])
        )

        tracking_cols = {"epoch", "step"}
        metric_cols = [c for c in raw_df.columns if c not in tracking_cols]

        #Extract base metric names (e.g., 'val/loss' -> 'loss')
        variables = set()
        for col in metric_cols:
            clean_var = col.split("/", 1)[1] if "/" in col else col
            if clean_var not in tracking_cols:
                variables.add(clean_var)

        parsed = ParsedMetrics(x_col=x_col, csv_path=resolved_path)

        #Reshape each metric into a tidy DataFrame
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