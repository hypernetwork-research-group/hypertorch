from pathlib import Path
import pandas as pd


class LogParser:
    """Finds and parses the metrics CSV from an experiment folder.

    Currently, it only looks at the latest metrics CSV,
    to be expanded with the ability to parse older experiments.

    Args:
        base_logs_dir: Root directory containing experiment run folders.

    Note: defaults to 'hypertorch_logs'.
    """

    def __init__(self, base_logs_dir: str | Path = "hypertorch_logs") -> None:
        self.base_logs_dir = Path(base_logs_dir)

    def find_latest_experiment_dir(self) -> Path:
        """Finds the most recently modified experiment folder inside hypertorch_logs.

        Raises:
            FileNotFoundError: If there are no subdirectories or if
                base_logs_dir does not exist.
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

        Raises:
            FileNotFoundError: If there is no CSV metrics file in the latest folder.
        """
        latest_exp_dir = self.find_latest_experiment_dir()

        csv_files = list(latest_exp_dir.rglob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV metric files found inside latest experiment folder: '{latest_exp_dir}'"
            )

        csv_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return csv_files[0]

    def load_latest_metrics(self) -> tuple[pd.DataFrame, Path]:
        """Loads the latest CSV into a Pandas DataFrame.

        Returns:
            A tuple of (DataFrame, CSV_File_Path).

        Raises:
            FileNotFoundError: If there are no subdirectories, if base_logs_dir
                does not exist, or if no CSV is found.
        """
        target_csv = self.find_latest_metrics_csv()
        return self.load_csv(target_csv)

    def load_csv(self, path: str | Path) -> tuple[pd.DataFrame, Path]:
        """Loads a CSV into a Pandas DataFrame from the given path.

        If a relative path is provided, it is resolved relative to
        ``base_logs_dir``. Absolute paths are used as-is.

        Args:
            path: Relative or absolute path to the target CSV file.

        Returns:
            A tuple of (DataFrame, CSV_File_Path).

        Raises:
            ValueError: If the file is not a CSV.
            FileNotFoundError: If the path does not exist or is not a file.
        """
        target_path = Path(path)
        if not target_path.is_absolute() and not target_path.is_relative_to(self.base_logs_dir):
            target_path = self.base_logs_dir / target_path

        if target_path.suffix.lower() != ".csv":
            raise ValueError(f"File '{target_path}' is not a CSV file.")

        if not target_path.is_file():
            raise FileNotFoundError(f"CSV file '{target_path}' does not exist.")

        df = pd.read_csv(target_path)
        return df, target_path
