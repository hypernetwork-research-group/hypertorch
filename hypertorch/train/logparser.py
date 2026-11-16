from pathlib import Path
import pandas as pd


class LogParser:
    """
    Finds and parses the metrics CSV from an experiment folder.

    Currently, it only looks at the latest metrics CSV,
    to be expanded with the ability to parse older experiments.

    Args:
        base_logs_dir: Root directory containing experiment run folders.

    Note: defaults to 'hypertorch_logs'.
    """

    def __init__(self, base_logs_dir: str | Path = "hypertorch_logs") -> None:
        self.base_logs_dir = Path(base_logs_dir)

    def find_latest_experiment_dir(self) -> Path:
        """
        Finds the most recently modified experiment folder inside hypertorch_logs.
        """
        if not self.base_logs_dir.exists():
            raise FileNotFoundError(f"Logs root directory '{self.base_logs_dir}' does not exist.")

        # Filter for subdirectories only (e.g. experiment_0, experiment_1)
        experiment_dirs = [p for p in self.base_logs_dir.iterdir() if p.is_dir()]
        if not experiment_dirs:
            raise FileNotFoundError(f"No experiment folders found inside '{self.base_logs_dir}'.")

        # Sort folders by newest first
        experiment_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return experiment_dirs[0]

    def find_latest_metrics_csv(self) -> Path:
        """
        Finds the most recently modified CSV inside the latest experiment folder.
        """
        latest_exp_dir = self.find_latest_experiment_dir()

        # Recursively scan inside that latest experiment folder
        csv_files = list(latest_exp_dir.rglob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV metric files found inside latest experiment folder: '{latest_exp_dir}'"
            )

        # Sort CSV files by newest first
        csv_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return csv_files[0]

    def load_metrics(self) -> tuple[pd.DataFrame, Path]:
        """
        Step 3: Loads the latest CSV into a Pandas DataFrame.

        Returns:
            A tuple of (DataFrame, CSV_File_Path).
        """
        target_csv = self.find_latest_metrics_csv()
        df = pd.read_csv(target_csv)
        return df, target_csv
