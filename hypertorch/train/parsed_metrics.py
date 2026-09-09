from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd


@dataclass
class ParsedMetrics:
    """Container holding tidy, per-metric DataFrames parsed from experiment logs.

    Each metric DataFrame has standard columns: [x_col, 'split', 'value'].
    """

    x_col: str
    csv_path: Path | None = None
    _metrics: dict[str, pd.DataFrame] = field(default_factory=dict, init=False)

    def add(self, name: str, df: pd.DataFrame) -> None:
        """Adds a tidy DataFrame for a specific metric variable.
        
        Args:
            name: Metric variable name (e.g., 'loss', 'f1').
            df: DataFrame containing (x_col, split, value).
        """
        self._metrics[name] = df

    def fetch(self, name: str) -> pd.DataFrame:
        """Retrieves the tidy DataFrame for a given metric.

        Args:
            name: Name of the metric to retrieve.

        Returns:
            The tidy metric DataFrame.

        Raises:
            KeyError: If the metric name does not exist.
        """
        if name not in self._metrics:
            available = ", ".join(self.names())
            raise KeyError(f"Metric '{name}' not found. Available metrics: [{available}]")
        return self._metrics[name]

    def all(self) -> dict[str, pd.DataFrame]:
        """Returns all parsed metric DataFrames.

        Returns:
            A shallow copy dictionary mapping metric names to DataFrames.
        """        
        return self._metrics.copy()

    def names(self) -> list[str]:
        """Lists all available metric names.

        Returns:
            Sorted list of metric names.
        """
        return sorted(self._metrics.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._metrics

    def __getitem__(self, name: str) -> pd.DataFrame:
        return self.fetch(name)

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        return len(self._metrics)

    def __repr__(self) -> str:
        return f"ParsedMetrics(x_col='{self.x_col}', metrics={self.names()})"