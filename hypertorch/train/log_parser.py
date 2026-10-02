from collections.abc import Iterator
import heapq
import os
from pathlib import Path
import re
import warnings

import pandas as pd
from hypertorch.types import ParsedMetrics


class LogParser:
    """Finds and parses experiment metric logs into tidy ParsedMetrics containers.

    Tracks directories and maps each directory to its associated CSV log paths.

    Raises:
        FileNotFoundError: If base_logs_dir does not exist.
        NotADirectoryError: If base_logs_dir is not a directory.
    """

    def __init__(self, base_logs_dir: str | Path = "hypertorch_logs") -> None:
        self._base_logs_dir: Path = self._validate_base_dir(base_logs_dir)
        self._dir_to_csvs: dict[
            Path, list[Path]
        ] = {}  # Once explored, retrieving csv is a O(1) operation

    @staticmethod
    def _validate_base_dir(path: str | Path) -> Path:
        """Validates that the base directory exists and is a directory.

        Raises:
            FileNotFoundError: If the directory does not exist.
            NotADirectoryError: If the path points to a file or special node.
        """
        resolved = Path(path).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Logs root directory '{resolved}' does not exist.")
        if not resolved.is_dir():
            raise NotADirectoryError(f"Logs root path '{resolved}' is not a directory.")
        return resolved

    @staticmethod
    def _safe_mtime(entry: os.DirEntry) -> float:
        """Safely fetches modification time from a DirEntry."""
        try:
            return entry.stat().st_mtime
        except OSError:
            return 0.0

    @classmethod
    def _iter_csvs(cls, root: str | Path, max_depth: int = 4) -> Iterator[Path]:
        """explore depth first CSV files newest-to-oldest, bypassing checkpoint folders.

        Args:
            root: Root path to begin scanning.
            max_depth: Maximum recursion depth from the root.

        Yields:
            Paths to matching .csv files.
        """
        skip_dirs = {"checkpoints", "checkpoint", "weights", "comparison"}

        def _walk(current: str, depth: int) -> Iterator[Path]:
            if depth > max_depth:
                return
            try:
                with os.scandir(current) as it:
                    entries = list(it)
            except OSError:
                return

            entries.sort(key=cls._safe_mtime, reverse=True)

            # Yield CSVs directly inside current directory
            for entry in entries:
                if entry.is_file() and entry.name.lower().endswith(".csv"):
                    yield Path(entry.path)

            # Recurse into subdirectories, pruning skip_dirs
            for entry in entries:
                if entry.is_dir() and entry.name.lower() not in skip_dirs:
                    yield from _walk(entry.path, depth + 1)

        yield from _walk(str(root), 1)

    @property
    def base_logs_dir(self) -> Path:
        """Root directory where experiment logs are located."""
        return self._base_logs_dir

    @property
    def directory_paths(self) -> list[Path]:
        """Queued directories containing CSV files ready for parsing."""
        return list(self._dir_to_csvs.keys())

    @property
    def dir_to_csvs(self) -> dict[Path, list[Path]]:
        """Mapping of queued directory paths to their linked CSV file paths."""
        return {d: list(csvs) for d, csvs in self._dir_to_csvs.items()}

    def move_base_dir(self, new_base_dir: str | Path) -> None:
        """Updates the root search directory without clearing queued directories.

        Raises:
            FileNotFoundError: If new_base_dir does not exist.
            NotADirectoryError: If new_base_dir is not a directory.
        """
        self._base_logs_dir = self._validate_base_dir(new_base_dir)

    def _normalize_path(self, path: str | Path) -> Path:
        """Resolves path against cwd if existing, or base_logs_dir otherwise."""
        target = Path(path)
        if not target.is_absolute() and not target.exists():
            target = self._base_logs_dir / target
        return target.resolve()

    def _register_csv(self, csv_path: Path) -> None:
        """Internal helper to link a CSV path with its parent folder."""
        resolved_csv = csv_path.resolve()
        run_dir = resolved_csv.parent.resolve()
        if run_dir not in self._dir_to_csvs:
            self._dir_to_csvs[run_dir] = []
        if resolved_csv not in self._dir_to_csvs[run_dir]:
            self._dir_to_csvs[run_dir].append(resolved_csv)

    def get_csvs_for_directory(self, directory: str | Path) -> list[Path]:
        """Retrieves linked CSV paths for a directory, scanning disk only on cache miss."""
        target_dir = self._normalize_path(directory)

        # Check in-memory map for target_dir or any of its subfolders
        matching_csvs = [
            csv
            for run_dir, csvs in self._dir_to_csvs.items()
            if run_dir == target_dir or run_dir.is_relative_to(target_dir)
            for csv in csvs
        ]
        if matching_csvs:
            return matching_csvs

        if not target_dir.exists():
            raise FileNotFoundError(f"Directory '{target_dir}' does not exist.")
        if not target_dir.is_dir():
            raise NotADirectoryError(f"Path '{target_dir}' is not a directory.")

        # Cache miss: discover and link on demand
        for csv_path in self._iter_csvs(target_dir, max_depth=4):
            self._register_csv(csv_path)

        matching_csvs = [
            csv
            for run_dir, csvs in self._dir_to_csvs.items()
            if run_dir == target_dir or run_dir.is_relative_to(target_dir)
            for csv in csvs
        ]
        if not matching_csvs:
            raise FileNotFoundError(f"No CSV files found inside directory '{target_dir}'.")

        return matching_csvs

    def discover_from_latest_dir(self, num: int = 1) -> None:
        """Finds the latest `num` experiment directories and links all CSVs within them.

        Args:
            num: Number of recent experiment folders to inspect. Defaults to 1.

        Raises:
            ValueError: If `num` is less than 1.
            FileNotFoundError: If base_logs_dir has no experiment folders or no CSVs.
            OSError: If directory traversal fails.
        """
        if num < 1:
            raise ValueError(f"discover_latest() requires num >= 1, but got {num}.")

        self._validate_base_dir(self._base_logs_dir)

        try:
            with os.scandir(self._base_logs_dir) as it:
                latest_exp_dirs = heapq.nlargest(
                    num,
                    (entry for entry in it if entry.is_dir()),
                    key=self._safe_mtime,
                )
        except OSError as err:
            raise OSError(f"Failed to scan directory '{self._base_logs_dir}': {err}") from err

        if not latest_exp_dirs:
            raise FileNotFoundError(f"No experiment folders found inside '{self._base_logs_dir}'.")

        found_any = False
        for entry in latest_exp_dirs:
            for csv_path in self._iter_csvs(entry.path, max_depth=4):
                self._register_csv(csv_path)
                found_any = True

        if not found_any:
            raise FileNotFoundError(
                "No CSV metric files found inside any experiment folder in "
                f"'{self._base_logs_dir}'."
            )

    def discover_latest_metrics(self, num: int = 1) -> None:
        """Finds the latest `num` CSV metric files across experiment folders and links them.

        Scans top-level experiment folders newest-to-oldest and halts traversal
        as soon as `num` individual CSV files are discovered.

        Args:
            num: Number of recent metric CSV files to discover. Defaults to 1.

        Raises:
            ValueError: If `num` is less than 1.
            FileNotFoundError: If base_logs_dir has no experiment folders or no CSVs.
            OSError: If directory traversal fails.
        """
        if num < 1:
            raise ValueError(f"discover_latest_metrics() requires num >= 1, but got {num}.")

        self._validate_base_dir(self._base_logs_dir)

        try:
            with os.scandir(self._base_logs_dir) as it:
                top_dirs = [entry for entry in it if entry.is_dir()]
        except OSError as err:
            raise OSError(f"Failed to scan directory '{self._base_logs_dir}': {err}") from err

        if not top_dirs:
            raise FileNotFoundError(f"No experiment folders found inside '{self._base_logs_dir}'.")

        top_dirs.sort(key=self._safe_mtime, reverse=True)

        found_count = 0
        for entry in top_dirs:
            for csv_path in self._iter_csvs(entry.path, max_depth=4):
                self._register_csv(csv_path)
                found_count += 1
                if found_count >= num:
                    return

        if found_count == 0:
            raise FileNotFoundError(
                "No CSV metric files found inside any experiment folder in "
                f"'{self._base_logs_dir}'."
            )

    def discover_directory(self, directory_name: str | Path) -> None:
        """Scans a specific directory, linking all subdirectories and their CSVs.

        Args:
            directory_name: Folder name or relative/absolute path to search within.

        Raises:
            FileNotFoundError: If the directory does not exist or contains no CSV files.
            NotADirectoryError: If the target path is not a directory.
        """
        target_dir = self._normalize_path(directory_name)

        if not target_dir.exists():
            raise FileNotFoundError(f"Directory '{target_dir}' does not exist.")

        if not target_dir.is_dir():
            raise NotADirectoryError(f"Path '{target_dir}' is not a directory.")

        found_any = False
        for csv_path in self._iter_csvs(target_dir, max_depth=4):
            self._register_csv(csv_path)
            found_any = True

        if not found_any:
            raise FileNotFoundError(f"No CSV files found inside '{target_dir}'.")

    def parse_from_path(self, csv_or_dir_path: str | Path) -> ParsedMetrics:
        """Parses a single metric run from an explicit CSV file or run folder.

        If a directory is given, uses the linked (newest) CSV file.

        Args:
            csv_or_dir_path: Path to a CSV file or directory containing one.

        Returns:
            A ParsedMetrics instance.

        Raises:
            FileNotFoundError: If the target path or CSV does not exist.
        """
        path = self._normalize_path(csv_or_dir_path)

        if path.is_dir():
            csv_list = self.get_csvs_for_directory(path)
            target_csv = csv_list[0]
        else:
            target_csv = self._resolve_and_validate_path(path)

        return self._parse_file(target_csv)

    def parse(self, *directories: str | Path) -> list[ParsedMetrics]:
        """Parses all CSV metrics linked to the specified directories.

        Args:
            directories: One or more directory paths to parse.

        Returns:
            A list of ParsedMetrics instances for all CSVs in those directories.

        Raises:
            ValueError: If no directories are provided.
        """
        if not directories:
            raise ValueError("No directory paths provided to parse().")

        parsed_metrics: list[ParsedMetrics] = []
        for d in directories:
            csv_list = self.get_csvs_for_directory(d)
            parsed_metrics.extend(self._parse_file(csv_file) for csv_file in csv_list)

        return parsed_metrics

    def parse_all(self) -> list[ParsedMetrics]:
        """Parses all CSV metrics across all queued directories in memory.

        Returns:
            A list of ParsedMetrics instances corresponding to all linked CSVs.

        Raises:
            ValueError: If no directories are queued.
        """
        if not self._dir_to_csvs:
            raise ValueError(
                "No directories queued in LogParser. "
                "Call discover_latest() or discover_directory() first, "
                "or pass paths directly to parse()."
            )

        parsed_metrics: list[ParsedMetrics] = []
        for csv_list in self._dir_to_csvs.values():
            parsed_metrics.extend(self._parse_file(csv_file) for csv_file in csv_list)

        return parsed_metrics

    def _parse_file(self, csv_path: Path) -> ParsedMetrics:
        """Internal worker that validates, reshapes, and loads CSV into ParsedMetrics."""
        raw_df, resolved_path = self._load_csv(csv_path)

        if resolved_path.is_relative_to(self._base_logs_dir):
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
        exp_id = match.group(1) if match else exp_dir.name

        # Determine model and version relative to the experiment root folder
        model_label = ""
        rel_to_exp = resolved_path.relative_to(exp_dir)
        if len(rel_to_exp.parts) >= 3:
            candidate_model = rel_to_exp.parts[0]
            candidate_version = rel_to_exp.parts[1].replace("version_", "")
            model_label = f"_{candidate_model}_{candidate_version}"
        elif len(rel_to_exp.parts) == 2:
            candidate_version = rel_to_exp.parts[0].replace("version_", "")
            model_label = f"_{candidate_version}"

        csv_stem = resolved_path.stem
        file_suffix = f"_{csv_stem}" if csv_stem not in {"metrics", "log"} else ""

        raw_identifier = f"{exp_id}{model_label}{file_suffix}"
        run_identifier = re.sub(r'[\/:*?"<>|]', "_", raw_identifier)

        x_col = (
            "epoch"
            if "epoch" in raw_df.columns
            else ("step" if "step" in raw_df.columns else raw_df.columns[0])
        )

        tracking_cols = {"epoch", "step", x_col}
        metric_cols = [c for c in raw_df.columns if c not in tracking_cols]

        variables: list[str] = []
        for col in metric_cols:
            clean_var = col.split("/", 1)[1] if "/" in col else col
            if clean_var not in tracking_cols and clean_var not in variables:
                variables.append(clean_var)

        parsed = ParsedMetrics(
            x_col=x_col,
            csv_path=resolved_path,
            experiment_dir=exp_dir,
            experiment_name=run_identifier,
        )

        for var_name in variables:
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
        """Internal helper to validate file existence and extensions."""
        target_path = self._normalize_path(path)

        if target_path.suffix.lower() != ".csv":
            raise ValueError(f"File '{target_path}' is not a CSV file.")

        if not target_path.is_file():
            raise FileNotFoundError(f"CSV file '{target_path}' does not exist.")

        return target_path

    def _load_csv(self, path: str | Path) -> tuple[pd.DataFrame, Path]:
        """Internal helper to load raw CSV data into a DataFrame."""
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
